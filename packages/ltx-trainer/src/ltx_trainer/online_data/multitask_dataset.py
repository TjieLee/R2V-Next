"""CPU-only manifest dataset for deterministic I2I and R2V online training."""

from __future__ import annotations

import hashlib
import json
import time
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ltx_trainer.online_data.constants import (
    IMAGE_FPS,
    IMAGE_NUM_FRAMES,
    IMAGE_TASK,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VIDEO_FPS,
    VIDEO_NUM_FRAMES,
    VIDEO_TASK,
    VLM_TARGET_INDICES,
)
from ltx_trainer.online_data.media_decoder import decode_image_rgb, decode_video_indices
from ltx_trainer.online_data.manifest_index import (
    default_manifest_index_path,
    read_jsonl_record_at,
    read_manifest_index,
)
from ltx_trainer.online_data.transforms import deterministic_resize_center_crop


@dataclass(frozen=True)
class SampleLoadError:
    sample_key: str
    task: str
    error_type: str
    message: str
    manifest_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_int_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"Manifest field {field!r} must be a list")
    return [int(item) for item in value]


def _validate_sample_plan_sha256(record: dict[str, Any]) -> None:
    payload = {key: value for key, value in record.items() if key != "sample_plan_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if str(record.get("sample_plan_sha256", "")) != expected:
        raise ValueError(f"sample_plan_sha256 mismatch for sample_key={record.get('sample_key')}")


def validate_manifest_record(record: dict[str, Any], index: int) -> None:
    required = {
        "sample_key",
        "sample_plan_sha256",
        "dataset_name",
        "task",
        "target_modality",
        "target_path",
        "reference_paths",
        "caption",
        "target_fps",
        "target_num_frames",
        "target_width",
        "target_height",
        "target_source_frame_indices",
        "vlm_target_frame_indices",
        "vlm_source_frame_indices",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"Manifest row {index} is missing required fields: {missing}")
    task = record["task"]
    if task not in {IMAGE_TASK, VIDEO_TASK}:
        raise ValueError(f"Manifest row {index} has unsupported task {task!r}")
    _validate_sample_plan_sha256(record)
    if int(record["target_width"]) != TARGET_WIDTH or int(record["target_height"]) != TARGET_HEIGHT:
        raise ValueError(f"Manifest row {index} must target {TARGET_WIDTH}x{TARGET_HEIGHT}")
    if not isinstance(record["reference_paths"], list) or not record["reference_paths"]:
        raise ValueError(f"Manifest row {index} must contain at least one reference path")
    target_indices = _as_int_list(record["target_source_frame_indices"], field="target_source_frame_indices")
    vlm_indices = _as_int_list(record["vlm_target_frame_indices"], field="vlm_target_frame_indices")
    if task == IMAGE_TASK:
        if record["target_modality"] != "image" or int(record["target_num_frames"]) != IMAGE_NUM_FRAMES:
            raise ValueError(f"I2I manifest row {index} must be a one-frame image")
        if float(record["target_fps"]) != IMAGE_FPS or target_indices != [0] or vlm_indices != [0]:
            raise ValueError(f"I2I manifest row {index} has invalid frame/fps metadata")
    else:
        if record["target_modality"] != "video" or int(record["target_num_frames"]) != VIDEO_NUM_FRAMES:
            raise ValueError(f"R2V manifest row {index} must contain exactly {VIDEO_NUM_FRAMES} frames")
        if float(record["target_fps"]) != VIDEO_FPS:
            raise ValueError(f"R2V manifest row {index} must use target_fps={VIDEO_FPS}")
        if len(target_indices) != VIDEO_NUM_FRAMES or any(a >= b for a, b in zip(target_indices, target_indices[1:])):
            raise ValueError(f"R2V manifest row {index} source indices must be 121 strictly increasing values")
        if tuple(vlm_indices) != VLM_TARGET_INDICES:
            raise ValueError(f"R2V manifest row {index} has invalid VLM target indices: {vlm_indices}")
        vlm_source_indices = _as_int_list(record["vlm_source_frame_indices"], field="vlm_source_frame_indices")
        expected_source_indices = [target_indices[target_index] for target_index in VLM_TARGET_INDICES]
        if vlm_source_indices != expected_source_indices:
            raise ValueError(f"R2V manifest row {index} has inconsistent VLM source indices")


class OnlineMultiTaskDataset(Dataset[dict[str, Any] | SampleLoadError]):
    """Read finalized online manifests without runtime random sampling."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        width: int = TARGET_WIDTH,
        height: int = TARGET_HEIGHT,
        max_ref_images: int | None = 4,
        return_load_errors: bool = True,
        vlm_reference_preprocess: Literal["original", "target_crop"] = "original",
        video_decoder: Literal["pyav", "opencv"] = "pyav",
        decode_timeout_seconds: float = 120.0,
        require_all_tasks: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Online manifest does not exist: {self.manifest_path}")
        self.width = int(width)
        self.height = int(height)
        self.max_ref_images = max_ref_images
        self.return_load_errors = return_load_errors
        self.vlm_reference_preprocess = vlm_reference_preprocess
        self.video_decoder = video_decoder
        self.decode_timeout_seconds = float(decode_timeout_seconds)
        self._manifest_handle: BinaryIO | None = None
        index_path = default_manifest_index_path(self.manifest_path)
        if index_path.is_file():
            self._offsets, self.task_indices = read_manifest_index(index_path)
        else:
            self._offsets, self.task_indices = self._scan_manifest_offsets()
        if not self._offsets:
            raise ValueError(f"Online manifest is empty: {self.manifest_path}")
        for task, indices in self.task_indices.items():
            if require_all_tasks and not indices:
                raise ValueError(f"Online manifest contains no {task!r} samples")

    def __len__(self) -> int:
        return len(self._offsets)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_manifest_handle"] = None
        return state

    def _scan_manifest_offsets(self) -> tuple[array, dict[str, array]]:
        offsets = array("Q")
        task_indices = {IMAGE_TASK: array("q"), VIDEO_TASK: array("q")}
        with self.manifest_path.open("rb") as handle:
            row_index = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                validate_manifest_record(record, row_index)
                offsets.append(offset)
                task_indices[str(record["task"])].append(row_index)
                row_index += 1
        return offsets, task_indices

    def _read_record(self, index: int) -> dict[str, Any]:
        if self._manifest_handle is None or self._manifest_handle.closed:
            self._manifest_handle = self.manifest_path.open("rb")
        record = read_jsonl_record_at(self._manifest_handle, self._offsets[index])
        validate_manifest_record(record, index)
        return record

    def __getitem__(self, index: int) -> dict[str, Any] | SampleLoadError:
        index = int(index)
        record = self._read_record(index)
        started = time.perf_counter()
        try:
            target = self._load_target(record)
            reference_paths = list(record["reference_paths"])
            if self.max_ref_images is not None:
                reference_paths = reference_paths[: self.max_ref_images]
            original_references = [decode_image_rgb(path) for path in reference_paths]
            references_vae = [
                deterministic_resize_center_crop(
                    reference.unsqueeze(0),
                    target_height=self.height,
                    target_width=self.width,
                )[0]
                for reference in original_references
            ]
            references_vlm = (
                original_references
                if self.vlm_reference_preprocess == "original"
                else references_vae
            )
            target_indices = torch.tensor(record["target_source_frame_indices"], dtype=torch.long)
            vlm_target_indices = torch.tensor(record["vlm_target_frame_indices"], dtype=torch.long)
            return {
                "sample_key": str(record["sample_key"]),
                "sample_plan_sha256": str(record["sample_plan_sha256"]),
                "task": str(record["task"]),
                "target_modality": str(record["target_modality"]),
                "target_pixels": target,
                "reference_pixels_vae": references_vae,
                "reference_images_vlm": references_vlm,
                "reference_pixels": references_vae,
                "vlm_reference_preprocess": self.vlm_reference_preprocess,
                "video_decoder": self.video_decoder,
                "caption": str(record["caption"]),
                "target_fps": float(record["target_fps"]),
                "target_num_frames": int(record["target_num_frames"]),
                "target_source_frame_indices": target_indices,
                "vlm_target_frame_indices": vlm_target_indices,
                "vlm_source_frame_indices": torch.tensor(record["vlm_source_frame_indices"], dtype=torch.long),
                "reference_paths": [str(path) for path in reference_paths],
                "manifest_index": int(index),
                "data_decode_ms": (time.perf_counter() - started) * 1000.0,
            }
        except Exception as exc:
            if not self.return_load_errors:
                raise
            return SampleLoadError(
                sample_key=str(record.get("sample_key", f"row-{index}")),
                task=str(record.get("task", "unknown")),
                error_type=type(exc).__name__,
                message=str(exc),
                manifest_index=int(index),
            )

    def _load_target(self, record: dict[str, Any]) -> Tensor:
        if record["task"] == IMAGE_TASK:
            frames = decode_image_rgb(record["target_path"]).unsqueeze(0)
        else:
            source_indices = _as_int_list(
                record["target_source_frame_indices"], field="target_source_frame_indices"
            )
            if len(source_indices) != VIDEO_NUM_FRAMES:
                raise ValueError(
                    f"insufficient_frames_for_121_at_24fps: planned {len(source_indices)} frames"
                )
            frames = decode_video_indices(
                record["target_path"],
                source_indices,
                decoder=self.video_decoder,
                timeout_seconds=self.decode_timeout_seconds,
            )
        return deterministic_resize_center_crop(
            frames,
            target_height=self.height,
            target_width=self.width,
            crop_xyxy=record.get("crop_xyxy"),
        )


def collate_online_raw_batch(samples: Sequence[dict[str, Any] | SampleLoadError]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty online batch")
    errors = [sample for sample in samples if isinstance(sample, SampleLoadError)]
    if errors:
        return {"sample_load_errors": errors}
    records = [sample for sample in samples if isinstance(sample, dict)]
    tasks = {record["task"] for record in records}
    if len(tasks) != 1:
        raise ValueError(f"An online microbatch must contain one task/modality, got {sorted(tasks)}")
    target_shapes = {tuple(record["target_pixels"].shape) for record in records}
    if len(target_shapes) != 1:
        raise ValueError(f"Online target tensors in a microbatch must share shape, got {sorted(target_shapes)}")
    return {
        "sample_key": [record["sample_key"] for record in records],
        "sample_plan_sha256": [record["sample_plan_sha256"] for record in records],
        "task": [record["task"] for record in records],
        "target_modality": [record["target_modality"] for record in records],
        "target_pixels": torch.stack([record["target_pixels"] for record in records], dim=0),
        "reference_pixels_vae": [record["reference_pixels_vae"] for record in records],
        "reference_images_vlm": [record["reference_images_vlm"] for record in records],
        "reference_pixels": [record["reference_pixels_vae"] for record in records],
        "vlm_reference_preprocess": [record["vlm_reference_preprocess"] for record in records],
        "video_decoder": [record["video_decoder"] for record in records],
        "caption": [record["caption"] for record in records],
        "target_fps": torch.tensor([record["target_fps"] for record in records], dtype=torch.float32),
        "target_num_frames": torch.tensor(
            [record["target_num_frames"] for record in records], dtype=torch.long
        ),
        "target_source_frame_indices": torch.stack(
            [record["target_source_frame_indices"] for record in records], dim=0
        ),
        "vlm_target_frame_indices": torch.stack(
            [record["vlm_target_frame_indices"] for record in records], dim=0
        ),
        "vlm_source_frame_indices": torch.stack(
            [record["vlm_source_frame_indices"] for record in records], dim=0
        ),
        "reference_paths": [record["reference_paths"] for record in records],
        "manifest_index": torch.tensor([record["manifest_index"] for record in records], dtype=torch.long),
        "data_decode_ms": torch.tensor([record["data_decode_ms"] for record in records], dtype=torch.float32),
    }
