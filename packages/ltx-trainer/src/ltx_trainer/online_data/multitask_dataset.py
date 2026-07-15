"""CPU-only manifest dataset for deterministic I2I and R2V online training."""

from __future__ import annotations

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
    IMAGE_TASK,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VIDEO_NUM_FRAMES,
    VIDEO_TASK,
)
from ltx_trainer.online_data.media_decoder import decode_image_rgb, decode_video_indices
from ltx_trainer.online_data.manifest_index import (
    default_manifest_index_path,
    read_jsonl_record_at,
    read_manifest_index,
)
from ltx_trainer.online_data.manifest_schema import (
    as_int_list as _as_int_list,
    validate_manifest_record,
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
        cpu_transform_chunk_frames: int = 4,
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
        self.cpu_transform_chunk_frames = int(cpu_transform_chunk_frames)
        self._manifest_handle: BinaryIO | None = None
        index_path = default_manifest_index_path(self.manifest_path)
        if index_path.is_file():
            self._offsets, self.task_indices = read_manifest_index(
                index_path,
                manifest_path=self.manifest_path,
            )
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
                    chunk_frames=self.cpu_transform_chunk_frames,
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
            chunk_frames=self.cpu_transform_chunk_frames,
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
