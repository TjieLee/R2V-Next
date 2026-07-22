"""Dependency-light schema validation for finalized online manifest rows."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ltx_trainer.online_data.constants import (
    IMAGE_FPS,
    IMAGE_NUM_FRAMES,
    IMAGE_TASK,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VIDEO_FPS,
    VIDEO_NUM_FRAMES,
    VIDEO_TASK,
)


def as_int_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"Manifest field {field!r} must be a list")
    return [int(item) for item in value]


def _validate_sample_plan_sha256(record: dict[str, Any]) -> None:
    payload = {key: value for key, value in record.items() if key != "sample_plan_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if str(record.get("sample_plan_sha256", "")) != expected:
        raise ValueError(f"sample_plan_sha256 mismatch for sample_key={record.get('sample_key')}")


def _require_str(record: dict[str, Any], field: str, index: int) -> str:
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest row {index} field {field!r} must be a non-empty string")
    return value


def _strictly_increasing(values: list[int]) -> bool:
    return bool(values) and all(left < right for left, right in zip(values, values[1:]))


def validate_manifest_record(record: dict[str, Any], index: int) -> None:
    required = {
        "sample_key",
        "sample_plan_sha256",
        "dataset_name",
        "adapter_name",
        "source_record_id",
        "task",
        "target_modality",
        "target_path",
        "reference_paths",
        "caption",
        "crop_xyxy",
        "clip_start_frame",
        "clip_end_frame",
        "original_fps",
        "target_fps",
        "target_num_frames",
        "target_width",
        "target_height",
        "target_source_frame_indices",
        "semantic_anchor_target_indices",
        "semantic_anchor_source_indices",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"Manifest row {index} is missing required fields: {missing}")
    forbidden = {"vlm_target_frame_indices", "vlm_source_frame_indices"} & record.keys()
    if forbidden:
        raise ValueError(f"Manifest row {index} contains removed fields: {sorted(forbidden)}")

    for field in ("sample_key", "dataset_name", "adapter_name", "source_record_id", "target_path"):
        _require_str(record, field, index)
    task = _require_str(record, "task", index)
    if task not in {IMAGE_TASK, VIDEO_TASK}:
        raise ValueError(f"Manifest row {index} has unsupported task {task!r}")
    _validate_sample_plan_sha256(record)

    if int(record["target_width"]) != TARGET_WIDTH or int(record["target_height"]) != TARGET_HEIGHT:
        raise ValueError(f"Manifest row {index} must target {TARGET_WIDTH}x{TARGET_HEIGHT}")
    if not isinstance(record["reference_paths"], list) or not record["reference_paths"]:
        raise ValueError(f"Manifest row {index} must contain at least one reference path")
    if any(not isinstance(path, str) or not path for path in record["reference_paths"]):
        raise ValueError(f"Manifest row {index} contains an invalid reference path")

    target_indices = as_int_list(record["target_source_frame_indices"], field="target_source_frame_indices")
    anchor_target_indices = as_int_list(
        record["semantic_anchor_target_indices"],
        field="semantic_anchor_target_indices",
    )
    anchor_source_indices = as_int_list(
        record["semantic_anchor_source_indices"],
        field="semantic_anchor_source_indices",
    )
    if len(anchor_target_indices) != len(anchor_source_indices):
        raise ValueError(f"Manifest row {index} semantic anchor index lengths differ")

    if task == IMAGE_TASK:
        if record["target_modality"] != "image" or int(record["target_num_frames"]) != IMAGE_NUM_FRAMES:
            raise ValueError(f"I2I manifest row {index} must be a one-frame image")
        if float(record["target_fps"]) != IMAGE_FPS or target_indices != [0]:
            raise ValueError(f"I2I manifest row {index} has invalid frame/fps metadata")
        if anchor_target_indices != [0] or anchor_source_indices != [0]:
            raise ValueError(f"I2I manifest row {index} must use the single semantic anchor [0]")
        return

    if record["target_modality"] != "video" or int(record["target_num_frames"]) != VIDEO_NUM_FRAMES:
        raise ValueError(f"R2V manifest row {index} must contain exactly {VIDEO_NUM_FRAMES} frames")
    if float(record["target_fps"]) != VIDEO_FPS:
        raise ValueError(f"R2V manifest row {index} must use target_fps={VIDEO_FPS}")
    if len(target_indices) != VIDEO_NUM_FRAMES or not _strictly_increasing(target_indices):
        raise ValueError(
            f"R2V manifest row {index} source indices must be "
            f"{VIDEO_NUM_FRAMES} strictly increasing values"
        )
    if not _strictly_increasing(anchor_target_indices):
        raise ValueError(f"R2V manifest row {index} target semantic anchors must be strictly increasing")
    if not _strictly_increasing(anchor_source_indices):
        raise ValueError(f"R2V manifest row {index} source semantic anchors must be strictly increasing")
    if anchor_target_indices[0] != 0 or anchor_target_indices[-1] != VIDEO_NUM_FRAMES - 1:
        raise ValueError(f"R2V manifest row {index} semantic anchors must include first and last target frames")
    if any(target_index < 0 or target_index >= VIDEO_NUM_FRAMES for target_index in anchor_target_indices):
        raise ValueError(f"R2V manifest row {index} contains an out-of-range target semantic anchor")
    expected_sources = [target_indices[target_index] for target_index in anchor_target_indices]
    if anchor_source_indices != expected_sources:
        raise ValueError(f"R2V manifest row {index} semantic source anchors do not match the target plan")
