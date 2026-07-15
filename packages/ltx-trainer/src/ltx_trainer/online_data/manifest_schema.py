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
    VLM_TARGET_INDICES,
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
    target_indices = as_int_list(
        record["target_source_frame_indices"],
        field="target_source_frame_indices",
    )
    vlm_indices = as_int_list(
        record["vlm_target_frame_indices"],
        field="vlm_target_frame_indices",
    )
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
        if len(target_indices) != VIDEO_NUM_FRAMES or any(
            left >= right for left, right in zip(target_indices, target_indices[1:])
        ):
            raise ValueError(f"R2V manifest row {index} source indices must be 121 strictly increasing values")
        if tuple(vlm_indices) != VLM_TARGET_INDICES:
            raise ValueError(f"R2V manifest row {index} has invalid VLM target indices: {vlm_indices}")
        vlm_source_indices = as_int_list(
            record["vlm_source_frame_indices"],
            field="vlm_source_frame_indices",
        )
        expected_source_indices = [target_indices[target_index] for target_index in VLM_TARGET_INDICES]
        if vlm_source_indices != expected_source_indices:
            raise ValueError(f"R2V manifest row {index} has inconsistent VLM source indices")
