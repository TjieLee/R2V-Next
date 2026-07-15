"""Schema inspection and deterministic online-manifest construction utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

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
from ltx_trainer.online_data.media_decoder import probe_video


class ManifestReject(ValueError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    return str(value)


def read_annotation_rows(path: str | Path) -> list[dict[str, Any]]:
    annotation_path = Path(path).expanduser().resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")
    suffix = annotation_path.suffix.lower()
    if suffix == ".parquet":
        rows = pd.read_parquet(annotation_path).to_dict("records")
    elif suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in annotation_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        rows = list(payload.values()) if isinstance(payload, dict) else payload
    elif suffix == ".csv":
        with annotation_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"Unsupported annotation format: {annotation_path.suffix}")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"Annotation must contain object rows: {annotation_path}")
    return [{str(key): _json_safe(value) for key, value in row.items()} for row in rows]


def inspect_annotation(path: str | Path, *, sample_count: int = 3) -> dict[str, Any]:
    rows = read_annotation_rows(path)
    fields = sorted({key for row in rows for key in row})
    field_report: dict[str, Any] = {}
    for field in fields:
        values = [row.get(field) for row in rows]
        non_null = [value for value in values if value is not None]
        type_counts = Counter(type(value).__name__ for value in non_null)
        field_report[field] = {
            "types": dict(sorted(type_counts.items())),
            "null_fraction": (len(values) - len(non_null)) / max(1, len(values)),
        }
    role_terms = {
        "media_or_target": ("target", "tgt", "image", "video", "path"),
        "source_or_reference": ("source", "src", "reference", "ref"),
        "prompt_or_instruction": ("prompt", "instruction", "caption", "text"),
        "crop_or_face_cut": ("crop", "face_cut", "cut"),
    }
    role_candidates = {
        role: [field for field in fields if any(term in field.lower() for term in terms)]
        for role, terms in role_terms.items()
    }
    return {
        "path": str(Path(path).expanduser().resolve()),
        "row_count": len(rows),
        "fields": field_report,
        "available_columns": fields,
        "role_candidates_not_adapter_mappings": role_candidates,
        "examples": rows[:sample_count],
    }


def load_multitask_data_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), list):
        raise ValueError(f"Multi-task config must contain a datasets list: {config_path}")
    return payload


def parse_path_list(value: Any, *, field: str) -> list[str]:
    if isinstance(value, list):
        paths = [str(item) for item in value if str(item).strip()]
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            return parse_path_list(json.loads(stripped), field=field)
        paths = [part.strip() for part in stripped.replace(";", "|").split("|") if part.strip()]
    else:
        raise ValueError(f"Field {field!r} must contain a path list, got {type(value).__name__}")
    if not paths:
        raise ValueError(f"Field {field!r} contains no paths")
    return paths


def parse_numeric_list(value: Any, *, field: str) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"Field {field!r} must contain four numeric values")
    return [float(item) for item in value]


def parse_pair(value: Any, *, field: str) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Field {field!r} must contain two integer values")
    return [int(item) for item in value]


def resolve_media_path(value: Any, *, data_root: str | Path | None) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and data_root is not None:
        path = Path(data_root).expanduser() / path
    return str(path.resolve())


def stable_sample_key(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def finalize_manifest_record(record: dict[str, Any]) -> dict[str, Any]:
    plan_payload = {key: value for key, value in record.items() if key != "sample_plan_sha256"}
    record["sample_plan_sha256"] = stable_sample_key(plan_payload)
    return record


def validate_sample_plan_sha256(record: Mapping[str, Any]) -> None:
    saved = str(record.get("sample_plan_sha256", ""))
    expected = stable_sample_key({key: value for key, value in record.items() if key != "sample_plan_sha256"})
    if saved != expected:
        raise ValueError(f"sample_plan_sha256 mismatch for sample_key={record.get('sample_key')}")


def build_i2i_record(
    row: Mapping[str, Any],
    *,
    dataset_name: str,
    data_root: str | Path | None,
    target_field: str,
    reference_field: str,
    caption_field: str,
    crop_field: str | None = None,
) -> dict[str, Any]:
    available = sorted(row)
    required = [target_field, reference_field, caption_field]
    missing = [field for field in required if field not in row]
    if missing:
        raise ManifestReject(
            "i2i_schema_mismatch",
            f"I2I adapter fields are missing: {missing}. Available columns: {available}",
        )
    target_path = resolve_media_path(row[target_field], data_root=data_root)
    reference_paths = [
        resolve_media_path(value, data_root=data_root)
        for value in parse_path_list(row[reference_field], field=reference_field)
    ]
    crop_xyxy = None
    if crop_field is not None and row.get(crop_field) is not None:
        crop_xyxy = parse_numeric_list(row[crop_field], field=crop_field)
    identity = {
        "dataset_name": dataset_name,
        "task": IMAGE_TASK,
        "target_path": target_path,
        "reference_paths": reference_paths,
        "caption": str(row[caption_field]),
        "crop_xyxy": crop_xyxy,
        "face_cut": None,
    }
    sample_key = stable_sample_key(identity)
    return finalize_manifest_record(
        {
            "sample_key": sample_key,
            "dataset_name": dataset_name,
            "task": IMAGE_TASK,
            "target_modality": "image",
            "target_path": target_path,
            "reference_paths": reference_paths,
            "caption": str(row[caption_field]),
            "crop_xyxy": crop_xyxy,
            "face_cut": None,
            "original_fps": IMAGE_FPS,
            "target_fps": IMAGE_FPS,
            "target_num_frames": IMAGE_NUM_FRAMES,
            "target_width": TARGET_WIDTH,
            "target_height": TARGET_HEIGHT,
            "target_source_frame_indices": [0],
            "vlm_target_frame_indices": [0],
            "vlm_source_frame_indices": [0],
        }
    )


def build_r2v_record(
    row: Mapping[str, Any],
    *,
    dataset_name: str,
    data_root: str | Path | None,
    manifest_seed: int,
) -> dict[str, Any]:
    required = {"video_path", "text", "crop", "face_cut", "ref_images"}
    missing = sorted(required - row.keys())
    if missing:
        raise ManifestReject(
            "r2v_schema_mismatch",
            f"OpenS2V adapter fields are missing: {missing}. Available columns: {sorted(row)}",
        )
    target_path = resolve_media_path(row["video_path"], data_root=data_root)
    reference_paths = [
        resolve_media_path(value, data_root=data_root)
        for value in parse_path_list(row["ref_images"], field="ref_images")
    ]
    crop_raw = parse_numeric_list(row["crop"], field="crop")
    # OpenS2V stores (start_x, end_x, start_y, end_y).
    crop_xyxy = [crop_raw[0], crop_raw[2], crop_raw[1], crop_raw[3]]
    face_cut = parse_pair(row["face_cut"], field="face_cut")
    if face_cut[1] <= face_cut[0]:
        raise ManifestReject("invalid_face_cut", f"Invalid face_cut={face_cut} for {target_path}")
    header = probe_video(target_path)
    original_fps = float(header["fps"])
    if not math.isfinite(original_fps) or original_fps <= 0:
        raise ManifestReject("invalid_video_fps", f"Could not determine FPS for {target_path}")
    frame_count = int(header["frame_count"])
    end_frame = min(face_cut[1], frame_count) if frame_count > 0 else face_cut[1]
    identity = {
        "dataset_name": dataset_name,
        "task": VIDEO_TASK,
        "target_path": target_path,
        "reference_paths": reference_paths,
        "caption": str(row["text"]),
        "crop_xyxy": crop_xyxy,
        "face_cut": [face_cut[0], end_frame],
    }
    sample_key = stable_sample_key(identity)
    max_start = end_frame - 1 - round((VIDEO_NUM_FRAMES - 1) * original_fps / VIDEO_FPS)
    if max_start < face_cut[0]:
        raise ManifestReject(
            "insufficient_frames_for_121_at_24fps",
            f"No exact 121-frame/24-fps clip fits face_cut={[face_cut[0], end_frame]} at fps={original_fps}",
        )
    rng_seed = int(stable_sample_key({"sample_key": sample_key, "manifest_seed": manifest_seed})[:16], 16)
    sample_start = random.Random(rng_seed).randint(face_cut[0], max_start)
    source_indices = [round(sample_start + index * original_fps / VIDEO_FPS) for index in range(VIDEO_NUM_FRAMES)]
    if (
        len(source_indices) != VIDEO_NUM_FRAMES
        or len(source_indices) != len(set(source_indices))
        or any(left >= right for left, right in zip(source_indices, source_indices[1:]))
        or source_indices[0] < face_cut[0]
        or source_indices[-1] >= end_frame
    ):
        raise ManifestReject(
            "insufficient_frames_for_121_at_24fps",
            f"Exact source indices are not strictly increasing inside face_cut for {target_path}",
        )
    vlm_source_indices = [source_indices[index] for index in VLM_TARGET_INDICES]
    return finalize_manifest_record(
        {
            "sample_key": sample_key,
            "dataset_name": dataset_name,
            "task": VIDEO_TASK,
            "target_modality": "video",
            "target_path": target_path,
            "reference_paths": reference_paths,
            "caption": str(row["text"]),
            "crop_xyxy": crop_xyxy,
            "face_cut": [face_cut[0], end_frame],
            "original_fps": original_fps,
            "target_fps": VIDEO_FPS,
            "target_num_frames": VIDEO_NUM_FRAMES,
            "target_width": TARGET_WIDTH,
            "target_height": TARGET_HEIGHT,
            "target_source_frame_indices": source_indices,
            "vlm_target_frame_indices": list(VLM_TARGET_INDICES),
            "vlm_source_frame_indices": vlm_source_indices,
        }
    )


def deduplicate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record["sample_key"])
        if key in unique and unique[key]["sample_plan_sha256"] != record["sample_plan_sha256"]:
            raise ValueError(f"sample_key collision with different plans: {key}")
        unique[key] = record
    return [unique[key] for key in sorted(unique)]
