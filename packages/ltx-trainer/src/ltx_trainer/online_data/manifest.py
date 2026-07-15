"""Schema inspection and deterministic online-manifest construction utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import random
import warnings
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd
import yaml
from PIL import Image, ImageOps

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

logger = logging.getLogger(__name__)


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


def iter_annotation_rows(
    path: str | Path,
    *,
    batch_size: int = 4096,
) -> Iterator[dict[str, Any]]:
    """Yield normalized annotation rows while keeping memory bounded."""
    annotation_path = Path(path).expanduser().resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    suffix = annotation_path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("Streaming parquet annotations requires pyarrow") from exc
        parquet_file = parquet.ParquetFile(annotation_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield {str(key): _json_safe(value) for key, value in row.items()}
    elif suffix == ".jsonl":
        with annotation_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError(f"JSONL row {line_number} is not an object: {annotation_path}")
                yield {str(key): _json_safe(value) for key, value in row.items()}
    elif suffix == ".json":
        warnings.warn(
            f"JSON list/object annotations are loaded in memory; prefer JSONL for large files: {annotation_path}",
            RuntimeWarning,
            stacklevel=2,
        )
        with annotation_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.values() if isinstance(payload, dict) else payload
        if not isinstance(rows, (list, tuple)) and not hasattr(rows, "__iter__"):
            raise ValueError(f"JSON annotation must contain rows: {annotation_path}")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"JSON annotation contains a non-object row: {annotation_path}")
            yield {str(key): _json_safe(value) for key, value in row.items()}
    elif suffix == ".csv":
        with annotation_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                yield {str(key): _json_safe(value) for key, value in row.items()}
    else:
        raise ValueError(f"Unsupported annotation format: {annotation_path.suffix}")


def read_annotation_rows(path: str | Path) -> list[dict[str, Any]]:
    """Backward-compatible eager wrapper for callers that explicitly need a list."""
    return list(iter_annotation_rows(path))


def inspect_annotation(path: str | Path, *, sample_count: int = 3) -> dict[str, Any]:
    row_count = 0
    examples: list[dict[str, Any]] = []
    type_counts: dict[str, Counter[str]] = {}
    null_counts: dict[str, int] = {}
    for row in iter_annotation_rows(path):
        if len(examples) < sample_count:
            examples.append(row)
        existing_fields = set(type_counts)
        row_fields = set(row)
        for missing_field in existing_fields - row_fields:
            null_counts[missing_field] += 1
        for field, value in row.items():
            if field not in type_counts:
                type_counts[field] = Counter()
                null_counts[field] = row_count
            if value is None:
                null_counts[field] += 1
            else:
                type_counts[field][type(value).__name__] += 1
        row_count += 1
    fields = sorted(type_counts)
    field_report = {
        field: {
            "types": dict(sorted(type_counts[field].items())),
            "null_fraction": null_counts[field] / max(1, row_count),
        }
        for field in fields
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
        "row_count": row_count,
        "fields": field_report,
        "available_columns": fields,
        "role_candidates_not_adapter_mappings": role_candidates,
        "examples": examples,
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


def _require_nonempty_caption(value: Any) -> str:
    caption = str(value).strip() if value is not None else ""
    if not caption:
        raise ManifestReject("empty_caption", "Caption/instruction is empty")
    return caption


ImageValidator = Callable[[str], tuple[int, int]]


def _validate_reference_paths(
    paths: list[str],
    *,
    image_validator: ImageValidator | None = None,
) -> None:
    for path in paths:
        image_path = Path(path)
        try:
            if image_validator is not None:
                image_validator(str(image_path))
            else:
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                with Image.open(image_path) as image:
                    image.verify()
        except Exception as exc:
            raise ManifestReject("missing_reference", f"Reference image is unreadable: {image_path}") from exc


def _resolved_reference_paths(
    value: Any,
    *,
    field: str,
    data_root: str | Path | None,
) -> list[str]:
    try:
        paths = parse_path_list(value, field=field)
    except Exception as exc:
        raise ManifestReject("missing_reference", f"No usable reference paths in field {field!r}") from exc
    return [resolve_media_path(path, data_root=data_root) for path in paths]


def _validate_crop_xyxy(
    crop: list[float] | None,
    *,
    width: int,
    height: int,
) -> None:
    if crop is None:
        return
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ManifestReject(
            "invalid_crop",
            f"crop_xyxy={crop} is outside media size {width}x{height}",
        )


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
    image_validator: ImageValidator | None = None,
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
    if image_validator is None and not Path(target_path).is_file():
        raise ManifestReject("missing_target", f"Target image does not exist: {target_path}")
    reference_paths = _resolved_reference_paths(
        row[reference_field],
        field=reference_field,
        data_root=data_root,
    )
    crop_xyxy = None
    if crop_field is not None and row.get(crop_field) is not None:
        try:
            crop_xyxy = parse_numeric_list(row[crop_field], field=crop_field)
        except Exception as exc:
            raise ManifestReject("invalid_crop", f"Invalid crop field {crop_field!r}") from exc
    caption = _require_nonempty_caption(row[caption_field])
    _validate_reference_paths(reference_paths, image_validator=image_validator)
    try:
        if image_validator is not None:
            target_width, target_height = image_validator(target_path)
        else:
            with Image.open(target_path) as target_image:
                target_image.verify()
            with Image.open(target_path) as target_image:
                target_width, target_height = ImageOps.exif_transpose(target_image).size
    except Exception as exc:
        raise ManifestReject("missing_target", f"Target image is unreadable: {target_path}") from exc
    _validate_crop_xyxy(crop_xyxy, width=target_width, height=target_height)
    identity = {
        "dataset_name": dataset_name,
        "task": IMAGE_TASK,
        "target_path": target_path,
        "reference_paths": reference_paths,
        "caption": caption,
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
            "caption": caption,
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
    video_header: Mapping[str, float | int] | None = None,
    image_validator: ImageValidator | None = None,
    target_path_validated: bool = False,
) -> dict[str, Any]:
    required = {"video_path", "text", "crop", "face_cut", "ref_images"}
    missing = sorted(required - row.keys())
    if missing:
        raise ManifestReject(
            "r2v_schema_mismatch",
            f"OpenS2V adapter fields are missing: {missing}. Available columns: {sorted(row)}",
        )
    target_path = resolve_media_path(row["video_path"], data_root=data_root)
    if not target_path_validated and not Path(target_path).is_file():
        raise ManifestReject("missing_target", f"Target video does not exist: {target_path}")
    reference_paths = _resolved_reference_paths(
        row["ref_images"],
        field="ref_images",
        data_root=data_root,
    )
    try:
        crop_raw = parse_numeric_list(row["crop"], field="crop")
    except Exception as exc:
        raise ManifestReject("invalid_crop", "Invalid OpenS2V crop") from exc
    # OpenS2V stores (start_x, end_x, start_y, end_y).
    crop_xyxy = [crop_raw[0], crop_raw[2], crop_raw[1], crop_raw[3]]
    try:
        face_cut = parse_pair(row["face_cut"], field="face_cut")
    except Exception as exc:
        raise ManifestReject("invalid_face_cut", "Invalid OpenS2V face_cut") from exc
    if face_cut[1] <= face_cut[0]:
        raise ManifestReject("invalid_face_cut", f"Invalid face_cut={face_cut} for {target_path}")
    caption = _require_nonempty_caption(row["text"])
    _validate_reference_paths(reference_paths, image_validator=image_validator)
    try:
        header = dict(video_header) if video_header is not None else probe_video(target_path)
        original_fps = float(header["fps"])
        frame_count = int(header["frame_count"])
        video_width = int(header["width"])
        video_height = int(header["height"])
    except Exception as exc:
        raise ManifestReject("invalid_video_header", f"Could not read video header: {target_path}") from exc
    if not math.isfinite(original_fps) or original_fps <= 0:
        raise ManifestReject("invalid_video_header", f"Could not determine FPS for {target_path}")
    if frame_count <= 0 or video_width <= 0 or video_height <= 0:
        raise ManifestReject("invalid_video_header", f"Incomplete video header for {target_path}: {header}")
    _validate_crop_xyxy(crop_xyxy, width=video_width, height=video_height)
    if not (0 <= face_cut[0] < face_cut[1] <= frame_count):
        raise ManifestReject("invalid_face_cut", f"Invalid face_cut={face_cut} for frame_count={frame_count}")
    end_frame = face_cut[1]
    identity = {
        "dataset_name": dataset_name,
        "task": VIDEO_TASK,
        "target_path": target_path,
        "reference_paths": reference_paths,
        "caption": caption,
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
            "caption": caption,
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
