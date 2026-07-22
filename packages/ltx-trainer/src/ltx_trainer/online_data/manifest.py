"""Schema inspection and deterministic canonical online-manifest construction."""

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

from ltx_trainer.online_data.adapters import (
    AdapterReject,
    CanonicalR2VSource,
    get_r2v_adapter,
)
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
from ltx_trainer.online_data.media_decoder import probe_video

logger = logging.getLogger(__name__)


class ManifestReject(ValueError):
    def __init__(self, reason: str, message: str) -> None:
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


def _explicit_row_id(row: Mapping[str, Any], fallback: int | str) -> str:
    for field in ("source_record_id", "key", "id"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return str(fallback)


def iter_annotation_items(
    path: str | Path,
    *,
    batch_size: int = 4096,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(row_id,row)`` for JSONL/list/dict, parquet, and CSV sources."""
    annotation_path = Path(path).expanduser().resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    suffix = annotation_path.suffix.lower()
    if suffix == ".parquet":
        try:
            from pyarrow import parquet  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("Streaming parquet annotations requires pyarrow") from exc
        global_index = 0
        parquet_file = parquet.ParquetFile(annotation_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for raw_row in batch.to_pylist():
                row = {str(key): _json_safe(value) for key, value in raw_row.items()}
                yield _explicit_row_id(row, global_index), row
                global_index += 1
        return
    if suffix == ".jsonl":
        with annotation_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw_row = json.loads(line)
                if not isinstance(raw_row, Mapping):
                    raise ValueError(f"JSONL row {line_number} is not an object: {annotation_path}")
                row = {str(key): _json_safe(value) for key, value in raw_row.items()}
                yield _explicit_row_id(row, line_number), row
        return
    if suffix == ".json":
        warnings.warn(
            f"JSON list/object annotations are loaded in memory; prefer JSONL for large files: {annotation_path}",
            RuntimeWarning,
            stacklevel=2,
        )
        with annotation_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping):
            iterator = payload.items()
        elif isinstance(payload, list):
            iterator = enumerate(payload)
        else:
            raise ValueError(f"JSON annotation must be a list or dict-of-records: {annotation_path}")
        for fallback_id, raw_row in iterator:
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"JSON annotation contains a non-object row: {annotation_path}")
            row = {str(key): _json_safe(value) for key, value in raw_row.items()}
            yield _explicit_row_id(row, fallback_id), row
        return
    if suffix == ".csv":
        with annotation_path.open("r", encoding="utf-8", newline="") as handle:
            for index, raw_row in enumerate(csv.DictReader(handle)):
                row = {str(key): _json_safe(value) for key, value in raw_row.items()}
                yield _explicit_row_id(row, index), row
        return
    raise ValueError(f"Unsupported annotation format: {annotation_path.suffix}")


def iter_annotation_rows(path: str | Path, *, batch_size: int = 4096) -> Iterator[dict[str, Any]]:
    for _row_id, row in iter_annotation_items(path, batch_size=batch_size):
        yield row


def read_annotation_rows(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_annotation_rows(path))


def inspect_annotation(path: str | Path, *, sample_count: int = 3) -> dict[str, Any]:
    row_count = 0
    examples: list[dict[str, Any]] = []
    type_counts: dict[str, Counter[str]] = {}
    null_counts: dict[str, int] = {}
    for _row_id, row in iter_annotation_items(path):
        if len(examples) < sample_count:
            examples.append(row)
        for missing_field in set(type_counts) - set(row):
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
    return {
        "path": str(Path(path).expanduser().resolve()),
        "row_count": row_count,
        "fields": {
            field: {
                "types": dict(sorted(type_counts[field].items())),
                "null_fraction": null_counts[field] / max(1, row_count),
            }
            for field in fields
        },
        "available_columns": fields,
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


def resolve_media_path(value: Any, *, data_root: str | Path | None) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and data_root is not None:
        path = Path(data_root).expanduser() / path
    return str(path.resolve())


ImageValidator = Callable[[str], tuple[int, int]]


def _validate_reference_paths(
    paths: list[str],
    *,
    image_validator: ImageValidator | None = None,
    reason: str = "missing_reference",
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
            raise ManifestReject(reason, f"Reference image is unreadable: {image_path}") from exc


def _validate_crop_xyxy(crop: list[float] | None, *, width: int, height: int) -> None:
    if crop is None:
        return
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ManifestReject("invalid_crop", f"crop_xyxy={crop} is outside media size {width}x{height}")


def stable_sample_key(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def finalize_manifest_record(record: dict[str, Any]) -> dict[str, Any]:
    record["sample_plan_sha256"] = stable_sample_key(
        {key: value for key, value in record.items() if key != "sample_plan_sha256"}
    )
    return record


def validate_sample_plan_sha256(record: Mapping[str, Any]) -> None:
    saved = str(record.get("sample_plan_sha256", ""))
    expected = stable_sample_key({key: value for key, value in record.items() if key != "sample_plan_sha256"})
    if saved != expected:
        raise ValueError(f"sample_plan_sha256 mismatch for sample_key={record.get('sample_key')}")


def uniform_integer_indices(*, start: int, end: int, count: int) -> list[int]:
    if count <= 0 or end < start:
        raise ValueError("uniform integer sampling requires count>0 and end>=start")
    if count == 1:
        return [start]
    if count > end - start + 1:
        raise ValueError("uniform integer sampling count exceeds inclusive range")
    result = [round(start + index * (end - start) / (count - 1)) for index in range(count)]
    if result[0] != start or result[-1] != end or any(a >= b for a, b in zip(result, result[1:])):
        raise ValueError(f"uniform integer sampling failed for start={start}, end={end}, count={count}")
    return result


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
    missing = [field for field in (target_field, reference_field, caption_field) if field not in row]
    if missing:
        raise ManifestReject("i2i_schema_mismatch", f"I2I adapter fields are missing: {missing}")
    target_path = resolve_media_path(row[target_field], data_root=data_root)
    reference_paths = [
        resolve_media_path(path, data_root=data_root)
        for path in parse_path_list(row[reference_field], field=reference_field)
    ]
    caption = str(row[caption_field] or "").strip()
    if not caption:
        raise ManifestReject("empty_caption", "Caption/instruction is empty")
    crop_xyxy = None
    if crop_field is not None and row.get(crop_field) is not None:
        crop_xyxy = parse_numeric_list(row[crop_field], field=crop_field)
    _validate_reference_paths(reference_paths, image_validator=image_validator)
    try:
        if image_validator is not None:
            target_width, target_height = image_validator(target_path)
        else:
            with Image.open(target_path) as image:
                image.verify()
            with Image.open(target_path) as image:
                target_width, target_height = ImageOps.exif_transpose(image).size
    except Exception as exc:
        raise ManifestReject("missing_target", f"Target image is unreadable: {target_path}") from exc
    _validate_crop_xyxy(crop_xyxy, width=target_width, height=target_height)
    source_record_id = str(row.get("source_record_id", row.get("key", target_path)))
    identity = {
        "dataset_name": dataset_name,
        "source_record_id": source_record_id,
        "task": IMAGE_TASK,
        "target_path": target_path,
        "reference_paths": reference_paths,
        "caption": caption,
        "crop_xyxy": crop_xyxy,
    }
    return finalize_manifest_record(
        {
            "sample_key": stable_sample_key(identity),
            "dataset_name": dataset_name,
            "adapter_name": "i2i",
            "source_record_id": source_record_id,
            "task": IMAGE_TASK,
            "target_modality": "image",
            "target_path": target_path,
            "reference_paths": reference_paths,
            "caption": caption,
            "crop_xyxy": crop_xyxy,
            "clip_start_frame": 0,
            "clip_end_frame": 1,
            "original_fps": IMAGE_FPS,
            "target_fps": IMAGE_FPS,
            "target_num_frames": IMAGE_NUM_FRAMES,
            "target_width": TARGET_WIDTH,
            "target_height": TARGET_HEIGHT,
            "target_source_frame_indices": [0],
            "semantic_anchor_target_indices": [0],
            "semantic_anchor_source_indices": [0],
        }
    )


def normalize_r2v_source(
    row: Mapping[str, Any],
    *,
    row_id: str,
    dataset_name: str,
    dataset_type: str,
    data_root: str | Path | None,
    adapter_config: Mapping[str, Any] | None,
    manifest_seed: int,
) -> CanonicalR2VSource:
    config = dict(adapter_config or {})
    config["manifest_seed"] = manifest_seed
    try:
        return get_r2v_adapter(dataset_type).normalize(
            row,
            row_id=row_id,
            dataset_name=dataset_name,
            data_root=data_root,
            config=config,
        )
    except AdapterReject as exc:
        raise ManifestReject(exc.reason, str(exc)) from exc


def build_canonical_r2v_record(
    canonical: CanonicalR2VSource,
    *,
    manifest_seed: int,
    anchor_frame_ratio: float = 0.10,
    video_header: Mapping[str, float | int] | None = None,
    image_validator: ImageValidator | None = None,
    target_path_validated: bool = False,
) -> dict[str, Any]:
    target_path = canonical.video_path
    missing_target_reason = "phantom_missing_target" if canonical.adapter_name == "phantom" else "missing_target"
    if not target_path_validated and not Path(target_path).is_file():
        raise ManifestReject(missing_target_reason, f"Target video does not exist: {target_path}")
    try:
        header = dict(video_header) if video_header is not None else probe_video(target_path)
        original_fps = float(header["fps"])
        frame_count = int(header["frame_count"])
        video_width = int(header["width"])
        video_height = int(header["height"])
    except Exception as exc:
        raise ManifestReject("invalid_video_header", f"Could not read video header: {target_path}") from exc
    if not math.isfinite(original_fps) or original_fps <= 0 or frame_count <= 0:
        raise ManifestReject("invalid_video_header", f"Incomplete video header for {target_path}: {header}")
    if original_fps < VIDEO_FPS:
        raise ManifestReject(
            "source_fps_below_24",
            f"Source FPS {original_fps} is below required target FPS {VIDEO_FPS}: {target_path}",
        )
    _validate_crop_xyxy(canonical.crop_xyxy, width=video_width, height=video_height)
    clip_start = canonical.clip_start_frame
    clip_end = canonical.clip_end_frame if canonical.clip_end_frame is not None else frame_count
    if not 0 <= clip_start < clip_end <= frame_count:
        raise ManifestReject("invalid_clip", f"Invalid clip [{clip_start},{clip_end}) for frame_count={frame_count}")

    identity = {
        "dataset_name": canonical.dataset_name,
        "adapter_name": canonical.adapter_name,
        "source_record_id": canonical.source_record_id,
        "task": VIDEO_TASK,
        "target_path": target_path,
        "reference_paths": canonical.reference_paths,
        "caption": canonical.caption,
        "crop_xyxy": canonical.crop_xyxy,
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
    }
    sample_key = stable_sample_key(identity)
    max_start = clip_end - 1 - round((VIDEO_NUM_FRAMES - 1) * original_fps / VIDEO_FPS)
    if max_start < clip_start:
        raise ManifestReject(
            "insufficient_frames_for_121_at_24fps",
            f"No exact 121-frame/24-fps clip fits [{clip_start},{clip_end}) at fps={original_fps}",
        )
    rng_seed = int(stable_sample_key({"sample_key": sample_key, "manifest_seed": manifest_seed})[:16], 16)
    sample_start = random.Random(rng_seed).randint(clip_start, max_start)
    source_indices = [round(sample_start + index * original_fps / VIDEO_FPS) for index in range(VIDEO_NUM_FRAMES)]
    if (
        len(source_indices) != VIDEO_NUM_FRAMES
        or any(left >= right for left, right in zip(source_indices, source_indices[1:]))
        or source_indices[0] < clip_start
        or source_indices[-1] >= clip_end
    ):
        raise ManifestReject("insufficient_frames_for_121_at_24fps", "Source index plan is not strictly increasing")

    references = list(canonical.reference_paths)
    reference_reason = "phantom_missing_reference" if canonical.adapter_name == "phantom" else "missing_reference"
    if not references:
        raise ManifestReject(reference_reason, f"No references for {canonical.source_record_id}")
    _validate_reference_paths(references, image_validator=image_validator, reason=reference_reason)
    num_anchors = max(1, round(VIDEO_NUM_FRAMES * anchor_frame_ratio))
    anchor_target_indices = uniform_integer_indices(start=0, end=VIDEO_NUM_FRAMES - 1, count=num_anchors)
    anchor_source_indices = [source_indices[index] for index in anchor_target_indices]
    record = {
        "sample_key": sample_key,
        "dataset_name": canonical.dataset_name,
        "adapter_name": canonical.adapter_name,
        "source_record_id": canonical.source_record_id,
        "task": VIDEO_TASK,
        "target_modality": "video",
        "target_path": target_path,
        "reference_paths": references,
        "caption": canonical.caption,
        "crop_xyxy": canonical.crop_xyxy,
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
        "original_fps": original_fps,
        "target_fps": VIDEO_FPS,
        "target_num_frames": VIDEO_NUM_FRAMES,
        "target_width": TARGET_WIDTH,
        "target_height": TARGET_HEIGHT,
        "target_source_frame_indices": source_indices,
        "semantic_anchor_target_indices": anchor_target_indices,
        "semantic_anchor_source_indices": anchor_source_indices,
        **_json_safe(canonical.metadata),
    }
    return finalize_manifest_record(record)


def deduplicate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record["sample_key"])
        if key in unique and unique[key]["sample_plan_sha256"] != record["sample_plan_sha256"]:
            raise ValueError(f"sample_key collision with different plans: {key}")
        unique[key] = record
    return [unique[key] for key in sorted(unique)]
