"""Schema adapters and read-only preflight for external R2V evaluation sets."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import statistics
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

from PIL import Image

from ltx_trainer.online_data.constants import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VIDEO_FPS,
    VIDEO_NUM_FRAMES,
)
from ltx_trainer.online_inference.read_only_sources import ReadOnlySourcePolicy

SCHEMA_VERSION = 1
TARGET_FRAMES = VIDEO_NUM_FRAMES
TARGET_FPS = VIDEO_FPS
DATASET_SCHEMAS = {"videoxfun_test", "opens2v_open_domain", "custom64"}

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


class ExternalEvalSchemaError(ValueError):
    """A source record cannot be normalized safely."""


def stable_sample_key(dataset_name: str, source_record_id: str) -> str:
    value = f"{dataset_name}\0{source_record_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def stable_sample_seed(base_seed: int, dataset_name: str, source_record_id: str) -> int:
    value = f"{base_seed}\0{dataset_name}\0{source_record_id}".encode("utf-8")
    # torch.Generator accepts signed 64-bit seeds; keep the value portable.
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & ((1 << 63) - 1)


def sanitize_output_id(value: str) -> str:
    source = str(value).strip()
    if not source:
        raise ExternalEvalSchemaError("source record ID must not be empty")
    if "/" in source or "\\" in source or ".." in source:
        raise ExternalEvalSchemaError(f"unsafe source record ID {source!r}")
    if _CONTROL_CHARACTERS.search(source):
        raise ExternalEvalSchemaError(f"source record ID contains a control character: {source!r}")
    normalized = _UNSAFE_FILENAME.sub("_", source).strip("._-")
    if not normalized or normalized in {".", ".."}:
        raise ExternalEvalSchemaError(f"source record ID has no safe filename component: {source!r}")
    return normalized


def _required_text(record: dict[str, Any], key: str, *, record_id: str) -> str:
    value = str(record.get(key, "")).strip()
    if not value:
        raise ExternalEvalSchemaError(f"{record_id}: {key} must not be empty")
    return value


def _is_absolute_path(value: str) -> bool:
    return Path(value).is_absolute() or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _reference_paths(
    values: Any,
    *,
    record_id: str,
    input_parent: Path,
    require_absolute: bool,
) -> tuple[list[str], list[str]]:
    if not isinstance(values, list):
        raise ExternalEvalSchemaError(f"{record_id}: reference paths must be a list")
    if not 1 <= len(values) <= 4:
        raise ExternalEvalSchemaError(f"{record_id}: expected 1..4 references, got {len(values)}")
    original = [str(value) for value in values]
    resolved: list[str] = []
    for value in original:
        if not value.strip():
            raise ExternalEvalSchemaError(f"{record_id}: reference path must not be empty")
        is_absolute = _is_absolute_path(value)
        path = Path(value).expanduser()
        if require_absolute and not is_absolute:
            raise ExternalEvalSchemaError(f"{record_id}: reference path must be absolute: {value}")
        if not is_absolute:
            path = input_parent / path
        resolved.append(value if is_absolute and not path.is_absolute() else str(path.resolve()))
    return original, resolved


def _normalized_record(
    *,
    dataset_name: str,
    source_json: Path,
    source_record_id: str,
    caption: str,
    original_reference_paths: list[str],
    reference_paths: list[str],
    dataset_metadata: dict[str, Any],
) -> dict[str, Any]:
    output_id = sanitize_output_id(source_record_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "source_json": str(source_json),
        "source_record_id": source_record_id,
        "output_id": output_id,
        "sample_key": stable_sample_key(dataset_name, source_record_id),
        "task": "r2v",
        "caption": caption,
        "reference_paths": reference_paths,
        "original_reference_paths": original_reference_paths,
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "num_frames": TARGET_FRAMES,
        "fps": TARGET_FPS,
        "dataset_metadata": dataset_metadata,
    }


def normalize_external_dataset(
    *,
    dataset_schema: str,
    input_json: str | Path,
    payload: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize every record while collecting record-local schema errors."""
    if dataset_schema not in DATASET_SCHEMAS:
        raise ExternalEvalSchemaError(
            f"dataset_schema must be one of {sorted(DATASET_SCHEMAS)}, got {dataset_schema!r}"
        )
    source_json = Path(input_json).expanduser().resolve()
    parent = source_json.parent
    if dataset_schema == "opens2v_open_domain":
        if not isinstance(payload, dict):
            raise ExternalEvalSchemaError("OpenS2V JSON must be an object keyed by sample ID")
        source_rows = [(str(key), value) for key, value in payload.items()]
    else:
        if not isinstance(payload, list):
            raise ExternalEvalSchemaError(f"{dataset_schema} JSON must be a list")
        source_rows = [(str(index), value) for index, value in enumerate(payload)]

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for source_index, row in source_rows:
        try:
            if not isinstance(row, dict):
                raise ExternalEvalSchemaError(f"record {source_index} must be an object")
            if dataset_schema == "videoxfun_test":
                source_id = _required_text(row, "id", record_id=source_index)
                caption = _required_text(row, "prompt", record_id=source_id)
                original, references = _reference_paths(
                    row.get("ref_images"),
                    record_id=source_id,
                    input_parent=parent,
                    require_absolute=True,
                )
                metadata = {}
            elif dataset_schema == "custom64":
                video_name = _required_text(row, "video_name", record_id=source_index)
                if Path(video_name).name != video_name or ".." in video_name:
                    raise ExternalEvalSchemaError(
                        f"{source_index}: video_name must be a plain filename, got {video_name!r}"
                    )
                source_id = Path(video_name).stem
                caption = _required_text(row, "caption", record_id=source_id)
                original, references = _reference_paths(
                    row.get("ref_image_paths"),
                    record_id=source_id,
                    input_parent=parent,
                    require_absolute=True,
                )
                metadata = {"video_name": video_name}
            else:
                source_id = source_index.strip()
                if not source_id:
                    raise ExternalEvalSchemaError("OpenS2V source ID must not be empty")
                caption = _required_text(row, "prompt", record_id=source_id)
                original, references = _reference_paths(
                    row.get("img_paths"),
                    record_id=source_id,
                    input_parent=parent,
                    require_absolute=False,
                )
                metadata = {
                    "synthesis_flag": row.get("synthesis_flag"),
                    "class_label": row.get("class_label"),
                    "schema_group": source_id.split("_", 1)[0],
                }
            records.append(
                _normalized_record(
                    dataset_name=dataset_schema,
                    source_json=source_json,
                    source_record_id=source_id,
                    caption=caption,
                    original_reference_paths=original,
                    reference_paths=references,
                    dataset_metadata=metadata,
                )
            )
        except (ExternalEvalSchemaError, TypeError, ValueError) as exc:
            errors.append(
                {
                    "source_index": source_index,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    source_ids = Counter(record["source_record_id"] for record in records)
    output_ids = Counter(record["output_id"] for record in records)
    for identifier, count in source_ids.items():
        if count > 1:
            errors.append(
                {
                    "source_index": identifier,
                    "error_type": "DuplicateSourceId",
                    "message": f"duplicate source ID: {identifier}",
                }
            )
    for identifier, count in output_ids.items():
        if count > 1:
            colliding = [
                record["source_record_id"]
                for record in records
                if record["output_id"] == identifier
            ]
            errors.append(
                {
                    "source_index": identifier,
                    "error_type": "NormalizedOutputIdCollision",
                    "message": f"normalized output ID {identifier!r} collides for {colliding}",
                }
            )
    return records, errors


def validate_normalized_record(record: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "dataset_name",
        "source_json",
        "source_record_id",
        "output_id",
        "sample_key",
        "task",
        "caption",
        "reference_paths",
        "original_reference_paths",
        "width",
        "height",
        "num_frames",
        "fps",
        "dataset_metadata",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ExternalEvalSchemaError(f"normalized record is missing fields: {missing}")
    if int(record["schema_version"]) != SCHEMA_VERSION or record["task"] != "r2v":
        raise ExternalEvalSchemaError("normalized record must be schema v1 task='r2v'")
    if record["dataset_name"] not in DATASET_SCHEMAS:
        raise ExternalEvalSchemaError(f"unknown normalized dataset {record['dataset_name']!r}")
    if not Path(str(record["source_json"])).is_absolute():
        raise ExternalEvalSchemaError("normalized source_json must be absolute")
    if sanitize_output_id(str(record["output_id"])) != record["output_id"]:
        raise ExternalEvalSchemaError(f"output_id is not canonical: {record['output_id']!r}")
    expected_output_id = sanitize_output_id(str(record["source_record_id"]))
    if record["output_id"] != expected_output_id:
        raise ExternalEvalSchemaError("output_id does not match the normalized source_record_id")
    caption = str(record["caption"]).strip()
    references = list(record["reference_paths"])
    if not caption or not 1 <= len(references) <= 4:
        raise ExternalEvalSchemaError("normalized record requires a caption and 1..4 references")
    if any(not _is_absolute_path(str(reference)) for reference in references):
        raise ExternalEvalSchemaError("normalized reference_paths must all be absolute")
    if len(record["original_reference_paths"]) != len(references):
        raise ExternalEvalSchemaError("original and resolved reference path counts differ")
    geometry = (
        int(record["width"]),
        int(record["height"]),
        int(record["num_frames"]),
        float(record["fps"]),
    )
    expected_geometry = (TARGET_WIDTH, TARGET_HEIGHT, TARGET_FRAMES, TARGET_FPS)
    if geometry != expected_geometry:
        raise ExternalEvalSchemaError(
            f"external R2V geometry must be {expected_geometry}, got {geometry}"
        )
    expected_key = stable_sample_key(str(record["dataset_name"]), str(record["source_record_id"]))
    if record["sample_key"] != expected_key:
        raise ExternalEvalSchemaError("normalized record sample_key does not match stable SHA256")
    return record


def load_normalized_manifest(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(validate_normalized_record(json.loads(line)))
            except (json.JSONDecodeError, ExternalEvalSchemaError) as exc:
                raise ExternalEvalSchemaError(f"manifest line {line_number}: {exc}") from exc
    if not records:
        raise ExternalEvalSchemaError("normalized manifest is empty")
    output_ids = [record["output_id"] for record in records]
    if len(output_ids) != len(set(output_ids)):
        raise ExternalEvalSchemaError("normalized manifest contains duplicate output IDs")
    return records


def center_crop_risk(
    source_width: int,
    source_height: int,
    *,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
) -> dict[str, Any]:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source image dimensions must be positive")
    scale = max(target_height / source_height, target_width / source_width)
    scaled_height = max(target_height, int(round(source_height * scale)))
    scaled_width = max(target_width, int(round(source_width * scale)))
    crop_top = (scaled_height - target_height) // 2
    crop_left = (scaled_width - target_width) // 2
    crop_bottom = scaled_height - target_height - crop_top
    crop_right = scaled_width - target_width - crop_left
    retained = (target_width * target_height) / (scaled_width * scaled_height)
    source_ratio = source_width / source_height
    target_ratio = target_width / target_height
    return {
        "source_width": source_width,
        "source_height": source_height,
        "source_aspect_ratio": source_ratio,
        "target_aspect_ratio": target_ratio,
        "scaled_width": scaled_width,
        "scaled_height": scaled_height,
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
        "crop_left": crop_left,
        "crop_right": crop_right,
        "retained_area_ratio": retained,
        "severe_crop_risk": retained < 0.70,
        "vertical_crop_risk": crop_top > 0 or crop_bottom > 0,
        "horizontal_crop_risk": crop_left > 0 or crop_right > 0,
        "square_to_wide_risk": abs(source_ratio - 1.0) <= 0.05 and target_ratio > 1.5,
        "portrait_to_wide_risk": source_ratio < 1.0 and target_ratio > 1.0,
    }


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def build_preflight_report(
    records: list[dict[str, Any]],
    *,
    policy: ReadOnlySourcePolicy,
    schema_errors: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    missing: list[str] = []
    non_files: list[str] = []
    decode_failures: list[dict[str, str]] = []
    image_details: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    extensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    ratios: Counter[str] = Counter()
    source_snapshots: list[dict[str, Any]] = []
    for record in records:
        for reference_index, value in enumerate(record["reference_paths"]):
            try:
                path = policy.assert_read_path(value)
            except PermissionError as exc:
                decode_failures.append({"path": str(value), "error": str(exc)})
                continue
            if not path.exists():
                missing.append(str(path))
                continue
            if not path.is_file():
                non_files.append(str(path))
                continue
            source_snapshots.append(policy.snapshot(path).__dict__)
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    mode = image.mode
            except (OSError, ValueError) as exc:
                decode_failures.append({"path": str(path), "error": str(exc)})
                continue
            risk = center_crop_risk(width, height)
            row = {
                "dataset_name": record["dataset_name"],
                "source_record_id": record["source_record_id"],
                "output_id": record["output_id"],
                "reference_index": reference_index,
                "reference_path": str(path),
                **risk,
            }
            crop_rows.append(row)
            image_details.append(
                {
                    "source_record_id": record["source_record_id"],
                    "reference_index": reference_index,
                    "path": str(path),
                    "width": width,
                    "height": height,
                    "mode": mode,
                }
            )
            extensions[path.suffix.lower() or "<none>"] += 1
            modes[mode] += 1
            ratios[f"{width / height:.2f}"] += 1

    reference_lists = Counter(tuple(record["reference_paths"]) for record in records)
    captions = [record["caption"] for record in records]
    groups = Counter(
        str(record["dataset_metadata"].get("schema_group", "<none>"))
        for record in records
    )
    synthesis = Counter(
        str(record["dataset_metadata"].get("synthesis_flag", "<none>"))
        for record in records
    )
    class_labels: Counter[str] = Counter()
    for record in records:
        labels = record["dataset_metadata"].get("class_label")
        if isinstance(labels, list):
            class_labels.update(str(value) for value in labels)
    report = {
        "schema_version": SCHEMA_VERSION,
        "record_count": len(records),
        "actual_count": len(records),
        "schema_errors": list(schema_errors or []),
        "schema_error_count": len(schema_errors or []),
        "duplicate_source_ids": [
            value for value, count in Counter(r["source_record_id"] for r in records).items() if count > 1
        ],
        "duplicate_normalized_output_ids": [
            value for value, count in Counter(r["output_id"] for r in records).items() if count > 1
        ],
        "empty_prompt_count": sum(not caption.strip() for caption in captions),
        "reference_count_distribution": dict(
            sorted(Counter(len(record["reference_paths"]) for record in records).items())
        ),
        "missing_reference_paths": missing,
        "non_file_reference_paths": non_files,
        "decode_failures": decode_failures,
        "image_details": image_details,
        "image_mode_distribution": dict(sorted(modes.items())),
        "aspect_ratio_distribution": dict(sorted(ratios.items())),
        "reference_extension_distribution": dict(sorted(extensions.items())),
        "duplicate_reference_lists": [
            {"reference_paths": list(values), "count": count}
            for values, count in reference_lists.items()
            if count > 1
        ],
        "prompt_character_statistics": _summary([float(len(value)) for value in captions]),
        "prompt_token_like_statistics": _summary(
            [float(len(re.findall(r"\S+", value))) for value in captions]
        ),
        "opens2v_schema_group_distribution": dict(sorted(groups.items())),
        "synthesis_flag_distribution": dict(sorted(synthesis.items())),
        "class_label_distribution": dict(sorted(class_labels.items())),
        "crop_risk_counts": {
            key: sum(bool(row[key]) for row in crop_rows)
            for key in (
                "severe_crop_risk",
                "vertical_crop_risk",
                "horizontal_crop_risk",
                "square_to_wide_risk",
                "portrait_to_wide_risk",
            )
        },
        "source_snapshots": source_snapshots,
        "preflight_passed": not (
            schema_errors or missing or non_files or decode_failures
        ),
    }
    return report, crop_rows


def manifest_jsonl(records: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)


def crop_rows_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


__all__ = [
    "DATASET_SCHEMAS",
    "SCHEMA_VERSION",
    "TARGET_FPS",
    "TARGET_FRAMES",
    "TARGET_HEIGHT",
    "TARGET_WIDTH",
    "ExternalEvalSchemaError",
    "build_preflight_report",
    "center_crop_risk",
    "crop_rows_csv",
    "load_normalized_manifest",
    "manifest_jsonl",
    "normalize_external_dataset",
    "sanitize_output_id",
    "stable_sample_key",
    "stable_sample_seed",
    "validate_normalized_record",
]
