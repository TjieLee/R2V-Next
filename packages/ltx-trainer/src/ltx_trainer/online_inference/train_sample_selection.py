"""Bounded-memory selection of deterministic samples from online manifests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Sequence

from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_data.manifest_index import (
    default_manifest_index_path,
    read_jsonl_record_at,
    read_manifest_index,
)
from ltx_trainer.online_data.manifest_schema import validate_manifest_record

SELECTION_VERSION = 1
SUPPORTED_TASKS = (IMAGE_TASK, VIDEO_TASK)


@dataclass(frozen=True)
class SelectionResult:
    samples: list[dict[str, Any]]
    summary: dict[str, Any]


def _resolve_media_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _scan_compact_index(manifest_path: Path) -> tuple[array, dict[str, array]]:
    offsets = array("Q")
    task_indices = {IMAGE_TASK: array("q"), VIDEO_TASK: array("q")}
    with manifest_path.open("rb") as handle:
        row_index = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            task = str(record.get("task"))
            if task not in task_indices:
                raise ValueError(f"Manifest row {row_index} has unsupported task {task!r}")
            offsets.append(offset)
            task_indices[task].append(row_index)
            row_index += 1
    return offsets, task_indices


def _load_compact_index(manifest_path: Path) -> tuple[array, dict[str, array], Path | None]:
    index_path = default_manifest_index_path(manifest_path)
    if index_path.is_file():
        offsets, task_indices = read_manifest_index(
            index_path,
            manifest_path=manifest_path,
        )
        return offsets, task_indices, index_path
    offsets, task_indices = _scan_compact_index(manifest_path)
    return offsets, task_indices, None


def _coprime_stride(length: int, rng: random.Random) -> tuple[int, int]:
    if length <= 1:
        return 0, 1
    start = rng.randrange(length)
    stride = rng.randrange(1, length)
    while math.gcd(stride, length) != 1:
        stride += 1
        if stride >= length:
            stride = 1
    return start, stride


def _task_row_order(indices: Sequence[int], *, seed: int, task: str) -> Iterable[int]:
    if not indices:
        return
    task_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{task}".encode("utf-8")).digest()[:8],
        byteorder="little",
    )
    start, stride = _coprime_stride(len(indices), random.Random(task_seed))
    for position in range(len(indices)):
        yield int(indices[(start + position * stride) % len(indices)])


def _selection_entry(
    record: dict[str, Any],
    *,
    manifest_path: Path,
    manifest_index: int,
) -> dict[str, Any]:
    reference_paths = [
        str(_resolve_media_path(str(path), manifest_path))
        for path in record["reference_paths"]
    ]
    return {
        "selection_version": SELECTION_VERSION,
        "manifest_path": str(manifest_path),
        "manifest_index": int(manifest_index),
        "sample_key": str(record["sample_key"]),
        "sample_plan_sha256": str(record["sample_plan_sha256"]),
        "dataset_name": str(record["dataset_name"]),
        "task": str(record["task"]),
        "target_modality": str(record["target_modality"]),
        "caption": str(record["caption"]),
        "reference_paths": reference_paths,
        "reference_count": len(reference_paths),
        "target_path": str(_resolve_media_path(str(record["target_path"]), manifest_path)),
        "target_source_frame_indices": record.get("target_source_frame_indices"),
        "vlm_target_frame_indices": record.get("vlm_target_frame_indices"),
        "vlm_source_frame_indices": record.get("vlm_source_frame_indices"),
        "crop_xyxy": record.get("crop_xyxy"),
        "face_cut": record.get("face_cut"),
        "width": int(record["target_width"]),
        "height": int(record["target_height"]),
        "num_frames": int(record["target_num_frames"]),
        "fps": float(record["target_fps"]),
    }


def _validate_candidate(
    record: dict[str, Any],
    *,
    manifest_path: Path,
    manifest_index: int,
    max_caption_chars: int | None,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        validate_manifest_record(record, manifest_index)
    except (TypeError, ValueError) as exc:
        return None, f"invalid_manifest_record:{type(exc).__name__}"
    caption = str(record["caption"])
    if not caption.strip():
        return None, "empty_caption"
    if max_caption_chars is not None and len(caption) > max_caption_chars:
        return None, "caption_too_long"
    references = list(record["reference_paths"])
    if not 1 <= len(references) <= 4:
        return None, "reference_count_out_of_range"
    for reference in references:
        if not _resolve_media_path(str(reference), manifest_path).is_file():
            return None, "missing_reference"
    # This checks metadata availability only. The target is never opened or decoded.
    if not _resolve_media_path(str(record["target_path"]), manifest_path).is_file():
        return None, "missing_target"
    return (
        _selection_entry(
            record,
            manifest_path=manifest_path,
            manifest_index=manifest_index,
        ),
        None,
    )


def _read_candidate(
    handle: BinaryIO,
    offsets: Sequence[int],
    row_index: int,
    *,
    manifest_path: Path,
    max_caption_chars: int | None,
) -> tuple[dict[str, Any] | None, str | None]:
    record = read_jsonl_record_at(handle, int(offsets[row_index]))
    return _validate_candidate(
        record,
        manifest_path=manifest_path,
        manifest_index=row_index,
        max_caption_chars=max_caption_chars,
    )


def _select_explicit_indices(
    *,
    manifest_path: Path,
    offsets: Sequence[int],
    manifest_indices: Sequence[int],
    tasks: set[str],
    max_caption_chars: int | None,
    skipped: Counter[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    with manifest_path.open("rb") as handle:
        for row_index in manifest_indices:
            if not 0 <= row_index < len(offsets):
                raise ValueError(f"Manifest index {row_index} is outside [0, {len(offsets) - 1}]")
            candidate, reason = _read_candidate(
                handle,
                offsets,
                row_index,
                manifest_path=manifest_path,
                max_caption_chars=max_caption_chars,
            )
            if candidate is None:
                skipped[str(reason)] += 1
                continue
            if candidate["task"] not in tasks:
                skipped["task_filtered"] += 1
                continue
            selected.append(candidate)
    return selected


def _select_explicit_keys(
    *,
    manifest_path: Path,
    offsets: Sequence[int],
    sample_keys: Sequence[str],
    tasks: set[str],
    max_caption_chars: int | None,
    skipped: Counter[str],
) -> list[dict[str, Any]]:
    requested = list(dict.fromkeys(str(key) for key in sample_keys))
    requested_set = set(requested)
    found: dict[str, dict[str, Any]] = {}
    with manifest_path.open("rb") as handle:
        for row_index, offset in enumerate(offsets):
            record = read_jsonl_record_at(handle, int(offset))
            sample_key = str(record.get("sample_key", ""))
            if sample_key not in requested_set:
                continue
            candidate, reason = _validate_candidate(
                record,
                manifest_path=manifest_path,
                manifest_index=row_index,
                max_caption_chars=max_caption_chars,
            )
            if candidate is None:
                skipped[str(reason)] += 1
            elif candidate["task"] not in tasks:
                skipped["task_filtered"] += 1
            else:
                found[sample_key] = candidate
            if len(found) == len(requested_set):
                break
    missing = [key for key in requested if key not in found]
    if missing:
        raise ValueError(f"Requested sample keys were not selectable: {missing}")
    return [found[key] for key in requested]


def _select_stratified_task(
    *,
    task: str,
    samples_per_task: int,
    seed: int,
    manifest_path: Path,
    offsets: Sequence[int],
    task_indices: Sequence[int],
    max_caption_chars: int | None,
    skipped: Counter[str],
) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, Any]] = {}
    fallback: list[dict[str, Any]] = []
    fallback_limit = max(samples_per_task * 2, 8)
    with manifest_path.open("rb") as handle:
        for row_index in _task_row_order(task_indices, seed=seed, task=task):
            candidate, reason = _read_candidate(
                handle,
                offsets,
                row_index,
                manifest_path=manifest_path,
                max_caption_chars=max_caption_chars,
            )
            if candidate is None:
                skipped[str(reason)] += 1
                continue
            reference_count = int(candidate["reference_count"])
            buckets.setdefault(reference_count, candidate)
            if len(fallback) < fallback_limit:
                fallback.append(candidate)
            desired_bucket_count = min(samples_per_task, 4)
            if len(buckets) >= desired_bucket_count and len(fallback) >= samples_per_task:
                break

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for reference_count in (1, 2, 3, 4):
        candidate = buckets.get(reference_count)
        if candidate is None or len(selected) >= samples_per_task:
            continue
        selected.append(candidate)
        selected_keys.add(str(candidate["sample_key"]))
    for candidate in fallback:
        sample_key = str(candidate["sample_key"])
        if sample_key in selected_keys:
            continue
        selected.append(candidate)
        selected_keys.add(sample_key)
        if len(selected) >= samples_per_task:
            break
    if len(selected) < samples_per_task:
        raise RuntimeError(
            f"Could select only {len(selected)}/{samples_per_task} valid {task} samples"
        )
    return selected


def _select_deterministic_task(
    *,
    task: str,
    samples_per_task: int,
    seed: int,
    manifest_path: Path,
    offsets: Sequence[int],
    task_indices: Sequence[int],
    max_caption_chars: int | None,
    skipped: Counter[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    with manifest_path.open("rb") as handle:
        for row_index in _task_row_order(task_indices, seed=seed, task=task):
            candidate, reason = _read_candidate(
                handle,
                offsets,
                row_index,
                manifest_path=manifest_path,
                max_caption_chars=max_caption_chars,
            )
            if candidate is None:
                skipped[str(reason)] += 1
                continue
            selected.append(candidate)
            if len(selected) >= samples_per_task:
                break
    if len(selected) < samples_per_task:
        raise RuntimeError(
            f"Could select only {len(selected)}/{samples_per_task} valid {task} samples"
        )
    return selected


def select_online_train_samples(
    manifest_path: str | Path,
    *,
    tasks: Sequence[str] = SUPPORTED_TASKS,
    samples_per_task: int = 4,
    seed: int = 42,
    manifest_indices: Sequence[int] | None = None,
    sample_keys: Sequence[str] | None = None,
    stratify_by_reference_count: bool = True,
    max_caption_chars: int | None = None,
) -> SelectionResult:
    """Select online rows without materializing or decoding the full manifest."""
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Online manifest does not exist: {manifest}")
    normalized_tasks = tuple(dict.fromkeys(str(task).strip() for task in tasks))
    if not normalized_tasks or any(task not in SUPPORTED_TASKS for task in normalized_tasks):
        raise ValueError(f"tasks must be a non-empty subset of {SUPPORTED_TASKS}, got {normalized_tasks}")
    if samples_per_task < 1:
        raise ValueError("samples_per_task must be >= 1")
    if max_caption_chars is not None and max_caption_chars < 1:
        raise ValueError("max_caption_chars must be >= 1")
    if manifest_indices and sample_keys:
        raise ValueError("manifest_indices and sample_keys are mutually exclusive")

    offsets, task_indices, index_path = _load_compact_index(manifest)
    skipped: Counter[str] = Counter()
    task_set = set(normalized_tasks)
    if manifest_indices:
        samples = _select_explicit_indices(
            manifest_path=manifest,
            offsets=offsets,
            manifest_indices=[int(index) for index in manifest_indices],
            tasks=task_set,
            max_caption_chars=max_caption_chars,
            skipped=skipped,
        )
        selection_mode = "explicit_manifest_indices"
    elif sample_keys:
        samples = _select_explicit_keys(
            manifest_path=manifest,
            offsets=offsets,
            sample_keys=sample_keys,
            tasks=task_set,
            max_caption_chars=max_caption_chars,
            skipped=skipped,
        )
        selection_mode = "explicit_sample_keys"
    else:
        samples = []
        for task in normalized_tasks:
            if stratify_by_reference_count:
                samples.extend(
                    _select_stratified_task(
                        task=task,
                        samples_per_task=samples_per_task,
                        seed=seed,
                        manifest_path=manifest,
                        offsets=offsets,
                        task_indices=task_indices[task],
                        max_caption_chars=max_caption_chars,
                        skipped=skipped,
                    )
                )
            else:
                samples.extend(
                    _select_deterministic_task(
                        task=task,
                        samples_per_task=samples_per_task,
                        seed=seed,
                        manifest_path=manifest,
                        offsets=offsets,
                        task_indices=task_indices[task],
                        max_caption_chars=max_caption_chars,
                        skipped=skipped,
                    )
                )
        selection_mode = "deterministic_stratified" if stratify_by_reference_count else "deterministic"

    keys = [str(sample["sample_key"]) for sample in samples]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Selection produced duplicate sample_key values")
    task_counts = Counter(str(sample["task"]) for sample in samples)
    reference_counts = {
        task: dict(
            sorted(
                Counter(
                    int(sample["reference_count"])
                    for sample in samples
                    if sample["task"] == task
                ).items()
            )
        )
        for task in normalized_tasks
    }
    summary = {
        "selection_version": SELECTION_VERSION,
        "selection_mode": selection_mode,
        "manifest_path": str(manifest),
        "manifest_index_path": str(index_path) if index_path is not None else None,
        "manifest_row_count": len(offsets),
        "tasks": list(normalized_tasks),
        "samples_per_task": samples_per_task,
        "seed": seed,
        "stratify_by_reference_count": stratify_by_reference_count,
        "max_caption_chars": max_caption_chars,
        "selected_count": len(samples),
        "selected_by_task": dict(sorted(task_counts.items())),
        "selected_reference_counts": reference_counts,
        "skipped_reasons": dict(sorted(skipped.items())),
        "target_files_decoded": 0,
    }
    return SelectionResult(samples=samples, summary=summary)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_sample_dir_name(sample: dict[str, Any]) -> str:
    sample_key = str(sample["sample_key"])
    safe_key = "".join(character if character.isalnum() or character in "-_" else "_" for character in sample_key)
    if len(safe_key) > 96:
        suffix = hashlib.sha256(sample_key.encode("utf-8")).hexdigest()[:12]
        safe_key = f"{safe_key[:80]}_{suffix}"
    return f"{sample['task']}_{safe_key}"


def _link_or_copy_reference(source: Path, destination: Path) -> str:
    try:
        destination.symlink_to(os.path.relpath(source, destination.parent))
        return "relative_symlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_selection_bundle(
    result: SelectionResult,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Publish a complete selection bundle atomically."""
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Selection output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        samples_root = temporary / "samples"
        samples_root.mkdir()
        published_samples: list[dict[str, Any]] = []
        link_modes: Counter[str] = Counter()
        for sample in result.samples:
            sample_dir = samples_root / _safe_sample_dir_name(sample)
            sample_dir.mkdir()
            _atomic_write_text(sample_dir / "prompt.txt", f"{sample['caption']}\n")
            reference_exports: list[str] = []
            for index, value in enumerate(sample["reference_paths"]):
                source = Path(value)
                suffix = source.suffix.lower() or ".png"
                destination_ref = sample_dir / f"reference_{index:02d}{suffix}"
                link_modes[_link_or_copy_reference(source, destination_ref)] += 1
                reference_exports.append(str(destination_ref.relative_to(temporary)))
            published = dict(sample)
            published["sample_dir"] = str(sample_dir.relative_to(temporary))
            published["reference_exports"] = reference_exports
            _atomic_write_text(
                sample_dir / "sample.json",
                json.dumps(published, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            published_samples.append(published)

        _atomic_write_text(
            temporary / "selected_samples.jsonl",
            "".join(
                json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n"
                for sample in published_samples
            ),
        )
        summary = dict(result.summary)
        summary["reference_export_modes"] = dict(sorted(link_modes.items()))
        summary["output_dir"] = str(destination)
        _atomic_write_text(
            temporary / "selection_summary.json",
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination
