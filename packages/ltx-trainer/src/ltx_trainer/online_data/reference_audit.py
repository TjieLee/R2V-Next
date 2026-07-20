"""Read-only audit helpers for online-manifest reference images."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image


def collect_reference_occurrences(
    manifest_path: str | Path,
    *,
    limit: int | None = None,
) -> tuple[dict[Path, list[dict[str, Any]]], int]:
    """Stream a manifest and group occurrences without scanning source directories."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    manifest = Path(manifest_path).expanduser().resolve()
    occurrences: dict[Path, list[dict[str, Any]]] = {}
    record_count = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if limit is not None and record_count >= limit:
                break
            manifest_index = record_count
            record = json.loads(line)
            reference_paths = record.get("reference_paths")
            if not isinstance(reference_paths, list):
                raise ValueError(
                    f"manifest_index={manifest_index} reference_paths must be a list"
                )
            for reference_index, value in enumerate(reference_paths):
                path = Path(str(value)).expanduser()
                if not path.is_absolute():
                    path = manifest.parent / path
                path = path.resolve()
                occurrences.setdefault(path, []).append(
                    {
                        "manifest_index": manifest_index,
                        "sample_key": str(record.get("sample_key", f"row-{manifest_index}")),
                        "task": str(record.get("task", "unknown")),
                        "reference_index": reference_index,
                        "reference_path": str(path),
                    }
                )
            record_count += 1
    return occurrences, record_count


def audit_reference_path(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reference_path": str(path),
        "status": "ok",
        "reason": "ok",
        "mode": None,
        "width": None,
        "height": None,
        "dtype": None,
        "shape": None,
        "degenerate_vlm_reference_geometry": False,
    }
    if not path.exists():
        result.update(status="missing", reason="missing_reference_path")
        return result
    if not path.is_file():
        result.update(status="non_file", reason="non_file_reference_path")
        return result
    try:
        with Image.open(path) as image:
            mode = image.mode
            rgb = image.convert("RGB")
            rgb.load()
            width, height = rgb.size
    except (OSError, ValueError) as exc:
        result.update(
            status="decode_failure",
            reason="reference_decode_failure",
            error_type=type(exc).__name__,
            message=str(exc),
        )
        return result
    degenerate = width <= 1 or height <= 1
    result.update(
        mode=mode,
        width=width,
        height=height,
        dtype="torch.uint8",
        shape=[height, width, 3],
        degenerate_vlm_reference_geometry=degenerate,
        reason="degenerate_vlm_reference_geometry" if degenerate else "ok",
    )
    return result


def audit_online_manifest_references(
    manifest_path: str | Path,
    *,
    limit: int | None = None,
    workers: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    occurrences, record_count = collect_reference_occurrences(
        manifest_path,
        limit=limit,
    )
    paths = list(occurrences)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        unique_results = list(executor.map(audit_reference_path, paths))

    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    unique_degenerate_paths: list[str] = []
    for path, audit in zip(paths, unique_results, strict=True):
        status_counts[str(audit["status"])] += 1
        if audit["degenerate_vlm_reference_geometry"]:
            unique_degenerate_paths.append(str(path))
        for occurrence in occurrences[path]:
            rows.append({**occurrence, **audit})

    summary = {
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "record_count": record_count,
        "reference_occurrence_count": len(rows),
        "unique_reference_count": len(paths),
        "duplicate_reference_occurrence_count": len(rows) - len(paths),
        "unique_status_counts": dict(sorted(status_counts.items())),
        "degenerate_unique_reference_count": len(unique_degenerate_paths),
        "degenerate_reference_occurrence_count": sum(
            bool(row["degenerate_vlm_reference_geometry"]) for row in rows
        ),
        "degenerate_reference_paths": unique_degenerate_paths,
        "missing_unique_reference_count": status_counts["missing"],
        "decode_failure_unique_reference_count": status_counts["decode_failure"],
    }
    return rows, summary


__all__ = [
    "audit_online_manifest_references",
    "audit_reference_path",
    "collect_reference_occurrences",
]
