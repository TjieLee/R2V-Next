"""Build a deterministic low-memory I2I/R2V online manifest."""

from __future__ import annotations

import json
import os
import resource
import sqlite3
import stat
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any

import typer
from PIL import Image, ImageOps

from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_data.manifest import (
    ManifestReject,
    build_i2i_record,
    build_r2v_record,
    iter_annotation_rows,
    load_multitask_data_config,
    probe_video,
    resolve_media_path,
)
from ltx_trainer.online_data.manifest_index import (
    build_manifest_offset_index,
    default_manifest_index_path,
)
from ltx_trainer.online_data.path_safety import assert_write_path_allowed

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


class _ManifestCollisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class _MediaSignature:
    path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class _ValidationResult:
    payload: dict[str, Any] | None
    error_reason: str | None = None
    error_message: str | None = None

    def unwrap_image_size(self) -> tuple[int, int]:
        if self.error_reason is not None or self.payload is None:
            raise RuntimeError(self.error_message or "Image validation failed")
        return int(self.payload["width"]), int(self.payload["height"])

    def as_probe_result(self) -> tuple[dict[str, Any] | None, ManifestReject | None]:
        if self.error_reason is None:
            return self.payload, None
        return None, ManifestReject(self.error_reason, self.error_message or self.error_reason)


def _verified_image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).size


class _MediaValidationCache:
    """Bounded same-build validation cache backed by the builder's temporary SQLite DB."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.hits = 0
        self.misses = 0
        connection.execute(
            """
            CREATE TABLE media_validation_cache (
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                ok INTEGER NOT NULL,
                payload_json TEXT,
                error_reason TEXT,
                error_message TEXT,
                PRIMARY KEY (kind, path, size, mtime_ns)
            ) WITHOUT ROWID
            """
        )

    @staticmethod
    def signature(path: str) -> _MediaSignature:
        resolved = str(Path(path).expanduser().resolve())
        try:
            metadata = Path(resolved).stat()
            if not stat.S_ISREG(metadata.st_mode):
                raise FileNotFoundError(resolved)
            return _MediaSignature(resolved, int(metadata.st_size), int(metadata.st_mtime_ns))
        except OSError:
            return _MediaSignature(resolved, -1, -1)

    def get(self, kind: str, signature: _MediaSignature) -> _ValidationResult | None:
        row = self.connection.execute(
            """
            SELECT ok, payload_json, error_reason, error_message
            FROM media_validation_cache
            WHERE kind = ? AND path = ? AND size = ? AND mtime_ns = ?
            """,
            (kind, signature.path, signature.size, signature.mtime_ns),
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        payload = json.loads(row[1]) if row[1] is not None else None
        return _ValidationResult(
            payload=payload,
            error_reason=None if int(row[0]) else str(row[2]),
            error_message=None if int(row[0]) else str(row[3]),
        )

    def put(self, kind: str, signature: _MediaSignature, result: _ValidationResult) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO media_validation_cache
            (kind, path, size, mtime_ns, ok, payload_json, error_reason, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kind,
                signature.path,
                signature.size,
                signature.mtime_ns,
                int(result.error_reason is None),
                json.dumps(result.payload, sort_keys=True) if result.payload is not None else None,
                result.error_reason,
                result.error_message,
            ),
        )

    def validate_image(self, path: str) -> tuple[int, int]:
        signature = self.signature(path)
        result = self.get("image", signature)
        if result is None:
            if signature.size < 0:
                result = _ValidationResult(None, "missing_media", f"Image does not exist: {signature.path}")
            else:
                try:
                    width, height = _verified_image_size(signature.path)
                    result = _ValidationResult({"width": width, "height": height})
                except Exception as exc:
                    result = _ValidationResult(None, "invalid_image", f"Unreadable image {signature.path}: {exc}")
            self.put("image", signature, result)
        return result.unwrap_image_size()


def _temporary_path(path: Path) -> Path:
    return Path(f"{path}.tmp.{os.getpid()}")


def _flush_and_sync(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            _flush_and_sync(handle)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _batched(rows: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    iterator = iter(rows)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def _safe_probe(
    row: dict[str, Any],
    data_root: str | Path | None,
) -> tuple[dict[str, Any] | None, ManifestReject | None]:
    try:
        if "video_path" not in row:
            return None, ManifestReject("r2v_schema_mismatch", "video_path field is missing")
        target_path = resolve_media_path(row["video_path"], data_root=data_root)
        if not Path(target_path).is_file():
            return None, ManifestReject("missing_target", f"Target video does not exist: {target_path}")
        return dict(probe_video(target_path)), None
    except Exception as exc:
        return None, ManifestReject("invalid_video_header", str(exc))


def _probe_resolved_video(path: str) -> _ValidationResult:
    try:
        if not Path(path).is_file():
            return _ValidationResult(None, "missing_target", f"Target video does not exist: {path}")
        return _ValidationResult(dict(probe_video(path)))
    except Exception as exc:
        return _ValidationResult(None, "invalid_video_header", str(exc))


def _iter_rows_with_bounded_probes(
    rows: Iterable[dict[str, Any]],
    *,
    data_root: str | Path | None,
    workers: int,
    batch_size: int,
    validation_cache: _MediaValidationCache | None = None,
) -> Iterator[tuple[dict[str, Any], dict[str, Any] | None, ManifestReject | None]]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for row_batch in _batched(rows, batch_size):
            if validation_cache is None:
                results = list(executor.map(partial(_safe_probe, data_root=data_root), row_batch))
                for row, (header, error) in zip(row_batch, results, strict=True):
                    yield row, header, error
                continue

            requests: list[_MediaSignature | _ValidationResult] = []
            results_by_signature: dict[_MediaSignature, _ValidationResult] = {}
            misses: dict[_MediaSignature, str] = {}
            for row in row_batch:
                if "video_path" not in row:
                    requests.append(
                        _ValidationResult(None, "r2v_schema_mismatch", "video_path field is missing")
                    )
                    continue
                target_path = resolve_media_path(row["video_path"], data_root=data_root)
                signature = validation_cache.signature(target_path)
                requests.append(signature)
                cached = validation_cache.get("video", signature)
                if cached is not None:
                    results_by_signature[signature] = cached
                elif signature.size < 0:
                    result = _ValidationResult(
                        None,
                        "missing_target",
                        f"Target video does not exist: {signature.path}",
                    )
                    validation_cache.put("video", signature, result)
                    results_by_signature[signature] = result
                else:
                    misses.setdefault(signature, signature.path)

            missing_signatures = list(misses)
            probed = executor.map(_probe_resolved_video, [misses[item] for item in missing_signatures])
            for signature, result in zip(missing_signatures, probed, strict=True):
                validation_cache.put("video", signature, result)
                results_by_signature[signature] = result

            for row, request in zip(row_batch, requests, strict=True):
                result = request if isinstance(request, _ValidationResult) else results_by_signature[request]
                header, error = result.as_probe_result()
                yield row, header, error


def _peak_rss_gb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024.0
    return peak_bytes / (1024.0**3)


@app.command()
def main(  # noqa: PLR0913, PLR0915
    train_data_config: str = typer.Option(..., "--train-data-config"),
    output: str = typer.Option(..., "--output"),
    reject_output: str | None = typer.Option(None, "--reject-output"),
    summary_output: str | None = typer.Option(None, "--summary-output"),
    manifest_seed: int = typer.Option(42, "--manifest-seed"),
    annotation_batch_size: int = typer.Option(4096, "--annotation-batch-size", min=1),
    probe_workers: int = typer.Option(8, "--probe-workers", min=1),
    probe_batch_size: int = typer.Option(256, "--probe-batch-size", min=1),
    i2i_target_field: str = typer.Option(..., "--i2i-target-field"),
    i2i_reference_field: str = typer.Option(..., "--i2i-reference-field"),
    i2i_caption_field: str = typer.Option(..., "--i2i-caption-field"),
    i2i_crop_field: str | None = typer.Option(None, "--i2i-crop-field"),
) -> None:
    started = time.perf_counter()
    output_path = assert_write_path_allowed(output)
    reject_path = assert_write_path_allowed(
        reject_output or str(output_path.with_name(output_path.stem + "_rejected.jsonl"))
    )
    summary_path = assert_write_path_allowed(
        summary_output or str(output_path.with_name(output_path.stem + "_summary.json"))
    )
    index_path = assert_write_path_allowed(default_manifest_index_path(output_path))
    for path in (output_path, reject_path, summary_path, index_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    output_temporary = _temporary_path(output_path)
    reject_temporary = _temporary_path(reject_path)
    index_temporary = _temporary_path(index_path)
    sqlite_path = assert_write_path_allowed(
        output_path.with_suffix(output_path.suffix + f".dedup.{os.getpid()}.sqlite")
    )
    config = load_multitask_data_config(train_data_config)
    raw_rows = 0
    accepted_rows = 0
    rejected_rows = 0
    duplicate_rows = 0
    task_counts: Counter[str] = Counter()
    reject_reason_counts: Counter[str] = Counter()

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(sqlite_path)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-16384")
        connection.execute(
            "CREATE TABLE dedup (sample_key TEXT PRIMARY KEY, sample_plan_sha256 TEXT NOT NULL) WITHOUT ROWID"
        )
        validation_cache = _MediaValidationCache(connection)

        with output_temporary.open("w", encoding="utf-8") as accepted_handle, reject_temporary.open(
            "w", encoding="utf-8"
        ) as reject_handle:
            for dataset in config["datasets"]:
                if not isinstance(dataset, dict):
                    raise ValueError("Each datasets entry must be a mapping")
                task = str(dataset.get("task", ""))
                dataset_name = str(dataset.get("name", task))
                annotation_path = dataset.get("ann_path") or dataset.get("parquet") or dataset.get("path")
                if annotation_path is None:
                    raise ValueError(f"Dataset {dataset_name!r} has no ann_path/parquet/path")
                rows: Iterable[dict[str, Any]] = iter_annotation_rows(
                    annotation_path,
                    batch_size=annotation_batch_size,
                )
                max_samples = dataset.get("max_samples")
                if max_samples is not None:
                    rows = islice(rows, int(max_samples))
                data_root = dataset.get("data_root", config.get("data_root"))
                if task == VIDEO_TASK:
                    row_stream: Iterable[
                        tuple[dict[str, Any], dict[str, Any] | None, ManifestReject | None]
                    ] = (
                        _iter_rows_with_bounded_probes(
                            rows,
                            data_root=data_root,
                            workers=probe_workers,
                            batch_size=probe_batch_size,
                            validation_cache=validation_cache,
                        )
                    )
                else:
                    row_stream = ((row, None, None) for row in rows)

                for row_index, (row, video_header, probe_error) in enumerate(row_stream):
                    raw_rows += 1
                    try:
                        if task == IMAGE_TASK:
                            record = build_i2i_record(
                                row,
                                dataset_name=dataset_name,
                                data_root=data_root,
                                target_field=i2i_target_field,
                                reference_field=i2i_reference_field,
                                caption_field=i2i_caption_field,
                                crop_field=i2i_crop_field,
                                image_validator=validation_cache.validate_image,
                            )
                        elif task == VIDEO_TASK:
                            if probe_error is not None or video_header is None:
                                raise probe_error or ManifestReject("invalid_video_header", "Video probe failed")
                            record = build_r2v_record(
                                row,
                                dataset_name=dataset_name,
                                data_root=data_root,
                                manifest_seed=manifest_seed,
                                video_header=video_header,
                                image_validator=validation_cache.validate_image,
                                target_path_validated=True,
                            )
                        else:
                            raise ValueError(f"Unsupported task {task!r} in dataset {dataset_name!r}")

                        sample_key = str(record["sample_key"])
                        plan_sha = str(record["sample_plan_sha256"])
                        existing = connection.execute(
                            "SELECT sample_plan_sha256 FROM dedup WHERE sample_key = ?",
                            (sample_key,),
                        ).fetchone()
                        if existing is not None:
                            if str(existing[0]) != plan_sha:
                                raise _ManifestCollisionError(
                                    f"sample_key collision with different plans: {sample_key}"
                                )
                            duplicate_rows += 1
                            continue
                        connection.execute("INSERT INTO dedup VALUES (?, ?)", (sample_key, plan_sha))
                        accepted_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        accepted_rows += 1
                        task_counts[task] += 1
                        if accepted_rows % 10_000 == 0:
                            connection.commit()
                    except Exception as exc:
                        if isinstance(exc, _ManifestCollisionError):
                            raise
                        reason = exc.reason if isinstance(exc, ManifestReject) else type(exc).__name__
                        reject_handle.write(
                            json.dumps(
                                {
                                    "dataset_name": dataset_name,
                                    "row_index": row_index,
                                    "task": task,
                                    "reason": reason,
                                    "message": str(exc),
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        rejected_rows += 1
                        reject_reason_counts[reason] += 1
            connection.commit()
            _flush_and_sync(accepted_handle)
            _flush_and_sync(reject_handle)

        if any(task_counts[task] == 0 for task in (IMAGE_TASK, VIDEO_TASK)):
            raise RuntimeError(f"Built manifest does not contain both tasks: {dict(task_counts)}")
        build_manifest_offset_index(output_temporary, index_temporary)
        output_temporary.replace(output_path)
        reject_temporary.replace(reject_path)
        index_temporary.replace(index_path)
        summary = {
            "raw_rows": raw_rows,
            "accepted_rows": accepted_rows,
            "rejected_rows": rejected_rows,
            "duplicate_rows": duplicate_rows,
            "task_counts": dict(sorted(task_counts.items())),
            "reject_reason_counts": dict(sorted(reject_reason_counts.items())),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_gb": _peak_rss_gb(),
            "manifest_path": str(output_path),
            "manifest_index_path": str(index_path),
            "media_validation_cache_hits": validation_cache.hits,
            "media_validation_cache_misses": validation_cache.misses,
        }
        _atomic_write_json(summary_path, summary)
        typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        if connection is not None:
            connection.close()
        sqlite_path.unlink(missing_ok=True)
        output_temporary.unlink(missing_ok=True)
        reject_temporary.unlink(missing_ok=True)
        index_temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    app()
