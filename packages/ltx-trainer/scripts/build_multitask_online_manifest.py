"""Build a deterministic low-memory I2I/R2V online manifest."""

from __future__ import annotations

import json
import os

try:
    import resource
except ImportError:  # pragma: no cover - exercised on Windows
    resource = None
import sqlite3
import stat
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, TypeVar

import typer
from PIL import Image, ImageOps

from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_data.manifest import (
    CanonicalR2VSource,
    ManifestReject,
    build_canonical_r2v_record,
    build_i2i_record,
    iter_annotation_items,
    load_multitask_data_config,
    normalize_r2v_source,
    parse_path_list,
    probe_video,
    resolve_media_path,
)
from ltx_trainer.online_data.manifest_index import (
    build_manifest_offset_index,
    default_manifest_index_path,
)
from ltx_trainer.online_data.path_safety import assert_write_path_allowed

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)
_T = TypeVar("_T")


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
        self.owner_thread_id = threading.get_ident()
        self.image_cache_hits = 0
        self.image_cache_misses = 0
        self.video_cache_hits = 0
        self.video_cache_misses = 0
        self.image_probe_submitted = 0
        self.video_probe_submitted = 0
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

    @property
    def hits(self) -> int:
        return self.image_cache_hits + self.video_cache_hits

    @property
    def misses(self) -> int:
        return self.image_cache_misses + self.video_cache_misses

    def _assert_owner_thread(self) -> None:
        if threading.get_ident() != self.owner_thread_id:
            raise RuntimeError("Media validation SQLite cache may only be accessed by its owner thread")

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
        self._assert_owner_thread()
        row = self.connection.execute(
            """
            SELECT ok, payload_json, error_reason, error_message
            FROM media_validation_cache
            WHERE kind = ? AND path = ? AND size = ? AND mtime_ns = ?
            """,
            (kind, signature.path, signature.size, signature.mtime_ns),
        ).fetchone()
        if row is None:
            if kind == "image":
                self.image_cache_misses += 1
            else:
                self.video_cache_misses += 1
            return None
        if kind == "image":
            self.image_cache_hits += 1
        else:
            self.video_cache_hits += 1
        payload = json.loads(row[1]) if row[1] is not None else None
        return _ValidationResult(
            payload=payload,
            error_reason=None if int(row[0]) else str(row[2]),
            error_message=None if int(row[0]) else str(row[3]),
        )

    def put(self, kind: str, signature: _MediaSignature, result: _ValidationResult) -> None:
        self._assert_owner_thread()
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

    def _prefetch(
        self,
        kind: str,
        paths: Iterable[str],
        *,
        executor: ThreadPoolExecutor,
    ) -> None:
        self._assert_owner_thread()
        missing: dict[_MediaSignature, str] = {}
        seen: set[_MediaSignature] = set()
        for path in paths:
            signature = self.signature(path)
            if signature in seen:
                continue
            seen.add(signature)
            if self.get(kind, signature) is None:
                missing[signature] = signature.path

        signatures = list(missing)
        if kind == "image":
            self.image_probe_submitted += len(signatures)
            results = executor.map(_probe_resolved_image, [missing[item] for item in signatures])
        elif kind == "video":
            self.video_probe_submitted += len(signatures)
            results = executor.map(_probe_resolved_video, [missing[item] for item in signatures])
        else:  # pragma: no cover - internal programming error
            raise ValueError(f"Unsupported media validation kind: {kind}")
        for signature, result in zip(signatures, results, strict=True):
            self.put(kind, signature, result)

    def prefetch_images(
        self,
        paths: Iterable[str],
        *,
        executor: ThreadPoolExecutor,
    ) -> None:
        self._prefetch("image", paths, executor=executor)

    def prefetch_videos(
        self,
        paths: Iterable[str],
        *,
        executor: ThreadPoolExecutor,
    ) -> None:
        self._prefetch("video", paths, executor=executor)

    def _require(self, kind: str, path: str) -> _ValidationResult:
        signature = self.signature(path)
        result = self.get(kind, signature)
        if result is None:
            raise RuntimeError(
                f"Media validation cache miss after batch prefetch: kind={kind}, path={signature.path}"
            )
        return result

    def require_image(self, path: str) -> tuple[int, int]:
        return self._require("image", path).unwrap_image_size()

    def require_video(self, path: str) -> _ValidationResult:
        return self._require("video", path)

    validate_image = require_image


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


def _batched(rows: Iterable[_T], batch_size: int) -> Iterator[list[_T]]:
    iterator = iter(rows)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def _probe_resolved_image(path: str) -> _ValidationResult:
    try:
        if not Path(path).is_file():
            return _ValidationResult(None, "missing_media", f"Image does not exist: {path}")
        width, height = _verified_image_size(path)
        return _ValidationResult({"width": width, "height": height})
    except Exception as exc:
        return _ValidationResult(None, "invalid_image", f"Unreadable image {path}: {exc}")


def _probe_resolved_video(path: str) -> _ValidationResult:
    try:
        if not Path(path).is_file():
            return _ValidationResult(None, "missing_target", f"Target video does not exist: {path}")
        return _ValidationResult(dict(probe_video(path)))
    except Exception as exc:
        return _ValidationResult(None, "invalid_video_header", str(exc))


def _collect_i2i_media_paths(
    row: Mapping[str, Any],
    *,
    data_root: str | Path | None,
    target_field: str,
    reference_field: str,
) -> list[str]:
    paths: list[str] = []
    try:
        if target_field in row:
            paths.append(resolve_media_path(row[target_field], data_root=data_root))
    except Exception:
        pass
    try:
        if reference_field in row:
            paths.extend(
                resolve_media_path(path, data_root=data_root)
                for path in parse_path_list(row[reference_field], field=reference_field)
            )
    except Exception:
        pass
    return paths


def _iter_canonical_rows_with_bounded_probes(
    rows: Iterable[tuple[str, dict[str, Any]]],
    *,
    dataset_name: str,
    dataset_type: str,
    data_root: str | Path | None,
    adapter_config: Mapping[str, Any] | None,
    manifest_seed: int,
    executor: ThreadPoolExecutor,
    batch_size: int,
    validation_cache: _MediaValidationCache,
) -> Iterator[tuple[str, CanonicalR2VSource | None, dict[str, Any] | None, ManifestReject | None]]:
    """Normalize R2V rows, probe canonical paths in bounded parallel batches, and preserve order."""
    for row_batch in _batched(rows, batch_size):
        canonicals: list[CanonicalR2VSource | None] = []
        normalization_errors: list[ManifestReject | None] = []
        for source_record_id, row in row_batch:
            try:
                canonical = normalize_r2v_source(
                    row,
                    row_id=source_record_id,
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    data_root=data_root,
                    adapter_config=adapter_config,
                    manifest_seed=manifest_seed,
                )
            except ManifestReject as exc:
                canonicals.append(None)
                normalization_errors.append(exc)
                continue
            canonicals.append(canonical)
            normalization_errors.append(None)

        validation_cache.prefetch_videos(
            (canonical.video_path for canonical in canonicals if canonical is not None),
            executor=executor,
        )
        validation_cache.prefetch_images(
            (
                path
                for canonical in canonicals
                if canonical is not None
                for path in canonical.reference_paths
            ),
            executor=executor,
        )

        for (source_record_id, _row), canonical, error in zip(
            row_batch,
            canonicals,
            normalization_errors,
            strict=True,
        ):
            if error is not None:
                yield source_record_id, None, None, error
                continue
            if canonical is None:  # pragma: no cover - guarded by paired error state
                raise RuntimeError("Normalized R2V source is missing without an error")
            header, probe_error = validation_cache.require_video(canonical.video_path).as_probe_result()
            yield source_record_id, canonical, header, probe_error


def _peak_rss_gb() -> float:
    if resource is None:
        return 0.0
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

        with (
            ThreadPoolExecutor(
                max_workers=probe_workers,
                thread_name_prefix="manifest-media",
            ) as executor,
            output_temporary.open("w", encoding="utf-8") as accepted_handle,
            reject_temporary.open("w", encoding="utf-8") as reject_handle,
        ):
            def accept_record(record: dict[str, Any]) -> None:
                nonlocal accepted_rows, duplicate_rows
                sample_key = str(record["sample_key"])
                plan_sha = str(record["sample_plan_sha256"])
                existing = connection.execute(
                    "SELECT sample_plan_sha256 FROM dedup WHERE sample_key = ?",
                    (sample_key,),
                ).fetchone()
                if existing is not None:
                    if str(existing[0]) != plan_sha:
                        raise _ManifestCollisionError(f"sample_key collision with different plans: {sample_key}")
                    duplicate_rows += 1
                    return
                connection.execute("INSERT INTO dedup VALUES (?, ?)", (sample_key, plan_sha))
                accepted_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                accepted_rows += 1
                task_counts[str(record["task"])] += 1
                if accepted_rows % 10_000 == 0:
                    connection.commit()

            def reject_record(
                *,
                dataset_name: str,
                row_index: int,
                task: str,
                source_record_id: str,
                exc: Exception,
            ) -> None:
                nonlocal rejected_rows
                reason = exc.reason if isinstance(exc, ManifestReject) else type(exc).__name__
                reject_handle.write(
                    json.dumps(
                        {
                            "dataset_name": dataset_name,
                            "row_index": row_index,
                            "task": task,
                            "reason": reason,
                            "source_record_id": source_record_id,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                rejected_rows += 1
                reject_reason_counts[reason] += 1

            for dataset in config["datasets"]:
                if not isinstance(dataset, dict):
                    raise ValueError("Each datasets entry must be a mapping")
                task = str(dataset.get("task", ""))
                dataset_name = str(dataset.get("name", task))
                annotation_path = dataset.get("ann_path") or dataset.get("parquet") or dataset.get("path")
                if annotation_path is None:
                    raise ValueError(f"Dataset {dataset_name!r} has no ann_path/parquet/path")
                rows: Iterable[tuple[str, dict[str, Any]]] = iter_annotation_items(
                    annotation_path,
                    batch_size=annotation_batch_size,
                )
                max_samples = dataset.get("max_samples")
                if max_samples is not None:
                    rows = islice(rows, int(max_samples))
                data_root = dataset.get("data_root", config.get("data_root"))
                dataset_type = str(dataset.get("dataset_type", ""))
                adapter_config = dataset.get("adapter")

                if task == VIDEO_TASK:
                    probed_rows = _iter_canonical_rows_with_bounded_probes(
                        rows,
                        dataset_name=dataset_name,
                        dataset_type=dataset_type,
                        data_root=data_root,
                        adapter_config=adapter_config,
                        manifest_seed=manifest_seed,
                        executor=executor,
                        batch_size=probe_batch_size,
                        validation_cache=validation_cache,
                    )
                    for row_index, (source_record_id, canonical, video_header, probe_error) in enumerate(probed_rows):
                        raw_rows += 1
                        try:
                            if probe_error is not None:
                                raise probe_error
                            if canonical is None or video_header is None:
                                raise RuntimeError("R2V canonical row was not probed")
                            record = build_canonical_r2v_record(
                                canonical,
                                manifest_seed=manifest_seed,
                                anchor_frame_ratio=float(dataset.get("anchor_frame_ratio", 0.10)),
                                video_header=video_header,
                                image_validator=validation_cache.require_image,
                                target_path_validated=True,
                            )
                            accept_record(record)
                        except Exception as exc:
                            if isinstance(exc, _ManifestCollisionError):
                                raise
                            reject_record(
                                dataset_name=dataset_name,
                                row_index=row_index,
                                task=task,
                                source_record_id=source_record_id,
                                exc=exc,
                            )
                    continue

                if task != IMAGE_TASK:
                    raise ValueError(f"Unsupported task {task!r} in dataset {dataset_name!r}")

                row_index = 0
                for row_batch in _batched(rows, probe_batch_size):
                    image_paths = [
                        path
                        for _source_record_id, row in row_batch
                        for path in _collect_i2i_media_paths(
                            row,
                            data_root=data_root,
                            target_field=i2i_target_field,
                            reference_field=i2i_reference_field,
                        )
                    ]
                    validation_cache.prefetch_images(image_paths, executor=executor)
                    for source_record_id, row in row_batch:
                        raw_rows += 1
                        try:
                            record = build_i2i_record(
                                row,
                                dataset_name=dataset_name,
                                data_root=data_root,
                                target_field=i2i_target_field,
                                reference_field=i2i_reference_field,
                                caption_field=i2i_caption_field,
                                crop_field=i2i_crop_field,
                                image_validator=validation_cache.require_image,
                            )
                            accept_record(record)
                        except Exception as exc:
                            if isinstance(exc, _ManifestCollisionError):
                                raise
                            reject_record(
                                dataset_name=dataset_name,
                                row_index=row_index,
                                task=task,
                                source_record_id=source_record_id,
                                exc=exc,
                            )
                        row_index += 1
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
            "image_probe_submitted": validation_cache.image_probe_submitted,
            "video_probe_submitted": validation_cache.video_probe_submitted,
            "image_cache_hits": validation_cache.image_cache_hits,
            "image_cache_misses": validation_cache.image_cache_misses,
            "video_cache_hits": validation_cache.video_cache_hits,
            "video_cache_misses": validation_cache.video_cache_misses,
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
