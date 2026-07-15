"""Sharded, resumable, deterministic construction of online manifests."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import resource
import shutil
import socket
import sqlite3
import struct
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_NUM_FRAMES, VIDEO_TASK
from ltx_trainer.online_data.manifest import (
    ManifestReject,
    build_i2i_record,
    build_r2v_record,
    iter_annotation_rows,
    load_multitask_data_config,
    parse_numeric_list,
    parse_pair,
    parse_path_list,
    probe_video,
)
from ltx_trainer.online_data.manifest_index import (
    build_manifest_offset_index,
    default_manifest_index_path,
    validate_manifest_index,
)
from ltx_trainer.online_data.video_probe_pool import PersistentVideoProbePool, VideoProbeError

JSONL_INDEX_MAGIC = b"LTXJIDX1"
JSONL_INDEX_HEADER = struct.Struct("<Q")
JSONL_INDEX_ENTRY = struct.Struct("<Q")
BUILD_METADATA_VERSION = 1
SHARD_MARKER_VERSION = 1
R2V_FILTER_SEMANTIC_VERSION = 2
TASK_ORDER = (IMAGE_TASK, VIDEO_TASK)
logger = logging.getLogger(__name__)


class ManifestCollisionError(RuntimeError):
    """Raised when a stable sample key maps to two different sample plans."""


class MediaValidationError(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class AnnotationSource:
    task: str
    dataset_name: str
    dataset_order: int
    annotation_path: str
    data_root: str | None
    row_count: int
    task_start_row: int
    task_end_row: int
    size: int
    mtime_ns: int
    fingerprint: str
    suffix: str
    jsonl_index_path: str | None = None


@dataclass(frozen=True)
class BuildOptions:
    shards_per_task: int = 8
    image_workers: int = 16
    video_workers: int = 8
    max_in_flight: int = 256
    annotation_batch_size: int = 4096
    probe_timeout_seconds: float = 60.0
    video_probe_mode: str = "persistent"
    video_probe_max_tasks_per_worker: int = 1000
    resume_build: bool = True
    progress_interval_seconds: float = 10.0
    manifest_seed: int = 42
    i2i_target_field: str = "image"
    i2i_reference_field: str = "edit_image"
    i2i_caption_field: str = "prompt"
    i2i_crop_field: str | None = None
    max_samples_per_task: int | None = None
    recover_stale_locks: bool = False
    stale_lock_seconds: float = 21_600.0

    def validate(self) -> None:
        integer_values = {
            "shards_per_task": self.shards_per_task,
            "image_workers": self.image_workers,
            "video_workers": self.video_workers,
            "max_in_flight": self.max_in_flight,
            "annotation_batch_size": self.annotation_batch_size,
            "video_probe_max_tasks_per_worker": self.video_probe_max_tasks_per_worker,
        }
        for name, value in integer_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.probe_timeout_seconds <= 0:
            raise ValueError("probe_timeout_seconds must be positive")
        if self.video_probe_mode not in {"persistent", "isolated"}:
            raise ValueError("video_probe_mode must be 'persistent' or 'isolated'")
        if self.progress_interval_seconds <= 0:
            raise ValueError("progress_interval_seconds must be positive")
        if self.max_samples_per_task is not None and self.max_samples_per_task <= 0:
            raise ValueError("max_samples_per_task must be positive when set")


@dataclass
class RowBuildResult:
    task_row_index: int
    source_row_index: int
    dataset_order: int
    dataset_name: str
    record: dict[str, Any] | None = None
    rejection: dict[str, Any] | None = None


@dataclass(frozen=True)
class R2VPrefilterResult:
    target_path: str
    reference_paths: tuple[str, ...]
    caption: str
    crop_xyxy: tuple[float, float, float, float]
    face_cut: tuple[int, int]


class BuildRuntimeStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counts[name] += value

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def update(self, values: Mapping[str, int]) -> None:
        with self._lock:
            self._counts.update(values)


def prefilter_r2v_annotation(
    row: Mapping[str, Any],
    *,
    data_root: str | Path | None,
) -> R2VPrefilterResult:
    """Reject annotation-only impossibilities without touching any media."""
    required = {"video_path", "text", "crop", "face_cut", "ref_images"}
    missing = sorted(required - row.keys())
    if missing:
        raise ManifestReject(
            "r2v_schema_mismatch",
            f"OpenS2V adapter fields are missing: {missing}. Available columns: {sorted(row)}",
        )
    caption = str(row["text"]).strip() if row["text"] is not None else ""
    if not caption:
        raise ManifestReject("empty_caption", "Caption/instruction is empty")
    try:
        crop_raw = parse_numeric_list(row["crop"], field="crop")
    except Exception as exc:
        raise ManifestReject("invalid_crop", "Invalid OpenS2V crop") from exc
    if not all(math.isfinite(value) for value in crop_raw):
        raise ManifestReject("invalid_crop", f"OpenS2V crop contains non-finite values: {crop_raw}")
    if crop_raw[1] <= crop_raw[0] or crop_raw[3] <= crop_raw[2]:
        raise ManifestReject("invalid_crop", f"OpenS2V crop has non-positive extent: {crop_raw}")
    try:
        face_cut = parse_pair(row["face_cut"], field="face_cut")
    except Exception as exc:
        raise ManifestReject("invalid_face_cut", "Invalid OpenS2V face_cut") from exc
    if face_cut[0] < 0 or face_cut[1] <= face_cut[0]:
        raise ManifestReject("invalid_face_cut", f"Invalid face_cut={face_cut}")
    try:
        reference_values = parse_path_list(row["ref_images"], field="ref_images")
    except Exception as exc:
        raise ManifestReject("missing_reference", "No usable reference paths in ref_images") from exc
    if face_cut[1] - face_cut[0] < VIDEO_NUM_FRAMES:
        raise ManifestReject(
            "insufficient_face_cut_span_for_121",
            f"face_cut={face_cut} contains fewer than {VIDEO_NUM_FRAMES} source frames",
        )
    def annotation_path(value: Any) -> str:
        path = Path(str(value)).expanduser()
        if not path.is_absolute() and data_root is not None:
            path = Path(data_root).expanduser() / path
        return os.path.abspath(path)  # noqa: PTH100 - Stage A must not resolve or stat media.

    return R2VPrefilterResult(
        target_path=annotation_path(row["video_path"]),
        reference_paths=tuple(annotation_path(path) for path in reference_values),
        caption=caption,
        crop_xyxy=(crop_raw[0], crop_raw[2], crop_raw[1], crop_raw[3]),
        face_cut=(face_cut[0], face_cut[1]),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_tasks(value: str | Iterable[str]) -> list[str]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    requested = {str(item).strip().lower() for item in raw if str(item).strip()}
    unsupported = requested - set(TASK_ORDER)
    if unsupported:
        raise ValueError(f"Unsupported tasks: {sorted(unsupported)}")
    if not requested:
        raise ValueError("At least one task must be selected")
    return [task for task in TASK_ORDER if task in requested]


def canonical_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.uname().sysname == "Darwin":
        return value / (1024**3)
    return value / (1024**2)


def _flush_and_sync(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            _flush_and_sync(handle)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _source_fingerprint(path: Path) -> tuple[os.stat_result, str]:
    stat_result = path.stat()
    payload = {
        "path": str(path),
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
    }
    return stat_result, canonical_fingerprint(payload)


class ShardLock:
    """Exclusive lock whose stale takeover always requires explicit opt-in."""

    def __init__(
        self,
        path: str | Path,
        *,
        config_fingerprint: str,
        recover_stale: bool = False,
        stale_after_seconds: float = 21_600.0,
    ) -> None:
        self.path = Path(path)
        self.config_fingerprint = config_fingerprint
        self.recover_stale = recover_stale
        self.stale_after_seconds = stale_after_seconds
        self._owned = False

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _existing_lock_is_active(self, payload: Mapping[str, Any]) -> bool:
        hostname = str(payload.get("hostname", ""))
        pid = int(payload.get("pid", -1))
        if hostname == socket.gethostname():
            return self._pid_is_alive(pid)
        started_epoch = float(payload.get("started_epoch", time.time()))
        return time.time() - started_epoch < self.stale_after_seconds

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": utc_now(),
            "started_epoch": time.time(),
            "config_fingerprint": self.config_fingerprint,
        }
        while True:
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                try:
                    existing = json.loads(self.path.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
                if self._existing_lock_is_active(existing):
                    raise RuntimeError(f"Active shard lock exists: {self.path}")
                if not self.recover_stale:
                    raise RuntimeError(
                        f"Stale shard lock exists: {self.path}. "
                        "Use --recover-stale-locks to take it over."
                    )
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
                _flush_and_sync(handle)
            self._owned = True
            return

    def release(self) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False

    def __enter__(self) -> ShardLock:
        self.acquire()
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()


def _jsonl_index_header(path: Path, fingerprint: str, row_count: int) -> dict[str, Any]:
    stat_result = path.stat()
    return {
        "version": 1,
        "source_path": str(path),
        "source_size": stat_result.st_size,
        "source_mtime_ns": stat_result.st_mtime_ns,
        "source_fingerprint": fingerprint,
        "row_count": row_count,
        "offset_entry_size": JSONL_INDEX_ENTRY.size,
    }


def read_jsonl_offset_index(path: str | Path) -> tuple[dict[str, Any], int]:
    index_path = Path(path)
    with index_path.open("rb") as handle:
        if handle.read(len(JSONL_INDEX_MAGIC)) != JSONL_INDEX_MAGIC:
            raise ValueError(f"Invalid JSONL offset index magic: {index_path}")
        packed_length = handle.read(JSONL_INDEX_HEADER.size)
        if len(packed_length) != JSONL_INDEX_HEADER.size:
            raise ValueError(f"Truncated JSONL offset index: {index_path}")
        (header_length,) = JSONL_INDEX_HEADER.unpack(packed_length)
        header = json.loads(handle.read(header_length))
        entries_start = len(JSONL_INDEX_MAGIC) + JSONL_INDEX_HEADER.size + header_length
    expected_size = entries_start + int(header["row_count"]) * JSONL_INDEX_ENTRY.size
    if index_path.stat().st_size != expected_size:
        raise ValueError(f"JSONL offset index size mismatch: {index_path}")
    return header, entries_start


def build_jsonl_offset_index(
    source_path: str | Path,
    index_dir: str | Path,
    *,
    recover_stale_locks: bool = False,
) -> tuple[Path, dict[str, Any]]:
    source = Path(source_path).expanduser().resolve()
    _, fingerprint = _source_fingerprint(source)
    destination = Path(index_dir) / f"{fingerprint}.jsonl_offsets.idx"
    if destination.exists():
        try:
            header, _ = read_jsonl_offset_index(destination)
            expected = _jsonl_index_header(source, fingerprint, int(header["row_count"]))
            if all(header.get(key) == value for key, value in expected.items()):
                return destination, header
        except Exception:
            pass

    lock = ShardLock(
        f"{destination}.lock",
        config_fingerprint=fingerprint,
        recover_stale=recover_stale_locks,
    )
    with lock:
        if destination.exists():
            try:
                header, _ = read_jsonl_offset_index(destination)
                expected = _jsonl_index_header(source, fingerprint, int(header["row_count"]))
                if all(header.get(key) == value for key, value in expected.items()):
                    return destination, header
            except Exception:
                destination.unlink(missing_ok=True)
        raw_offsets = Path(f"{destination}.offsets.tmp.{os.getpid()}")
        temporary = Path(f"{destination}.tmp.{os.getpid()}")
        row_count = 0
        try:
            with source.open("rb") as source_handle, raw_offsets.open("wb") as offsets_handle:
                while True:
                    offset = source_handle.tell()
                    line = source_handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    json.loads(line)
                    offsets_handle.write(JSONL_INDEX_ENTRY.pack(offset))
                    row_count += 1
                _flush_and_sync(offsets_handle)
            header = _jsonl_index_header(source, fingerprint, row_count)
            encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
            with temporary.open("wb") as output:
                output.write(JSONL_INDEX_MAGIC)
                output.write(JSONL_INDEX_HEADER.pack(len(encoded_header)))
                output.write(encoded_header)
                with raw_offsets.open("rb") as offsets_handle:
                    shutil.copyfileobj(offsets_handle, output, length=1024 * 1024)
                _flush_and_sync(output)
            temporary.replace(destination)
            return destination, header
        finally:
            raw_offsets.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)


def _annotation_row_count(
    path: Path,
    *,
    index_dir: Path,
    recover_stale_locks: bool,
) -> tuple[int, str | None]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("Parallel parquet manifests require pyarrow") from exc
        return int(parquet.ParquetFile(path).metadata.num_rows), None
    if suffix == ".jsonl":
        index, header = build_jsonl_offset_index(
            path,
            index_dir,
            recover_stale_locks=recover_stale_locks,
        )
        return int(header["row_count"]), str(index)
    return sum(1 for _ in iter_annotation_rows(path)), None


def discover_annotation_sources(
    train_data_config: str | Path,
    *,
    shard_root: str | Path,
    tasks: Iterable[str],
    max_samples_per_task: int | None = None,
    recover_stale_locks: bool = False,
) -> dict[str, list[AnnotationSource]]:
    config = load_multitask_data_config(train_data_config)
    selected = set(parse_tasks(tasks))
    root = Path(shard_root)
    index_dir = root / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    task_offsets = {task: 0 for task in selected}
    sources: dict[str, list[AnnotationSource]] = {task: [] for task in selected}
    global_root = config.get("data_root")
    for dataset_order, dataset in enumerate(config["datasets"]):
        task = str(dataset.get("task", "")).lower()
        if task not in selected:
            continue
        annotation_value = dataset.get("ann_path") or dataset.get("parquet") or dataset.get("path")
        if annotation_value is None:
            raise ValueError(f"Dataset {dataset.get('name', dataset_order)!r} has no annotation path")
        annotation_path = Path(annotation_value).expanduser().resolve()
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")
        count, jsonl_index = _annotation_row_count(
            annotation_path,
            index_dir=index_dir,
            recover_stale_locks=recover_stale_locks,
        )
        remaining = None
        if max_samples_per_task is not None:
            remaining = max(0, max_samples_per_task - task_offsets[task])
            count = min(count, remaining)
        if count == 0:
            continue
        stat_result, fingerprint = _source_fingerprint(annotation_path)
        data_root = dataset.get("data_root", global_root)
        start = task_offsets[task]
        end = start + count
        sources[task].append(
            AnnotationSource(
                task=task,
                dataset_name=str(dataset.get("name", f"dataset_{dataset_order}")),
                dataset_order=dataset_order,
                annotation_path=str(annotation_path),
                data_root=str(Path(data_root).expanduser().resolve()) if data_root is not None else None,
                row_count=count,
                task_start_row=start,
                task_end_row=end,
                size=stat_result.st_size,
                mtime_ns=stat_result.st_mtime_ns,
                fingerprint=fingerprint,
                suffix=annotation_path.suffix.lower(),
                jsonl_index_path=jsonl_index,
            )
        )
        task_offsets[task] = end
    for task in selected:
        if not sources[task]:
            raise ValueError(f"No configured annotation sources found for task {task}")
    return sources


def _iter_jsonl_range(source: AnnotationSource, start: int, end: int) -> Iterator[tuple[int, dict[str, Any]]]:
    if source.jsonl_index_path is None:
        raise ValueError(f"JSONL source has no offset index: {source.annotation_path}")
    header, entries_start = read_jsonl_offset_index(source.jsonl_index_path)
    if header["source_fingerprint"] != source.fingerprint:
        raise ValueError(f"Stale JSONL offset index for {source.annotation_path}")
    with Path(source.jsonl_index_path).open("rb") as index_handle, Path(source.annotation_path).open(
        "rb"
    ) as source_handle:
        index_handle.seek(entries_start + start * JSONL_INDEX_ENTRY.size)
        for row_index in range(start, end):
            packed = index_handle.read(JSONL_INDEX_ENTRY.size)
            if len(packed) != JSONL_INDEX_ENTRY.size:
                raise ValueError(f"Truncated JSONL offset index for {source.annotation_path}")
            (offset,) = JSONL_INDEX_ENTRY.unpack(packed)
            source_handle.seek(offset)
            row = json.loads(source_handle.readline())
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row {row_index} is not an object: {source.annotation_path}")
            yield row_index, {str(key): value for key, value in row.items()}


def _iter_parquet_range(
    source: AnnotationSource,
    start: int,
    end: int,
    *,
    batch_size: int,
) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as parquet  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("Parallel parquet manifests require pyarrow") from exc
    parquet_file = parquet.ParquetFile(source.annotation_path)
    row_group_start = 0
    for row_group in range(parquet_file.num_row_groups):
        group_rows = parquet_file.metadata.row_group(row_group).num_rows
        group_end = row_group_start + group_rows
        overlap_start = max(start, row_group_start)
        overlap_end = min(end, group_end)
        if overlap_start < overlap_end:
            batch_start = row_group_start
            for batch in parquet_file.iter_batches(batch_size=batch_size, row_groups=[row_group]):
                batch_end = batch_start + batch.num_rows
                local_start = max(overlap_start, batch_start) - batch_start
                local_end = min(overlap_end, batch_end) - batch_start
                if local_start < local_end:
                    rows = batch.slice(local_start, local_end - local_start).to_pylist()
                    absolute_start = batch_start + local_start
                    for offset, row in enumerate(rows):
                        yield absolute_start + offset, {str(key): value for key, value in row.items()}
                batch_start = batch_end
        row_group_start = group_end
        if row_group_start >= end:
            break


def iter_source_range(
    source: AnnotationSource,
    start: int,
    end: int,
    *,
    batch_size: int,
) -> Iterator[tuple[int, dict[str, Any]]]:
    if not 0 <= start <= end <= source.row_count:
        raise ValueError(f"Invalid source range [{start}, {end}) for {source.dataset_name}")
    if source.suffix == ".jsonl":
        yield from _iter_jsonl_range(source, start, end)
        return
    if source.suffix == ".parquet":
        yield from _iter_parquet_range(source, start, end, batch_size=batch_size)
        return
    for row_index, row in enumerate(iter_annotation_rows(source.annotation_path, batch_size=batch_size)):
        if row_index >= end:
            break
        if row_index >= start:
            yield row_index, row


def iter_task_range(
    sources: Iterable[AnnotationSource],
    start: int,
    end: int,
    *,
    batch_size: int,
) -> Iterator[tuple[int, AnnotationSource, int, dict[str, Any]]]:
    for source in sources:
        overlap_start = max(start, source.task_start_row)
        overlap_end = min(end, source.task_end_row)
        if overlap_start >= overlap_end:
            continue
        local_start = overlap_start - source.task_start_row
        local_end = overlap_end - source.task_start_row
        for source_row_index, row in iter_source_range(
            source,
            local_start,
            local_end,
            batch_size=batch_size,
        ):
            yield source.task_start_row + source_row_index, source, source_row_index, row


def shard_bounds(total_rows: int, shards: int, shard_id: int) -> tuple[int, int]:
    if shards <= 0 or not 0 <= shard_id < shards:
        raise ValueError(f"Invalid shard request: total={total_rows}, shards={shards}, id={shard_id}")
    return math.floor(shard_id * total_rows / shards), math.floor((shard_id + 1) * total_rows / shards)


def _probe_child(path: str, connection: Any) -> None:
    try:
        connection.send((True, dict(probe_video(path))))
    except BaseException as exc:  # native decoder errors must cross the process boundary
        connection.send((False, type(exc).__name__, str(exc)))
    finally:
        connection.close()


def probe_video_isolated(path: str, timeout_seconds: float) -> dict[str, Any]:
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(target=_probe_child, args=(path, child_connection), daemon=True)
    process.start()
    child_connection.close()
    try:
        if not parent_connection.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=5.0)
            raise MediaValidationError("video_probe_timeout", f"Video probe timed out after {timeout_seconds}s: {path}")
        result = parent_connection.recv()
        process.join(timeout=5.0)
        if not result[0]:
            raise MediaValidationError("invalid_video_header", f"{result[1]}: {result[2]}")
        return dict(result[1])
    finally:
        parent_connection.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)


class PersistentMediaCache:
    """Task-local SQLite cache with in-flight de-duplication and serialized writes."""

    def __init__(
        self,
        path: str | Path,
        *,
        probe_timeout_seconds: float,
        probe_runner: Callable[[str, float], Mapping[str, Any]] = probe_video_isolated,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.probe_timeout_seconds = probe_timeout_seconds
        self.probe_runner = probe_runner
        self._lock = threading.Lock()
        self._inflight: dict[tuple[str, str, int, int], Future[dict[str, Any]]] = {}
        self.hits = 0
        self.misses = 0
        self.image_hits = 0
        self.image_misses = 0
        self.video_hits = 0
        self.video_misses = 0
        self.probe_success = 0
        self.probe_timeout = 0
        self.invalid_video_header = 0
        self._connection = self._open_connection()
        self._write_queue: queue.Queue[
            tuple[
                tuple[str, str, int, int],
                Mapping[str, Any] | None,
                MediaValidationError | None,
                threading.Event,
                list[BaseException],
            ]
            | None
        ] = queue.Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"manifest-cache-writer-{self.path.parent.name}",
            daemon=True,
        )
        self._writer_thread.start()

    def _open_connection(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.path, timeout=60.0, check_same_thread=False)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=60000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS media_validation (
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    ok INTEGER NOT NULL,
                    payload_json TEXT,
                    reason TEXT,
                    message TEXT,
                    PRIMARY KEY(kind, path, size, mtime_ns)
                )
                """
            )
            connection.commit()
            connection.execute("SELECT COUNT(*) FROM media_validation").fetchone()
            return connection
        except sqlite3.DatabaseError:
            if connection is not None:
                connection.close()
            self.path.unlink(missing_ok=True)
            Path(f"{self.path}-wal").unlink(missing_ok=True)
            Path(f"{self.path}-shm").unlink(missing_ok=True)
            connection = sqlite3.connect(self.path, timeout=60.0, check_same_thread=False)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE media_validation (
                    kind TEXT NOT NULL, path TEXT NOT NULL, size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL, ok INTEGER NOT NULL, payload_json TEXT,
                    reason TEXT, message TEXT,
                    PRIMARY KEY(kind, path, size, mtime_ns)
                )
                """
            )
            connection.commit()
            return connection

    @staticmethod
    def _signature(kind: str, path: str) -> tuple[str, str, int, int]:
        resolved = str(Path(path).expanduser().resolve())
        stat_result = Path(resolved).stat()
        return kind, resolved, stat_result.st_size, stat_result.st_mtime_ns

    def _lookup(self, signature: tuple[str, str, int, int]) -> tuple[bool, dict[str, Any]] | None:
        row = self._connection.execute(
            "SELECT ok, payload_json, reason, message FROM media_validation "
            "WHERE kind=? AND path=? AND size=? AND mtime_ns=?",
            signature,
        ).fetchone()
        if row is None:
            return None
        if not row[0]:
            raise MediaValidationError(str(row[2]), str(row[3]))
        return True, json.loads(row[1])

    def _store(
        self,
        signature: tuple[str, str, int, int],
        *,
        payload: Mapping[str, Any] | None = None,
        error: MediaValidationError | None = None,
    ) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO media_validation "
            "(kind,path,size,mtime_ns,ok,payload_json,reason,message) VALUES (?,?,?,?,?,?,?,?)",
            (
                *signature,
                error is None,
                json.dumps(payload) if payload is not None else None,
                error.reason if error else None,
                str(error) if error else None,
            ),
        )
        self._connection.commit()

    def _writer_loop(self) -> None:
        while True:
            request = self._write_queue.get()
            if request is None:
                return
            signature, payload, error, completed, failures = request
            try:
                with self._lock:
                    self._store(signature, payload=payload, error=error)
            except BaseException as exc:
                failures.append(exc)
            finally:
                completed.set()

    def _persist(
        self,
        signature: tuple[str, str, int, int],
        *,
        payload: Mapping[str, Any] | None = None,
        error: MediaValidationError | None = None,
    ) -> None:
        completed = threading.Event()
        failures: list[BaseException] = []
        self._write_queue.put((signature, payload, error, completed, failures))
        completed.wait()
        if failures:
            raise RuntimeError(f"Could not write media cache {self.path}") from failures[0]

    def _get_or_compute(
        self,
        kind: str,
        path: str,
        compute: Callable[[str], Mapping[str, Any]],
    ) -> dict[str, Any]:
        try:
            signature = self._signature(kind, path)
        except Exception as exc:
            raise MediaValidationError("missing_media", f"Media is missing or unreadable: {path}") from exc
        owner = False
        with self._lock:
            try:
                cached = self._lookup(signature)
            except MediaValidationError:
                self._record_cache_hit(kind)
                raise
            if cached is not None:
                self._record_cache_hit(kind)
                return cached[1]
            future = self._inflight.get(signature)
            if future is None:
                future = Future()
                self._inflight[signature] = future
                self._record_cache_miss(kind)
                owner = True
            else:
                self._record_cache_hit(kind)
        if not owner:
            return future.result()
        try:
            payload = dict(compute(signature[1]))
        except MediaValidationError as exc:
            try:
                self._persist(signature, error=exc)
            except BaseException as persist_exc:
                with self._lock:
                    future.set_exception(persist_exc)
                    self._inflight.pop(signature, None)
                raise persist_exc from exc
            with self._lock:
                future.set_exception(exc)
                self._inflight.pop(signature, None)
            raise
        except Exception as exc:
            reason = "invalid_video_header" if kind == "video" else "invalid_image"
            wrapped = MediaValidationError(reason, str(exc))
            try:
                self._persist(signature, error=wrapped)
            except BaseException as persist_exc:
                with self._lock:
                    future.set_exception(persist_exc)
                    self._inflight.pop(signature, None)
                raise persist_exc from wrapped
            with self._lock:
                future.set_exception(wrapped)
                self._inflight.pop(signature, None)
            raise wrapped from exc
        try:
            self._persist(signature, payload=payload)
        except BaseException as exc:
            with self._lock:
                future.set_exception(exc)
                self._inflight.pop(signature, None)
            raise
        with self._lock:
            future.set_result(payload)
            self._inflight.pop(signature, None)
        return payload

    def _record_cache_hit(self, kind: str) -> None:
        self.hits += 1
        if kind == "video":
            self.video_hits += 1
        else:
            self.image_hits += 1

    def _record_cache_miss(self, kind: str) -> None:
        self.misses += 1
        if kind == "video":
            self.video_misses += 1
        else:
            self.image_misses += 1

    def validate_image(self, path: str) -> tuple[int, int]:
        def compute(resolved: str) -> Mapping[str, Any]:
            with Image.open(resolved) as image:
                image.verify()
            with Image.open(resolved) as image:
                width, height = ImageOps.exif_transpose(image).size
            return {"width": int(width), "height": int(height)}

        result = self._get_or_compute("image", path, compute)
        return int(result["width"]), int(result["height"])

    def probe_video(self, path: str) -> dict[str, Any]:
        def compute(resolved: str) -> Mapping[str, Any]:
            try:
                result = self.probe_runner(resolved, self.probe_timeout_seconds)
            except VideoProbeError as exc:
                error = MediaValidationError(exc.reason, str(exc))
                with self._lock:
                    if error.reason == "video_probe_timeout":
                        self.probe_timeout += 1
                    else:
                        self.invalid_video_header += 1
                raise error from exc
            except MediaValidationError as exc:
                with self._lock:
                    if exc.reason == "video_probe_timeout":
                        self.probe_timeout += 1
                    else:
                        self.invalid_video_header += 1
                raise
            with self._lock:
                self.probe_success += 1
            return result

        return self._get_or_compute("video", path, compute)

    def close(self) -> None:
        self._write_queue.put(None)
        self._writer_thread.join()
        with self._lock:
            self._connection.close()


def _semantic_build_payload(task: str, options: BuildOptions, sources: Iterable[AnnotationSource]) -> dict[str, Any]:
    payload = {
        "version": BUILD_METADATA_VERSION,
        "task": task,
        "shards_per_task": options.shards_per_task,
        "probe_timeout_seconds": options.probe_timeout_seconds,
        "manifest_seed": options.manifest_seed,
        "i2i_target_field": options.i2i_target_field,
        "i2i_reference_field": options.i2i_reference_field,
        "i2i_caption_field": options.i2i_caption_field,
        "i2i_crop_field": options.i2i_crop_field,
        "max_samples_per_task": options.max_samples_per_task,
        "sources": [
            {
                "dataset_name": source.dataset_name,
                "dataset_order": source.dataset_order,
                "annotation_path": source.annotation_path,
                "data_root": source.data_root,
                "row_count": source.row_count,
                "task_start_row": source.task_start_row,
                "task_end_row": source.task_end_row,
                "fingerprint": source.fingerprint,
            }
            for source in sources
        ],
    }
    if task == VIDEO_TASK:
        payload["r2v_filter_semantic_version"] = R2V_FILTER_SEMANTIC_VERSION
    return payload


def task_build_fingerprint(task: str, options: BuildOptions, sources: Iterable[AnnotationSource]) -> str:
    return canonical_fingerprint(_semantic_build_payload(task, options, sources))


def _update_build_metadata(
    shard_root: Path,
    *,
    train_data_config: Path,
    task: str,
    options: BuildOptions,
    sources: list[AnnotationSource],
    build_fingerprint: str,
) -> None:
    metadata_path = shard_root / "build_metadata.json"
    lock_path = shard_root / "build_metadata.lock"
    deadline = time.monotonic() + 60.0
    while True:
        try:
            lock = ShardLock(
                lock_path,
                config_fingerprint="build_metadata",
                recover_stale=options.recover_stale_locks,
                stale_after_seconds=options.stale_lock_seconds,
            )
            lock.acquire()
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)
    try:
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            payload = {
                "version": BUILD_METADATA_VERSION,
                "created_at": utc_now(),
                "train_data_config": str(train_data_config),
                "tasks": {},
            }
        existing_config = payload.get("train_data_config")
        if existing_config != str(train_data_config):
            raise ValueError(
                f"Shard root belongs to a different data config: {existing_config} != {train_data_config}"
            )
        task_payload = {
            "task": task,
            "expected_shards": options.shards_per_task,
            "total_rows": sum(source.row_count for source in sources),
            "build_config_fingerprint": build_fingerprint,
            "image_workers": options.image_workers,
            "video_workers": options.video_workers,
            "video_probe_mode": options.video_probe_mode,
            "video_probe_max_tasks_per_worker": options.video_probe_max_tasks_per_worker,
            "max_in_flight": options.max_in_flight,
            "sources": [asdict(source) for source in sources],
            "updated_at": utc_now(),
        }
        existing = payload["tasks"].get(task)
        if existing is not None and existing.get("build_config_fingerprint") != build_fingerprint:
            # Existing shards remain untouched; explicit shard validation will rebuild them.
            task_payload["previous_build_config_fingerprint"] = existing.get("build_config_fingerprint")
        payload["tasks"][task] = task_payload
        payload["updated_at"] = utc_now()
        atomic_write_json(metadata_path, payload)
    finally:
        lock.release()


def _ordered_parallel_map(
    items: Iterable[tuple[int, AnnotationSource, int, dict[str, Any]]],
    worker: Callable[[tuple[int, AnnotationSource, int, dict[str, Any]]], RowBuildResult],
    *,
    workers: int,
    max_in_flight: int,
    heartbeat_seconds: float | None = None,
) -> Iterator[tuple[RowBuildResult | None, int]]:
    iterator = iter(items)
    futures: dict[Future[RowBuildResult], int] = {}
    buffered: dict[int, RowBuildResult] = {}
    exhausted = False
    next_index: int | None = None
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="manifest-row") as executor:
        while futures or not exhausted:
            while not exhausted and len(futures) + len(buffered) < max_in_flight:
                try:
                    item = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                item_index = item[0]
                if next_index is None:
                    next_index = item_index
                futures[executor.submit(worker, item)] = item_index
            if not futures:
                break
            done, _ = wait(
                futures,
                timeout=heartbeat_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                yield None, len(futures) + len(buffered)
                continue
            for future in done:
                index = futures.pop(future)
                buffered[index] = future.result()
            while next_index is not None and next_index in buffered:
                result = buffered.pop(next_index)
                next_index += 1
                yield result, len(futures) + len(buffered)


def _make_rejection(
    *,
    task: str,
    source: AnnotationSource,
    source_row_index: int,
    task_row_index: int,
    exc: Exception,
) -> dict[str, Any]:
    reason = getattr(exc, "reason", type(exc).__name__)
    if isinstance(exc, MediaValidationError) and exc.reason == "missing_media":
        reason = "missing_target"
    return {
        "dataset_name": source.dataset_name,
        "dataset_order": source.dataset_order,
        "row_index": source_row_index,
        "task": task,
        "reason": reason,
        "message": str(exc),
        "_build_task_row_index": task_row_index,
        "_build_source_row_index": source_row_index,
        "_build_dataset_order": source.dataset_order,
    }


def _build_row(
    item: tuple[int, AnnotationSource, int, dict[str, Any]],
    *,
    task: str,
    options: BuildOptions,
    cache: PersistentMediaCache,
    runtime_stats: BuildRuntimeStats | None = None,
) -> RowBuildResult:
    task_row_index, source, source_row_index, row = item
    try:
        if task == IMAGE_TASK:
            record = build_i2i_record(
                row,
                dataset_name=source.dataset_name,
                data_root=source.data_root,
                target_field=options.i2i_target_field,
                reference_field=options.i2i_reference_field,
                caption_field=options.i2i_caption_field,
                crop_field=options.i2i_crop_field,
                image_validator=cache.validate_image,
            )
        elif task == VIDEO_TASK:
            try:
                prefilter = prefilter_r2v_annotation(row, data_root=source.data_root)
            except Exception:
                if runtime_stats is not None:
                    runtime_stats.increment("annotation_prefilter_rejected")
                raise
            if runtime_stats is not None:
                runtime_stats.increment("rows_sent_to_video_probe")
            reference_validation_started = False

            def validate_reference(path: str) -> tuple[int, int]:
                nonlocal reference_validation_started
                if not reference_validation_started:
                    reference_validation_started = True
                    if runtime_stats is not None:
                        runtime_stats.increment("reference_validation_started")
                return cache.validate_image(path)

            try:
                header = cache.probe_video(prefilter.target_path)
                record = build_r2v_record(
                    row,
                    dataset_name=source.dataset_name,
                    data_root=source.data_root,
                    manifest_seed=options.manifest_seed,
                    video_header=header,
                    image_validator=validate_reference,
                    target_path_validated=True,
                )
            except Exception:
                if runtime_stats is not None and not reference_validation_started:
                    runtime_stats.increment("reference_validation_skipped_due_to_video_reject")
                raise
        else:
            raise ValueError(f"Unsupported task: {task}")
        record.update(
            {
                "_build_task_row_index": task_row_index,
                "_build_source_row_index": source_row_index,
                "_build_dataset_order": source.dataset_order,
            }
        )
        return RowBuildResult(
            task_row_index=task_row_index,
            source_row_index=source_row_index,
            dataset_order=source.dataset_order,
            dataset_name=source.dataset_name,
            record=record,
        )
    except Exception as exc:
        return RowBuildResult(
            task_row_index=task_row_index,
            source_row_index=source_row_index,
            dataset_order=source.dataset_order,
            dataset_name=source.dataset_name,
            rejection=_make_rejection(
                task=task,
                source=source,
                source_row_index=source_row_index,
                task_row_index=task_row_index,
                exc=exc,
            ),
        )


def shard_paths(task_dir: Path, shard_id: int) -> dict[str, Path]:
    stem = task_dir / f"shard_{shard_id:05d}"
    return {
        "accepted": Path(f"{stem}.accepted.jsonl"),
        "rejected": Path(f"{stem}.rejected.jsonl"),
        "summary": Path(f"{stem}.summary.json"),
        "done": Path(f"{stem}.done.json"),
        "lock": Path(f"{stem}.lock"),
    }


def validate_done_marker(
    marker_path: str | Path,
    *,
    expected_task: str | None = None,
    expected_shard_id: int | None = None,
    expected_build_fingerprint: str | None = None,
    expected_sources: Iterable[AnnotationSource] | None = None,
) -> dict[str, Any]:
    path = Path(marker_path)
    marker = json.loads(path.read_text(encoding="utf-8"))
    if int(marker.get("version", -1)) != SHARD_MARKER_VERSION:
        raise ValueError(f"Unsupported shard marker version: {path}")
    checks = {
        "task": expected_task,
        "shard_id": expected_shard_id,
        "build_config_fingerprint": expected_build_fingerprint,
    }
    for key, expected in checks.items():
        if expected is not None and marker.get(key) != expected:
            raise ValueError(f"Shard marker {key} mismatch for {path}: {marker.get(key)} != {expected}")
    if expected_sources is not None:
        expected_fingerprints = {source.annotation_path: source.fingerprint for source in expected_sources}
        if marker.get("source_fingerprints") != expected_fingerprints:
            raise ValueError(f"Shard source fingerprint mismatch: {path}")
    paths = marker.get("files", {})
    marker_suffix = ".done.json"
    if not path.name.endswith(marker_suffix):
        raise ValueError(f"Unexpected shard marker filename: {path}")
    stem = path.name[: -len(marker_suffix)]
    for name in ("accepted", "rejected", "summary"):
        artifact = Path(paths[name]["path"])
        expected_artifact = path.parent / f"{stem}.{name}.{'jsonl' if name != 'summary' else 'json'}"
        if artifact.resolve() != expected_artifact.resolve():
            raise ValueError(
                f"Shard marker artifact path mismatch: {artifact} != {expected_artifact}"
            )
        if not artifact.is_file():
            raise ValueError(f"Shard artifact is missing: {artifact}")
        digest = file_sha256(artifact)
        if digest != paths[name]["sha256"]:
            raise ValueError(f"Shard artifact SHA256 mismatch: {artifact}")
    return marker


def _cleanup_shard_artifacts(paths: Mapping[str, Path]) -> None:
    for name in ("accepted", "rejected", "summary", "done"):
        paths[name].unlink(missing_ok=True)
        for temporary in paths[name].parent.glob(f"{paths[name].name}.tmp.*"):
            temporary.unlink(missing_ok=True)


def _format_eta(seconds: float | None) -> str | None:
    if seconds is None or not math.isfinite(seconds):
        return None
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _write_progress(  # noqa: PLR0913
    shard_root: Path,
    *,
    task: str,
    shard_id: int,
    total_shards: int,
    processed: int,
    total: int,
    accepted: int,
    rejected: int,
    workers: int,
    in_flight: int,
    started: float,
    interval_started: float,
    interval_rows: int,
    cache: PersistentMediaCache,
    runtime_stats: BuildRuntimeStats,
    probe_pool: PersistentVideoProbePool | None,
) -> None:
    now = time.perf_counter()
    elapsed = max(now - started, 1.0e-9)
    interval_elapsed = max(now - interval_started, 1.0e-9)
    average_rate = processed / elapsed
    current_rate = interval_rows / interval_elapsed
    eta = (total - processed) / average_rate if average_rate > 0 else None
    runtime = runtime_stats.snapshot()
    payload = {
        "version": 1,
        "updated_at": utc_now(),
        "task": task,
        "shard": shard_id + 1,
        "shards": total_shards,
        "processed": processed,
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "workers": workers,
        "in_flight": in_flight,
        "rate_current": current_rate,
        "rate_avg": average_rate,
        "eta_seconds": eta,
        "eta": _format_eta(eta),
        "stage": (
            "media_validation"
            if task != VIDEO_TASK or runtime.get("rows_sent_to_video_probe", 0)
            else "annotation_prefilter"
        ),
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "image_cache_hits": cache.image_hits,
        "image_cache_misses": cache.image_misses,
        "video_cache_hits": cache.video_hits,
        "video_cache_misses": cache.video_misses,
        "annotation_prefilter_rejected": runtime.get("annotation_prefilter_rejected", 0),
        "rows_sent_to_video_probe": runtime.get("rows_sent_to_video_probe", 0),
        "video_probe_success": cache.probe_success,
        "video_probe_timeout": cache.probe_timeout,
        "video_probe_invalid": cache.invalid_video_header,
        "video_probe_submissions": (
            probe_pool.probe_submit_count if probe_pool is not None else cache.video_misses
        ),
        "video_probe_worker_starts": probe_pool.worker_start_count if probe_pool is not None else 0,
        "video_probe_worker_restarts": probe_pool.worker_restart_count if probe_pool is not None else 0,
        "reference_validation_started": runtime.get("reference_validation_started", 0),
        "reference_validation_skipped_due_to_video_reject": runtime.get(
            "reference_validation_skipped_due_to_video_reject",
            0,
        ),
    }
    atomic_write_json(shard_root / f"build_{task}.progress.json", payload)
    logger.info(
        f"task={task} shard={shard_id + 1}/{total_shards} "
        f"processed={processed}/{total} accepted={accepted} rejected={rejected} "
        f"workers={workers} in_flight={in_flight} rate_current={current_rate:.1f} "
        f"rate_avg={average_rate:.1f} eta={payload['eta'] or 'unknown'} "
        f"cache_hits={cache.hits} cache_misses={cache.misses}",
    )


def run_r2v_prefilter(
    train_data_config: str | Path,
    *,
    shard_root: str | Path,
    options: BuildOptions,
) -> dict[str, Any]:
    """Scan R2V annotations without opening or statting target/reference media."""
    options.validate()
    root = Path(shard_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    sources = discover_annotation_sources(
        train_data_config,
        shard_root=root,
        tasks=[VIDEO_TASK],
        max_samples_per_task=options.max_samples_per_task,
        recover_stale_locks=options.recover_stale_locks,
    )[VIDEO_TASK]
    total_rows = sum(source.row_count for source in sources)
    accepted_path = root / "r2v_prefilter.accepted.jsonl"
    rejected_path = root / "r2v_prefilter.rejected.jsonl"
    summary_path = root / "r2v_prefilter_summary.json"
    accepted_temp = Path(f"{accepted_path}.tmp.{os.getpid()}")
    rejected_temp = Path(f"{rejected_path}.tmp.{os.getpid()}")
    started = time.perf_counter()
    passed = 0
    rejected = 0
    reject_reasons: Counter[str] = Counter()
    span_histogram: Counter[str] = Counter()
    ref_count_histogram: Counter[str] = Counter()
    try:
        rows = iter_task_range(
            sources,
            0,
            total_rows,
            batch_size=options.annotation_batch_size,
        )
        with accepted_temp.open("w", encoding="utf-8") as accepted_handle, rejected_temp.open(
            "w",
            encoding="utf-8",
        ) as rejected_handle:
            for task_row_index, source, source_row_index, row in rows:
                try:
                    histogram_face_cut = parse_pair(row.get("face_cut"), field="face_cut")
                except Exception:
                    pass
                else:
                    span_histogram[str(histogram_face_cut[1] - histogram_face_cut[0])] += 1
                try:
                    histogram_references = parse_path_list(row.get("ref_images"), field="ref_images")
                except Exception:
                    pass
                else:
                    ref_count_histogram[str(len(histogram_references))] += 1
                try:
                    prefilter_r2v_annotation(row, data_root=source.data_root)
                except Exception as exc:
                    rejection = _make_rejection(
                        task=VIDEO_TASK,
                        source=source,
                        source_row_index=source_row_index,
                        task_row_index=task_row_index,
                        exc=exc,
                    )
                    rejected_handle.write(json.dumps(rejection, ensure_ascii=False, sort_keys=True) + "\n")
                    rejected += 1
                    reject_reasons[str(rejection["reason"])] += 1
                    continue
                accepted_handle.write(
                    json.dumps(
                        {
                            "dataset_name": source.dataset_name,
                            "dataset_order": source.dataset_order,
                            "source_row_index": source_row_index,
                            "task_row_index": task_row_index,
                            "row": row,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                passed += 1
            _flush_and_sync(accepted_handle)
            _flush_and_sync(rejected_handle)
        accepted_temp.replace(accepted_path)
        rejected_temp.replace(rejected_path)
        elapsed = time.perf_counter() - started
        summary = {
            "version": 1,
            "build_mode": "r2v_annotation_prefilter_only",
            "raw_rows": total_rows,
            "prefilter_passed": passed,
            "prefilter_rejected": rejected,
            "reject_reason_counts": dict(sorted(reject_reasons.items())),
            "face_cut_span_histogram": dict(sorted(span_histogram.items(), key=lambda item: int(item[0]))),
            "ref_count_histogram": dict(sorted(ref_count_histogram.items(), key=lambda item: int(item[0]))),
            "elapsed_seconds": elapsed,
            "rows_per_second": total_rows / max(elapsed, 1.0e-9),
            "peak_rss_gb": peak_rss_gb(),
            "accepted_path": str(accepted_path),
            "rejected_path": str(rejected_path),
            "accepted_sha256": file_sha256(accepted_path),
            "rejected_sha256": file_sha256(rejected_path),
            "media_open_count": 0,
        }
        atomic_write_json(summary_path, summary)
        return summary
    finally:
        accepted_temp.unlink(missing_ok=True)
        rejected_temp.unlink(missing_ok=True)


def build_task_shards(  # noqa: PLR0912, PLR0915
    train_data_config: str | Path,
    *,
    task: str,
    shard_root: str | Path,
    options: BuildOptions,
    probe_runner: Callable[[str, float], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    options.validate()
    task = parse_tasks([task])[0]
    root = Path(shard_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = Path(train_data_config).expanduser().resolve()
    sources = discover_annotation_sources(
        config_path,
        shard_root=root,
        tasks=[task],
        max_samples_per_task=options.max_samples_per_task,
        recover_stale_locks=options.recover_stale_locks,
    )[task]
    total_rows = sum(source.row_count for source in sources)
    build_fingerprint = task_build_fingerprint(task, options, sources)
    task_dir = root / task
    task_dir.mkdir(parents=True, exist_ok=True)
    _update_build_metadata(
        root,
        train_data_config=config_path,
        task=task,
        options=options,
        sources=sources,
        build_fingerprint=build_fingerprint,
    )
    probe_pool: PersistentVideoProbePool | None = None
    if probe_runner is None and task == VIDEO_TASK and options.video_probe_mode == "persistent":
        probe_pool = PersistentVideoProbePool(
            workers=options.video_workers,
            timeout_seconds=options.probe_timeout_seconds,
            max_tasks_per_worker=options.video_probe_max_tasks_per_worker,
        )

        def persistent_probe(path: str, _timeout_seconds: float) -> Mapping[str, Any]:
            assert probe_pool is not None
            return probe_pool.submit(path).result()

        effective_probe_runner = persistent_probe
    else:
        effective_probe_runner = probe_runner or probe_video_isolated
    try:
        cache = PersistentMediaCache(
            task_dir / "media_cache.sqlite",
            probe_timeout_seconds=options.probe_timeout_seconds,
            probe_runner=effective_probe_runner,
        )
    except BaseException:
        if probe_pool is not None:
            probe_pool.close(wait=False)
        raise
    started = time.perf_counter()
    runtime_stats = BuildRuntimeStats()
    aggregate = Counter()
    reject_reasons: Counter[str] = Counter()
    completed_shards = 0
    workers = options.image_workers if task == IMAGE_TASK else options.video_workers
    try:
        for shard_id in range(options.shards_per_task):
            start_row, end_row = shard_bounds(total_rows, options.shards_per_task, shard_id)
            paths = shard_paths(task_dir, shard_id)
            if options.resume_build and paths["done"].exists():
                try:
                    marker = validate_done_marker(
                        paths["done"],
                        expected_task=task,
                        expected_shard_id=shard_id,
                        expected_build_fingerprint=build_fingerprint,
                        expected_sources=sources,
                    )
                except Exception as exc:
                    logger.warning("Rebuilding invalid shard %s/%d: %s", task, shard_id, exc)
                else:
                    aggregate.update(
                        raw_rows=int(marker["raw_rows"]),
                        accepted_rows=int(marker["accepted_rows"]),
                        rejected_rows=int(marker["rejected_rows"]),
                        duplicate_rows_local=int(marker["duplicate_rows_local"]),
                    )
                    reject_reasons.update(marker.get("reject_reason_counts", {}))
                    runtime_stats.update(marker.get("runtime_stats", {}))
                    completed_shards += 1
                    continue

            lock = ShardLock(
                paths["lock"],
                config_fingerprint=build_fingerprint,
                recover_stale=options.recover_stale_locks,
                stale_after_seconds=options.stale_lock_seconds,
            )
            with lock:
                _cleanup_shard_artifacts(paths)
                shard_started = time.perf_counter()
                accepted_temp = Path(f"{paths['accepted']}.tmp.{os.getpid()}")
                rejected_temp = Path(f"{paths['rejected']}.tmp.{os.getpid()}")
                summary_temp = Path(f"{paths['summary']}.tmp.{os.getpid()}")
                local = Counter(
                    raw_rows=0,
                    accepted_rows=0,
                    rejected_rows=0,
                    duplicate_rows_local=0,
                )
                local_reasons: Counter[str] = Counter()
                local_keys: dict[str, str] = {}
                progress_started = time.perf_counter()
                progress_rows = 0
                shard_runtime_start = runtime_stats.snapshot()
                try:
                    rows = iter_task_range(
                        sources,
                        start_row,
                        end_row,
                        batch_size=options.annotation_batch_size,
                    )
                    worker = partial(
                        _build_row,
                        task=task,
                        options=options,
                        cache=cache,
                        runtime_stats=runtime_stats,
                    )
                    with accepted_temp.open("w", encoding="utf-8") as accepted_handle, rejected_temp.open(
                        "w", encoding="utf-8"
                    ) as rejected_handle:
                        for result, in_flight in _ordered_parallel_map(
                            rows,
                            worker,
                            workers=workers,
                            max_in_flight=options.max_in_flight,
                            heartbeat_seconds=options.progress_interval_seconds,
                        ):
                            if result is not None:
                                local["raw_rows"] += 1
                                progress_rows += 1
                                if result.record is not None:
                                    key = str(result.record["sample_key"])
                                    plan = str(result.record["sample_plan_sha256"])
                                    previous = local_keys.get(key)
                                    if previous is not None:
                                        if previous != plan:
                                            raise ManifestCollisionError(
                                                f"sample_key collision with different plans inside shard: {key}"
                                            )
                                        local["duplicate_rows_local"] += 1
                                    else:
                                        local_keys[key] = plan
                                        accepted_handle.write(
                                            json.dumps(result.record, ensure_ascii=False, sort_keys=True) + "\n"
                                        )
                                        local["accepted_rows"] += 1
                                else:
                                    assert result.rejection is not None
                                    rejected_handle.write(
                                        json.dumps(result.rejection, ensure_ascii=False, sort_keys=True) + "\n"
                                    )
                                    local["rejected_rows"] += 1
                                    local_reasons[str(result.rejection["reason"])] += 1
                            now = time.perf_counter()
                            if now - progress_started >= options.progress_interval_seconds:
                                _write_progress(
                                    root,
                                    task=task,
                                    shard_id=shard_id,
                                    total_shards=options.shards_per_task,
                                    processed=aggregate["raw_rows"] + local["raw_rows"],
                                    total=total_rows,
                                    accepted=aggregate["accepted_rows"] + local["accepted_rows"],
                                    rejected=aggregate["rejected_rows"] + local["rejected_rows"],
                                    workers=workers,
                                    in_flight=in_flight,
                                    started=started,
                                    interval_started=progress_started,
                                    interval_rows=progress_rows,
                                    cache=cache,
                                    runtime_stats=runtime_stats,
                                    probe_pool=probe_pool,
                                )
                                progress_started = now
                                progress_rows = 0
                        _flush_and_sync(accepted_handle)
                        _flush_and_sync(rejected_handle)
                    if local["raw_rows"] != end_row - start_row:
                        raise RuntimeError(
                            f"Shard row count mismatch for {task}/{shard_id}: "
                            f"processed={local['raw_rows']}, expected={end_row - start_row}"
                        )
                    elapsed = time.perf_counter() - shard_started
                    runtime_snapshot = runtime_stats.snapshot()
                    shard_runtime = {
                        key: value - shard_runtime_start.get(key, 0)
                        for key, value in runtime_snapshot.items()
                        if value - shard_runtime_start.get(key, 0)
                    }
                    summary = {
                        "version": SHARD_MARKER_VERSION,
                        "task": task,
                        "shard_id": shard_id,
                        "source_start_row": start_row,
                        "source_end_row": end_row,
                        **dict(local),
                        "reject_reason_counts": dict(sorted(local_reasons.items())),
                        "elapsed_seconds": elapsed,
                        "average_rows_per_second": local["raw_rows"] / max(elapsed, 1.0e-9),
                        "peak_rss_gb": peak_rss_gb(),
                        "media_cache_hits": cache.hits,
                        "media_cache_misses": cache.misses,
                        "image_cache_hits": cache.image_hits,
                        "image_cache_misses": cache.image_misses,
                        "video_cache_hits": cache.video_hits,
                        "video_cache_misses": cache.video_misses,
                        "probe_success": cache.probe_success,
                        "probe_timeouts": cache.probe_timeout,
                        "invalid_video_header": cache.invalid_video_header,
                        "video_probe_worker_restarts": (
                            probe_pool.worker_restart_count if probe_pool is not None else 0
                        ),
                        "runtime_stats": shard_runtime,
                    }
                    with summary_temp.open("w", encoding="utf-8") as summary_handle:
                        json.dump(summary, summary_handle, ensure_ascii=False, indent=2, sort_keys=True)
                        summary_handle.write("\n")
                        _flush_and_sync(summary_handle)
                    accepted_temp.replace(paths["accepted"])
                    rejected_temp.replace(paths["rejected"])
                    summary_temp.replace(paths["summary"])
                    source_fingerprints = {source.annotation_path: source.fingerprint for source in sources}
                    overlapping = [
                        source
                        for source in sources
                        if max(start_row, source.task_start_row) < min(end_row, source.task_end_row)
                    ]
                    marker = {
                        "version": SHARD_MARKER_VERSION,
                        "completed_at": utc_now(),
                        "task": task,
                        "dataset_name": overlapping[0].dataset_name if len(overlapping) == 1 else "multiple",
                        "shard_id": shard_id,
                        "source_start_row": start_row,
                        "source_end_row": end_row,
                        **dict(local),
                        "reject_reason_counts": dict(sorted(local_reasons.items())),
                        "source_annotation_path": overlapping[0].annotation_path if len(overlapping) == 1 else None,
                        "source_annotation_size": overlapping[0].size if len(overlapping) == 1 else None,
                        "source_annotation_mtime_ns": overlapping[0].mtime_ns if len(overlapping) == 1 else None,
                        "source_annotation_fingerprint": overlapping[0].fingerprint if len(overlapping) == 1 else None,
                        "source_fingerprints": source_fingerprints,
                        "source_ranges": [
                            {
                                "dataset_name": source.dataset_name,
                                "dataset_order": source.dataset_order,
                                "annotation_path": source.annotation_path,
                                "source_start_row": max(start_row, source.task_start_row) - source.task_start_row,
                                "source_end_row": min(end_row, source.task_end_row) - source.task_start_row,
                            }
                            for source in overlapping
                        ],
                        "build_config_fingerprint": build_fingerprint,
                        "runtime_stats": shard_runtime,
                        "elapsed_seconds": elapsed,
                        "peak_rss_gb": peak_rss_gb(),
                        "files": {
                            "accepted": {"path": str(paths["accepted"]), "sha256": file_sha256(paths["accepted"])},
                            "rejected": {"path": str(paths["rejected"]), "sha256": file_sha256(paths["rejected"])},
                            "summary": {"path": str(paths["summary"]), "sha256": file_sha256(paths["summary"])},
                        },
                    }
                    atomic_write_json(paths["done"], marker)
                    aggregate.update(local)
                    reject_reasons.update(local_reasons)
                    completed_shards += 1
                finally:
                    accepted_temp.unlink(missing_ok=True)
                    rejected_temp.unlink(missing_ok=True)
                    summary_temp.unlink(missing_ok=True)
        _write_progress(
            root,
            task=task,
            shard_id=max(0, options.shards_per_task - 1),
            total_shards=options.shards_per_task,
            processed=aggregate["raw_rows"],
            total=total_rows,
            accepted=aggregate["accepted_rows"],
            rejected=aggregate["rejected_rows"],
            workers=workers,
            in_flight=0,
            started=started,
            interval_started=started,
            interval_rows=aggregate["raw_rows"],
            cache=cache,
            runtime_stats=runtime_stats,
            probe_pool=probe_pool,
        )
        return {
            "task": task,
            "completed_shards": completed_shards,
            "expected_shards": options.shards_per_task,
            "build_config_fingerprint": build_fingerprint,
            "total_rows": total_rows,
            **dict(aggregate),
            "reject_reason_counts": dict(sorted(reject_reasons.items())),
            "elapsed_seconds": time.perf_counter() - started,
            "media_cache_hits": cache.hits,
            "media_cache_misses": cache.misses,
            "image_cache_hits": cache.image_hits,
            "image_cache_misses": cache.image_misses,
            "video_cache_hits": cache.video_hits,
            "video_cache_misses": cache.video_misses,
            "probe_timeouts": cache.probe_timeout,
            "video_probe_mode": options.video_probe_mode,
            "video_probe_submissions": (
                probe_pool.probe_submit_count if probe_pool is not None else cache.video_misses
            ),
            "video_probe_worker_starts": probe_pool.worker_start_count if probe_pool is not None else 0,
            "video_probe_worker_restarts": probe_pool.worker_restart_count if probe_pool is not None else 0,
            "runtime_stats": runtime_stats.snapshot(),
        }
    finally:
        cache.close()
        if probe_pool is not None:
            probe_pool.close()


def _strip_build_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_build_")}


def _prepare_merge_database(path: Path) -> sqlite3.Connection:
    path.unlink(missing_ok=True)
    Path(f"{path}-wal").unlink(missing_ok=True)
    Path(f"{path}-shm").unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE samples (sample_key TEXT PRIMARY KEY, sample_plan_sha256 TEXT NOT NULL)"
    )
    return connection


def merge_manifest_shards(  # noqa: PLR0912, PLR0915
    shard_root: str | Path,
    *,
    tasks: Iterable[str],
    output: str | Path,
    reject_output: str | Path,
    summary_output: str | Path,
    dedup_db: str | Path | None = None,
    require_all_shards: bool = True,
) -> dict[str, Any]:
    root = Path(shard_root).expanduser().resolve()
    selected_tasks = parse_tasks(tasks)
    metadata = json.loads((root / "build_metadata.json").read_text(encoding="utf-8"))
    output_path = Path(output).expanduser().resolve()
    reject_path = Path(reject_output).expanduser().resolve()
    summary_path = Path(summary_output).expanduser().resolve()
    index_path = default_manifest_index_path(output_path)
    output_temp = Path(f"{output_path}.tmp.{os.getpid()}")
    reject_temp = Path(f"{reject_path}.tmp.{os.getpid()}")
    summary_temp = Path(f"{summary_path}.tmp.{os.getpid()}")
    index_temp = Path(f"{index_path}.tmp.{os.getpid()}")
    database_path = (
        Path(dedup_db).expanduser().resolve()
        if dedup_db
        else Path(f"{output_path}.dedup.tmp.{os.getpid()}.sqlite")
    )
    for path in (output_path, reject_path, summary_path, index_path, database_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    task_counts: Counter[str] = Counter()
    source_task_counts: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    task_elapsed: Counter[str] = Counter()
    cache_hits: Counter[str] = Counter()
    cache_misses: Counter[str] = Counter()
    probe_timeouts = 0
    shard_summaries: list[dict[str, Any]] = []
    raw_rows = rejected_rows = duplicate_rows = 0
    accepted_rows = 0
    connection: sqlite3.Connection | None = None
    try:
        connection = _prepare_merge_database(database_path)
        with output_temp.open("w", encoding="utf-8") as output_handle, reject_temp.open(
            "w", encoding="utf-8"
        ) as reject_handle:
            for task in selected_tasks:
                task_metadata = metadata.get("tasks", {}).get(task)
                if task_metadata is None:
                    if require_all_shards:
                        raise ValueError(f"Build metadata has no task {task}")
                    continue
                expected_shards = int(task_metadata["expected_shards"])
                expected_build = str(task_metadata["build_config_fingerprint"])
                sources = [AnnotationSource(**payload) for payload in task_metadata["sources"]]
                previous_task_row = -1
                for shard_id in range(expected_shards):
                    paths = shard_paths(root / task, shard_id)
                    if not paths["done"].exists():
                        if require_all_shards:
                            raise ValueError(f"Missing completed shard marker: {paths['done']}")
                        continue
                    marker = validate_done_marker(
                        paths["done"],
                        expected_task=task,
                        expected_shard_id=shard_id,
                        expected_build_fingerprint=expected_build,
                        expected_sources=sources,
                    )
                    expected_start, expected_end = shard_bounds(
                        int(task_metadata["total_rows"]), expected_shards, shard_id
                    )
                    if (marker["source_start_row"], marker["source_end_row"]) != (
                        expected_start,
                        expected_end,
                    ):
                        raise ValueError(f"Shard range mismatch: {paths['done']}")
                    raw_rows += int(marker["raw_rows"])
                    source_task_counts[task] += int(marker["raw_rows"])
                    rejected_rows += int(marker["rejected_rows"])
                    duplicate_rows += int(marker["duplicate_rows_local"])
                    reject_reasons.update(marker.get("reject_reason_counts", {}))
                    task_elapsed[task] += float(marker["elapsed_seconds"])
                    summary_payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
                    cache_hits[task] = max(cache_hits[task], int(summary_payload.get("media_cache_hits", 0)))
                    cache_misses[task] = max(cache_misses[task], int(summary_payload.get("media_cache_misses", 0)))
                    probe_timeouts = max(probe_timeouts, int(summary_payload.get("probe_timeouts", 0)))
                    shard_summaries.append(summary_payload)
                    observed_accepted = 0
                    with paths["accepted"].open("r", encoding="utf-8") as accepted_handle:
                        for line in accepted_handle:
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            observed_accepted += 1
                            task_row = int(record["_build_task_row_index"])
                            if not expected_start <= task_row < expected_end:
                                raise ValueError(
                                    f"Accepted row {task_row} is outside shard range "
                                    f"[{expected_start}, {expected_end}): {paths['accepted']}"
                                )
                            if task_row <= previous_task_row:
                                raise ValueError(f"Non-deterministic shard row order at {paths['accepted']}")
                            previous_task_row = task_row
                            key = str(record["sample_key"])
                            plan = str(record["sample_plan_sha256"])
                            existing = connection.execute(
                                "SELECT sample_plan_sha256 FROM samples WHERE sample_key=?", (key,)
                            ).fetchone()
                            if existing is not None:
                                if existing[0] != plan:
                                    raise ManifestCollisionError(
                                        f"sample_key collision with different plans during merge: {key}"
                                    )
                                duplicate_rows += 1
                                continue
                            connection.execute("INSERT INTO samples VALUES (?,?)", (key, plan))
                            clean = _strip_build_fields(record)
                            output_handle.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")
                            accepted_rows += 1
                            task_counts[task] += 1
                    if observed_accepted != int(marker["accepted_rows"]):
                        raise ValueError(
                            f"Shard accepted row count mismatch: marker={marker['accepted_rows']}, "
                            f"file={observed_accepted}, path={paths['accepted']}"
                        )
                    observed_rejected = 0
                    with paths["rejected"].open("r", encoding="utf-8") as rejected_handle:
                        for line in rejected_handle:
                            if line.strip():
                                observed_rejected += 1
                                reject_handle.write(line if line.endswith("\n") else f"{line}\n")
                    if observed_rejected != int(marker["rejected_rows"]):
                        raise ValueError(
                            f"Shard rejected row count mismatch: marker={marker['rejected_rows']}, "
                            f"file={observed_rejected}, path={paths['rejected']}"
                        )
            connection.commit()
            _flush_and_sync(output_handle)
            _flush_and_sync(reject_handle)
        build_manifest_offset_index(output_temp, index_temp)
        index_metadata = validate_manifest_index(output_temp, index_temp)
        if index_metadata.manifest_row_count != accepted_rows:
            raise ValueError(
                f"Merged manifest/index row mismatch: {accepted_rows} != {index_metadata.manifest_row_count}"
            )
        merge_elapsed = time.perf_counter() - started
        summary = {
            "raw_rows": raw_rows,
            "accepted_rows": accepted_rows,
            "rejected_rows": rejected_rows,
            "duplicate_rows": duplicate_rows,
            "task_counts": dict(sorted(task_counts.items())),
            "reject_reason_counts": dict(sorted(reject_reasons.items())),
            "elapsed_seconds": merge_elapsed + sum(task_elapsed.values()),
            "peak_rss_gb": peak_rss_gb(),
            "manifest_path": str(output_path),
            "manifest_index_path": str(index_path),
            "build_mode": "parallel_sharded_resume",
            "shards_per_task": {
                task: metadata["tasks"][task]["expected_shards"]
                for task in selected_tasks
                if task in metadata["tasks"]
            },
            "image_workers": metadata.get("tasks", {}).get(IMAGE_TASK, {}).get("image_workers", 0),
            "video_workers": metadata.get("tasks", {}).get(VIDEO_TASK, {}).get("video_workers", 0),
            "source_task_counts": dict(sorted(source_task_counts.items())),
            "accepted_task_counts": dict(sorted(task_counts.items())),
            "task_elapsed_seconds": dict(sorted(task_elapsed.items())),
            "task_average_rows_per_second": {
                task: source_task_counts[task] / max(task_elapsed[task], 1.0e-9) for task in selected_tasks
            },
            "media_cache_hits": dict(sorted(cache_hits.items())),
            "media_cache_misses": dict(sorted(cache_misses.items())),
            "probe_timeouts": probe_timeouts,
            "source_fingerprints": {
                task: {
                    source["annotation_path"]: source["fingerprint"]
                    for source in metadata["tasks"][task]["sources"]
                }
                for task in selected_tasks
                if task in metadata["tasks"]
            },
            "shard_summaries": shard_summaries,
            "merge_elapsed_seconds": merge_elapsed,
            "manifest_sha256": file_sha256(output_temp),
        }
        with summary_temp.open("w", encoding="utf-8") as summary_handle:
            json.dump(summary, summary_handle, ensure_ascii=False, indent=2, sort_keys=True)
            summary_handle.write("\n")
            _flush_and_sync(summary_handle)
        output_temp.replace(output_path)
        reject_temp.replace(reject_path)
        index_temp.replace(index_path)
        summary_temp.replace(summary_path)
        validate_manifest_index(output_path, index_path)
        return summary
    finally:
        if connection is not None:
            connection.close()
        for path in (output_temp, reject_temp, summary_temp, index_temp):
            path.unlink(missing_ok=True)
        database_path.unlink(missing_ok=True)
        Path(f"{database_path}-wal").unlink(missing_ok=True)
        Path(f"{database_path}-shm").unlink(missing_ok=True)
