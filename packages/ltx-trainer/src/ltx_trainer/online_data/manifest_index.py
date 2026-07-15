"""Versioned byte-offset index for large online JSONL manifests."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK

INDEX_MAGIC = b"LTXIDX02"
LEGACY_INDEX_MAGIC = b"LTXIDX01"
INDEX_HEADER = struct.Struct("<QQI32s")
INDEX_ENTRY = struct.Struct("<QB")
_TASK_TO_ID = {IMAGE_TASK: 0, VIDEO_TASK: 1}
_ID_TO_TASK = {value: key for key, value in _TASK_TO_ID.items()}


@dataclass(frozen=True)
class ManifestIndexMetadata:
    manifest_size_bytes: int
    manifest_row_count: int
    index_entry_size: int
    manifest_sha256: str


def default_manifest_index_path(manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    return Path(f"{path}.idx")


def _rebuild_error(message: str, path: Path) -> ValueError:
    return ValueError(f"{message}. Rebuild the online manifest index: {path}")


def _read_header(handle: BinaryIO, path: Path) -> ManifestIndexMetadata:
    magic = handle.read(len(INDEX_MAGIC))
    if magic == LEGACY_INDEX_MAGIC:
        raise _rebuild_error("Legacy LTXIDX01 index is not supported", path)
    if magic != INDEX_MAGIC:
        raise _rebuild_error("Invalid online manifest index header", path)
    packed = handle.read(INDEX_HEADER.size)
    if len(packed) != INDEX_HEADER.size:
        raise _rebuild_error("Truncated online manifest index header", path)
    manifest_size, row_count, entry_size, digest = INDEX_HEADER.unpack(packed)
    if entry_size != INDEX_ENTRY.size:
        raise _rebuild_error(
            f"Unsupported index entry size {entry_size}; expected {INDEX_ENTRY.size}",
            path,
        )
    expected_size = len(INDEX_MAGIC) + INDEX_HEADER.size + row_count * entry_size
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise _rebuild_error(
            f"Manifest index size mismatch: expected={expected_size}, actual={actual_size}",
            path,
        )
    return ManifestIndexMetadata(
        manifest_size_bytes=manifest_size,
        manifest_row_count=row_count,
        index_entry_size=entry_size,
        manifest_sha256=digest.hex(),
    )


def _manifest_fingerprint(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _validate_manifest_fingerprint(
    metadata: ManifestIndexMetadata,
    *,
    manifest_path: Path,
    index_path: Path,
) -> None:
    manifest_size = manifest_path.stat().st_size
    if manifest_size != metadata.manifest_size_bytes:
        raise _rebuild_error(
            "Manifest/index file-size mismatch: "
            f"manifest={manifest_size}, indexed={metadata.manifest_size_bytes}",
            index_path,
        )
    _, digest = _manifest_fingerprint(manifest_path)
    if digest != metadata.manifest_sha256:
        raise _rebuild_error(
            "Manifest/index SHA256 mismatch: "
            f"manifest={digest}, indexed={metadata.manifest_sha256}",
            index_path,
        )


def build_manifest_offset_index(
    manifest_path: str | Path,
    index_path: str | Path | None = None,
) -> Path:
    """Stream a finalized JSONL manifest into an atomic fixed-record v2 index."""
    manifest = Path(manifest_path).expanduser().resolve()
    destination = (
        Path(index_path).expanduser().resolve()
        if index_path is not None
        else default_manifest_index_path(manifest)
    )
    temporary = Path(f"{destination}.tmp.{os.getpid()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        digest = hashlib.sha256()
        row_count = 0
        with manifest.open("rb") as source, temporary.open("w+b") as output:
            output.write(INDEX_MAGIC)
            output.write(bytes(INDEX_HEADER.size))
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                digest.update(line)
                if not line.strip():
                    continue
                record = json.loads(line)
                task = str(record.get("task"))
                if task not in _TASK_TO_ID:
                    raise ValueError(f"Manifest row at byte {offset} has unsupported task {task!r}")
                output.write(INDEX_ENTRY.pack(offset, _TASK_TO_ID[task]))
                row_count += 1
            manifest_size = source.tell()
            output.seek(len(INDEX_MAGIC))
            output.write(
                INDEX_HEADER.pack(
                    manifest_size,
                    row_count,
                    INDEX_ENTRY.size,
                    digest.digest(),
                )
            )
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def read_manifest_index(
    index_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> tuple[array, dict[str, array]]:
    """Load compact offsets/task-local row ids and optionally verify manifest identity."""
    path = Path(index_path).expanduser().resolve()
    offsets = array("Q")
    task_indices = {IMAGE_TASK: array("q"), VIDEO_TASK: array("q")}
    with path.open("rb") as handle:
        metadata = _read_header(handle, path)
        if manifest_path is not None:
            _validate_manifest_fingerprint(
                metadata,
                manifest_path=Path(manifest_path).expanduser().resolve(),
                index_path=path,
            )
        for row_index in range(metadata.manifest_row_count):
            entry = handle.read(INDEX_ENTRY.size)
            if len(entry) != INDEX_ENTRY.size:
                raise _rebuild_error("Truncated online manifest index", path)
            offset, task_id = INDEX_ENTRY.unpack(entry)
            if task_id not in _ID_TO_TASK:
                raise _rebuild_error(f"Invalid task id {task_id} in online manifest index", path)
            offsets.append(offset)
            task_indices[_ID_TO_TASK[task_id]].append(row_index)
    return offsets, task_indices


def read_jsonl_record_at(handle: BinaryIO, offset: int) -> dict[str, Any]:
    handle.seek(int(offset))
    line = handle.readline()
    if not line:
        raise ValueError(f"No manifest row found at byte offset {offset}")
    record = json.loads(line)
    if not isinstance(record, dict):
        raise ValueError(f"Manifest row at byte offset {offset} is not an object")
    return record


def iter_index_entries(index_path: str | Path) -> Iterator[tuple[int, str]]:
    path = Path(index_path).expanduser().resolve()
    with path.open("rb") as handle:
        metadata = _read_header(handle, path)
        for _ in range(metadata.manifest_row_count):
            entry = handle.read(INDEX_ENTRY.size)
            if len(entry) != INDEX_ENTRY.size:
                raise _rebuild_error("Truncated online manifest index", path)
            offset, task_id = INDEX_ENTRY.unpack(entry)
            if task_id not in _ID_TO_TASK:
                raise _rebuild_error(f"Invalid task id {task_id} in online manifest index", path)
            yield offset, _ID_TO_TASK[task_id]


def validate_manifest_index(
    manifest_path: str | Path,
    index_path: str | Path | None = None,
) -> ManifestIndexMetadata:
    """Strictly verify fingerprint plus every indexed offset and task id."""
    manifest = Path(manifest_path).expanduser().resolve()
    index = (
        Path(index_path).expanduser().resolve()
        if index_path is not None
        else default_manifest_index_path(manifest)
    )
    with index.open("rb") as handle:
        metadata = _read_header(handle, index)
    _validate_manifest_fingerprint(metadata, manifest_path=manifest, index_path=index)

    entries = iter(iter_index_entries(index))
    observed_rows = 0
    with manifest.open("rb") as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                indexed_offset, indexed_task = next(entries)
            except StopIteration as exc:
                raise _rebuild_error("Manifest contains more rows than its index", index) from exc
            record = json.loads(line)
            task = str(record.get("task"))
            if indexed_offset != offset or indexed_task != task:
                raise _rebuild_error(
                    "Manifest index entry mismatch at row "
                    f"{observed_rows}: offset={indexed_offset}/{offset}, task={indexed_task!r}/{task!r}",
                    index,
                )
            observed_rows += 1
    if observed_rows != metadata.manifest_row_count:
        raise _rebuild_error(
            f"Manifest/index row mismatch: manifest={observed_rows}, index={metadata.manifest_row_count}",
            index,
        )
    try:
        extra_entry = next(entries)
    except StopIteration:
        extra_entry = None
    if extra_entry is not None:
        raise _rebuild_error("Manifest index contains trailing entries", index)
    return metadata
