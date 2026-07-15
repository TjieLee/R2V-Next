"""Compact byte-offset index for large online JSONL manifests."""

from __future__ import annotations

import json
import os
import struct
from array import array
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK

INDEX_MAGIC = b"LTXIDX01"
INDEX_ENTRY = struct.Struct("<QB")
_TASK_TO_ID = {IMAGE_TASK: 0, VIDEO_TASK: 1}
_ID_TO_TASK = {value: key for key, value in _TASK_TO_ID.items()}


def default_manifest_index_path(manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    return Path(f"{path}.idx")


def build_manifest_offset_index(
    manifest_path: str | Path,
    index_path: str | Path | None = None,
) -> Path:
    """Stream a finalized JSONL manifest into an atomic fixed-record index."""
    manifest = Path(manifest_path).expanduser().resolve()
    destination = (
        Path(index_path).expanduser().resolve()
        if index_path is not None
        else default_manifest_index_path(manifest)
    )
    temporary = Path(f"{destination}.tmp.{os.getpid()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with manifest.open("rb") as source, temporary.open("wb") as output:
            output.write(INDEX_MAGIC)
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                task = str(record.get("task"))
                if task not in _TASK_TO_ID:
                    raise ValueError(f"Manifest row at byte {offset} has unsupported task {task!r}")
                output.write(INDEX_ENTRY.pack(offset, _TASK_TO_ID[task]))
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def read_manifest_index(index_path: str | Path) -> tuple[array, dict[str, array]]:
    """Load compact offsets and task-local row ids without JSON objects."""
    path = Path(index_path).expanduser().resolve()
    offsets = array("Q")
    task_indices = {IMAGE_TASK: array("q"), VIDEO_TASK: array("q")}
    with path.open("rb") as handle:
        if handle.read(len(INDEX_MAGIC)) != INDEX_MAGIC:
            raise ValueError(f"Invalid online manifest index header: {path}")
        row_index = 0
        while entry := handle.read(INDEX_ENTRY.size):
            if len(entry) != INDEX_ENTRY.size:
                raise ValueError(f"Truncated online manifest index: {path}")
            offset, task_id = INDEX_ENTRY.unpack(entry)
            if task_id not in _ID_TO_TASK:
                raise ValueError(f"Invalid task id {task_id} in online manifest index: {path}")
            offsets.append(offset)
            task_indices[_ID_TO_TASK[task_id]].append(row_index)
            row_index += 1
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
        if handle.read(len(INDEX_MAGIC)) != INDEX_MAGIC:
            raise ValueError(f"Invalid online manifest index header: {path}")
        while entry := handle.read(INDEX_ENTRY.size):
            if len(entry) != INDEX_ENTRY.size:
                raise ValueError(f"Truncated online manifest index: {path}")
            offset, task_id = INDEX_ENTRY.unpack(entry)
            yield offset, _ID_TO_TASK[task_id]
