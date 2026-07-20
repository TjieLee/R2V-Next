"""Read-only source and writable-output boundaries for external evaluation."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


@dataclass(frozen=True)
class SourceSnapshot:
    path: str
    size_bytes: int
    mtime_ns: int
    inode: int


@dataclass(frozen=True)
class ReadOnlySourcePolicy:
    """Allow listed reads while confining every write beneath one owned root."""

    writable_root: Path
    allowed_roots: tuple[Path, ...] = ()
    allowed_files: frozenset[Path] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "writable_root", _resolved(self.writable_root))
        object.__setattr__(
            self,
            "allowed_roots",
            tuple(_resolved(path) for path in self.allowed_roots),
        )
        object.__setattr__(
            self,
            "allowed_files",
            frozenset(_resolved(path) for path in self.allowed_files),
        )

    def with_allowed_files(self, paths: list[str | Path]) -> "ReadOnlySourcePolicy":
        return ReadOnlySourcePolicy(
            writable_root=self.writable_root,
            allowed_roots=self.allowed_roots,
            allowed_files=self.allowed_files | frozenset(_resolved(path) for path in paths),
        )

    def assert_read_path(self, path: str | Path) -> Path:
        candidate = _resolved(path)
        if candidate in self.allowed_files or any(
            candidate.is_relative_to(root) for root in self.allowed_roots
        ):
            return candidate
        raise PermissionError(f"External evaluation source is not allowlisted: {candidate}")

    def assert_write_path(self, path: str | Path) -> Path:
        candidate = _resolved(path)
        if candidate == self.writable_root or candidate.is_relative_to(self.writable_root):
            return candidate
        raise PermissionError(
            f"External evaluation writes must stay under {self.writable_root}: {candidate}"
        )

    def ensure_directory(self, path: str | Path) -> Path:
        destination = self.assert_write_path(path)
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def read_json(self, path: str | Path) -> Any:
        source = self.assert_read_path(path)
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def snapshot(self, path: str | Path) -> SourceSnapshot:
        source = self.assert_read_path(path)
        stat = source.stat()
        return SourceSnapshot(
            path=str(source),
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            inode=stat.st_ino,
        )

    def atomic_write_text(self, path: str | Path, value: str) -> Path:
        destination = self.assert_write_path(path)
        self.ensure_directory(destination.parent)
        temporary = self.assert_write_path(
            destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
        )
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination

    def atomic_write_json(self, path: str | Path, payload: Any) -> Path:
        return self.atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def link_or_copy(self, source: str | Path, destination: str | Path) -> str:
        source_path = self.assert_write_path(source)
        destination_path = self.assert_write_path(destination)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        self.ensure_directory(destination_path.parent)
        temporary = self.assert_write_path(
            destination_path.with_name(f".{destination_path.name}.tmp.{os.getpid()}")
        )
        temporary.unlink(missing_ok=True)
        mode = "hardlink"
        try:
            os.link(source_path, temporary)
        except OSError:
            mode = "copy"
            shutil.copy2(source_path, temporary)
        os.replace(temporary, destination_path)
        return mode


__all__ = ["ReadOnlySourcePolicy", "SourceSnapshot"]
