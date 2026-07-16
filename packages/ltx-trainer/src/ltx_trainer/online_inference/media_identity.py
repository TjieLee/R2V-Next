"""Filesystem identity checks for strict-no-GT online inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class RawReferenceLoadError(RuntimeError):
    """A sample-local reference media failure that may be skipped safely."""


class TargetReferenceAliasError(RawReferenceLoadError):
    """A reference resolves to the target and would leak ground-truth media."""


@dataclass(frozen=True)
class MediaIdentity:
    resolved_path: Path
    device: int | None
    inode: int | None


def media_identity(path: str | Path) -> MediaIdentity:
    """Return a fail-closed filesystem identity for an existing media path."""
    source = Path(path).expanduser()
    try:
        resolved = source.resolve(strict=False)
        stat_result = source.stat()
    except (OSError, RuntimeError) as exc:
        raise RawReferenceLoadError(f"Could not establish media identity for {source}: {exc}") from exc
    return MediaIdentity(
        resolved_path=resolved,
        device=int(stat_result.st_dev),
        inode=int(stat_result.st_ino),
    )


def paths_alias(left: str | Path, right: str | Path) -> bool:
    """Return whether two existing paths identify the same file."""
    left_identity = media_identity(left)
    right_identity = media_identity(right)
    if left_identity.resolved_path == right_identity.resolved_path:
        return True
    try:
        if os.path.samefile(left, right):
            return True
    except OSError as exc:
        raise RawReferenceLoadError(
            f"Could not compare media paths {left!s} and {right!s}: {exc}"
        ) from exc
    return (
        left_identity.device is not None
        and left_identity.inode is not None
        and left_identity.device == right_identity.device
        and left_identity.inode == right_identity.inode
    )


def assert_references_do_not_alias_target(
    reference_paths: Sequence[str | Path],
    target_path: str | Path,
) -> None:
    """Reject direct, symbolic-link, and hard-link aliases to the target."""
    for index, reference_path in enumerate(reference_paths):
        if paths_alias(reference_path, target_path):
            raise TargetReferenceAliasError(
                "reference_aliases_target: "
                f"reference[{index}]={reference_path!s} aliases target={target_path!s}"
            )
