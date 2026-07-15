"""Write-path policy for manifests, logs, caches, and online training output."""

from __future__ import annotations

from pathlib import Path

_ALLOWED_WRITE_ROOT = Path("/mnt/workspace/litengjie")
_FORBIDDEN_WRITE_ROOTS = (
    Path("/mnt/workspace/liutao"),
    Path("/mnt/workspace/jiangyuxiang2"),
)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def assert_write_path_allowed(path: str | Path) -> Path:
    """Resolve and validate a write path without creating it."""
    resolved = Path(path).expanduser().resolve()
    allowed_root = _ALLOWED_WRITE_ROOT.resolve()
    for forbidden_root in _FORBIDDEN_WRITE_ROOTS:
        if _is_relative_to(resolved, forbidden_root.resolve()):
            raise ValueError(f"Writing under {forbidden_root} is forbidden: {resolved}")
    if not _is_relative_to(resolved, allowed_root):
        raise ValueError(f"Online training writes must stay under {_ALLOWED_WRITE_ROOT}: {resolved}")
    return resolved
