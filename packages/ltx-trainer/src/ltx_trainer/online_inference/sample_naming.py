"""Collision-resistant sample directory naming."""

from __future__ import annotations

import hashlib
import re

_UNSAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_-]+")


def sample_directory_name(
    *,
    task: str,
    sample_key: str,
    prefix_length: int = 72,
) -> str:
    """Build a stable name whose hash always covers the original full key."""
    if prefix_length < 1:
        raise ValueError("prefix_length must be positive")
    safe_task = _UNSAFE_COMPONENT.sub("_", str(task)).strip("_") or "task"
    safe_prefix = _UNSAFE_COMPONENT.sub("_", str(sample_key)).strip("_") or "sample"
    digest = hashlib.sha256(str(sample_key).encode("utf-8")).hexdigest()[:12]
    return f"{safe_task}_{safe_prefix[:prefix_length]}_{digest}"
