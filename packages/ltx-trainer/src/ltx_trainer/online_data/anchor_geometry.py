"""Canonical semantic anchor geometry shared by training and inference."""

from __future__ import annotations

import torch
from torch import Tensor


def uniform_anchor_indices(*, frame_count: int, anchor_count: int) -> list[int]:
    if frame_count < 1:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if anchor_count < 1:
        raise ValueError(f"anchor_count must be positive, got {anchor_count}")
    if anchor_count > frame_count:
        raise ValueError(
            "anchor_count cannot exceed frame_count: "
            f"anchor_count={anchor_count}, frame_count={frame_count}"
        )
    if anchor_count == 1:
        return [0]

    indices = [
        round(index * (frame_count - 1) / (anchor_count - 1))
        for index in range(anchor_count)
    ]
    if indices[0] != 0 or indices[-1] != frame_count - 1:
        raise RuntimeError(
            "Uniform anchor sampling must include both endpoints: "
            f"indices={indices}"
        )
    if any(left >= right for left, right in zip(indices, indices[1:], strict=False)):
        raise RuntimeError(f"Uniform anchor indices must be strictly increasing: indices={indices}")
    return indices


def normalized_anchor_timestamps(
    *,
    frame_count: int,
    anchor_count: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    indices = uniform_anchor_indices(
        frame_count=frame_count,
        anchor_count=anchor_count,
    )
    denominator = max(1, frame_count - 1)
    return torch.tensor(indices, device=device, dtype=dtype) / denominator


__all__ = [
    "normalized_anchor_timestamps",
    "uniform_anchor_indices",
]
