"""Deterministic crop/resize transforms for online image and video samples."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def normalize_crop_xyxy(crop: Sequence[int | float] | None, *, width: int, height: int) -> tuple[int, int, int, int]:
    if crop is None:
        return 0, 0, width, height
    if len(crop) != 4:
        raise ValueError(f"crop_xyxy must contain four values, got {crop}")
    x0, y0, x1, y1 = (int(round(float(value))) for value in crop)
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


def deterministic_resize_center_crop(
    frames: Tensor,
    *,
    target_height: int,
    target_width: int,
    crop_xyxy: Sequence[int | float] | None = None,
    chunk_frames: int = 4,
) -> Tensor:
    """Transform uint8 ``[T,H,W,C]`` frames with bounded float-memory use."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be [T,H,W,3], got {tuple(frames.shape)}")
    if frames.dtype != torch.uint8:
        raise ValueError(f"frames must be uint8, got {frames.dtype}")
    if not 1 <= chunk_frames <= 16:
        raise ValueError(f"chunk_frames must be in [1, 16], got {chunk_frames}")
    num_frames, source_height, source_width, _ = frames.shape
    x0, y0, x1, y1 = normalize_crop_xyxy(crop_xyxy, width=source_width, height=source_height)
    cropped = frames[:, y0:y1, x0:x1]
    crop_height, crop_width = cropped.shape[1:3]
    scale = max(target_height / crop_height, target_width / crop_width)
    resized_height = max(target_height, int(round(crop_height * scale)))
    resized_width = max(target_width, int(round(crop_width * scale)))
    top = (resized_height - target_height) // 2
    left = (resized_width - target_width) // 2
    output = torch.empty(
        (num_frames, target_height, target_width, 3),
        dtype=torch.uint8,
        device=frames.device,
    )
    for start in range(0, num_frames, chunk_frames):
        stop = min(start + chunk_frames, num_frames)
        chunk = cropped[start:stop].permute(0, 3, 1, 2).float()
        resized = F.interpolate(
            chunk,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        transformed = resized[:, :, top : top + target_height, left : left + target_width]
        output[start:stop].copy_(
            transformed.round()
            .clamp_(0, 255)
            .to(dtype=torch.uint8)
            .permute(0, 2, 3, 1)
        )
        del transformed, resized, chunk
    return output
