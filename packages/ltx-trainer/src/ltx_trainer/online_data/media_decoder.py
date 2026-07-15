"""Exact image/video decoding for finalized online manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import av
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor


def decode_image_rgb(path: str | Path) -> Tensor:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array)


def probe_video(path: str | Path) -> dict[str, float | int]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = (
            getattr(stream, "average_rate", None)
            or getattr(stream, "base_rate", None)
            or getattr(stream, "guessed_rate", None)
            or 0.0
        )
        fps = float(rate)
        frame_count = int(stream.frames or 0)
        if frame_count <= 0 and stream.duration is not None and stream.time_base is not None and fps > 0:
            frame_count = int(round(float(stream.duration * stream.time_base) * fps))
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": int(stream.width),
            "height": int(stream.height),
        }


def decode_video_indices(path: str | Path, indices: Sequence[int]) -> Tensor:
    requested = [int(index) for index in indices]
    if not requested:
        raise ValueError("At least one video frame index is required")
    if any(index < 0 for index in requested):
        raise ValueError(f"Video frame indices must be non-negative: {requested}")
    if requested != sorted(requested) or len(requested) != len(set(requested)):
        raise ValueError("Video frame indices must be strictly increasing and unique")

    wanted = set(requested)
    decoded: dict[int, Tensor] = {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, requested[0])
        positioned_at = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        if positioned_at != requested[0]:
            raise ValueError(
                f"Video backend could not seek exactly to frame {requested[0]} in {path}; got {positioned_at}"
            )
        for frame_index in range(requested[0], requested[-1] + 1):
            success, frame = capture.read()
            if not success:
                break
            if frame_index in wanted:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                decoded[frame_index] = torch.from_numpy(rgb.copy())
                if len(decoded) == len(requested):
                    break
    finally:
        capture.release()
    missing = [index for index in requested if index not in decoded]
    if missing:
        raise ValueError(f"Could not decode requested frames {missing[:10]} from {path}")
    return torch.stack([decoded[index] for index in requested], dim=0)
