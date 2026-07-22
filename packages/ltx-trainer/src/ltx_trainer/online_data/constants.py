"""Shared geometry constants for the 480p/121-frame online profile."""

from __future__ import annotations

IMAGE_TASK = "i2i"
VIDEO_TASK = "r2v"

TARGET_WIDTH = 832
TARGET_HEIGHT = 480
IMAGE_NUM_FRAMES = 1
IMAGE_FPS = 1.0
VIDEO_NUM_FRAMES = 121
VIDEO_FPS = 24.0


def uniform_indices(num_frames: int, sample_count: int) -> list[int]:
    """Return endpoint-inclusive deterministic uniform integer indices."""
    if num_frames <= 0 or sample_count <= 0:
        raise ValueError("num_frames and sample_count must be positive")
    if sample_count > num_frames:
        raise ValueError(f"Cannot select {sample_count} unique frames from {num_frames}")
    if sample_count == 1:
        return [0]
    return [round(index * (num_frames - 1) / (sample_count - 1)) for index in range(sample_count)]
