"""Deterministic online data path for multi-task multi-reference training."""

from ltx_trainer.online_data.constants import (
    IMAGE_TASK,
    MAX_VLM_FRAMES,
    RAW_VISUAL_DIM,
    TARGET_HEIGHT,
    TARGET_VISUAL_DIM,
    TARGET_WIDTH,
    TOKENS_PER_FRAME,
    VIDEO_TASK,
    VISUAL_TOKEN_CAPACITY,
    VLM_TARGET_INDICES,
)
from ltx_trainer.online_data.path_safety import assert_write_path_allowed

__all__ = [
    "IMAGE_TASK",
    "MAX_VLM_FRAMES",
    "RAW_VISUAL_DIM",
    "TARGET_HEIGHT",
    "TARGET_VISUAL_DIM",
    "TARGET_WIDTH",
    "TOKENS_PER_FRAME",
    "VIDEO_TASK",
    "VISUAL_TOKEN_CAPACITY",
    "VLM_TARGET_INDICES",
    "assert_write_path_allowed",
]
