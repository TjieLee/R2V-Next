"""Deterministic online data path for semantic-flow training."""

from ltx_trainer.online_data.constants import (
    IMAGE_TASK,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VIDEO_TASK,
)
from ltx_trainer.online_data.path_safety import assert_write_path_allowed

__all__ = [
    "IMAGE_TASK",
    "TARGET_HEIGHT",
    "TARGET_WIDTH",
    "VIDEO_TASK",
    "assert_write_path_allowed",
]
