"""Serializable online sampler state helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class OnlineDataState:
    task_schedule_cursor: int = 0
    image_permutation_epoch: int = 0
    image_cursor: int = 0
    video_permutation_epoch: int = 0
    video_cursor: int = 0
    microstep_in_optimizer_step: int = 0
    sampler_seed: int = 42

    def state_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "OnlineDataState":
        missing = [name for name in cls.__dataclass_fields__ if name not in state]
        if missing:
            raise ValueError(f"Online sampler state is missing fields: {missing}")
        return cls(**{name: int(state[name]) for name in cls.__dataclass_fields__})
