from __future__ import annotations

import pytest
import torch

from ltx_trainer.online_data.anchor_geometry import (
    normalized_anchor_timestamps,
    uniform_anchor_indices,
)


R2V_ANCHOR_INDICES = [0, 11, 22, 33, 44, 55, 65, 76, 87, 98, 109, 120]


def test_r2v_anchor_indices_and_normalized_timestamps_are_exact() -> None:
    assert uniform_anchor_indices(frame_count=121, anchor_count=12) == R2V_ANCHOR_INDICES
    timestamps = normalized_anchor_timestamps(
        frame_count=121,
        anchor_count=12,
        device=torch.device("cpu"),
    )
    assert torch.equal(timestamps, torch.tensor(R2V_ANCHOR_INDICES, dtype=torch.float32) / 120)
    assert not torch.equal(timestamps, torch.linspace(0.0, 1.0, 12))


def test_single_frame_anchor_geometry_is_zero() -> None:
    assert uniform_anchor_indices(frame_count=1, anchor_count=1) == [0]
    assert torch.equal(
        normalized_anchor_timestamps(
            frame_count=1,
            anchor_count=1,
            device=torch.device("cpu"),
        ),
        torch.tensor([0.0]),
    )


@pytest.mark.parametrize(
    ("frame_count", "anchor_count", "message"),
    [
        (2, 3, "anchor_count cannot exceed frame_count"),
        (0, 1, "frame_count must be positive"),
        (-1, 1, "frame_count must be positive"),
        (1, 0, "anchor_count must be positive"),
        (1, -1, "anchor_count must be positive"),
    ],
)
def test_invalid_anchor_geometry_fails_closed(
    frame_count: int,
    anchor_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        uniform_anchor_indices(frame_count=frame_count, anchor_count=anchor_count)
