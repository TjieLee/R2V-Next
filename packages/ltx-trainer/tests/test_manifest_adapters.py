from __future__ import annotations

import pytest

from ltx_trainer.online_data.adapters import CanonicalR2VSource
from ltx_trainer.online_data.manifest import (
    ManifestReject,
    build_canonical_r2v_record,
    build_strict_source_indices,
)


def _canonical_source(*, clip_start: int, clip_end: int) -> CanonicalR2VSource:
    return CanonicalR2VSource(
        dataset_name="fps_boundary",
        adapter_name="opens2v",
        source_record_id=f"clip-{clip_start}-{clip_end}",
        video_path="synthetic.mp4",
        caption="A subject moves in frame.",
        reference_paths=["reference.png"],
        crop_xyxy=None,
        clip_start_frame=clip_start,
        clip_end_frame=clip_end,
        metadata={},
    )


def _build_record(*, source_fps: float, clip_start: int, clip_end: int) -> dict[str, object]:
    return build_canonical_r2v_record(
        _canonical_source(clip_start=clip_start, clip_end=clip_end),
        manifest_seed=42,
        video_header={
            "fps": source_fps,
            "frame_count": max(clip_end, 400),
            "width": 832,
            "height": 480,
        },
        image_validator=lambda _path: (832, 480),
        target_path_validated=True,
    )


def test_23976_fps_source_builds_strict_121_frame_plan() -> None:
    record = _build_record(source_fps=23.976023976, clip_start=30, clip_end=151)

    indices = record["target_source_frame_indices"]
    assert record["original_fps"] == 23.976023976
    assert record["target_fps"] == 24.0
    assert record["target_num_frames"] == 121
    assert len(indices) == 121
    assert all(left < right for left, right in zip(indices, indices[1:]))


def test_23_fps_source_rejects_duplicate_source_indices() -> None:
    with pytest.raises(ManifestReject) as exc_info:
        _build_record(source_fps=23.0, clip_start=30, clip_end=146)

    assert exc_info.value.reason == "source_fps_too_low_for_unique_24fps_sampling"
    message = str(exc_info.value)
    assert "Source FPS 23.0" in message
    assert "target FPS 24.0" in message
    assert "121 unique source frames" in message
    assert "duplicate at target indices" in message
    assert "clip=[30,146)" in message


@pytest.mark.parametrize(
    ("source_fps", "clip_start", "clip_end"),
    [
        (25.0, 18, 91),
        (59.94005994, 10, 83),
    ],
)
def test_short_clip_keeps_insufficient_frames_reject(
    source_fps: float,
    clip_start: int,
    clip_end: int,
) -> None:
    with pytest.raises(ManifestReject) as exc_info:
        _build_record(source_fps=source_fps, clip_start=clip_start, clip_end=clip_end)

    assert exc_info.value.reason == "insufficient_frames_for_121_at_24fps"


def test_public_source_index_helper_accepts_23976_without_clip_metadata() -> None:
    indices = build_strict_source_indices(
        sample_start=0,
        source_fps=23.976023976,
        target_fps=24.0,
        target_frame_count=121,
    )

    assert len(indices) == 121
    assert all(left < right for left, right in zip(indices, indices[1:]))
