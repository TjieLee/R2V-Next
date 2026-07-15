from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest
import torch

from ltx_trainer.online_data import media_decoder
from ltx_trainer.online_data.media_decoder import decode_video_indices


def _write_long_gop_h264(path: Path, *, frame_count: int = 320) -> None:
    try:
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264", rate=24)
            stream.width = 64
            stream.height = 48
            stream.pix_fmt = "yuv420p"
            stream.gop_size = 120
            for index in range(frame_count):
                pixels = np.zeros((48, 64, 3), dtype=np.uint8)
                pixels[:, :, 0] = index % 256
                pixels[:, :, 1] = (index * 3) % 256
                pixels[:, :, 2] = (index * 7) % 256
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception as exc:
        pytest.skip(f"libx264 encoder is unavailable: {exc}")


def _sequential_ground_truth(path: Path, indices: list[int]) -> torch.Tensor:
    wanted = set(indices)
    decoded = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                decoded[index] = torch.from_numpy(frame.to_ndarray(format="rgb24").copy())
            if index >= indices[-1]:
                break
    return torch.stack([decoded[index] for index in indices])


def test_pyav_random_access_matches_sequential_long_gop_decode(tmp_path: Path) -> None:
    video = tmp_path / "long_gop.mp4"
    _write_long_gop_h264(video)
    indices = list(range(180, 301))
    expected = _sequential_ground_truth(video, indices)
    actual = decode_video_indices(video, indices, decoder="pyav", timeout_seconds=30.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_opencv_fallback_does_not_require_exact_reported_seek_position(tmp_path: Path) -> None:
    video = tmp_path / "opencv_fallback.mp4"
    _write_long_gop_h264(video, frame_count=80)
    indices = list(range(10, 31))
    decoded = decode_video_indices(video, indices, decoder="opencv", timeout_seconds=30.0)
    assert decoded.shape == (len(indices), 48, 64, 3)


def test_decode_timeout_propagates_to_dataset_retry_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "timeout.mp4"
    _write_long_gop_h264(video, frame_count=20)

    def _timeout(*_args, **_kwargs) -> None:
        raise TimeoutError("forced timeout")

    monkeypatch.setattr(media_decoder, "_check_deadline", _timeout)
    with pytest.raises(TimeoutError, match="forced timeout"):
        decode_video_indices(video, [0, 1], decoder="pyav", timeout_seconds=30.0)
