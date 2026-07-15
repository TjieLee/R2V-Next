"""Deterministic image and exact-index video decoding for online manifests."""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Sequence

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


def _validate_indices(indices: Sequence[int]) -> list[int]:
    requested = [int(index) for index in indices]
    if not requested:
        raise ValueError("At least one video frame index is required")
    if any(index < 0 for index in requested):
        raise ValueError(f"Video frame indices must be non-negative: {requested}")
    if requested != sorted(requested) or len(requested) != len(set(requested)):
        raise ValueError("Video frame indices must be strictly increasing and unique")
    return requested


def _check_deadline(started: float, timeout_seconds: float, path: str | Path) -> None:
    if time.monotonic() - started > timeout_seconds:
        raise TimeoutError(f"Video decode exceeded {timeout_seconds:.1f}s: {path}")


@contextmanager
def _hard_decode_timeout(timeout_seconds: float, path: str | Path) -> Iterator[None]:
    """Interrupt a stuck native decoder in a POSIX DataLoader worker."""
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"Video decode exceeded {timeout_seconds:.1f}s: {path}")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _frame_index_from_pts(frame: av.VideoFrame, stream: av.VideoStream, fps: float) -> int | None:
    if frame.pts is None or stream.time_base is None or fps <= 0:
        return None
    start_pts = int(stream.start_time or 0)
    return int(round(float((frame.pts - start_pts) * stream.time_base) * fps))


def _frame_to_tensor(frame: av.VideoFrame) -> Tensor:
    return torch.from_numpy(frame.to_ndarray(format="rgb24").copy())


def _decode_video_indices_pyav(
    path: str | Path,
    requested: list[int],
    *,
    timeout_seconds: float,
) -> Tensor:
    started = time.monotonic()
    timeout = (timeout_seconds, timeout_seconds)
    with av.open(str(path), timeout=timeout) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = (
            getattr(stream, "average_rate", None)
            or getattr(stream, "base_rate", None)
            or getattr(stream, "guessed_rate", None)
            or 0.0
        )
        fps = float(rate)
        if fps <= 0 or stream.time_base is None:
            raise ValueError(f"Cannot map presentation timestamps to frame indices for {path}")

        seek_margin = max(32, int(round(2.0 * fps)))
        seek_frame = max(0, requested[0] - seek_margin)
        start_pts = int(stream.start_time or 0)
        seek_pts = start_pts + int(round((seek_frame / fps) / float(stream.time_base)))
        container.seek(seek_pts, stream=stream, any_frame=False, backward=True)

        wanted = set(requested)
        decoded: dict[int, Tensor] = {}
        pts_mapping_valid = True
        for frame in container.decode(stream):
            _check_deadline(started, timeout_seconds, path)
            frame_index = _frame_index_from_pts(frame, stream, fps)
            if frame_index is None:
                pts_mapping_valid = False
                break
            if frame_index in wanted and frame_index not in decoded:
                decoded[frame_index] = _frame_to_tensor(frame)
            if frame_index > requested[-1] or len(decoded) == len(requested):
                break

        if not pts_mapping_valid or any(index not in decoded for index in requested):
            # Some encoders expose unreliable/missing PTS after random seek.
            # Rewind the same container and establish exact ordinal indices by
            # sequential decode rather than trusting a floating frame position.
            decoded.clear()
            container.seek(start_pts, stream=stream, any_frame=False, backward=True)
            for frame_index, frame in enumerate(container.decode(stream)):
                _check_deadline(started, timeout_seconds, path)
                if frame_index in wanted:
                    decoded[frame_index] = _frame_to_tensor(frame)
                if frame_index >= requested[-1] or len(decoded) == len(requested):
                    break

    missing = [index for index in requested if index not in decoded]
    if missing:
        raise ValueError(f"Could not decode requested PyAV frames {missing[:10]} from {path}")
    return torch.stack([decoded[index] for index in requested], dim=0)


def _decode_video_indices_opencv(
    path: str | Path,
    requested: list[int],
    *,
    timeout_seconds: float,
) -> Tensor:
    started = time.monotonic()
    wanted = set(requested)
    decoded: dict[int, Tensor] = {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        # Do not infer the decoded ordinal from CAP_PROP_POS_FRAMES. Some
        # long-GOP backends report an approximate position after seek, so the
        # explicit fallback establishes exact ordinals from frame zero.
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = 0
        while frame_index <= requested[-1]:
            _check_deadline(started, timeout_seconds, path)
            success, frame = capture.read()
            if not success:
                break
            if frame_index in wanted:
                decoded[frame_index] = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy())
            if len(decoded) == len(requested):
                break
            frame_index += 1
    finally:
        capture.release()
    missing = [index for index in requested if index not in decoded]
    if missing:
        raise ValueError(f"Could not decode requested OpenCV frames {missing[:10]} from {path}")
    return torch.stack([decoded[index] for index in requested], dim=0)


def decode_video_indices(
    path: str | Path,
    indices: Sequence[int],
    *,
    decoder: Literal["pyav", "opencv"] = "pyav",
    timeout_seconds: float = 120.0,
) -> Tensor:
    requested = _validate_indices(indices)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if decoder == "pyav":
        with _hard_decode_timeout(timeout_seconds, path):
            return _decode_video_indices_pyav(path, requested, timeout_seconds=timeout_seconds)
    if decoder == "opencv":
        with _hard_decode_timeout(timeout_seconds, path):
            return _decode_video_indices_opencv(path, requested, timeout_seconds=timeout_seconds)
    raise ValueError(f"Unsupported video decoder {decoder!r}")
