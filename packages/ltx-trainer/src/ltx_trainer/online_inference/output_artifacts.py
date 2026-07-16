"""Atomic image/video artifacts for online training-sample replay."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image, ImageDraw
from torch import Tensor

from ltx_trainer.online_data.constants import TARGET_HEIGHT, TARGET_WIDTH
from ltx_trainer.online_inference.sample_naming import sample_directory_name


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def tensor_frame_to_pil(frame: Tensor) -> Image.Image:
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError(f"Expected decoded frame [3,H,W], got {tuple(frame.shape)}")
    pixels = (
        frame.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(pixels, mode="RGB")


def atomic_save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.png")
    image.save(temporary, format="PNG")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_save_video(
    video: Tensor,
    path: Path,
    *,
    fps: float,
    save_video: Callable[..., None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.mp4")
    temporary.unlink(missing_ok=True)
    save_video(
        video_tensor=video,
        output_path=temporary,
        fps=fps,
        video_format="FCHW",
    )
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise RuntimeError(f"Temporary video is missing or empty: {temporary}")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_reference_montage(reference_paths: list[str], destination: Path) -> None:
    images: list[Image.Image] = []
    for value in reference_paths:
        with Image.open(value) as source:
            image = source.convert("RGB")
            image.thumbnail((320, 240), Image.Resampling.LANCZOS)
            images.append(image.copy())
    if not images:
        raise ValueError("Cannot create a reference montage without images")
    label_height = 28
    width = sum(image.width for image in images)
    height = max(image.height for image in images) + label_height
    montage = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(montage)
    left = 0
    for index, image in enumerate(images):
        montage.paste(image, (left, label_height))
        draw.text((left + 6, 6), f"ref {index}", fill="black")
        left += image.width
    atomic_save_png(montage, destination)


def save_reference_images(reference_paths: list[str], sample_dir: Path) -> list[str]:
    outputs: list[str] = []
    for index, value in enumerate(reference_paths):
        with Image.open(value) as source:
            image = source.convert("RGB")
            name = f"reference_{index:02d}.png"
            atomic_save_png(image, sample_dir / name)
            outputs.append(name)
    return outputs


def save_contact_sheet(
    video: Tensor,
    destination: Path,
    *,
    frame_count: int = 8,
    columns: int = 4,
) -> None:
    if video.ndim != 4 or video.shape[0] < 1:
        raise ValueError(f"Expected decoded video [F,C,H,W], got {tuple(video.shape)}")
    selected_frame_count = min(frame_count, int(video.shape[0]))
    indices = torch.linspace(0, video.shape[0] - 1, selected_frame_count).round().long().tolist()
    frames = [tensor_frame_to_pil(video[index]) for index in indices]
    thumb_width = 320
    thumb_height = round(thumb_width * TARGET_HEIGHT / TARGET_WIDTH)
    for frame in frames:
        frame.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), "black")
    for index, frame in enumerate(frames):
        sheet.paste(frame, ((index % columns) * thumb_width, (index // columns) * thumb_height))
    atomic_save_png(sheet, destination)


def sample_output_dir(output_root: Path, sample: dict[str, Any]) -> Path:
    task = str(sample["task"])
    return output_root / task / sample_directory_name(
        task=task,
        sample_key=str(sample["sample_key"]),
    )


def output_is_complete(sample_dir: Path) -> bool:
    marker = sample_dir / "success.json"
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    artifacts = [sample_dir / value for value in payload.get("artifacts", [])]
    return bool(artifacts) and all(path.is_file() and path.stat().st_size > 0 for path in artifacts)
