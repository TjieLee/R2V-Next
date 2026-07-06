#!/usr/bin/env python3
"""Precompute frozen Gemma/SigLIP target-video visual condition tokens.

Each output mirrors the target media path and stores connector-input visual
tokens produced by the frozen Gemma vision tower and multi-modal projector:

    visual_tokens: [K, D]
    visual_token_mask: [K]
    num_visual_tokens: int
    num_video_frames: int

These tokens are the Stage 1 GT visual-condition tokens and the Stage 2 MSE
teacher targets for the VLM planner placeholders.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import torch
import typer
from PIL import Image
from rich.progress import track
from transformers import AutoImageProcessor

from ltx_core.multicond.visual_tokens import extract_projected_visual_tokens
from ltx_core.utils import find_matching_file
from ltx_trainer import logger
from ltx_trainer.model_loader import load_text_encoder
from ltx_trainer.video_utils import read_video

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Encode sampled target-video frames into frozen SigLIP/projector GT visual tokens.",
)


def _read_rows(dataset_file: Path) -> list[dict[str, Any]]:
    suffix = dataset_file.suffix.lower()
    if suffix == ".json":
        data = json.loads(dataset_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return list(data.values())
        if isinstance(data, list):
            return data
        raise ValueError("JSON manifest must contain a list or dict of objects")
    if suffix == ".jsonl":
        return [json.loads(line) for line in dataset_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".csv":
        with dataset_file.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    raise ValueError(f"Unsupported manifest suffix: {dataset_file.suffix}")


def _resolve_path(path_value: str, root_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root_dir / path


def _output_relative(path: Path, data_root: Path) -> Path:
    try:
        return path.relative_to(data_root)
    except ValueError:
        return Path(*path.parts[1:]) if path.is_absolute() else path


def _atomic_save(data: dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + f".tmp.{os.getpid()}")
    torch.save(data, tmp_file)
    tmp_file.replace(output_file)


def _sample_video_frames(
    video_path: Path,
    *,
    sample_fps: float,
    max_source_frames: int | None,
    num_sampled_frames: int | None,
    max_sampled_frames: int | None,
) -> tuple[list[Image.Image], float, torch.Tensor]:
    video, source_fps = read_video(video_path, max_frames=max_source_frames)
    if video.shape[0] == 0:
        raise ValueError(f"No frames decoded from {video_path}")

    if num_sampled_frames is not None:
        if num_sampled_frames == 1:
            frame_indices = torch.tensor([0], dtype=torch.long)
        else:
            frame_indices = torch.linspace(0, video.shape[0] - 1, steps=num_sampled_frames).round().to(torch.long)
    else:
        stride = max(int(round(source_fps / sample_fps)), 1)
        frame_indices = torch.arange(0, video.shape[0], stride, dtype=torch.long)
    if max_sampled_frames is not None:
        frame_indices = frame_indices[:max_sampled_frames]
    if frame_indices.numel() == 0:
        frame_indices = torch.tensor([0], dtype=torch.long)

    sampled = video[frame_indices].clamp(0, 1)
    images = []
    for frame in sampled:
        array = (frame.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        images.append(Image.fromarray(array, mode="RGB"))
    return images, source_fps, frame_indices


@app.command()
def main(  # noqa: PLR0913
    dataset_file: str = typer.Argument(..., help="Flat CSV/JSON/JSONL manifest."),
    text_encoder_path: str = typer.Option(..., help="Local Gemma text encoder directory."),
    output_dir: str = typer.Option(..., help="Output gt_siglip_tokens directory."),
    video_column: str = typer.Option("video", help="Target video path column used for both sampling and output names."),
    root_dir: str | None = typer.Option(
        None,
        help="Root for relative video paths. Defaults to the manifest parent.",
    ),
    sample_fps: float = typer.Option(
        6.0,
        help="Target-video frame sampling rate for SigLIP teacher tokens.",
        gt=0.0,
    ),
    num_sampled_frames: int | None = typer.Option(
        None,
        help="Sample exactly this many target-video frames uniformly. Overrides --sample-fps when set.",
        min=1,
    ),
    max_source_frames: int | None = typer.Option(
        None,
        help="Decode at most this many source video frames before sampling. Set to the training bucket frame count.",
        min=1,
    ),
    max_sampled_frames: int | None = typer.Option(
        None,
        help="Optional cap on sampled target-video frames per sample. Use with --sample-fps to bound K.",
        min=1,
    ),
    expected_token_count: int | None = typer.Option(
        None,
        help="Optional assert: every saved sample must produce exactly this many visual tokens.",
        min=1,
    ),
    device: str = typer.Option("cuda", help="Torch device for SigLIP/projector extraction."),
    overwrite: bool = typer.Option(False, help="Rebuild files that already exist."),
) -> None:
    dataset_path = Path(dataset_file)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {dataset_path}")

    data_root = Path(root_dir) if root_dir is not None else dataset_path.parent
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    processor_root = str(find_matching_file(text_encoder_path, "preprocessor_config.json").parent)
    image_processor = AutoImageProcessor.from_pretrained(processor_root, local_files_only=True, use_fast=False)
    text_encoder = load_text_encoder(text_encoder_path, device=device, dtype=torch.bfloat16)
    text_encoder.eval()

    rows = _read_rows(dataset_path)
    processed = 0
    skipped = 0
    first_token_count: int | None = None

    for row in track(rows, description="Encoding target-video GT SigLIP visual tokens"):
        if video_column not in row:
            raise ValueError(f"Missing video column '{video_column}' in row: {row}")

        video_path = _resolve_path(str(row[video_column]), data_root)
        output_file = out_root / _output_relative(video_path, data_root).with_suffix(".pt")
        if output_file.exists() and not overwrite:
            skipped += 1
            continue

        images, source_fps, frame_indices = _sample_video_frames(
            video_path,
            sample_fps=sample_fps,
            max_source_frames=max_source_frames,
            num_sampled_frames=num_sampled_frames,
            max_sampled_frames=max_sampled_frames,
        )
        processed_images = image_processor(images=images, return_tensors="pt")
        pixel_values = processed_images["pixel_values"].to(device=device, dtype=torch.bfloat16)

        with torch.inference_mode():
            visual_batch = extract_projected_visual_tokens(text_encoder.model, pixel_values)

        tokens = visual_batch.tokens[0].cpu().contiguous()
        mask = visual_batch.mask[0].cpu().contiguous()
        token_count = int(mask.sum().item())
        tokens_per_frame = token_count // len(images)
        if first_token_count is None:
            first_token_count = token_count
            logger.info(f"Detected {first_token_count} GT visual tokens per sample")
        if expected_token_count is not None and token_count != expected_token_count:
            raise ValueError(
                f"{video_path} produced {token_count} visual tokens, expected {expected_token_count}. "
                "Use the detected count in planner_token_count or keep target-video sampling fixed."
            )

        save_data = {
            "visual_tokens": tokens,
            "visual_token_mask": mask,
            "num_visual_tokens": torch.tensor(token_count, dtype=torch.long),
            "num_video_frames": torch.tensor(len(images), dtype=torch.long),
            "tokens_per_frame": torch.tensor(tokens_per_frame, dtype=torch.long),
            "sampled_frame_indices": frame_indices.cpu().contiguous(),
            "source_fps": torch.tensor(source_fps, dtype=torch.float32),
            "sample_fps": torch.tensor(sample_fps, dtype=torch.float32),
            "num_sampled_frames_setting": torch.tensor(num_sampled_frames or -1, dtype=torch.long),
            "max_source_frames": torch.tensor(max_source_frames or -1, dtype=torch.long),
        }
        _atomic_save(save_data, output_file)
        processed += 1

    logger.info(
        f"Target-video GT SigLIP token preprocessing complete: {processed} encoded, {skipped} skipped -> {out_root}"
    )


if __name__ == "__main__":
    app()
