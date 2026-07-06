#!/usr/bin/env python3
"""Precompute frozen Gemma/SigLIP visual condition tokens.

Each output mirrors the target media path and stores connector-input visual
tokens produced by the frozen Gemma vision tower and multi-modal projector:

    visual_tokens: [K, D]
    visual_token_mask: [K]
    num_visual_tokens: int
    num_ref_images: int

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
from rich.progress import track
from transformers import AutoImageProcessor

from ltx_core.multicond.visual_tokens import extract_projected_visual_tokens
from ltx_core.utils import find_matching_file
from ltx_trainer import logger
from ltx_trainer.model_loader import load_text_encoder
from ltx_trainer.utils import open_image_as_srgb

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Encode reference images into frozen SigLIP/projector GT visual tokens.",
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


def _parse_reference_images(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if not isinstance(parsed, list):
                raise ValueError(f"reference_images JSON must decode to a list, got {type(parsed).__name__}")
            return [str(item) for item in parsed if str(item).strip()]
        for delimiter in ("|", ";", ","):
            if delimiter in stripped:
                return [part.strip() for part in stripped.split(delimiter) if part.strip()]
        return [stripped]
    raise ValueError(f"Unsupported reference image value type: {type(value).__name__}")


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


@app.command()
def main(  # noqa: PLR0913
    dataset_file: str = typer.Argument(..., help="Flat CSV/JSON/JSONL manifest."),
    text_encoder_path: str = typer.Option(..., help="Local Gemma text encoder directory."),
    output_dir: str = typer.Option(..., help="Output gt_siglip_tokens directory."),
    video_column: str = typer.Option("video", help="Target video path column used to mirror output names."),
    reference_column: str = typer.Option("reference_images", help="Column containing 1-N reference image paths."),
    root_dir: str | None = typer.Option(
        None,
        help="Root for relative video/reference paths. Defaults to the manifest parent.",
    ),
    max_ref_images: int | None = typer.Option(None, help="Optional cap on reference images per sample.", min=1),
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

    for row in track(rows, description="Encoding GT SigLIP visual tokens"):
        if video_column not in row:
            raise ValueError(f"Missing video column '{video_column}' in row: {row}")

        video_path = _resolve_path(str(row[video_column]), data_root)
        output_file = out_root / _output_relative(video_path, data_root).with_suffix(".pt")
        if output_file.exists() and not overwrite:
            skipped += 1
            continue

        ref_values = _parse_reference_images(row.get(reference_column))
        if max_ref_images is not None:
            ref_values = ref_values[:max_ref_images]
        if not ref_values:
            logger.warning(f"Skipping {video_path}: no reference images")
            skipped += 1
            continue

        images = [open_image_as_srgb(_resolve_path(value, data_root)) for value in ref_values]
        processed_images = image_processor(images=images, return_tensors="pt")
        pixel_values = processed_images["pixel_values"].to(device=device, dtype=torch.bfloat16)

        with torch.inference_mode():
            visual_batch = extract_projected_visual_tokens(text_encoder.model, pixel_values)

        tokens = visual_batch.tokens[0].cpu().contiguous()
        mask = visual_batch.mask[0].cpu().contiguous()
        token_count = int(mask.sum().item())
        if first_token_count is None:
            first_token_count = token_count
            logger.info(f"Detected {first_token_count} GT visual tokens per sample")
        if expected_token_count is not None and token_count != expected_token_count:
            raise ValueError(
                f"{video_path} produced {token_count} visual tokens, expected {expected_token_count}. "
                "Use the detected count in planner_token_count or fix reference image slots."
            )

        save_data = {
            "visual_tokens": tokens,
            "visual_token_mask": mask,
            "num_visual_tokens": torch.tensor(token_count, dtype=torch.long),
            "num_ref_images": torch.tensor(len(images), dtype=torch.long),
        }
        _atomic_save(save_data, output_file)
        processed += 1

    logger.info(
        f"GT SigLIP token preprocessing complete: {processed} encoded, {skipped} skipped -> {out_root}"
    )


if __name__ == "__main__":
    app()
