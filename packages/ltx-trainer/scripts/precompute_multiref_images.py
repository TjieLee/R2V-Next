#!/usr/bin/env python3

"""Precompute stacked multi-reference image latents for Stage 1 training.

The input metadata must contain a target media column used for output naming
and a reference image column containing either a JSON list or a delimited string
of image paths. Each output file mirrors the target media path and stores:

    latents: [R, C, 1, H, W]
    ref_valid_mask: [R]
    num_refs: int
    num_frames/height/width/fps metadata
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import typer
from rich.progress import track
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import crop, resize, to_tensor

from ltx_trainer import logger
from ltx_trainer.model_loader import load_video_vae_encoder
from ltx_trainer.utils import open_image_as_srgb
from process_videos import _atomic_save, _encode_video, _output_relative

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Encode 1-N reference images per sample into multi_reference_latents/.",
)


def _load_rows(dataset_file: Path) -> list[dict[str, Any]]:
    if dataset_file.suffix == ".csv":
        return pd.read_csv(dataset_file).to_dict(orient="records")
    if dataset_file.suffix == ".json":
        data = json.loads(dataset_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return list(data.values())
        if isinstance(data, list):
            return data
        raise ValueError("JSON metadata must be a list or dict of objects")
    if dataset_file.suffix == ".jsonl":
        rows = []
        with dataset_file.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported metadata format: {dataset_file.suffix}")


def _parse_ref_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError(f"Reference image JSON must decode to a list, got {type(parsed).__name__}")
            return [str(item) for item in parsed]
        for sep in ("|", ";", ","):
            if sep in text:
                return [part.strip() for part in text.split(sep) if part.strip()]
        return [text]
    raise ValueError(f"Unsupported reference_images value type: {type(value).__name__}")


def _resolve_path(value: str, data_root: Path, media_root: Path | None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if media_root is not None:
        return media_root / path
    return data_root / path


def _parse_resolution(value: str) -> tuple[int, int]:
    width_str, height_str = value.lower().split("x", 1)
    width, height = int(width_str), int(height_str)
    if width % 32 != 0 or height % 32 != 0:
        raise ValueError(f"Reference resolution must be divisible by 32, got {width}x{height}")
    return width, height


def _resize_and_crop_image(image: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    current_height, current_width = image.shape[2], image.shape[3]
    current_aspect = current_width / current_height
    target_aspect = target_width / target_height

    if current_aspect > target_aspect:
        new_width = int(current_width * target_height / current_height)
        image = resize(image, size=[target_height, new_width], interpolation=InterpolationMode.BICUBIC)
    else:
        new_height = int(current_height * target_width / current_width)
        image = resize(image, size=[new_height, target_width], interpolation=InterpolationMode.BICUBIC)

    current_height, current_width = image.shape[2], image.shape[3]
    image = image.squeeze(0)
    top = (current_height - target_height) // 2
    left = (current_width - target_width) // 2
    return crop(image, top=top, left=left, height=target_height, width=target_width)


def _load_ref_image(path: Path, target_height: int, target_width: int) -> torch.Tensor:
    image = open_image_as_srgb(path)
    image = to_tensor(image).unsqueeze(0)
    image = _resize_and_crop_image(image, target_height, target_width)
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.clamp_(0, 1)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    image = transform(image)
    return image.unsqueeze(1)


def _target_resolution_from_latents(
    *,
    target_path: Path,
    data_root: Path,
    target_latents_dir: Path,
) -> tuple[int, int]:
    rel = _output_relative(target_path, data_root).with_suffix(".pt")
    latent_file = target_latents_dir / rel
    if not latent_file.is_file():
        raise FileNotFoundError(f"Target latent file not found for resolution lookup: {latent_file}")
    latent_data = torch.load(latent_file, map_location="cpu", weights_only=True)
    return int(latent_data["width"]) * 32, int(latent_data["height"]) * 32


@app.command()
def main(  # noqa: PLR0913
    dataset_path: str = typer.Argument(..., help="CSV/JSON/JSONL metadata file"),
    model_path: str = typer.Option(..., help="Path to the LTX-2 checkpoint (.safetensors)"),
    output_dir: str = typer.Option(..., help="Output directory, usually .precomputed/multi_reference_latents"),
    media_column: str = typer.Option("video", help="Target media column used to mirror output names"),
    reference_column: str = typer.Option(
        "reference_images",
        help="Column containing 1-N reference image paths as a list or delimited string",
    ),
    media_root: str | None = typer.Option(None, help="Optional base directory for relative media paths"),
    target_latents_dir: str | None = typer.Option(
        None,
        help="Existing latents/ directory. When set, reference image resolution is read per target sample.",
    ),
    ref_resolution: str | None = typer.Option(
        None,
        help='Fallback fixed reference resolution as "WxH"; required when --target-latents-dir is omitted.',
    ),
    max_ref_images: int | None = typer.Option(None, help="Optional cap on reference images per sample", min=1),
    device: str = typer.Option("cuda", help="Torch device for VAE encoding"),
    vae_tiling: bool = typer.Option(False, help="Enable spatial VAE tiling"),
    overwrite: bool = typer.Option(False, help="Recompute outputs even if they already exist"),
) -> None:
    dataset_file = Path(dataset_path)
    data_root = dataset_file.parent
    media_root_path = Path(media_root) if media_root else None
    target_latents_path = Path(target_latents_dir) if target_latents_dir else None
    fixed_resolution = _parse_resolution(ref_resolution) if ref_resolution else None
    if target_latents_path is None and fixed_resolution is None:
        raise typer.BadParameter("Provide either --target-latents-dir or --ref-resolution")

    rows = _load_rows(dataset_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device)
    vae = load_video_vae_encoder(model_path, device=torch_device, dtype=torch.bfloat16)

    processed = 0
    skipped = 0
    for row in track(rows, description="Encoding multi-reference images"):
        target_path = _resolve_path(str(row[media_column]), data_root, media_root_path)
        output_file = output_path / _output_relative(target_path, data_root).with_suffix(".pt")
        if output_file.is_file() and not overwrite:
            skipped += 1
            continue

        ref_paths = [_resolve_path(path, data_root, media_root_path) for path in _parse_ref_list(row[reference_column])]
        if max_ref_images is not None:
            ref_paths = ref_paths[:max_ref_images]
        if not ref_paths:
            logger.warning(f"Skipping {target_path}: no reference images")
            skipped += 1
            continue

        if target_latents_path is not None:
            target_width, target_height = _target_resolution_from_latents(
                target_path=target_path,
                data_root=data_root,
                target_latents_dir=target_latents_path,
            )
        else:
            target_width, target_height = fixed_resolution

        refs = [_load_ref_image(path, target_height, target_width) for path in ref_paths]
        video = torch.stack(refs, dim=0)
        with torch.inference_mode():
            encoded = _encode_video(vae=vae, video=video, use_tiling=vae_tiling)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "latents": encoded["latents"].cpu().contiguous(),
            "ref_valid_mask": torch.ones(len(ref_paths), dtype=torch.bool),
            "num_refs": len(ref_paths),
            "num_frames": encoded["num_frames"],
            "height": encoded["height"],
            "width": encoded["width"],
            "fps": 1.0,
        }
        _atomic_save(save_data, output_file)
        processed += 1

    logger.info(f"Multi-reference preprocessing complete: {processed} encoded, {skipped} skipped -> {output_path}")


if __name__ == "__main__":
    app()
