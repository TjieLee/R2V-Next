#!/usr/bin/env python3
"""Precompute Stage 1 multi-reference VLM text conditions.

This builds connector-input features from:

    system prompt -> user text prompt -> reference image tokens

No planner placeholders are appended here. Stage 1 appends target-video GT
SigLIP/projector visual tokens later in the training strategy.
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
from transformers import AutoImageProcessor, AutoTokenizer, Gemma3Processor

from ltx_core.multicond.visual_tokens import extract_projected_visual_tokens, scatter_visual_tokens_into_embeddings
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_core.utils import find_matching_file
from ltx_trainer import logger
from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder
from ltx_trainer.utils import open_image_as_srgb

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Build Stage 1 VLM-aware conditions from text plus 1-N reference images.",
)


def _load_default_system_prompt(prompt_name: str) -> str:
    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "ltx-core"
        / "src"
        / "ltx_core"
        / "text_encoders"
        / "gemma"
        / "encoders"
        / "prompts"
        / prompt_name
    )
    return prompt_path.read_text(encoding="utf-8")


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


def _build_messages(system_prompt: str, user_prompt: str, num_images: int) -> list[dict[str, Any]]:
    if num_images <= 0:
        user_content: str | list[dict[str, str]] = f"User Raw Input Prompt: {user_prompt}."
    else:
        content: list[dict[str, str]] = [
            {"type": "text", "text": f"User Raw Input Prompt: {user_prompt}."},
            {"type": "text", "text": "Reference images:"},
        ]
        for index in range(num_images):
            content.extend(
                [
                    {"type": "text", "text": f"Reference image {index + 1}:"},
                    {"type": "image"},
                ]
            )
        user_content = content

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _atomic_save(data: dict[str, torch.Tensor], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + f".tmp.{os.getpid()}")
    torch.save(data, tmp_file)
    tmp_file.replace(output_file)


def _get_language_model(text_encoder: torch.nn.Module) -> torch.nn.Module:
    gemma_model = text_encoder.model.model
    language_model = getattr(gemma_model, "language_model", None)
    if language_model is None:
        raise ValueError("Gemma model does not expose language_model")
    return language_model


def _get_input_embeddings(language_model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(language_model, "get_input_embeddings"):
        embeddings = language_model.get_input_embeddings()
        if embeddings is not None:
            return embeddings
    embeddings = getattr(language_model, "embed_tokens", None)
    if embeddings is None:
        raise ValueError("Gemma language_model does not expose input embeddings")
    return embeddings


def _build_inputs_embeds(
    *,
    text_encoder: torch.nn.Module,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor | None,
    num_ref_images: int,
) -> torch.Tensor:
    language_model = _get_language_model(text_encoder)
    embed_tokens = _get_input_embeddings(language_model)
    inputs_embeds = embed_tokens(input_ids)

    if pixel_values is None:
        return inputs_embeds

    image_counts = torch.tensor([num_ref_images], device=pixel_values.device, dtype=torch.long)
    visual_batch = extract_projected_visual_tokens(
        text_encoder.model,
        pixel_values,
        image_counts=image_counts,
    )
    return scatter_visual_tokens_into_embeddings(
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        visual_tokens=visual_batch.tokens,
        visual_mask=visual_batch.mask,
        image_token_index=GEMMA3_CONFIG_FOR_LTX.image_token_index,
    )


@app.command()
def main(  # noqa: PLR0913
    dataset_file: str = typer.Argument(..., help="Flat CSV/JSON/JSONL manifest."),
    model_path: str = typer.Option(..., help="Path to the LTX-2 checkpoint (.safetensors)."),
    text_encoder_path: str = typer.Option(..., help="Local Gemma text encoder directory."),
    output_dir: str = typer.Option(..., help="Output directory, usually .precomputed/vlm_conditions."),
    video_column: str = typer.Option("video", help="Target video path column used for output names."),
    caption_column: str = typer.Option("caption", help="Column containing user/raw prompt text."),
    reference_column: str = typer.Option("reference_images", help="Column containing 1-N reference image paths."),
    root_dir: str | None = typer.Option(
        None,
        help="Root for relative video/reference paths. Defaults to the manifest parent.",
    ),
    system_prompt_path: str | None = typer.Option(
        None,
        help="Optional custom system prompt file. Defaults to the multi-reference video planner prompt.",
    ),
    max_ref_images: int | None = typer.Option(None, help="Optional cap on reference images per sample."),
    max_length: int = typer.Option(4096, help="Tokenizer max length."),
    device: str = typer.Option("cuda", help="Torch device for VLM condition extraction."),
    load_in_8bit: bool = typer.Option(False, help="Load Gemma in 8-bit mode."),
    overwrite: bool = typer.Option(False, help="Rebuild files that already exist."),
    skip_errors: bool = typer.Option(True, help="Skip unreadable/bad rows instead of stopping the whole shard."),
) -> None:
    dataset_path = Path(dataset_file)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {dataset_path}")
    if max_ref_images is not None and max_ref_images < 1:
        raise typer.BadParameter("--max-ref-images must be >= 1")
    if max_length < 1:
        raise typer.BadParameter("--max-length must be >= 1")

    data_root = Path(root_dir) if root_dir is not None else dataset_path.parent
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tokenizer_root = str(find_matching_file(text_encoder_path, "tokenizer.model").parent)
    processor_root = str(find_matching_file(text_encoder_path, "preprocessor_config.json").parent)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True, model_max_length=max_length)
    image_processor = AutoImageProcessor.from_pretrained(processor_root, local_files_only=True, use_fast=False)
    processor = Gemma3Processor(image_processor=image_processor, tokenizer=tokenizer)

    text_encoder = load_text_encoder(text_encoder_path, device=device, dtype=torch.bfloat16, load_in_8bit=load_in_8bit)
    text_encoder.eval()
    embeddings_processor = load_embeddings_processor(model_path, device=device, dtype=torch.bfloat16)
    embeddings_processor.eval()

    rows = _read_rows(dataset_path)
    default_system_prompt = _load_default_system_prompt("gemma_multiref_video_planner_system_prompt.txt")
    custom_system_prompt = Path(system_prompt_path).read_text(encoding="utf-8") if system_prompt_path else None
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for row in track(rows, description="Building Stage 1 multi-reference VLM conditions"):
        try:
            if video_column not in row:
                raise ValueError(f"Missing video column '{video_column}' in row: {row}")
            if caption_column not in row:
                raise ValueError(f"Missing caption column '{caption_column}' in row: {row}")

            video_path = _resolve_path(str(row[video_column]), data_root)
            output_file = out_root / _output_relative(video_path, data_root).with_suffix(".pt")
            if output_file.exists() and not overwrite:
                skipped_count += 1
                continue

            ref_values = _parse_reference_images(row.get(reference_column))
            if max_ref_images is not None:
                ref_values = ref_values[:max_ref_images]
            ref_paths = [_resolve_path(value, data_root) for value in ref_values]
            images = [open_image_as_srgb(path) for path in ref_paths]

            system_prompt = custom_system_prompt or default_system_prompt
            messages = _build_messages(
                system_prompt=system_prompt,
                user_prompt=str(row[caption_column]),
                num_images=len(images),
            )
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            processed = processor(
                text=text,
                images=images if images else None,
                return_tensors="pt",
                padding=False,
                max_length=max_length,
                truncation=True,
            )

            input_ids = processed["input_ids"].to(device=device, dtype=torch.long)
            attention_mask = processed["attention_mask"].to(device=device, dtype=torch.long)
            pixel_values = processed.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)

            with torch.inference_mode():
                inputs_embeds = _build_inputs_embeds(
                    text_encoder=text_encoder,
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    num_ref_images=len(images),
                )
                language_model = _get_language_model(text_encoder)
                outputs = language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states
                video_prompt_embeds, audio_prompt_embeds = embeddings_processor.feature_extractor(
                    hidden_states,
                    attention_mask,
                    "right",
                )

            save_data = {
                "video_prompt_embeds": video_prompt_embeds[0].cpu().contiguous(),
                "prompt_attention_mask": attention_mask[0].cpu().contiguous(),
                "num_ref_images": torch.tensor(len(images), dtype=torch.long),
            }
            if audio_prompt_embeds is not None:
                save_data["audio_prompt_embeds"] = audio_prompt_embeds[0].cpu().contiguous()
            _atomic_save(save_data, output_file)
            processed_count += 1
        except Exception as exc:
            if not skip_errors:
                raise
            failed_count += 1
            logger.warning(f"Skipping row due to Stage 1 VLM condition preprocessing error: {exc}")

    logger.info(
        f"Stage 1 VLM condition preprocessing complete: "
        f"{processed_count} encoded, {skipped_count} existing skipped, {failed_count} failed skipped -> {out_root}"
    )


if __name__ == "__main__":
    app()
