#!/usr/bin/env python3
"""Precompute tokenized Gemma/VLM planner inputs for Stage 2 training.

This script is intentionally lightweight: it does not run Gemma. It builds the
chat-template input expected by Gemma3Processor in the order:

    system prompt -> user text -> reference image tokens
    -> <image_start> -> <image_pad> * K -> <image_end>

It stores tensors under ``planner_vlm_inputs/`` with the same relative paths as
``latents/`` and ``conditions/``. The saved masks separate source reference
image tokens, text tokens, and target planner placeholder regions so Stage 2
CFG dropout can remove text and image conditions independently.
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

from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_core.utils import find_matching_file
from ltx_trainer import logger
from ltx_trainer.utils import open_image_as_srgb

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Build tokenized Gemma/VLM chat inputs for multi-reference Stage 2 planner training.",
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


def _tensorize_processor_output(processed: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key, value in processed.items():
        if isinstance(value, torch.Tensor):
            if key in {"input_ids", "attention_mask", "token_type_ids", "position_ids", "cache_position"}:
                value = value.squeeze(0)
            elif key == "pixel_values" and value.ndim == 5 and value.shape[0] == 1:
                value = value.squeeze(0)
            tensors[key] = value.cpu().contiguous()
    return tensors


def _compute_source_max_length(max_length: int, planner_token_count: int) -> int:
    source_max_length = max_length - planner_token_count - 2
    if source_max_length <= 0:
        raise typer.BadParameter("--max-length must be greater than --planner-token-count + 2 boundary tokens")
    return source_max_length


def _append_planner_placeholders(
    tensor_data: dict[str, torch.Tensor],
    *,
    planner_token_count: int,
    max_length: int,
    source_max_length: int,
    placeholder_token_id: int,
    start_token_id: int,
    end_token_id: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    for unused_key in ("token_type_ids", "position_ids", "cache_position"):
        tensor_data.pop(unused_key, None)

    input_ids = tensor_data["input_ids"].to(dtype=torch.long)
    attention_mask = tensor_data["attention_mask"].to(dtype=torch.long)
    if input_ids.ndim != 1:
        raise ValueError(f"Expected 1D input_ids after squeeze, got {tuple(input_ids.shape)}")
    planner_region_len = planner_token_count + 2
    if input_ids.shape[0] + planner_region_len > max_length:
        raise ValueError(
            "Source VLM input exceeded the reserved planner budget. This should not happen when "
            "processor(max_length=source_max_length) is respected; increase --max-length, reduce "
            "--planner-token-count, reduce --max-ref-images, or shorten the caption."
        )

    placeholder_ids = torch.full((planner_token_count,), placeholder_token_id, dtype=torch.long)
    placeholder_mask = torch.ones(planner_token_count, dtype=torch.bool)
    boundary_ids = torch.tensor([start_token_id, end_token_id], dtype=torch.long)

    input_ids = torch.cat([input_ids, boundary_ids[:1], placeholder_ids, boundary_ids[1:]], dim=0)
    planner_attention = torch.ones(planner_region_len, dtype=torch.long)
    attention_mask = torch.cat([attention_mask, planner_attention], dim=0)
    planner_placeholder_mask = torch.cat(
        [
            torch.zeros(input_ids.shape[0] - planner_token_count - 1, dtype=torch.bool),
            placeholder_mask,
            torch.zeros(1, dtype=torch.bool),
        ],
        dim=0,
    )
    planner_boundary_mask = torch.cat(
        [
            torch.zeros(input_ids.shape[0] - planner_region_len, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
            torch.zeros(planner_token_count, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
        ],
        dim=0,
    )
    planner_region_mask = planner_placeholder_mask | planner_boundary_mask

    pad_len = max_length - input_ids.shape[0]
    if pad_len > 0:
        input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)], dim=0)
        attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)], dim=0)
        planner_placeholder_mask = torch.cat([planner_placeholder_mask, torch.zeros(pad_len, dtype=torch.bool)], dim=0)
        planner_boundary_mask = torch.cat([planner_boundary_mask, torch.zeros(pad_len, dtype=torch.bool)], dim=0)
        planner_region_mask = torch.cat([planner_region_mask, torch.zeros(pad_len, dtype=torch.bool)], dim=0)

    tensor_data["input_ids"] = input_ids
    tensor_data["attention_mask"] = attention_mask
    tensor_data["planner_placeholder_mask"] = planner_placeholder_mask
    tensor_data["planner_boundary_mask"] = planner_boundary_mask
    tensor_data["planner_region_mask"] = planner_region_mask
    active_mask = attention_mask.to(dtype=torch.bool)
    ref_visual_token_mask = (
        (input_ids == GEMMA3_CONFIG_FOR_LTX.image_token_index) & ~planner_region_mask & active_mask
    )
    ref_image_region_mask = (
        (
            (input_ids == GEMMA3_CONFIG_FOR_LTX.image_token_index)
            | (input_ids == GEMMA3_CONFIG_FOR_LTX.boi_token_index)
            | (input_ids == GEMMA3_CONFIG_FOR_LTX.eoi_token_index)
        )
        & ~planner_region_mask
        & active_mask
    )
    text_token_mask = active_mask & ~ref_image_region_mask & ~planner_region_mask
    tensor_data["ref_visual_token_mask"] = ref_visual_token_mask.to(dtype=torch.bool)
    tensor_data["ref_image_region_mask"] = ref_image_region_mask.to(dtype=torch.bool)
    # Backward-compatible name used by older Stage 2 code for drop_ref masking.
    # It now represents the full reference image region, not only pure image-pad positions.
    tensor_data["gt_image_token_mask"] = ref_image_region_mask.to(dtype=torch.bool)
    tensor_data["text_token_mask"] = text_token_mask.to(dtype=torch.bool)
    tensor_data["source_max_length"] = torch.tensor(source_max_length, dtype=torch.long)
    tensor_data["ref_visual_token_count"] = ref_visual_token_mask.sum().to(dtype=torch.long)
    tensor_data["planner_token_count"] = torch.tensor(planner_token_count, dtype=torch.long)
    tensor_data["planner_placeholder_token_id"] = torch.tensor(placeholder_token_id, dtype=torch.long)
    tensor_data["planner_start_token_id"] = torch.tensor(start_token_id, dtype=torch.long)
    tensor_data["planner_end_token_id"] = torch.tensor(end_token_id, dtype=torch.long)
    return tensor_data


def _validate_reference_image_token_count(tensor_data: dict[str, torch.Tensor], num_ref_images: int) -> None:
    expected = num_ref_images * GEMMA3_CONFIG_FOR_LTX.mm_tokens_per_image
    actual_tensor = tensor_data.get("ref_visual_token_count")
    if actual_tensor is None:
        actual = int(tensor_data["ref_visual_token_mask"].sum().item())
    elif isinstance(actual_tensor, torch.Tensor):
        actual = int(actual_tensor.item())
    else:
        actual = int(actual_tensor)
    if actual != expected:
        raise ValueError(
            "Reference image tokens were truncated or mismatched. Increase --max-length, reduce "
            "--planner-token-count, reduce --max-ref-images, or shorten the caption. "
            f"Expected {expected} reference image tokens for {num_ref_images} images, got {actual}."
        )


def _atomic_save(data: dict[str, torch.Tensor], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + f".tmp.{os.getpid()}")
    torch.save(data, tmp_file)
    tmp_file.replace(output_file)


@app.command()
def main(
    dataset_file: str = typer.Argument(..., help="Flat CSV/JSON/JSONL manifest."),
    text_encoder_path: str = typer.Option(..., help="Local Gemma text encoder directory."),
    output_dir: str = typer.Option(..., help="Output planner_vlm_inputs directory."),
    video_column: str = typer.Option("video", help="Column containing target video path."),
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
    planner_token_count: int = typer.Option(
        1024,
        help="Fixed <image_pad> target planner placeholder count. Must match gt_siglip_tokens visual token count.",
    ),
    max_length: int = typer.Option(4096, help="Tokenizer max length."),
    overwrite: bool = typer.Option(False, help="Rebuild files that already exist."),
    skip_errors: bool = typer.Option(True, help="Skip unreadable/bad rows instead of stopping the whole shard."),
) -> None:
    dataset_path = Path(dataset_file)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {dataset_path}")
    if max_ref_images is not None and max_ref_images < 1:
        raise typer.BadParameter("--max-ref-images must be >= 1")
    if planner_token_count < 1:
        raise typer.BadParameter("--planner-token-count must be >= 1")
    if max_length < 1:
        raise typer.BadParameter("--max-length must be >= 1")
    source_max_length = _compute_source_max_length(max_length=max_length, planner_token_count=planner_token_count)

    data_root = Path(root_dir) if root_dir is not None else dataset_path.parent
    out_root = Path(output_dir)

    tokenizer_root = str(find_matching_file(text_encoder_path, "tokenizer.model").parent)
    processor_root = str(find_matching_file(text_encoder_path, "preprocessor_config.json").parent)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True, model_max_length=max_length)
    image_processor = AutoImageProcessor.from_pretrained(processor_root, local_files_only=True, use_fast=False)
    processor = Gemma3Processor(image_processor=image_processor, tokenizer=tokenizer)

    rows = _read_rows(dataset_path)
    default_multiref_prompt = _load_default_system_prompt("gemma_multiref_video_planner_system_prompt.txt")
    custom_system_prompt = Path(system_prompt_path).read_text(encoding="utf-8") if system_prompt_path else None
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for row in track(rows, description="Building planner VLM inputs"):
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

            system_prompt = custom_system_prompt or default_multiref_prompt
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
                max_length=source_max_length,
                truncation=True,
            )

            tensor_data = _tensorize_processor_output(processed)
            tensor_data = _append_planner_placeholders(
                tensor_data,
                planner_token_count=planner_token_count,
                max_length=max_length,
                source_max_length=source_max_length,
                placeholder_token_id=GEMMA3_CONFIG_FOR_LTX.image_token_index,
                start_token_id=GEMMA3_CONFIG_FOR_LTX.boi_token_index,
                end_token_id=GEMMA3_CONFIG_FOR_LTX.eoi_token_index,
                pad_token_id=pad_token_id,
            )
            tensor_data["num_ref_images"] = torch.tensor(len(images), dtype=torch.long)
            _validate_reference_image_token_count(tensor_data, len(images))
            _atomic_save(tensor_data, output_file)
            processed_count += 1
        except Exception as exc:
            if not skip_errors:
                raise
            failed_count += 1
            logger.warning(f"Skipping row due to planner VLM input preprocessing error: {exc}")

    logger.info(
        f"Planner VLM input preprocessing complete: "
        f"{processed_count} encoded, {skipped_count} existing skipped, {failed_count} failed skipped -> {out_root}"
    )


if __name__ == "__main__":
    app()
