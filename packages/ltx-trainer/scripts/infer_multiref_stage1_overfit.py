#!/usr/bin/env python3
"""Stage 1 multi-reference teacher-forcing inference for overfit samples.

This is not the final Stage 2 planner inference path. It uses precomputed
target-video GT SigLIP visual tokens for the positive/full-condition branch.
It also supports a text-only/no-SigLIP ablation branch that keeps the DiT
multi-reference latent stream while replacing the positive condition with
plain LTX text conditioning. Optional inference-time CFG/STG is supported.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import typer
import yaml
from einops import rearrange
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from rich.console import Console
from rich.progress import track
from safetensors.torch import load_file
from torch import Tensor

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_core.multicond.rope_mask_builder import build_multiref_sequence
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_core.types import VideoLatentShape
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder, load_transformer, load_video_vae_decoder
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.training_strategies.multi_reference_video import MultiReferenceVideoStrategy
from ltx_trainer.utils import open_image_as_srgb
from ltx_trainer.video_utils import save_video

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Run Stage 1 multi-reference teacher-forcing inference on one overfit sample.",
)

_DEFAULT_TILING = TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=192, tile_overlap_in_pixels=64),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=48, tile_overlap_in_frames=24),
)


class _TupleSafeLoader(yaml.SafeLoader):
    pass


def _construct_python_tuple(loader: yaml.SafeLoader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node))


_TupleSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple,
)


def _velocity_to_denoised(latent: Tensor, velocity: Tensor, timesteps: Tensor) -> Tensor:
    if latent.shape != velocity.shape:
        raise ValueError(f"Velocity shape {list(velocity.shape)} does not match latent shape {list(latent.shape)}")
    if timesteps.shape != latent.shape[:2]:
        raise ValueError(
            f"Timesteps shape {list(timesteps.shape)} must match packed token shape {list(latent.shape[:2])}"
        )
    denoise_timesteps = timesteps.unsqueeze(-1).to(device=latent.device, dtype=torch.float32)
    return (latent.to(torch.float32) - velocity.to(torch.float32) * denoise_timesteps).to(latent.dtype)


def _combine_cfg_denoised(denoised_pos: Tensor, denoised_neg: Tensor, guidance_scale: float) -> Tensor:
    return denoised_pos + (guidance_scale - 1.0) * (denoised_pos - denoised_neg)


def _stg_enabled(stg_scale: float) -> bool:
    return stg_scale != 0.0


def _parse_stg_blocks(value: str | None) -> list[int] | None:
    if value is None:
        return [28]
    text = value.strip().lower()
    if text in {"", "none", "all"}:
        return None
    blocks: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            blocks.append(int(part))
    return blocks or None


def _build_stg_perturbation_config(stg_blocks: list[int] | None) -> BatchedPerturbationConfig:
    return BatchedPerturbationConfig(
        perturbations=[
            PerturbationConfig(
                perturbations=[
                    Perturbation(
                        type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                        blocks=stg_blocks,
                    )
                ]
            )
        ]
    )


def _combine_multidirectional_denoised(
    *,
    denoised_pos: Tensor,
    denoised_neg: Tensor | None,
    denoised_no_ref: Tensor | None,
    denoised_siglip_isolated: Tensor | None,
    denoised_siglip_null: Tensor | None,
    denoised_stg: Tensor | None,
    guidance_scale: float,
    ref_guidance_scale: float,
    siglip_guidance_scale: float,
    stg_scale: float,
    guidance_rescale: float,
    target_seq_len: int | None = None,
) -> Tensor:
    if _cfg_enabled(guidance_scale):
        if denoised_neg is None:
            raise ValueError("guidance_scale != 1.0 requires denoised_neg")
        guided = denoised_neg + guidance_scale * (denoised_pos - denoised_neg)
    else:
        guided = denoised_pos

    if ref_guidance_scale != 0.0:
        if denoised_no_ref is None:
            raise ValueError("ref_guidance_scale != 0.0 requires denoised_no_ref")
        guided = guided + ref_guidance_scale * (denoised_pos - denoised_no_ref)

    if siglip_guidance_scale != 0.0:
        if denoised_siglip_isolated is None:
            raise ValueError("siglip_guidance_scale != 0 requires denoised_siglip_isolated")
        if denoised_siglip_null is None:
            raise ValueError("siglip_guidance_scale != 0 requires denoised_siglip_null")
        guided = guided + siglip_guidance_scale * (denoised_siglip_isolated - denoised_siglip_null)

    if _stg_enabled(stg_scale):
        if denoised_stg is None:
            raise ValueError("stg_scale != 0.0 requires denoised_stg")
        guided = guided + stg_scale * (denoised_pos - denoised_stg)

    if guidance_rescale != 0.0:
        if target_seq_len is None or target_seq_len <= 0:
            raise ValueError("guidance_rescale requires a positive target_seq_len")

        cond_target = denoised_pos[:, -target_seq_len:, :].float()
        guided_target = guided[:, -target_seq_len:, :].float()
        cond_std = cond_target.std()
        guided_std = guided_target.std().clamp(min=1.0e-8)
        factor = cond_std / guided_std
        factor = guidance_rescale * factor + (1.0 - guidance_rescale)
        guided = guided * factor.to(device=guided.device, dtype=guided.dtype)

    return guided


def _condition_shape(conditions: dict[str, Tensor | None] | None) -> list[int] | None:
    if conditions is None:
        return None
    context_key = _condition_feature_key(conditions)
    context = conditions[context_key]
    return _shape(context) if isinstance(context, Tensor) else None


def _cfg_enabled(guidance_scale: float) -> bool:
    return guidance_scale != 1.0


def _validate_guidance_rescale(guidance_rescale: float) -> None:
    if not 0.0 <= guidance_rescale <= 1.0:
        raise typer.BadParameter("--guidance-rescale must be in [0, 1]")


def _validate_siglip_guidance(siglip_guidance_scale: float, condition_mode: str) -> None:
    if not math.isfinite(siglip_guidance_scale):
        raise typer.BadParameter("--siglip-guidance-scale must be finite")
    if siglip_guidance_scale != 0.0 and condition_mode != "full_siglip":
        raise typer.BadParameter(
            "--siglip-guidance-scale is only supported with --condition-mode full_siglip"
        )


def _build_isolated_siglip_contexts(
    *,
    pos_context: Tensor,
    pos_context_mask: Tensor | None,
    visual_token_count: int,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    if visual_token_count <= 0:
        raise ValueError("Isolated SigLIP guidance requires visual_context_token_count > 0")
    if visual_token_count > pos_context.shape[1]:
        raise ValueError(
            "visual_context_token_count exceeds positive context length: "
            f"{visual_token_count} > {pos_context.shape[1]}"
        )

    text_token_count = pos_context.shape[1] - visual_token_count
    text_vlm_context = pos_context[:, :text_token_count, :]
    visual_context = pos_context[:, text_token_count:, :]
    null_text_siglip_context = torch.cat(
        [torch.zeros_like(text_vlm_context), visual_context],
        dim=1,
    )
    null_text_no_siglip_context = torch.zeros_like(pos_context)
    return (
        null_text_siglip_context,
        null_text_no_siglip_context,
        pos_context_mask,
        pos_context_mask,
    )


def _negative_ref_valid_mask(ref_valid_mask: Tensor, *, drop_ref_latents: bool) -> Tensor:
    return torch.zeros_like(ref_valid_mask) if drop_ref_latents else ref_valid_mask


_POSITIVE_CONDITION_MODES = {"full_siglip", "text_only_no_siglip"}


def _uses_full_siglip_condition(condition_mode: str) -> bool:
    return condition_mode == "full_siglip"


def _condition_mode_detail(condition_mode: str, *, guidance_scale: float, stg_scale: float) -> str:
    if condition_mode == "full_siglip":
        if _cfg_enabled(guidance_scale) and _stg_enabled(stg_scale):
            return "stage1_full_siglip_cfg_stg"
        if _cfg_enabled(guidance_scale):
            return "stage1_teacher_cfg_full_vs_negative_prompt_no_siglip"
        if _stg_enabled(stg_scale):
            return "stage1_teacher_stg_full_condition"
        return "stage1_teacher_gt_siglip_full_condition"

    if condition_mode == "text_only_no_siglip":
        if _cfg_enabled(guidance_scale) and _stg_enabled(stg_scale):
            return "stage1_text_only_no_siglip_cfg_stg"
        if _cfg_enabled(guidance_scale):
            return "stage1_text_only_no_siglip_cfg"
        if _stg_enabled(stg_scale):
            return "stage1_text_only_no_siglip_stg"
        return "stage1_text_only_no_siglip_with_reference_latents"

    raise ValueError(f"Unsupported condition mode: {condition_mode}")


def _read_manifest_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return list(data.values())
        if isinstance(data, list):
            return data
        raise ValueError(f"JSON manifest must contain a list or dict of objects: {path}")
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    raise ValueError(f"Unsupported manifest suffix: {path.suffix}")


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for pattern in ("*.json", "*.jsonl", "*.csv"):
            for file in sorted(path.glob(pattern)):
                rows.extend(_read_manifest_file(file))
        if not rows:
            raise FileNotFoundError(f"No manifest shards found in {path}")
        return rows
    if path.is_file():
        return _read_manifest_file(path)
    raise FileNotFoundError(f"Manifest does not exist: {path}")


def _resolve_path(value: str, root_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root_dir / path


def _output_relative(path: Path, data_root: Path) -> Path:
    try:
        return path.relative_to(data_root)
    except ValueError:
        return Path(*path.parts[1:]) if path.is_absolute() else path


def _normalize_precomputed_root(path: Path) -> Path:
    if (path / ".precomputed").is_dir():
        return path / ".precomputed"
    return path


def _parse_reference_images(value: Any) -> list[str]:
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
                raise ValueError("reference_images JSON must decode to a list")
            return [str(item) for item in parsed]
        for sep in ("|", ";", ","):
            if sep in text:
                return [part.strip() for part in text.split(sep) if part.strip()]
        return [text]
    raise ValueError(f"Unsupported reference_images value type: {type(value).__name__}")


def _load_pt_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing precomputed file: {path}")
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


def _unsqueeze_sample_dim(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Tensor):
            out[key] = value.unsqueeze(0)
        else:
            out[key] = torch.tensor([value]) if isinstance(value, (int, float, bool)) else value
    return out


def _build_single_sample_batch(
    *,
    latents: dict[str, Any],
    multi_reference_latents: dict[str, Any],
    conditions: dict[str, Any] | None = None,
    gt_visual_tokens: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latents = PrecomputedDataset._normalize_video_latents(latents)
    batch = {
        "latents": _unsqueeze_sample_dim(latents),
        "multi_ref_latents": _unsqueeze_sample_dim(multi_reference_latents),
    }
    if conditions is not None:
        batch["conditions"] = _unsqueeze_sample_dim(conditions)
    if gt_visual_tokens is not None:
        batch["gt_visual_tokens"] = _unsqueeze_sample_dim(gt_visual_tokens)
    return batch


def _move_nested_to_device(data: dict[str, Any], *, device: torch.device, dtype: torch.dtype | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Tensor):
            if dtype is not None and value.is_floating_point():
                out[key] = value.to(device=device, dtype=dtype)
            else:
                out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def _shape(value: Tensor) -> list[int]:
    return list(value.shape)


def _condition_feature_key(conditions: dict[str, Tensor | None]) -> str:
    return "video_prompt_embeds" if "video_prompt_embeds" in conditions else "prompt_embeds"


def _load_config(config_path: Path) -> LtxTrainerConfig:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    config_data = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_TupleSafeLoader)
    cfg = LtxTrainerConfig(**config_data)
    if cfg.training_strategy.name != "multi_reference_video":
        raise ValueError(
            f"Stage 1 teacher inference requires training_strategy.name='multi_reference_video', "
            f"got {cfg.training_strategy.name!r}."
        )
    if cfg.training_strategy.cfg_dropout_enabled:
        console.print(
            "[yellow]Stage1 teacher inference does not use training-time CFG dropout; "
            "overriding training_strategy.cfg_dropout_enabled=false. "
            "Inference-time CFG is controlled by --guidance-scale.[/yellow]"
        )
        cfg.training_strategy.cfg_dropout_enabled = False
    return cfg


def _setup_lora(transformer: torch.nn.Module, cfg: LtxTrainerConfig) -> torch.nn.Module:
    if cfg.lora is None:
        raise ValueError("model.training_mode='lora' requires a lora config")
    lora_config = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        target_modules=cfg.lora.target_modules,
        lora_dropout=cfg.lora.dropout,
        init_lora_weights=True,
    )
    return get_peft_model(transformer, lora_config)


def _load_checkpoint_weights(
    *,
    checkpoint_path: Path,
    cfg: LtxTrainerConfig,
    transformer: torch.nn.Module,
    embeddings_processor: torch.nn.Module,
    strategy: MultiReferenceVideoStrategy,
) -> dict[str, bool]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    state_dict = load_file(checkpoint_path)
    visual_projection_loaded = any(key.startswith("training_strategy.visual_token_projection.") for key in state_dict)
    visual_full_encoder_loaded = any(
        key.startswith("training_strategy.visual_full_encoder.") for key in state_dict
    )
    projection_required = (
        cfg.training_strategy.visual_token_source_dim
        != cfg.training_strategy.visual_token_target_dim
    )
    if cfg.training_strategy.visual_context_mode == "full_tokens_3d_sa" and not visual_full_encoder_loaded:
        raise ValueError(
            "Stage 1 inference with visual_context_mode='full_tokens_3d_sa' requires checkpoint keys under "
            "training_strategy.visual_full_encoder.*. Do not use a legacy Q-former checkpoint."
        )
    if (
        cfg.training_strategy.visual_context_mode == "full_tokens_3d_sa"
        and projection_required
        and not visual_projection_loaded
    ):
        raise ValueError(
            "Stage 1 full-token inference requires checkpoint keys under "
            "training_strategy.visual_token_projection.*."
        )

    strategy.load_extra_checkpoint_state_dict(state_dict)

    processor_state = {
        key.removeprefix("embeddings_processor."): value
        for key, value in state_dict.items()
        if key.startswith("embeddings_processor.")
    }
    connector_checkpoint_loaded = bool(processor_state)
    if processor_state:
        embeddings_processor.load_state_dict(processor_state, strict=False)
        console.print("Loaded embeddings_processor auxiliary checkpoint state")
    else:
        console.print("[yellow]No embeddings_processor.* weights found in checkpoint; using base connector weights.[/yellow]")

    if cfg.model.training_mode == "full":
        transformer_state = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("training_strategy.")
            and not key.startswith("embeddings_processor.")
            and not key.startswith("text_encoder.")
        }
        if transformer_state:
            transformer.load_state_dict(transformer_state, strict=True)
        return {
            "connector_checkpoint_loaded": connector_checkpoint_loaded,
            "visual_token_projection_checkpoint_loaded": visual_projection_loaded,
            "visual_full_encoder_checkpoint_loaded": visual_full_encoder_loaded,
        }

    lora_state = {
        key.replace("diffusion_model.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("diffusion_model.")
    }
    if not lora_state:
        console.print("[yellow]No diffusion_model.* LoRA weights found; loaded auxiliary state only.[/yellow]")
        return {
            "connector_checkpoint_loaded": connector_checkpoint_loaded,
            "visual_token_projection_checkpoint_loaded": visual_projection_loaded,
            "visual_full_encoder_checkpoint_loaded": visual_full_encoder_loaded,
        }
    base_model = transformer.get_base_model()
    try:
        set_peft_model_state_dict(base_model, lora_state)
    except RuntimeError as exc:
        raise RuntimeError(
            "LoRA config does not match checkpoint. Use the original training config or matching rank/target_modules."
        ) from exc
    console.print(f"Loaded LoRA checkpoint: {checkpoint_path}")
    return {
        "connector_checkpoint_loaded": connector_checkpoint_loaded,
        "visual_token_projection_checkpoint_loaded": visual_projection_loaded,
        "visual_full_encoder_checkpoint_loaded": visual_full_encoder_loaded,
    }


def _copy_sample_media(
    *,
    sample: dict[str, Any],
    sample_dir: Path,
    manifest_root: Path,
    video_column: str,
    reference_column: str,
) -> tuple[Path, list[Path]]:
    gt_path = _resolve_path(str(sample[video_column]), manifest_root)
    if not gt_path.is_file():
        raise FileNotFoundError(f"GT video does not exist: {gt_path}")
    gt_dst = sample_dir / "gt.mp4"
    shutil.copy2(gt_path, gt_dst)

    ref_outputs: list[Path] = []
    for idx, ref_value in enumerate(_parse_reference_images(sample.get(reference_column))):
        ref_path = _resolve_path(ref_value, manifest_root)
        if not ref_path.is_file():
            raise FileNotFoundError(f"Reference image does not exist: {ref_path}")
        dst = sample_dir / f"ref_{idx}.jpg"
        try:
            image = open_image_as_srgb(ref_path)
            image.save(dst, quality=95)
        except Exception:
            dst = dst.with_suffix(ref_path.suffix or ".jpg")
            shutil.copy2(ref_path, dst)
        ref_outputs.append(dst)
    return gt_dst, ref_outputs


def _load_sample_precomputed(
    *,
    row: dict[str, Any],
    manifest_root: Path,
    precomputed_root: Path,
    video_column: str,
    condition_mode: str,
    text_condition_source: Literal["precomputed", "online"] = "online",
    text_conditions_dir: str = "conditions",
) -> tuple[Path, dict[str, dict[str, Any]]]:
    video_path = _resolve_path(str(row[video_column]), manifest_root)
    rel_path = _output_relative(video_path, manifest_root).with_suffix(".pt")
    files = {
        "latents": precomputed_root / "latents" / rel_path,
        "multi_reference_latents": precomputed_root / "multi_reference_latents" / rel_path,
    }
    if _uses_full_siglip_condition(condition_mode):
        files.update(
            {
                "conditions": precomputed_root / "vlm_conditions" / rel_path,
                "gt_visual_tokens": precomputed_root / "gt_siglip_tokens" / rel_path,
            }
        )
    elif text_condition_source == "precomputed":
        text_condition_path = precomputed_root / text_conditions_dir / rel_path
        if not text_condition_path.is_file():
            raise FileNotFoundError(
                "Precomputed text-only condition is missing. Run preprocessing or use "
                "--text-condition-source online. "
                f"Expected: {text_condition_path}"
            )
        files["conditions"] = text_condition_path
    return rel_path, {key: _load_pt_file(path) for key, path in files.items()}


def _prepare_condition_context(
    *,
    strategy: MultiReferenceVideoStrategy,
    embeddings_processor: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Tensor], dict[str, list[int]]]:
    batch["conditions"] = _move_nested_to_device(batch["conditions"], device=device, dtype=dtype)
    batch["gt_visual_tokens"] = _move_nested_to_device(batch["gt_visual_tokens"], device=device, dtype=dtype)

    conditions = batch["conditions"]
    key = _condition_feature_key(conditions)
    original_shape = _shape(conditions[key])
    raw_visual = batch["gt_visual_tokens"][strategy.config.visual_token_key]
    raw_visual_shape = _shape(raw_visual)
    raw_loaded_visual, _ = strategy._load_raw_condition_visual_tokens(
        batch["gt_visual_tokens"],
        device=conditions[key].device,
        dtype=conditions[key].dtype,
    )
    projected_visual_shape = _shape(strategy._project_visual_tokens(raw_loaded_visual, target_dim=conditions[key].shape[-1]))

    source_dim = strategy.config.visual_token_source_dim
    if source_dim is not None and raw_visual.shape[-1] != source_dim:
        raise ValueError(f"GT visual token raw dim {raw_visual.shape[-1]} does not match config source dim {source_dim}")

    conditions = strategy.prepare_conditions(batch, conditions)
    pre_connector_key = _condition_feature_key(conditions)
    pre_connector_shape = _shape(conditions[pre_connector_key])
    target_dim = strategy.config.visual_token_target_dim
    if target_dim is not None and conditions[pre_connector_key].shape[-1] != target_dim:
        raise ValueError(
            f"Final pre-connector condition dim {conditions[pre_connector_key].shape[-1]} does not match "
            f"config target dim {target_dim}"
        )
    if conditions[pre_connector_key].shape[1] < original_shape[1]:
        raise ValueError(
            "Stage1 visual branch expects prepare_conditions() not to shrink text/VLM sequence length before "
            f"the connector: original {original_shape}, final {pre_connector_shape}"
        )
    added = conditions[pre_connector_key].shape[1] - original_shape[1]
    if added > 0:
        mask = conditions["prompt_attention_mask"]
        if bool(mask[:, original_shape[1] :].any()):
            raise ValueError(
                "Pre-connector added tokens must be padding-only. "
                "SigLIP/visual tokens must not be appended before the text connector."
            )

    video_features = conditions[pre_connector_key]
    audio_features = conditions.get("audio_prompt_embeds")
    additive_mask = convert_to_additive_mask(conditions["prompt_attention_mask"], video_features.dtype)
    video_embeds, audio_embeds, attention_mask = embeddings_processor.create_embeddings(
        video_features,
        audio_features,
        additive_mask,
    )
    conditions["video_prompt_embeds"] = video_embeds
    if audio_embeds is not None:
        conditions["audio_prompt_embeds"] = audio_embeds
    conditions["prompt_attention_mask"] = attention_mask

    text_context_shape = _shape(video_embeds)
    batch["conditions"] = conditions
    conditions = strategy.postprocess_conditions_after_connector(batch, conditions)
    visual_context_shape = batch.get("_visual_context_shape")

    shapes = {
        "original_condition_shape": original_shape,
        "original_feature_shape": original_shape,
        "raw_gt_visual_shape": raw_visual_shape,
        "projected_visual_shape": projected_visual_shape,
        "pre_connector_condition_shape": pre_connector_shape,
        "post_connector_text_condition_shape": text_context_shape,
        "visual_context_shape": visual_context_shape,
        "visual_context_token_count": batch.get("_visual_context_token_count"),
        "post_connector_condition_shape": _shape(conditions["video_prompt_embeds"]),
        "transformer_condition_shape": _shape(conditions["video_prompt_embeds"]),
    }
    return conditions, shapes


def _prepare_precomputed_text_condition(
    *,
    strategy: MultiReferenceVideoStrategy,
    embeddings_processor: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor | None]:
    if "conditions" not in batch:
        raise ValueError("Precomputed text-only condition is missing from the sample batch")
    batch["conditions"] = _move_nested_to_device(batch["conditions"], device=device, dtype=dtype)
    conditions = strategy.prepare_conditions(batch, batch["conditions"])
    feature_key = _condition_feature_key(conditions)
    video_features = conditions[feature_key]
    audio_features = conditions.get("audio_prompt_embeds")
    additive_mask = convert_to_additive_mask(conditions["prompt_attention_mask"], video_features.dtype)
    video_embeds, audio_embeds, attention_mask = embeddings_processor.create_embeddings(
        video_features,
        audio_features,
        additive_mask,
    )
    prepared: dict[str, Tensor | None] = {
        "video_prompt_embeds": video_embeds,
        "audio_prompt_embeds": audio_embeds,
        "prompt_attention_mask": attention_mask,
    }
    batch["conditions"] = prepared
    return prepared


def _encode_text_prompt_condition(
    *,
    cfg: LtxTrainerConfig,
    embeddings_processor: torch.nn.Module,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor | None]:
    if cfg.model.text_encoder_path is None:
        raise ValueError("text_only_no_siglip requires model.text_encoder_path to encode the text prompt.")
    text_encoder = load_text_encoder(
        gemma_model_path=cfg.model.text_encoder_path,
        device=device,
        dtype=dtype,
        load_in_8bit=cfg.acceleration.load_text_encoder_in_8bit,
    )
    text_encoder.eval()
    with torch.inference_mode():
        hidden_states, attention_mask = text_encoder.encode([prompt])[0]
        out = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
    conditions: dict[str, Tensor | None] = {
        "video_prompt_embeds": out.video_encoding.to(device=device, dtype=dtype),
        "audio_prompt_embeds": (
            out.audio_encoding.to(device=device, dtype=dtype) if out.audio_encoding is not None else None
        ),
        "prompt_attention_mask": None,
    }
    del text_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return conditions


def _encode_negative_prompt_condition(
    *,
    cfg: LtxTrainerConfig,
    embeddings_processor: torch.nn.Module,
    negative_prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor | None]:
    if cfg.model.text_encoder_path is None:
        raise ValueError("--guidance-scale > 1.0 requires model.text_encoder_path to encode the negative prompt.")
    return _encode_text_prompt_condition(
        cfg=cfg,
        embeddings_processor=embeddings_processor,
        prompt=negative_prompt,
        device=device,
        dtype=dtype,
    )


def _denoise_stage1(
    *,
    transformer: torch.nn.Module,
    strategy: MultiReferenceVideoStrategy,
    batch: dict[str, Any],
    positive_conditions: dict[str, Tensor],
    negative_conditions: dict[str, Tensor | None] | None,
    no_ref_conditions: dict[str, Tensor | None] | None = None,
    guidance_scale: float,
    cfg_drop_ref_latents_in_negative: bool,
    ref_guidance_scale: float,
    siglip_guidance_scale: float,
    guidance_rescale: float,
    stg_scale: float,
    stg_blocks: list[int] | None,
    num_inference_steps: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be >= 1")
    if ref_guidance_scale != 0.0 and cfg_drop_ref_latents_in_negative:
        raise ValueError(
            "Reference-latent guidance requires the CFG negative branch to keep reference latents. "
            "Use --cfg-keep-ref-latents-in-negative."
        )

    latents_data = _move_nested_to_device(batch["latents"], device=device, dtype=dtype)
    ref_data = _move_nested_to_device(batch["multi_ref_latents"], device=device, dtype=dtype)
    target_shape_tensor = latents_data["latents"]
    batch_size = target_shape_tensor.shape[0]
    if batch_size != 1:
        raise ValueError(f"Stage1 overfit inference expects batch size 1, got {batch_size}")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    target_latents = torch.randn(target_shape_tensor.shape, generator=generator, device=device, dtype=dtype)
    target_tokens = strategy._video_patchifier.patchify(target_latents)
    target_seq_len = target_tokens.shape[1]

    ref_latents = strategy._normalize_reference_latents(ref_data["latents"])
    ref_valid_mask = strategy._get_reference_valid_mask(ref_data, ref_latents)
    if strategy.config.max_ref_images_per_sample is not None:
        max_refs = strategy.config.max_ref_images_per_sample
        ref_latents = ref_latents[:, :max_refs]
        ref_valid_mask = ref_valid_mask[:, :max_refs]
    _, num_refs, _channels, ref_frames, ref_height, ref_width = ref_latents.shape

    num_frames = int(latents_data["num_frames"].flatten()[0].item())
    height = int(latents_data["height"].flatten()[0].item())
    width = int(latents_data["width"].flatten()[0].item())
    fps_value = strategy._first_scalar(latents_data.get("fps"), default=24.0)

    ref_tokens = strategy._video_patchifier.patchify(ref_latents.reshape(num_refs, *ref_latents.shape[2:]))
    ref_tokens = ref_tokens.reshape(1, num_refs, ref_tokens.shape[1], ref_tokens.shape[2])
    target_positions = strategy._get_video_positions(
        num_frames=num_frames,
        height=height,
        width=width,
        batch_size=1,
        fps=fps_value,
        device=device,
    )
    ref_positions = strategy._get_video_positions(
        num_frames=ref_frames,
        height=ref_height,
        width=ref_width,
        batch_size=num_refs,
        fps=strategy._first_scalar(ref_data.get("fps"), default=1.0),
        device=device,
    )
    ref_positions = ref_positions.reshape(1, num_refs, *ref_positions.shape[1:])
    ref_positions = strategy._scale_reference_positions(ref_positions, height, width, ref_height, ref_width)
    no_ref_valid_mask = torch.zeros_like(ref_valid_mask)

    sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(device=device).float()
    stepper = EulerDiffusionStep()
    pos_context_key = _condition_feature_key(positive_conditions)
    pos_context = positive_conditions[pos_context_key]
    pos_context_mask = positive_conditions.get("prompt_attention_mask")
    if no_ref_conditions is None:
        no_ref_context = pos_context
        no_ref_context_mask = pos_context_mask
    else:
        no_ref_context_key = _condition_feature_key(no_ref_conditions)
        no_ref_context = no_ref_conditions[no_ref_context_key]
        no_ref_context_mask = no_ref_conditions.get("prompt_attention_mask")
    if siglip_guidance_scale != 0.0:
        visual_token_count = int(batch.get("_visual_context_token_count", 0) or 0)
        (
            null_text_siglip_context,
            null_text_no_siglip_context,
            siglip_isolated_context_mask,
            siglip_null_context_mask,
        ) = _build_isolated_siglip_contexts(
            pos_context=pos_context,
            pos_context_mask=pos_context_mask,
            visual_token_count=visual_token_count,
        )
    else:
        null_text_siglip_context = None
        null_text_no_siglip_context = None
        siglip_isolated_context_mask = None
        siglip_null_context_mask = None
    if _cfg_enabled(guidance_scale):
        if negative_conditions is None:
            raise ValueError("guidance_scale > 1.0 requires negative_conditions")
        neg_context_key = _condition_feature_key(negative_conditions)
        neg_context = negative_conditions[neg_context_key]
        neg_context_mask = negative_conditions.get("prompt_attention_mask")
    else:
        neg_context = None
        neg_context_mask = None
    stg_perturbation_config = _build_stg_perturbation_config(stg_blocks) if _stg_enabled(stg_scale) else None

    transformer.eval()
    with torch.inference_mode():
        for step_idx, sigma in enumerate(track(sigmas[:-1], description="Stage 1 denoising")):
            sigma_batch = sigma.unsqueeze(0)
            target_timesteps = torch.full(
                (1, target_seq_len),
                float(sigma.item()),
                dtype=torch.float32,
                device=device,
            )
            target_loss_mask = torch.ones((1, target_seq_len), dtype=torch.bool, device=device)
            packed = build_multiref_sequence(
                ref_tokens=ref_tokens,
                ref_positions=ref_positions,
                ref_valid_mask=ref_valid_mask,
                target_tokens=target_tokens,
                target_positions=target_positions,
                target_timesteps=target_timesteps,
                target_loss_mask=target_loss_mask,
                reference_time_stride=strategy.config.reference_time_stride,
            )
            video_pos = Modality(
                enabled=True,
                latent=packed.latents,
                sigma=sigma_batch,
                timesteps=packed.timesteps,
                positions=packed.positions,
                context=pos_context,
                context_mask=pos_context_mask,
                attention_mask=packed.attention_mask,
            )
            velocity_pos, _ = transformer(video=video_pos, audio=None, perturbations=None)
            if velocity_pos is None:
                raise RuntimeError("Transformer returned no video velocity during Stage 1 inference")
            denoised_pos = _velocity_to_denoised(video_pos.latent, velocity_pos, packed.timesteps)
            denoised_neg = None
            denoised_no_ref = None
            denoised_siglip_isolated = None
            denoised_siglip_null = None
            denoised_stg = None

            if _cfg_enabled(guidance_scale):
                negative_ref_valid_mask = _negative_ref_valid_mask(
                    ref_valid_mask,
                    drop_ref_latents=cfg_drop_ref_latents_in_negative,
                )
                if cfg_drop_ref_latents_in_negative:
                    packed_neg = build_multiref_sequence(
                        ref_tokens=ref_tokens,
                        ref_positions=ref_positions,
                        ref_valid_mask=negative_ref_valid_mask,
                        target_tokens=target_tokens,
                        target_positions=target_positions,
                        target_timesteps=target_timesteps,
                        target_loss_mask=target_loss_mask,
                        reference_time_stride=strategy.config.reference_time_stride,
                    )
                else:
                    packed_neg = packed
                video_neg = Modality(
                    enabled=True,
                    latent=packed_neg.latents,
                    sigma=sigma_batch,
                    timesteps=packed_neg.timesteps,
                    positions=packed_neg.positions,
                    context=neg_context,
                    context_mask=neg_context_mask,
                    attention_mask=packed_neg.attention_mask,
                )
                velocity_neg, _ = transformer(video=video_neg, audio=None, perturbations=None)
                if velocity_neg is None:
                    raise RuntimeError("Transformer returned no negative-branch video velocity during Stage 1 CFG")
                denoised_neg = _velocity_to_denoised(video_neg.latent, velocity_neg, packed_neg.timesteps)

            if ref_guidance_scale != 0.0:
                packed_no_ref = build_multiref_sequence(
                    ref_tokens=ref_tokens,
                    ref_positions=ref_positions,
                    ref_valid_mask=no_ref_valid_mask,
                    target_tokens=target_tokens,
                    target_positions=target_positions,
                    target_timesteps=target_timesteps,
                    target_loss_mask=target_loss_mask,
                    reference_time_stride=strategy.config.reference_time_stride,
                )
                video_no_ref = Modality(
                    enabled=True,
                    latent=packed_no_ref.latents,
                    sigma=sigma_batch,
                    timesteps=packed_no_ref.timesteps,
                    positions=packed_no_ref.positions,
                    context=no_ref_context,
                    context_mask=no_ref_context_mask,
                    attention_mask=packed_no_ref.attention_mask,
                )
                velocity_no_ref, _ = transformer(video=video_no_ref, audio=None, perturbations=None)
                if velocity_no_ref is None:
                    raise RuntimeError("Transformer returned no full-without-reference-latents velocity")
                denoised_no_ref = _velocity_to_denoised(
                    video_no_ref.latent,
                    velocity_no_ref,
                    packed_no_ref.timesteps,
                )

            if siglip_guidance_scale != 0.0:
                video_siglip_isolated = Modality(
                    enabled=True,
                    latent=packed.latents,
                    sigma=sigma_batch,
                    timesteps=packed.timesteps,
                    positions=packed.positions,
                    context=null_text_siglip_context,
                    context_mask=siglip_isolated_context_mask,
                    attention_mask=packed.attention_mask,
                )
                velocity_siglip_isolated, _ = transformer(
                    video=video_siglip_isolated,
                    audio=None,
                    perturbations=None,
                )
                if velocity_siglip_isolated is None:
                    raise RuntimeError("Transformer returned no isolated-SigLIP video velocity")
                denoised_siglip_isolated = _velocity_to_denoised(
                    video_siglip_isolated.latent,
                    velocity_siglip_isolated,
                    packed.timesteps,
                )

                video_siglip_null = Modality(
                    enabled=True,
                    latent=packed.latents,
                    sigma=sigma_batch,
                    timesteps=packed.timesteps,
                    positions=packed.positions,
                    context=null_text_no_siglip_context,
                    context_mask=siglip_null_context_mask,
                    attention_mask=packed.attention_mask,
                )
                velocity_siglip_null, _ = transformer(
                    video=video_siglip_null,
                    audio=None,
                    perturbations=None,
                )
                if velocity_siglip_null is None:
                    raise RuntimeError("Transformer returned no null-SigLIP video velocity")
                denoised_siglip_null = _velocity_to_denoised(
                    video_siglip_null.latent,
                    velocity_siglip_null,
                    packed.timesteps,
                )

            if _stg_enabled(stg_scale):
                velocity_stg, _ = transformer(video=video_pos, audio=None, perturbations=stg_perturbation_config)
                if velocity_stg is None:
                    raise RuntimeError("Transformer returned no STG-branch video velocity during Stage 1 STG")
                denoised_stg = _velocity_to_denoised(video_pos.latent, velocity_stg, packed.timesteps)

            denoised_video = _combine_multidirectional_denoised(
                denoised_pos=denoised_pos,
                denoised_neg=denoised_neg,
                denoised_no_ref=denoised_no_ref,
                denoised_siglip_isolated=denoised_siglip_isolated,
                denoised_siglip_null=denoised_siglip_null,
                denoised_stg=denoised_stg,
                guidance_scale=guidance_scale,
                ref_guidance_scale=ref_guidance_scale,
                siglip_guidance_scale=siglip_guidance_scale,
                stg_scale=stg_scale,
                guidance_rescale=guidance_rescale,
                target_seq_len=target_seq_len,
            )
            next_packed = stepper.step(packed.latents, denoised_video, sigmas, step_idx)
            target_tokens = next_packed[:, -target_seq_len:, :]

    return strategy._video_patchifier.unpatchify(
        target_tokens,
        output_shape=VideoLatentShape.from_torch_shape(target_shape_tensor.shape),
    )


def _decode_video_latents(
    *,
    vae_decoder: torch.nn.Module,
    latents: Tensor,
    device: torch.device,
    decode_tile: bool,
) -> Tensor:
    vae_decoder.to(device)
    latents = latents.to(device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        if decode_tile:
            chunks = list(vae_decoder.tiled_decode(latents, tiling_config=_DEFAULT_TILING))
            video = torch.cat(chunks, dim=2)
        else:
            video = vae_decoder(latents)
    video = ((video + 1.0) / 2.0).clamp(0.0, 1.0)
    return rearrange(video, "1 c f h w -> f c h w").float().cpu()


def _load_transformer_processor_and_strategy(
    *,
    cfg: LtxTrainerConfig,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, torch.nn.Module, MultiReferenceVideoStrategy, dict[str, bool]]:
    transformer = load_transformer(cfg.model.model_path, device=device, dtype=dtype)
    embeddings_processor = load_embeddings_processor(cfg.model.model_path, device=device, dtype=dtype)
    embeddings_processor.requires_grad_(False)

    strategy = get_training_strategy(cfg.training_strategy)
    if not isinstance(strategy, MultiReferenceVideoStrategy):
        raise TypeError(f"Expected MultiReferenceVideoStrategy, got {type(strategy).__name__}")
    strategy.attach_models(transformer=transformer, embeddings_processor=embeddings_processor, text_encoder=None)

    projection = strategy.get_trainable_modules().get("visual_token_projection")
    if projection is None:
        raise ValueError(
            "visual_token_projection is not enabled. Set training_strategy.visual_token_source_dim=3840 and "
            "visual_token_target_dim=4096 in the config."
        )
    console.print(
        "visual_token_projection enabled: "
        f"source_dim={cfg.training_strategy.visual_token_source_dim}, "
        f"target_dim={cfg.training_strategy.visual_token_target_dim}"
    )

    if cfg.model.training_mode == "lora":
        transformer = _setup_lora(transformer, cfg)
    checkpoint_flags = _load_checkpoint_weights(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        transformer=transformer,
        embeddings_processor=embeddings_processor,
        strategy=strategy,
    )
    transformer.requires_grad_(False).eval()
    embeddings_processor.requires_grad_(False).eval()
    for module in strategy.get_trainable_modules().values():
        module.requires_grad_(False).eval()

    return transformer, embeddings_processor, strategy, checkpoint_flags


def _load_models_and_strategy(
    *,
    cfg: LtxTrainerConfig,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, MultiReferenceVideoStrategy, dict[str, bool]]:
    transformer, embeddings_processor, strategy, checkpoint_flags = _load_transformer_processor_and_strategy(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=dtype,
    )

    vae_decoder = load_video_vae_decoder(cfg.model.model_path, device=device, dtype=dtype)
    vae_decoder.requires_grad_(False).eval()
    return transformer, embeddings_processor, vae_decoder, strategy, checkpoint_flags


@dataclass
class Stage1InferenceRuntime:
    cfg: LtxTrainerConfig
    transformer: torch.nn.Module
    embeddings_processor: torch.nn.Module
    vae_decoder: torch.nn.Module
    strategy: MultiReferenceVideoStrategy
    checkpoint_flags: dict[str, bool]
    negative_conditions: dict[str, Tensor | None] | None
    device: torch.device
    dtype: torch.dtype
    config_path: Path
    checkpoint_path: Path
    negative_prompt: str | None


def _load_inference_runtime(
    *,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    guidance_scale: float,
    negative_prompt: str | None,
    keep_feature_extractor: bool = False,
) -> Stage1InferenceRuntime:
    cfg = _load_config(config_path)
    transformer, embeddings_processor, strategy, checkpoint_flags = _load_transformer_processor_and_strategy(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=dtype,
    )
    effective_negative_prompt = negative_prompt or cfg.validation.negative_prompt
    if _cfg_enabled(guidance_scale):
        negative_conditions = _encode_negative_prompt_condition(
            cfg=cfg,
            embeddings_processor=embeddings_processor,
            negative_prompt=effective_negative_prompt,
            device=device,
            dtype=dtype,
        )
        console.print(f"negative condition shape: {_condition_shape(negative_conditions)}")
    else:
        negative_conditions = None
        effective_negative_prompt = None

    if not keep_feature_extractor and hasattr(embeddings_processor, "feature_extractor"):
        embeddings_processor.feature_extractor = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    vae_decoder = load_video_vae_decoder(cfg.model.model_path, device=device, dtype=dtype)
    vae_decoder.requires_grad_(False).eval()

    return Stage1InferenceRuntime(
        cfg=cfg,
        transformer=transformer,
        embeddings_processor=embeddings_processor,
        vae_decoder=vae_decoder,
        strategy=strategy,
        checkpoint_flags=checkpoint_flags,
        negative_conditions=negative_conditions,
        device=device,
        dtype=dtype,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        negative_prompt=effective_negative_prompt,
    )


def _expected_generated_path(output_dir: Path, sample_index: int, condition_mode: str) -> Path:
    return output_dir / f"sample_{sample_index}" / f"generated_{condition_mode}.mp4"


def _temporary_output_path(final_path: Path) -> Path:
    return final_path.with_name(f".{final_path.stem}.tmp{final_path.suffix}")


def _metadata_path_for_video(video_path: Path) -> Path:
    return video_path.parent / "metadata.json"


def _temporary_metadata_path(video_path: Path) -> Path:
    return video_path.parent / ".metadata.tmp.json"


def _sample_is_complete(
    *,
    output_dir: Path,
    sample_index: int,
    condition_mode: str,
) -> bool:
    video_path = _expected_generated_path(output_dir, sample_index, condition_mode)
    metadata_path = _metadata_path_for_video(video_path)
    if not video_path.is_file() or video_path.stat().st_size <= 0 or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(metadata, dict):
        return False
    return (
        metadata.get("sample_index") == sample_index
        and metadata.get("condition_mode") == condition_mode
        and metadata.get("generated") == str(video_path)
    )


def _cleanup_stale_sample_outputs(
    *,
    output_dir: Path,
    sample_index: int,
    condition_mode: str,
) -> None:
    final_video = _expected_generated_path(output_dir, sample_index, condition_mode)
    _temporary_output_path(final_video).unlink(missing_ok=True)
    _temporary_metadata_path(final_video).unlink(missing_ok=True)


def _save_sample_outputs_atomically(
    *,
    video_tensor: Tensor,
    final_video: Path,
    metadata: dict[str, Any],
    fps: float,
) -> None:
    final_video.parent.mkdir(parents=True, exist_ok=True)
    temp_video = _temporary_output_path(final_video)
    final_metadata = _metadata_path_for_video(final_video)
    temp_metadata = _temporary_metadata_path(final_video)
    temp_video.unlink(missing_ok=True)
    temp_metadata.unlink(missing_ok=True)

    save_video(video_tensor=video_tensor, output_path=temp_video, fps=fps, video_format="FCHW")
    if not temp_video.is_file() or temp_video.stat().st_size <= 0:
        raise RuntimeError(f"Temporary generated video is missing or empty: {temp_video}")
    temp_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp_video.replace(final_video)
    temp_metadata.replace(final_metadata)


def _select_sample_indices(
    *,
    row_count: int,
    start_index: int,
    end_index: int | None,
    shard_index: int,
    num_shards: int,
) -> list[int]:
    upper_bound = row_count if end_index is None else min(end_index, row_count)
    return [index for index in range(start_index, upper_bound) if index % num_shards == shard_index]


def _text_only_condition_shapes(conditions: dict[str, Tensor | None]) -> dict[str, Any]:
    text_condition_shape = _condition_shape(conditions)
    return {
        "original_condition_shape": None,
        "original_feature_shape": None,
        "raw_gt_visual_shape": None,
        "projected_visual_shape": None,
        "pre_connector_condition_shape": None,
        "post_connector_condition_shape": text_condition_shape,
        "transformer_condition_shape": text_condition_shape,
        "text_only_prompt_condition_shape": text_condition_shape,
        "post_connector_text_condition_shape": text_condition_shape,
        "visual_context_shape": None,
        "visual_context_token_count": 0,
    }


def _run_one_sample(  # noqa: PLR0913, PLR0915
    *,
    runtime: Stage1InferenceRuntime,
    rows: list[dict[str, Any]],
    sample_index: int,
    manifest_root: Path,
    precomputed_root: Path,
    output_dir: Path,
    video_column: str,
    caption_column: str,
    reference_column: str,
    condition_mode: str,
    text_condition_source: Literal["precomputed", "online"],
    text_conditions_dir: str,
    copy_media: bool,
    fps: float | None,
    decode_tile: bool,
    guidance_scale: float,
    cfg_negative_mode: str,
    cfg_drop_ref_latents_in_negative: bool,
    ref_guidance_scale: float,
    siglip_guidance_scale: float,
    guidance_rescale: float,
    stg_scale: float,
    stg_blocks: list[int] | None,
    stg_mode: str,
    num_inference_steps: int,
    seed: int,
) -> Path:
    sample = rows[sample_index]
    if video_column not in sample:
        raise ValueError(f"Selected sample has no video column {video_column!r}")
    sample_dir = output_dir / f"sample_{sample_index}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    if copy_media:
        gt_copy, ref_copies = _copy_sample_media(
            sample=sample,
            sample_dir=sample_dir,
            manifest_root=manifest_root,
            video_column=video_column,
            reference_column=reference_column,
        )
    else:
        gt_copy = None
        ref_copies = []

    rel_path, precomputed = _load_sample_precomputed(
        row=sample,
        manifest_root=manifest_root,
        precomputed_root=precomputed_root,
        video_column=video_column,
        condition_mode=condition_mode,
        text_condition_source=text_condition_source,
        text_conditions_dir=text_conditions_dir,
    )
    batch = _build_single_sample_batch(
        latents=precomputed["latents"],
        multi_reference_latents=precomputed["multi_reference_latents"],
        conditions=precomputed.get("conditions"),
        gt_visual_tokens=precomputed.get("gt_visual_tokens"),
    )

    positive_prompt = str(sample.get(caption_column, ""))
    if condition_mode == "full_siglip":
        conditions, condition_shapes = _prepare_condition_context(
            strategy=runtime.strategy,
            embeddings_processor=runtime.embeddings_processor,
            batch=batch,
            device=runtime.device,
            dtype=runtime.dtype,
        )
    elif text_condition_source == "precomputed":
        conditions = _prepare_precomputed_text_condition(
            strategy=runtime.strategy,
            embeddings_processor=runtime.embeddings_processor,
            batch=batch,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        condition_shapes = _text_only_condition_shapes(conditions)
    else:
        if not positive_prompt.strip():
            raise ValueError(f"Sample has empty prompt in caption column {caption_column!r}")
        conditions = _encode_text_prompt_condition(
            cfg=runtime.cfg,
            embeddings_processor=runtime.embeddings_processor,
            prompt=positive_prompt,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        condition_shapes = _text_only_condition_shapes(conditions)

    console.print(
        f"sample={sample_index} mode={condition_mode} "
        f"condition_shape={condition_shapes.get('post_connector_condition_shape')}"
    )
    generated_latents = _denoise_stage1(
        transformer=runtime.transformer,
        strategy=runtime.strategy,
        batch=batch,
        positive_conditions=conditions,
        negative_conditions=runtime.negative_conditions,
        guidance_scale=guidance_scale,
        cfg_drop_ref_latents_in_negative=cfg_drop_ref_latents_in_negative,
        ref_guidance_scale=ref_guidance_scale,
        siglip_guidance_scale=siglip_guidance_scale,
        guidance_rescale=guidance_rescale,
        stg_scale=stg_scale,
        stg_blocks=stg_blocks,
        num_inference_steps=num_inference_steps,
        seed=seed,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    decoded = _decode_video_latents(
        vae_decoder=runtime.vae_decoder,
        latents=generated_latents,
        device=runtime.device,
        decode_tile=decode_tile,
    )

    latent_fps = runtime.strategy._first_scalar(batch["latents"].get("fps"), default=24.0)
    output_fps = float(fps) if fps is not None else float(latent_fps)
    generated_path = _expected_generated_path(output_dir, sample_index, condition_mode)

    reference_paths = [
        str(_resolve_path(value, manifest_root))
        for value in _parse_reference_images(sample.get(reference_column))
    ]
    metadata = {
        "sample_index": sample_index,
        "checkpoint": str(runtime.checkpoint_path),
        "config": str(runtime.config_path),
        "prompt": positive_prompt,
        "gt_path": str(_resolve_path(str(sample[video_column]), manifest_root)),
        "gt_copy": str(gt_copy) if gt_copy is not None else None,
        "reference_paths": reference_paths,
        "reference_copies": [str(path) for path in ref_copies],
        "precomputed_relative_path": str(rel_path),
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "cfg_enabled": _cfg_enabled(guidance_scale),
        "negative_prompt": runtime.negative_prompt if _cfg_enabled(guidance_scale) else None,
        "cfg_negative_mode": cfg_negative_mode,
        "cfg_drop_ref_latents_in_negative": cfg_drop_ref_latents_in_negative,
        "negative_condition_shape": _condition_shape(runtime.negative_conditions),
        "ref_guidance_scale": ref_guidance_scale,
        "ref_guidance_enabled": ref_guidance_scale != 0.0,
        "siglip_guidance_scale": siglip_guidance_scale,
        "siglip_guidance_enabled": siglip_guidance_scale != 0.0,
        "siglip_guidance_formula": "null_text_with_siglip_and_refs - null_text_without_siglip_with_refs",
        "guidance_rescale": guidance_rescale,
        "guidance_rescale_enabled": guidance_rescale != 0.0,
        "stg_scale": stg_scale,
        "stg_enabled": _stg_enabled(stg_scale),
        "stg_blocks": stg_blocks,
        "stg_mode": stg_mode,
        "guidance_formula": (
            "x_negative_with_ref + cfg * (x_full - x_negative_with_ref) "
            "+ ref * (x_full - x_full_without_ref_latents) "
            "+ siglip * (x_null_text_with_siglip - x_null_text_without_siglip) "
            "+ stg * (x_full - x_stg)"
        ),
        "condition_mode": condition_mode,
        "condition_mode_detail": _condition_mode_detail(
            condition_mode,
            guidance_scale=guidance_scale,
            stg_scale=stg_scale,
        ),
        "uses_gt_siglip_visual_tokens": condition_mode == "full_siglip",
        "uses_vlm_reference_image_context": condition_mode == "full_siglip",
        "uses_text_only_prompt_condition": condition_mode == "text_only_no_siglip",
        "keeps_reference_latent_condition": True,
        "text_only_prompt_condition_shape": condition_shapes.get("text_only_prompt_condition_shape"),
        "visual_branch_enabled": (
            bool(getattr(runtime.strategy.config, "visual_branch_enabled", False))
            and condition_mode == "full_siglip"
        ),
        "visual_context_mode": runtime.strategy.config.visual_context_mode,
        "visual_context_shape": condition_shapes.get("visual_context_shape"),
        "visual_context_token_count": condition_shapes.get("visual_context_token_count"),
        "visual_full_sa_num_heads": runtime.strategy.config.visual_full_sa_num_heads,
        "visual_full_sa_depth": runtime.strategy.config.visual_full_sa_depth,
        "visual_gate": (
            float(runtime.strategy._visual_gate.value.detach().float().cpu().item())
            if getattr(runtime.strategy, "_visual_gate", None) is not None
            else None
        ),
        "connector_checkpoint_loaded": runtime.checkpoint_flags["connector_checkpoint_loaded"],
        "visual_token_projection_checkpoint_loaded": runtime.checkpoint_flags[
            "visual_token_projection_checkpoint_loaded"
        ],
        "visual_full_encoder_checkpoint_loaded": runtime.checkpoint_flags[
            "visual_full_encoder_checkpoint_loaded"
        ],
        "original_feature_shape": condition_shapes["original_feature_shape"],
        "raw_gt_visual_shape": condition_shapes["raw_gt_visual_shape"],
        "projected_visual_shape": condition_shapes["projected_visual_shape"],
        "pre_connector_condition_shape": condition_shapes["pre_connector_condition_shape"],
        "post_connector_condition_shape": condition_shapes["post_connector_condition_shape"],
        "final_condition_shape": condition_shapes["post_connector_condition_shape"],
        "reference_latent_shape": _shape(batch["multi_ref_latents"]["latents"]),
        "target_latent_shape": _shape(batch["latents"]["latents"]),
        "fps": output_fps,
        "decode_tile": decode_tile,
        "generated": str(generated_path),
    }
    _save_sample_outputs_atomically(
        video_tensor=decoded,
        final_video=generated_path,
        metadata=metadata,
        fps=output_fps,
    )
    console.print(f"[green]Saved generated video:[/green] {generated_path}")

    del batch, conditions, generated_latents, decoded, precomputed
    return generated_path


def _run_selected_samples(
    *,
    selected_indices: list[int],
    output_dir: Path,
    condition_mode: str,
    checkpoint_path: Path,
    shard_index: int,
    num_shards: int,
    skip_existing: bool,
    continue_on_error: bool,
    gc_interval: int,
    device: torch.device,
    run_sample: Callable[[int], Path],
    write_summary: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / f"failures_shard_{shard_index}.jsonl"
    failures_path.unlink(missing_ok=True)
    start_time = time.perf_counter()
    num_success = 0
    num_skipped = 0
    num_failed = 0

    for position, sample_index in enumerate(selected_indices, start=1):
        expected_video = _expected_generated_path(output_dir, sample_index, condition_mode)
        _cleanup_stale_sample_outputs(
            output_dir=output_dir,
            sample_index=sample_index,
            condition_mode=condition_mode,
        )
        if skip_existing and _sample_is_complete(
            output_dir=output_dir,
            sample_index=sample_index,
            condition_mode=condition_mode,
        ):
            num_skipped += 1
            console.print(f"[cyan]Skipping existing sample {sample_index}:[/cyan] {expected_video}")
            if gc_interval > 0 and position % gc_interval == 0:
                gc.collect()
            continue

        sample_start = time.perf_counter()
        recover_cuda_oom = False
        try:
            generated_path = run_sample(sample_index)
            num_success += 1
            elapsed = time.perf_counter() - sample_start
            allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            console.print(
                f"sample={sample_index} elapsed={elapsed:.2f}s output={generated_path} "
                f"cuda_allocated={allocated} cuda_peak={peak}"
            )
        except Exception as exc:
            num_failed += 1
            failure = {
                "sample_index": sample_index,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            with failures_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(failure, ensure_ascii=False) + "\n")
            console.print(f"[red]Sample {sample_index} failed:[/red] {type(exc).__name__}: {exc}")
            recover_cuda_oom = device.type == "cuda" and (
                isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            )
            if not continue_on_error:
                raise
        finally:
            if recover_cuda_oom:
                gc.collect()
                torch.cuda.empty_cache()
            elif gc_interval > 0 and position % gc_interval == 0:
                gc.collect()

    summary = {
        "condition_mode": condition_mode,
        "checkpoint": str(checkpoint_path),
        "num_selected": len(selected_indices),
        "num_success": num_success,
        "num_skipped": num_skipped,
        "num_failed": num_failed,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    if write_summary:
        summary_path = output_dir / f"batch_summary_shard_{shard_index}.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


@app.command()
def main(  # noqa: PLR0913, PLR0915
    config: str = typer.Option(..., help="Stage 1 overfit/training YAML config."),
    checkpoint: str = typer.Option(..., help="Stage 1 checkpoint safetensors file."),
    manifest: str = typer.Option(..., help="Overfit manifest JSON/JSONL/CSV."),
    precomputed_root: str = typer.Option(..., help="Root containing latents/, vlm_conditions/, etc."),
    output_dir: str = typer.Option(..., help="Directory for sample_<index>/ outputs."),
    sample_index: int | None = typer.Option(
        None,
        "--sample-index",
        help="Single sample index. Defaults to 0 when no batch range is requested.",
    ),
    all_samples: bool = typer.Option(
        False,
        "--all-samples/--no-all-samples",
        help="Process all samples in the selected global-index range.",
    ),
    start_index: int = typer.Option(0, "--start-index", help="Inclusive batch range start."),
    end_index: int | None = typer.Option(None, "--end-index", help="Exclusive batch range end."),
    shard_index: int = typer.Option(0, "--shard-index", help="Global modulo shard index."),
    num_shards: int = typer.Option(1, "--num-shards", help="Number of independent persistent workers."),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip non-empty generated videos that already exist.",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help="Continue processing the shard after a sample failure.",
    ),
    copy_media: bool | None = typer.Option(
        None,
        "--copy-media/--no-copy-media",
        help="Copy GT/reference media. Defaults to true for single sample and false for batch mode.",
    ),
    text_condition_source: Literal["precomputed", "online"] | None = typer.Option(
        None,
        "--text-condition-source",
        help="Text-only source. Defaults to online for single sample and precomputed for batch mode.",
    ),
    text_conditions_dir: str = typer.Option(
        "conditions",
        "--text-conditions-dir",
        help="Precomputed text-only condition directory.",
    ),
    gc_interval: int = typer.Option(20, "--gc-interval", help="Run Python GC after every N attempted samples."),
    device: str = typer.Option("cuda", help="Torch device, e.g. cuda or cuda:0."),
    seed: int = typer.Option(42, help="Random seed for target latent initialization."),
    num_inference_steps: int = typer.Option(50, help="Number of denoising steps."),
    video_column: str = typer.Option("video", help="Manifest video column."),
    caption_column: str = typer.Option("caption", help="Manifest caption column."),
    reference_column: str = typer.Option("reference_images", help="Manifest reference image column."),
    root_dir: str | None = typer.Option(None, help="Root for relative manifest media paths. Defaults to manifest parent."),
    fps: float | None = typer.Option(None, help="Override output fps. Defaults to latent metadata fps."),
    decode_tile: bool = typer.Option(True, "--decode-tile/--no-decode-tile", help="Use tiled VAE decode."),
    guidance_scale: float = typer.Option(
        1.0,
        "--guidance-scale",
        help="CFG guidance scale. 1.0 disables CFG. Recommended 1.2-2.0 for Stage1 checkpoints trained without CFG.",
    ),
    ref_guidance_scale: float = typer.Option(
        0.0,
        "--ref-guidance-scale",
        help=(
            "Independent reference-latent guidance scale. "
            "Adds scale * (full - full_without_reference_latents). "
            "0 disables this branch."
        ),
    ),
    siglip_guidance_scale: float = typer.Option(
        0.0,
        "--siglip-guidance-scale",
        help=(
            "Isolated SigLIP visual guidance. Adds scale * "
            "(null_text_with_siglip_and_refs - null_text_without_siglip_with_refs). "
            "0 disables this branch. Negative values reverse the direction for diagnostics."
        ),
    ),
    guidance_rescale: float = typer.Option(
        0.0,
        "--guidance-rescale",
        help=(
            "Rescale the final guided prediction toward the standard deviation "
            "of the full conditional prediction. 0 disables. "
            "LTX-2.3 standard default is 0.7."
        ),
    ),
    negative_prompt: str | None = typer.Option(
        None,
        "--negative-prompt",
        help="Negative prompt for CFG. Defaults to config.validation.negative_prompt.",
    ),
    cfg_negative_mode: str = typer.Option(
        "negative_prompt_no_siglip",
        "--cfg-negative-mode",
        help="CFG negative branch mode. Currently supports 'negative_prompt_no_siglip'.",
    ),
    cfg_drop_ref_latents_in_negative: bool = typer.Option(
        False,
        "--cfg-drop-ref-latents-in-negative/--cfg-keep-ref-latents-in-negative",
        help=(
            "If true, negative branch masks reference latent tokens as well. "
            "Default false keeps reference latents shared between positive and negative branches."
        ),
    ),
    stg_scale: float = typer.Option(
        0.0,
        "--stg-scale",
        help="STG scale. 0.0 disables STG. Final formula adds stg_scale * (x_prompt - x_stg).",
    ),
    stg_blocks: str | None = typer.Option(
        "28",
        "--stg-blocks",
        help=(
            "Comma-separated transformer block indices for STG video self-attention skipping. "
            "Use empty string, none, or all to apply to all blocks if supported."
        ),
    ),
    stg_mode: Literal["stg_v"] = typer.Option(
        "stg_v",
        "--stg-mode",
        help="STG mode for this video-only Stage1 inference script. Only stg_v is supported.",
    ),
    condition_mode: str = typer.Option(
        "full_siglip",
        "--condition-mode",
        help=(
            "Positive condition mode. full_siglip uses Stage1 full teacher condition with GT SigLIP tokens. "
            "text_only_no_siglip uses text-only Gemma/LTX condition and keeps reference latents."
        ),
    ),
) -> None:
    if sample_index is not None and sample_index < 0:
        raise typer.BadParameter("--sample-index must be >= 0")
    if all_samples and sample_index is not None:
        raise typer.BadParameter("--sample-index and --all-samples cannot be used together")
    if start_index < 0:
        raise typer.BadParameter("--start-index must be >= 0")
    if end_index is not None and end_index <= start_index:
        raise typer.BadParameter("--end-index must be greater than --start-index")
    if num_shards <= 0:
        raise typer.BadParameter("--num-shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise typer.BadParameter("--shard-index must satisfy 0 <= shard_index < num_shards")
    if gc_interval < 0:
        raise typer.BadParameter("--gc-interval must be >= 0")
    if num_inference_steps < 1:
        raise typer.BadParameter("--num-inference-steps must be >= 1")
    if guidance_scale < 1.0:
        raise typer.BadParameter("--guidance-scale must be >= 1.0")
    if ref_guidance_scale < 0.0:
        raise typer.BadParameter("--ref-guidance-scale must be >= 0.0")
    _validate_guidance_rescale(guidance_rescale)
    _validate_siglip_guidance(siglip_guidance_scale, condition_mode)
    if ref_guidance_scale != 0.0 and cfg_drop_ref_latents_in_negative:
        raise typer.BadParameter(
            "--ref-guidance-scale requires --cfg-keep-ref-latents-in-negative so the CFG negative branch is N_R"
        )
    if stg_scale < 0.0:
        raise typer.BadParameter("--stg-scale must be >= 0.0")
    if stg_mode != "stg_v":
        raise typer.BadParameter("--stg-mode currently only supports 'stg_v'")
    if cfg_negative_mode != "negative_prompt_no_siglip":
        raise typer.BadParameter("--cfg-negative-mode currently only supports 'negative_prompt_no_siglip'")
    if condition_mode not in _POSITIVE_CONDITION_MODES:
        raise typer.BadParameter(
            "--condition-mode must be one of: " + ", ".join(sorted(_POSITIVE_CONDITION_MODES))
        )

    range_requested = (
        all_samples
        or start_index != 0
        or end_index is not None
        or shard_index != 0
        or num_shards != 1
    )
    batch_mode = sample_index is None and range_requested
    manifest_path = Path(manifest)
    manifest_root = Path(root_dir) if root_dir is not None else manifest_path.parent
    rows = _read_manifest(manifest_path)
    if sample_index is not None:
        if sample_index >= len(rows):
            raise typer.BadParameter(f"--sample-index must be in [0, {len(rows) - 1}]")
        selected_indices = [sample_index]
    elif batch_mode:
        selected_indices = _select_sample_indices(
            row_count=len(rows),
            start_index=start_index,
            end_index=end_index,
            shard_index=shard_index,
            num_shards=num_shards,
        )
    else:
        if not rows:
            raise ValueError("Manifest contains no samples")
        selected_indices = [0]

    resolved_text_condition_source = text_condition_source or (
        "precomputed" if batch_mode and condition_mode == "text_only_no_siglip" else "online"
    )
    if batch_mode and condition_mode == "text_only_no_siglip" and resolved_text_condition_source == "online":
        raise typer.BadParameter(
            "Batch text_only_no_siglip inference requires --text-condition-source precomputed"
        )
    resolved_copy_media = copy_media if copy_media is not None else not batch_mode

    torch_device = torch.device(device)
    dtype = torch.bfloat16
    config_path = Path(config)
    checkpoint_path = Path(checkpoint)
    runtime = _load_inference_runtime(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=torch_device,
        dtype=dtype,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        keep_feature_extractor=(
            condition_mode == "text_only_no_siglip" and resolved_text_condition_source == "online"
        ),
    )
    parsed_stg_blocks = _parse_stg_blocks(stg_blocks)
    normalized_precomputed_root = _normalize_precomputed_root(Path(precomputed_root))
    output_dir_path = Path(output_dir)

    def run_sample(index: int) -> Path:
        return _run_one_sample(
            runtime=runtime,
            rows=rows,
            sample_index=index,
            manifest_root=manifest_root,
            precomputed_root=normalized_precomputed_root,
            output_dir=output_dir_path,
            video_column=video_column,
            caption_column=caption_column,
            reference_column=reference_column,
            condition_mode=condition_mode,
            text_condition_source=resolved_text_condition_source,
            text_conditions_dir=text_conditions_dir,
            copy_media=resolved_copy_media,
            fps=fps,
            decode_tile=decode_tile,
            guidance_scale=guidance_scale,
            cfg_negative_mode=cfg_negative_mode,
            cfg_drop_ref_latents_in_negative=cfg_drop_ref_latents_in_negative,
            ref_guidance_scale=ref_guidance_scale,
            siglip_guidance_scale=siglip_guidance_scale,
            guidance_rescale=guidance_rescale,
            stg_scale=stg_scale,
            stg_blocks=parsed_stg_blocks,
            stg_mode=stg_mode,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

    summary = _run_selected_samples(
        selected_indices=selected_indices,
        output_dir=output_dir_path,
        condition_mode=condition_mode,
        checkpoint_path=checkpoint_path,
        shard_index=shard_index,
        num_shards=num_shards,
        skip_existing=skip_existing,
        continue_on_error=continue_on_error if batch_mode else False,
        gc_interval=gc_interval,
        device=torch_device,
        run_sample=run_sample,
        write_summary=batch_mode,
    )
    if batch_mode:
        console.print(f"[green]Persistent shard complete:[/green] {json.dumps(summary, ensure_ascii=False)}")
        if summary["num_failed"] > 0:
            console.print(
                f"[red]Shard completed with {summary['num_failed']} failed samples[/red]"
            )
            raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
