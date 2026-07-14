#!/usr/bin/env python3
"""Persistent Stage 2 multi-reference planner inference for overfit samples."""

from __future__ import annotations

import gc
import importlib.util
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
import typer
import yaml
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from rich.console import Console
from safetensors.torch import load_file
from torch import Tensor, nn

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.model_loader import (
    load_embeddings_processor,
    load_text_encoder,
    load_transformer,
    load_video_vae_decoder,
)
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.training_strategies.multi_reference_planner_stage2 import (
    MultiReferencePlannerStage2Strategy,
)


def _load_stage1_helpers() -> Any:
    script = Path(__file__).with_name("infer_multiref_stage1_overfit.py")
    spec = importlib.util.spec_from_file_location("infer_multiref_stage1_shared", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Stage 1 inference helpers from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage1 = _load_stage1_helpers()
console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Run persistent Stage 2 planner inference without GT visual-token conditioning.",
)

_CONDITION_MODE = "stage2_planner"
_GT_METADATA_KEYS = ("tokens_per_frame", "sampled_frame_indices", "source_fps")
RefGuidanceMode = Literal["synchronized", "shared_planner_latent_only"]
_DEFAULT_REF_GUIDANCE_MODE: RefGuidanceMode = "synchronized"


@dataclass
class Stage2InferenceRuntime:
    cfg: LtxTrainerConfig
    transformer: nn.Module
    embeddings_processor: nn.Module
    text_encoder: nn.Module
    vae_decoder: nn.Module
    strategy: MultiReferencePlannerStage2Strategy
    checkpoint_flags: dict[str, bool]
    device: torch.device
    dtype: torch.dtype
    config_path: Path
    checkpoint_path: Path
    negative_prompt: str | None
    negative_conditions: dict[str, Tensor | None] | None


@dataclass(frozen=True)
class _GuidanceConditionBundle:
    positive_conditions: dict[str, Tensor]
    negative_conditions: dict[str, Tensor | None] | None
    no_ref_conditions: dict[str, Tensor | None] | None
    inference_diagnostics: dict[str, Any]
    no_ref_diagnostics: dict[str, Any] | None
    planner_forward_count: int
    shared_visual_context: Tensor | None = None
    shared_visual_mask: Tensor | None = None


def _load_config(config_path: Path) -> LtxTrainerConfig:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    config_data = yaml.load(config_path.read_text(encoding="utf-8"), Loader=stage1._TupleSafeLoader)
    cfg = LtxTrainerConfig(**config_data)
    strategy = cfg.training_strategy
    if strategy.name != "multi_reference_planner_stage2":
        raise ValueError(
            "Stage 2 inference requires training_strategy.name='multi_reference_planner_stage2', "
            f"got {strategy.name!r}"
        )
    if not strategy.use_online_vlm:
        raise ValueError("Stage 2 inference requires training_strategy.use_online_vlm=true")
    if strategy.planner_token_count != 2048:
        raise ValueError(f"Stage 2 inference requires planner_token_count=2048, got {strategy.planner_token_count}")
    if strategy.visual_context_mode != "full_tokens_3d_sa":
        raise ValueError("Stage 2 inference requires visual_context_mode='full_tokens_3d_sa'")
    if strategy.visual_connector_enabled:
        raise ValueError("Stage 2 inference requires visual_connector_enabled=false")
    strategy.cfg_dropout_enabled = False
    strategy.gemma_gradient_checkpointing = False
    return cfg


def _setup_dit_lora(transformer: nn.Module, cfg: LtxTrainerConfig) -> nn.Module:
    if cfg.model.training_mode != "lora" or cfg.lora is None:
        raise ValueError("Stage 2 inference requires the Stage 1 DiT LoRA config")
    return get_peft_model(
        transformer,
        LoraConfig(
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            init_lora_weights=True,
        ),
    )


def _setup_gemma_lora(text_encoder: nn.Module, cfg: LtxTrainerConfig) -> None:
    lora = cfg.text_encoder_lora
    if not lora.enabled:
        raise ValueError("Stage 2 inference requires text_encoder_lora.enabled=true")
    gemma_model = text_encoder.model.model
    language_model = getattr(gemma_model, "language_model", None)
    if language_model is None:
        raise ValueError("Gemma model does not expose model.language_model")
    language_model.requires_grad_(False)
    gemma_model.language_model = get_peft_model(
        language_model,
        LoraConfig(
            r=lora.rank,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=lora.target_modules,
            init_lora_weights=True,
        ),
    )


def _validate_stage2_checkpoint_state(state_dict: dict[str, Tensor]) -> None:
    required_components = {
        "Stage 1 DiT LoRA": lambda key: key.startswith("diffusion_model.") and "lora_" in key,
        "Gemma LoRA": lambda key: (
            key.startswith("text_encoder.model.model.language_model.") and "lora_" in key
        ),
        "planner_tokens": lambda key: key.startswith("training_strategy.planner_tokens."),
        "visual_token_projection": lambda key: key.startswith(
            "training_strategy.visual_token_projection."
        ),
        "visual_full_encoder": lambda key: key.startswith("training_strategy.visual_full_encoder."),
        "video_connector": lambda key: key.startswith("embeddings_processor.video_connector."),
    }
    missing = [
        name
        for name, predicate in required_components.items()
        if not any(predicate(key) for key in state_dict)
    ]
    if missing:
        raise RuntimeError(f"Stage 2 checkpoint is missing required components: {missing}")
    MultiReferencePlannerStage2Strategy.validate_checkpoint_state_dict(state_dict)


def _load_checkpoint_weights(
    *,
    checkpoint_path: Path,
    transformer: nn.Module,
    embeddings_processor: nn.Module,
    text_encoder: nn.Module,
    strategy: MultiReferencePlannerStage2Strategy,
) -> dict[str, bool]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    state_dict = load_file(checkpoint_path)
    _validate_stage2_checkpoint_state(state_dict)

    strategy.load_extra_checkpoint_state_dict(state_dict)

    processor_state = {
        key.removeprefix("embeddings_processor."): value
        for key, value in state_dict.items()
        if key.startswith("embeddings_processor.")
    }
    missing, unexpected = embeddings_processor.load_state_dict(processor_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected embeddings processor checkpoint keys: {unexpected}")
    if not any(key.startswith("video_connector.") for key in processor_state):
        raise RuntimeError("Stage 2 checkpoint is missing embeddings_processor.video_connector.*")

    text_state = {
        key.removeprefix("text_encoder."): value
        for key, value in state_dict.items()
        if key.startswith("text_encoder.")
    }
    _text_missing, text_unexpected = text_encoder.load_state_dict(text_state, strict=False)
    if text_unexpected:
        raise RuntimeError(f"Unexpected Gemma checkpoint keys: {text_unexpected}")

    dit_state = {
        key.removeprefix("diffusion_model."): value
        for key, value in state_dict.items()
        if key.startswith("diffusion_model.")
    }
    try:
        set_peft_model_state_dict(transformer.get_base_model(), dit_state)
    except RuntimeError as exc:
        raise RuntimeError("Stage 1 DiT LoRA config does not match the Stage 2 checkpoint") from exc

    del state_dict
    return {
        "dit_lora_checkpoint_loaded": bool(dit_state),
        "gemma_lora_checkpoint_loaded": bool(text_state),
        "planner_checkpoint_loaded": True,
        "visual_token_projection_checkpoint_loaded": True,
        "visual_full_encoder_checkpoint_loaded": True,
        "connector_checkpoint_loaded": not missing or bool(processor_state),
    }


def _disable_gradient_checkpointing(transformer: nn.Module, strategy: MultiReferencePlannerStage2Strategy) -> None:
    base_transformer = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
    if hasattr(base_transformer, "set_gradient_checkpointing"):
        base_transformer.set_gradient_checkpointing(False)
    language_model = strategy._get_language_model()
    language_model = getattr(language_model, "module", language_model)
    checkpoint_model = language_model.get_base_model() if hasattr(language_model, "get_base_model") else language_model
    disable = getattr(checkpoint_model, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()


def _resolve_negative_prompt(
    *,
    cli_negative_prompt: str | None,
    config_negative_prompt: str | None,
    guidance_scale: float,
) -> str | None:
    cli_prompt = cli_negative_prompt.strip() if cli_negative_prompt else None
    config_prompt = config_negative_prompt.strip() if config_negative_prompt else None
    effective_prompt = cli_prompt or config_prompt
    if guidance_scale > 1.0 and not effective_prompt:
        raise ValueError("--guidance-scale > 1 requires a non-empty negative prompt")
    return effective_prompt


def _load_inference_runtime(
    *,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    guidance_scale: float,
    negative_prompt: str | None,
) -> Stage2InferenceRuntime:
    cfg = _load_config(config_path)
    effective_negative_prompt = _resolve_negative_prompt(
        cli_negative_prompt=negative_prompt,
        config_negative_prompt=cfg.validation.negative_prompt,
        guidance_scale=guidance_scale,
    )
    transformer = load_transformer(cfg.model.model_path, device=device, dtype=dtype)
    transformer = _setup_dit_lora(transformer, cfg)
    embeddings_processor = load_embeddings_processor(cfg.model.model_path, device=device, dtype=dtype)
    text_encoder = load_text_encoder(
        gemma_model_path=cfg.model.text_encoder_path,
        device=device,
        dtype=dtype,
        load_in_8bit=cfg.acceleration.load_text_encoder_in_8bit,
    )
    _setup_gemma_lora(text_encoder, cfg)

    strategy = get_training_strategy(cfg.training_strategy)
    if not isinstance(strategy, MultiReferencePlannerStage2Strategy):
        raise TypeError(f"Expected MultiReferencePlannerStage2Strategy, got {type(strategy).__name__}")
    base_transformer = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
    strategy.attach_models(
        transformer=base_transformer,
        embeddings_processor=embeddings_processor,
        text_encoder=text_encoder,
    )
    checkpoint_flags = _load_checkpoint_weights(
        checkpoint_path=checkpoint_path,
        transformer=transformer,
        embeddings_processor=embeddings_processor,
        text_encoder=text_encoder,
        strategy=strategy,
    )

    _disable_gradient_checkpointing(transformer, strategy)
    transformer.requires_grad_(False).eval()
    embeddings_processor.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    for module in strategy.get_trainable_modules().values():
        module.requires_grad_(False).eval()
    negative_conditions = None
    if guidance_scale > 1.0:
        negative_conditions = _encode_negative_prompt_condition(
            text_encoder=text_encoder,
            embeddings_processor=embeddings_processor,
            strategy=strategy,
            negative_prompt=effective_negative_prompt,
            device=device,
            dtype=dtype,
        )
        console.print(f"Cached negative prompt condition: {stage1._condition_shape(negative_conditions)}")
    if hasattr(embeddings_processor, "feature_extractor"):
        embeddings_processor.feature_extractor = None

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    vae_decoder = load_video_vae_decoder(cfg.model.model_path, device=device, dtype=dtype)
    vae_decoder.requires_grad_(False).eval()
    return Stage2InferenceRuntime(
        cfg=cfg,
        transformer=transformer,
        embeddings_processor=embeddings_processor,
        text_encoder=text_encoder,
        vae_decoder=vae_decoder,
        strategy=strategy,
        checkpoint_flags=checkpoint_flags,
        device=device,
        dtype=dtype,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        negative_prompt=effective_negative_prompt,
        negative_conditions=negative_conditions,
    )


def _load_sample_precomputed(
    *,
    row: dict[str, Any],
    manifest_root: Path,
    precomputed_root: Path,
    video_column: str,
    need_gt: bool,
    strict_no_gt: bool,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    if strict_no_gt and need_gt:
        raise ValueError("strict-no-GT mode cannot request GT SigLIP data")
    video_path = stage1._resolve_path(str(row[video_column]), manifest_root)
    rel_path = stage1._output_relative(video_path, manifest_root).with_suffix(".pt")
    files = {
        "latents": precomputed_root / "latents" / rel_path,
        "multi_reference_latents": precomputed_root / "multi_reference_latents" / rel_path,
        "conditions": precomputed_root / "vlm_conditions" / rel_path,
        "text_conditions": precomputed_root / "conditions" / rel_path,
        "planner_vlm_inputs": precomputed_root / "planner_vlm_inputs" / rel_path,
    }
    if need_gt:
        files["gt_siglip_tokens"] = precomputed_root / "gt_siglip_tokens" / rel_path
    return rel_path, {name: stage1._load_pt_file(path) for name, path in files.items()}


def _build_batch(precomputed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    batch = stage1._build_single_sample_batch(
        latents=precomputed["latents"],
        multi_reference_latents=precomputed["multi_reference_latents"],
        conditions=precomputed["conditions"],
    )
    batch["planner_vlm_inputs"] = stage1._unsqueeze_sample_dim(precomputed["planner_vlm_inputs"])
    batch["text_conditions"] = stage1._unsqueeze_sample_dim(precomputed["text_conditions"])
    return batch


def _extract_gt_position_metadata(gt_data: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in _GT_METADATA_KEYS if key not in gt_data]
    if missing:
        raise ValueError(f"GT SigLIP metadata is missing fields: {missing}")
    return stage1._unsqueeze_sample_dim({key: gt_data[key] for key in _GT_METADATA_KEYS})


def _uniform_position_metadata(latents_metadata: dict[str, Any]) -> dict[str, Tensor]:
    num_video_frames = int(latents_metadata["num_frames"].flatten()[0].item())
    if num_video_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_video_frames}")
    sampled = torch.linspace(0, num_video_frames - 1, steps=8).round().to(dtype=torch.long).unsqueeze(0)
    fps = latents_metadata.get("fps")
    if isinstance(fps, Tensor):
        source_fps = fps.flatten()[:1].to(dtype=torch.float32)
    else:
        source_fps = torch.tensor([float(fps or 24.0)], dtype=torch.float32)
    return {
        "tokens_per_frame": torch.tensor([256], dtype=torch.long),
        "sampled_frame_indices": sampled,
        "source_fps": source_fps,
    }


def _compute_planner_metrics(
    *,
    predicted_tokens: Tensor,
    predicted_mask: Tensor,
    gt_data: dict[str, Any],
    visual_token_key: str,
) -> dict[str, float]:
    gt_tokens = gt_data.get(visual_token_key)
    if not isinstance(gt_tokens, Tensor):
        raise ValueError(f"GT diagnostics require {visual_token_key!r}")
    if gt_tokens.ndim == 2:
        gt_tokens = gt_tokens.unsqueeze(0)
    gt_mask = gt_data.get("visual_token_mask")
    if isinstance(gt_mask, Tensor):
        if gt_mask.ndim == 1:
            gt_mask = gt_mask.unsqueeze(0)
        mask = predicted_mask.detach().cpu().bool() & gt_mask.detach().cpu().bool()
    else:
        mask = predicted_mask.detach().cpu().bool()
    predicted = predicted_tokens.detach().float().cpu()
    target = gt_tokens.detach().float().cpu()
    if predicted.shape != target.shape:
        raise ValueError(f"Predicted token shape {tuple(predicted.shape)} != GT {tuple(target.shape)}")
    selected_predicted = predicted[mask]
    selected_target = target[mask]
    if selected_predicted.numel() == 0:
        raise ValueError("No valid tokens remain for planner diagnostics")
    return {
        "planner_siglip_mse": F.mse_loss(selected_predicted, selected_target).item(),
        "planner_siglip_cosine": F.cosine_similarity(selected_predicted, selected_target, dim=-1).mean().item(),
        "predicted_token_mean": selected_predicted.mean().item(),
        "predicted_token_std": selected_predicted.std().item(),
        "predicted_token_norm": selected_predicted.norm(dim=-1).mean().item(),
        "gt_token_mean": selected_target.mean().item(),
        "gt_token_std": selected_target.std().item(),
        "gt_token_norm": selected_target.norm(dim=-1).mean().item(),
    }


def _save_predicted_tokens(
    *,
    output_dir: Path,
    rel_path: Path,
    inference_diagnostics: dict[str, Any],
    planner_metrics: dict[str, float],
) -> Path:
    output_path = output_dir / "planner_predictions" / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "predicted_visual_tokens": inference_diagnostics["predicted_visual_tokens"][0].detach().cpu(),
            "predicted_visual_token_mask": inference_diagnostics["predicted_visual_token_mask"][0].detach().cpu(),
            "token_positions": inference_diagnostics["token_positions"][0].detach().cpu(),
            "diagnostics": {
                **planner_metrics,
                "planner_raw_shape": inference_diagnostics["planner_raw_shape"],
                "planner_projected_shape": inference_diagnostics["planner_projected_shape"],
                "visual_context_shape": inference_diagnostics["visual_context_shape"],
            },
        },
        output_path,
    )
    return output_path


def _autocast_context(device: torch.device, dtype: torch.dtype) -> Any:
    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def _encode_negative_prompt_condition(
    *,
    text_encoder: nn.Module,
    embeddings_processor: nn.Module,
    strategy: MultiReferencePlannerStage2Strategy,
    negative_prompt: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor | None]:
    if not negative_prompt:
        raise ValueError("Negative prompt must be non-empty")
    language_model = strategy._get_language_model()
    language_model = getattr(language_model, "module", language_model)
    disable_adapter = getattr(language_model, "disable_adapter", None)
    adapter_context = disable_adapter() if callable(disable_adapter) else nullcontext()
    with adapter_context, torch.inference_mode(), _autocast_context(device, dtype):
        hidden_states, attention_mask = text_encoder.encode([negative_prompt])[0]
        encoded = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
    return {
        "video_prompt_embeds": encoded.video_encoding.to(device=device, dtype=dtype),
        "audio_prompt_embeds": (
            encoded.audio_encoding.to(device=device, dtype=dtype)
            if encoded.audio_encoding is not None
            else None
        ),
        "prompt_attention_mask": None,
    }


def _prepare_guidance_condition_bundle(
    *,
    strategy: MultiReferencePlannerStage2Strategy,
    conditions: dict[str, Tensor],
    text_conditions: dict[str, Tensor],
    planner_vlm_inputs: dict[str, Any],
    latents_metadata: dict[str, Any],
    visual_position_metadata: dict[str, Any],
    negative_text_conditions: dict[str, Tensor | None] | None,
    guidance_scale: float,
    ref_guidance_scale: float,
    ref_guidance_mode: RefGuidanceMode,
) -> _GuidanceConditionBundle:
    """Build Stage 2 guidance contexts while making planner sharing explicit."""
    if ref_guidance_mode == "synchronized":
        positive_conditions, inference_diagnostics = strategy.prepare_inference_conditions(
            conditions=conditions,
            planner_vlm_inputs=planner_vlm_inputs,
            latents_metadata=latents_metadata,
            visual_position_metadata=visual_position_metadata,
        )
        if ref_guidance_scale != 0.0:
            no_ref_conditions, no_ref_diagnostics = strategy.prepare_inference_conditions(
                conditions=text_conditions,
                planner_vlm_inputs=planner_vlm_inputs,
                latents_metadata=latents_metadata,
                visual_position_metadata=visual_position_metadata,
                drop_reference_images=True,
            )
        else:
            no_ref_conditions = None
            no_ref_diagnostics = None
        return _GuidanceConditionBundle(
            positive_conditions=positive_conditions,
            negative_conditions=negative_text_conditions,
            no_ref_conditions=no_ref_conditions,
            inference_diagnostics=inference_diagnostics,
            no_ref_diagnostics=no_ref_diagnostics,
            planner_forward_count=1 + int(ref_guidance_scale != 0.0),
        )

    if ref_guidance_mode != "shared_planner_latent_only":
        raise ValueError(f"Unknown ref guidance mode: {ref_guidance_mode!r}")
    parts = MultiReferencePlannerStage2Strategy.prepare_inference_condition_parts(
        strategy,
        conditions=conditions,
        planner_vlm_inputs=planner_vlm_inputs,
        latents_metadata=latents_metadata,
        visual_position_metadata=visual_position_metadata,
    )
    if stage1._cfg_enabled(guidance_scale):
        if negative_text_conditions is None:
            raise ValueError("Shared-planner CFG requires cached negative text conditions")
        negative_conditions = MultiReferencePlannerStage2Strategy.append_shared_inference_visual_context(
            strategy,
            negative_text_conditions,
            visual_context=parts.visual_context,
            visual_mask=parts.visual_mask,
        )
    else:
        negative_conditions = None
    return _GuidanceConditionBundle(
        positive_conditions=parts.final_conditions,
        negative_conditions=negative_conditions,
        no_ref_conditions=None,
        inference_diagnostics=parts.diagnostics,
        no_ref_diagnostics=None,
        planner_forward_count=1,
        shared_visual_context=parts.visual_context,
        shared_visual_mask=parts.visual_mask,
    )


def _guidance_metadata(
    *,
    ref_guidance_mode: RefGuidanceMode,
    planner_forward_count: int,
    ref_guidance_enabled: bool,
) -> dict[str, Any]:
    if ref_guidance_mode == "shared_planner_latent_only":
        return {
            "ref_guidance_mode": ref_guidance_mode,
            "planner_forward_count": planner_forward_count,
            "negative_uses_shared_planner": True,
            "no_ref_uses_shared_planner": True,
            "no_ref_branch_is_synchronized": False,
            "ref_guidance_formula": (
                "full_shared_planner_with_refs - full_shared_planner_without_ref_latents"
            ),
            "guidance_formula": (
                "N_shared_planner_with_refs + cfg*(P_shared_planner_with_refs-N_shared_planner_with_refs) + "
                "ref*(P_shared_planner_with_refs-P_shared_planner_without_ref_latents) + "
                "stg*(P_shared_planner_with_refs-S_stg_of_P)"
            ),
        }
    return {
        "ref_guidance_mode": "synchronized",
        "planner_forward_count": planner_forward_count,
        "negative_uses_shared_planner": False,
        "no_ref_uses_shared_planner": False,
        "no_ref_branch_is_synchronized": ref_guidance_enabled,
        "ref_guidance_formula": "full_reference_planner_with_refs - synchronized_text_only_planner_without_refs",
        "guidance_formula": (
            "negative_with_refs + cfg*(full-negative_with_refs) + "
            "ref*(full-no_ref_sync) + stg*(full-stg)"
        ),
    }


def _run_one_sample(  # noqa: PLR0913, PLR0915
    *,
    runtime: Stage2InferenceRuntime,
    rows: list[dict[str, Any]],
    sample_index: int,
    manifest_root: Path,
    precomputed_root: Path,
    output_dir: Path,
    video_column: str,
    caption_column: str,
    reference_column: str,
    copy_media: bool,
    fps: float | None,
    decode_tile: bool,
    position_source: Literal["gt_metadata", "uniform_target"],
    compute_gt_siglip_metrics: bool,
    strict_no_gt: bool,
    save_predicted_tokens: bool,
    guidance_scale: float,
    ref_guidance_scale: float,
    ref_guidance_mode: RefGuidanceMode,
    guidance_rescale: float,
    stg_scale: float,
    stg_blocks: list[int] | None,
    num_inference_steps: int,
    seed: int,
) -> Path:
    sample_start = time.perf_counter()
    sample = rows[sample_index]
    if video_column not in sample:
        raise ValueError(f"Selected sample has no video column {video_column!r}")
    prompt = str(sample.get(caption_column, ""))
    if not prompt.strip():
        raise ValueError(f"Sample has empty prompt in caption column {caption_column!r}")
    sample_dir = output_dir / f"sample_{sample_index}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    if copy_media:
        gt_copy, ref_copies = stage1._copy_sample_media(
            sample=sample,
            sample_dir=sample_dir,
            manifest_root=manifest_root,
            video_column=video_column,
            reference_column=reference_column,
        )
    else:
        gt_copy = None
        ref_copies = []

    need_gt = not strict_no_gt and (position_source == "gt_metadata" or compute_gt_siglip_metrics)
    rel_path, precomputed = _load_sample_precomputed(
        row=sample,
        manifest_root=manifest_root,
        precomputed_root=precomputed_root,
        video_column=video_column,
        need_gt=need_gt,
        strict_no_gt=strict_no_gt,
    )
    batch = _build_batch(precomputed)
    gt_data = precomputed.get("gt_siglip_tokens")
    if position_source == "gt_metadata":
        if gt_data is None:
            raise ValueError("--position-source gt_metadata requires gt_siglip_tokens metadata")
        visual_position_metadata = _extract_gt_position_metadata(gt_data)
    else:
        visual_position_metadata = _uniform_position_metadata(batch["latents"])

    batch["conditions"] = stage1._move_nested_to_device(
        batch["conditions"],
        device=runtime.device,
        dtype=runtime.dtype,
    )
    batch["planner_vlm_inputs"] = stage1._move_nested_to_device(
        batch["planner_vlm_inputs"],
        device=runtime.device,
        dtype=runtime.dtype,
    )
    batch["text_conditions"] = stage1._move_nested_to_device(
        batch["text_conditions"],
        device=runtime.device,
        dtype=runtime.dtype,
    )
    visual_position_metadata = stage1._move_nested_to_device(
        visual_position_metadata,
        device=runtime.device,
    )

    with torch.inference_mode(), _autocast_context(runtime.device, runtime.dtype):
        guidance_conditions = _prepare_guidance_condition_bundle(
            strategy=runtime.strategy,
            conditions=batch["conditions"],
            text_conditions=batch["text_conditions"],
            planner_vlm_inputs=batch["planner_vlm_inputs"],
            latents_metadata=batch["latents"],
            visual_position_metadata=visual_position_metadata,
            negative_text_conditions=runtime.negative_conditions,
            guidance_scale=guidance_scale,
            ref_guidance_scale=ref_guidance_scale,
            ref_guidance_mode=ref_guidance_mode,
        )
    conditions = guidance_conditions.positive_conditions
    negative_conditions = guidance_conditions.negative_conditions
    no_ref_conditions = guidance_conditions.no_ref_conditions
    inference_diagnostics = guidance_conditions.inference_diagnostics
    no_ref_diagnostics = guidance_conditions.no_ref_diagnostics

    planner_metrics: dict[str, float] = {}
    if compute_gt_siglip_metrics:
        if gt_data is None:
            raise ValueError("--compute-gt-siglip-metrics requires gt_siglip_tokens")
        planner_metrics = _compute_planner_metrics(
            predicted_tokens=inference_diagnostics["predicted_visual_tokens"],
            predicted_mask=inference_diagnostics["predicted_visual_token_mask"],
            gt_data=gt_data,
            visual_token_key=runtime.strategy.config.visual_token_key,
        )

    prediction_path = None
    if save_predicted_tokens:
        prediction_path = _save_predicted_tokens(
            output_dir=output_dir,
            rel_path=rel_path,
            inference_diagnostics=inference_diagnostics,
            planner_metrics=planner_metrics,
        )
    inference_diagnostics.pop("predicted_visual_tokens")
    inference_diagnostics.pop("predicted_visual_token_mask")
    inference_diagnostics.pop("token_positions")
    if no_ref_diagnostics is not None:
        no_ref_diagnostics.pop("predicted_visual_tokens")
        no_ref_diagnostics.pop("predicted_visual_token_mask")
        no_ref_diagnostics.pop("token_positions")

    generated_latents = stage1._denoise_stage1(
        transformer=runtime.transformer,
        strategy=runtime.strategy,
        batch=batch,
        positive_conditions=conditions,
        negative_conditions=negative_conditions,
        no_ref_conditions=no_ref_conditions,
        guidance_scale=guidance_scale,
        cfg_drop_ref_latents_in_negative=False,
        ref_guidance_scale=ref_guidance_scale,
        siglip_guidance_scale=0.0,
        guidance_rescale=guidance_rescale,
        stg_scale=stg_scale,
        stg_blocks=stg_blocks,
        num_inference_steps=num_inference_steps,
        seed=seed,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    decoded = stage1._decode_video_latents(
        vae_decoder=runtime.vae_decoder,
        latents=generated_latents,
        device=runtime.device,
        decode_tile=decode_tile,
    )

    latent_fps = runtime.strategy._first_scalar(batch["latents"].get("fps"), default=24.0)
    output_fps = float(fps) if fps is not None else float(latent_fps)
    sampled_frame_indices = visual_position_metadata["sampled_frame_indices"].detach().cpu().flatten().tolist()
    num_video_frames = int(batch["latents"]["num_frames"].flatten()[0].item())
    generated_path = stage1._expected_generated_path(output_dir, sample_index, _CONDITION_MODE)
    reference_paths = [
        str(stage1._resolve_path(value, manifest_root))
        for value in stage1._parse_reference_images(sample.get(reference_column))
    ]
    metadata = {
        "sample_index": sample_index,
        "checkpoint": str(runtime.checkpoint_path),
        "config": str(runtime.config_path),
        "prompt": prompt,
        "reference_paths": reference_paths,
        "reference_copies": [str(path) for path in ref_copies],
        "gt_copy": str(gt_copy) if gt_copy is not None else None,
        "generated": str(generated_path),
        "condition_mode": _CONDITION_MODE,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "position_source": position_source,
        "uses_gt_visual_tokens": False,
        "uses_gt_visual_metadata": position_source == "gt_metadata",
        "strict_no_gt": strict_no_gt,
        "compute_gt_siglip_metrics": compute_gt_siglip_metrics,
        "negative_prompt": runtime.negative_prompt if guidance_scale > 1.0 else None,
        "cfg_negative_mode": (
            "negative_prompt_shared_planner_keep_refs"
            if ref_guidance_mode == "shared_planner_latent_only"
            else "negative_prompt_no_visual_keep_refs"
        ),
        "guidance_scale": guidance_scale,
        "ref_guidance_scale": ref_guidance_scale,
        "guidance_rescale": guidance_rescale,
        "stg_scale": stg_scale,
        "stg_blocks": stg_blocks,
        **_guidance_metadata(
            ref_guidance_mode=ref_guidance_mode,
            planner_forward_count=guidance_conditions.planner_forward_count,
            ref_guidance_enabled=ref_guidance_scale != 0.0,
        ),
        "uniform_sampled_frame_indices": sampled_frame_indices if position_source == "uniform_target" else None,
        "num_video_frames": num_video_frames,
        "planner_raw_shape": inference_diagnostics["planner_raw_shape"],
        "planner_projected_shape": inference_diagnostics["planner_projected_shape"],
        "visual_context_shape": inference_diagnostics["visual_context_shape"],
        "final_condition_shape": inference_diagnostics["final_condition_shape"],
        "reference_latent_shape": stage1._shape(batch["multi_ref_latents"]["latents"]),
        "target_latent_shape": stage1._shape(batch["latents"]["latents"]),
        "fps": output_fps,
        "elapsed_seconds": time.perf_counter() - sample_start,
        "decode_tile": decode_tile,
        "predicted_tokens_path": str(prediction_path) if prediction_path is not None else None,
        **runtime.checkpoint_flags,
        **planner_metrics,
    }
    for key in (
        "planner_siglip_mse",
        "planner_siglip_cosine",
        "predicted_token_mean",
        "predicted_token_std",
        "predicted_token_norm",
        "gt_token_mean",
        "gt_token_std",
        "gt_token_norm",
    ):
        metadata.setdefault(key, None)

    stage1._save_sample_outputs_atomically(
        video_tensor=decoded,
        final_video=generated_path,
        metadata=metadata,
        fps=output_fps,
    )
    console.print(f"[green]Saved Stage 2 planner video:[/green] {generated_path}")
    del batch, conditions, no_ref_conditions, inference_diagnostics, no_ref_diagnostics
    del generated_latents, decoded, precomputed
    return generated_path


def _add_summary_metrics(
    summary: dict[str, Any],
    *,
    output_dir: Path,
    selected_indices: list[int],
    shard_index: int,
) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {
        "planner_siglip_mse": [],
        "planner_siglip_cosine": [],
    }
    for sample_index in selected_indices:
        metadata_path = output_dir / f"sample_{sample_index}" / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for name in metric_values:
            value = metadata.get(name)
            if isinstance(value, int | float):
                metric_values[name].append(float(value))
    summary["mean_planner_siglip_mse"] = (
        sum(metric_values["planner_siglip_mse"]) / len(metric_values["planner_siglip_mse"])
        if metric_values["planner_siglip_mse"]
        else None
    )
    summary["mean_planner_siglip_cosine"] = (
        sum(metric_values["planner_siglip_cosine"]) / len(metric_values["planner_siglip_cosine"])
        if metric_values["planner_siglip_cosine"]
        else None
    )
    summary_path = output_dir / f"batch_summary_shard_{shard_index}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _validate_guidance(
    *,
    guidance_scale: float,
    ref_guidance_scale: float,
    siglip_guidance_scale: float,
    stg_scale: float,
) -> None:
    if guidance_scale < 1.0:
        raise typer.BadParameter("--guidance-scale must be >= 1.0")
    if ref_guidance_scale < 0.0:
        raise typer.BadParameter("--ref-guidance-scale must be >= 0.0")
    if siglip_guidance_scale != 0.0:
        raise typer.BadParameter("Stage 2 held-out inference requires --siglip-guidance-scale 0.0")
    if stg_scale < 0.0:
        raise typer.BadParameter("--stg-scale must be >= 0.0")


def _validate_strict_no_gt(
    *,
    strict_no_gt: bool,
    position_source: str,
    compute_gt_siglip_metrics: bool,
) -> None:
    if strict_no_gt and position_source != "uniform_target":
        raise typer.BadParameter("--strict-no-gt requires --position-source uniform_target")
    if strict_no_gt and compute_gt_siglip_metrics:
        raise typer.BadParameter("--strict-no-gt requires --no-compute-gt-siglip-metrics")


@app.command()
def main(  # noqa: PLR0913, PLR0915
    config: str = typer.Option(..., help="Stage 2 overfit YAML config."),
    checkpoint: str = typer.Option(..., help="Stage 2 checkpoint safetensors file."),
    manifest: str = typer.Option(..., help="Overfit manifest JSON/JSONL/CSV."),
    precomputed_root: str = typer.Option(..., help="Root containing Stage 2 precomputed directories."),
    output_dir: str = typer.Option(..., help="Directory for sample_<index>/ outputs."),
    sample_index: int | None = typer.Option(None, "--sample-index", help="Single global manifest index."),
    all_samples: bool = typer.Option(False, "--all-samples/--no-all-samples"),
    start_index: int = typer.Option(0, "--start-index"),
    end_index: int | None = typer.Option(None, "--end-index"),
    shard_index: int = typer.Option(0, "--shard-index"),
    num_shards: int = typer.Option(1, "--num-shards"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--no-skip-existing"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--fail-fast"),
    copy_media: bool | None = typer.Option(None, "--copy-media/--no-copy-media"),
    gc_interval: int = typer.Option(20, "--gc-interval"),
    device: str = typer.Option("cuda"),
    seed: int = typer.Option(42),
    num_inference_steps: int = typer.Option(50),
    video_column: str = typer.Option("video"),
    caption_column: str = typer.Option("caption"),
    reference_column: str = typer.Option("reference_images"),
    root_dir: str | None = typer.Option(None),
    fps: float | None = typer.Option(None),
    decode_tile: bool = typer.Option(True, "--decode-tile/--no-decode-tile"),
    position_source: Literal["gt_metadata", "uniform_target"] = typer.Option(
        "uniform_target",
        "--position-source",
    ),
    compute_gt_siglip_metrics: bool = typer.Option(
        False,
        "--compute-gt-siglip-metrics/--no-compute-gt-siglip-metrics",
    ),
    strict_no_gt: bool = typer.Option(
        True,
        "--strict-no-gt/--allow-gt-diagnostics",
    ),
    save_predicted_tokens: bool = typer.Option(
        False,
        "--save-predicted-tokens/--no-save-predicted-tokens",
    ),
    negative_prompt: str | None = typer.Option(None, "--negative-prompt"),
    guidance_scale: float = typer.Option(2.0, "--guidance-scale"),
    ref_guidance_scale: float = typer.Option(2.0, "--ref-guidance-scale"),
    ref_guidance_mode: RefGuidanceMode = typer.Option(
        _DEFAULT_REF_GUIDANCE_MODE,
        "--ref-guidance-mode",
        help="Reference guidance context semantics.",
    ),
    guidance_rescale: float = typer.Option(0.0, "--guidance-rescale"),
    siglip_guidance_scale: float = typer.Option(0.0, "--siglip-guidance-scale"),
    stg_scale: float = typer.Option(1.0, "--stg-scale"),
    stg_blocks: str | None = typer.Option(None, "--stg-blocks"),
) -> None:
    _validate_guidance(
        guidance_scale=guidance_scale,
        ref_guidance_scale=ref_guidance_scale,
        siglip_guidance_scale=siglip_guidance_scale,
        stg_scale=stg_scale,
    )
    stage1._validate_guidance_rescale(guidance_rescale)
    _validate_strict_no_gt(
        strict_no_gt=strict_no_gt,
        position_source=position_source,
        compute_gt_siglip_metrics=compute_gt_siglip_metrics,
    )
    if sample_index is not None and sample_index < 0:
        raise typer.BadParameter("--sample-index must be >= 0")
    if all_samples and sample_index is not None:
        raise typer.BadParameter("--sample-index and --all-samples cannot be used together")
    if start_index < 0:
        raise typer.BadParameter("--start-index must be >= 0")
    if end_index is not None and end_index <= start_index:
        raise typer.BadParameter("--end-index must be greater than --start-index")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise typer.BadParameter("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    if gc_interval < 0 or num_inference_steps < 1:
        raise typer.BadParameter("--gc-interval must be >= 0 and --num-inference-steps must be >= 1")

    manifest_path = Path(manifest)
    manifest_root = Path(root_dir) if root_dir is not None else manifest_path.parent
    rows = stage1._read_manifest(manifest_path)
    if not rows:
        raise ValueError("Manifest contains no samples")
    range_requested = all_samples or start_index != 0 or end_index is not None or shard_index != 0 or num_shards != 1
    batch_mode = sample_index is None and range_requested
    if sample_index is not None:
        if sample_index >= len(rows):
            raise typer.BadParameter(f"--sample-index must be in [0, {len(rows) - 1}]")
        selected_indices = [sample_index]
    elif batch_mode:
        selected_indices = stage1._select_sample_indices(
            row_count=len(rows),
            start_index=start_index,
            end_index=end_index,
            shard_index=shard_index,
            num_shards=num_shards,
        )
    else:
        selected_indices = [0]
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
    )
    if stg_blocks is None:
        resolved_stg_blocks = runtime.cfg.validation.stg_blocks
        if resolved_stg_blocks is None:
            resolved_stg_blocks = [29]
    else:
        resolved_stg_blocks = stage1._parse_stg_blocks(stg_blocks)
    normalized_precomputed_root = stage1._normalize_precomputed_root(Path(precomputed_root))
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
            copy_media=resolved_copy_media,
            fps=fps,
            decode_tile=decode_tile,
            position_source=position_source,
            compute_gt_siglip_metrics=compute_gt_siglip_metrics,
            strict_no_gt=strict_no_gt,
            save_predicted_tokens=save_predicted_tokens,
            guidance_scale=guidance_scale,
            ref_guidance_scale=ref_guidance_scale,
            ref_guidance_mode=ref_guidance_mode,
            guidance_rescale=guidance_rescale,
            stg_scale=stg_scale,
            stg_blocks=resolved_stg_blocks,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

    summary = stage1._run_selected_samples(
        selected_indices=selected_indices,
        output_dir=output_dir_path,
        condition_mode=_CONDITION_MODE,
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
    (output_dir_path / f"failures_shard_{shard_index}.jsonl").touch(exist_ok=True)
    if batch_mode:
        summary = _add_summary_metrics(
            summary,
            output_dir=output_dir_path,
            selected_indices=selected_indices,
            shard_index=shard_index,
        )
        console.print(f"[green]Persistent Stage 2 shard complete:[/green] {json.dumps(summary, ensure_ascii=False)}")
        if summary["num_failed"] > 0:
            raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
