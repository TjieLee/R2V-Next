"""Stable Stage 3 checkpoint loading for raw online-sample inference."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from ltx_trainer.model_loader import (
    load_embeddings_processor,
    load_text_encoder,
    load_transformer,
    load_video_vae_decoder,
    load_video_vae_encoder,
)
from ltx_trainer.online_data.online_batch_encoder import OnlineBatchEncoder
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.training_strategies.multi_reference_planner_stage2 import (
    MultiReferencePlannerStage2Strategy,
)

_STEP_PATTERN = re.compile(r"(?:^|_)step_(\d+)(?:\.|$)")


def _load_script_module(filename: str, module_name: str) -> Any:
    script = Path(__file__).resolve().parents[3] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load inference helpers from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_legacy_inference_helpers() -> tuple[Any, Any]:
    stage2 = _load_script_module(
        "infer_multiref_stage2_overfit.py",
        "jd_ltx_online_inference_stage2_helpers",
    )
    return stage2.stage1, stage2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int:
    match = _STEP_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Checkpoint filename contains no step: {path.name}")
    return int(match.group(1))


def _load_ready_marker(marker_path: Path) -> dict[str, Any]:
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    step = int(payload["global_step"])
    if marker_path.name != f"checkpoint_step_{step:05d}.ready.json":
        raise ValueError(f"Ready marker filename/global_step mismatch: {marker_path}")
    checkpoint_path = Path(str(payload["checkpoint_path"])).expanduser().resolve()
    training_state_path = Path(str(payload["training_state_path"])).expanduser().resolve()
    if not checkpoint_path.is_file() or not training_state_path.is_file():
        raise FileNotFoundError(
            "Ready marker points to incomplete publication: "
            f"checkpoint={checkpoint_path}, training_state={training_state_path}"
        )
    if checkpoint_step(checkpoint_path) != step:
        raise ValueError("Ready marker and checkpoint filename steps differ")
    if checkpoint_path.stat().st_size != int(payload["checkpoint_size_bytes"]):
        raise ValueError(f"Ready-marker checkpoint size mismatch: {checkpoint_path}")
    if str(payload.get("metadata_global_step")) != str(step):
        raise ValueError(f"Ready-marker metadata_global_step does not match {step}")
    if payload.get("metadata_training_phase") != "stage3":
        raise ValueError("Online inference accepts only Stage 3 ready markers")
    return {**payload, "checkpoint_path": str(checkpoint_path)}


def resolve_checkpoint(
    *,
    checkpoint: str | Path | None,
    latest_ready_dir: str | Path | None,
) -> tuple[Path, Path | None, dict[str, Any] | None]:
    if (checkpoint is None) == (latest_ready_dir is None):
        raise ValueError("Specify exactly one of checkpoint or latest_ready_dir")
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path, None, None

    ready_dir = Path(str(latest_ready_dir)).expanduser().resolve()
    markers = sorted(
        ready_dir.glob("checkpoint_step_*.ready.json"),
        key=lambda path: int(path.name.removeprefix("checkpoint_step_").removesuffix(".ready.json")),
    )
    if not markers:
        raise FileNotFoundError(f"No checkpoint ready marker found in {ready_dir}")
    marker_path = markers[-1]
    marker = _load_ready_marker(marker_path)
    return Path(marker["checkpoint_path"]), marker_path, marker


def audit_checkpoint(path: Path) -> dict[str, Any]:
    component_predicates = {
        "dit_lora": lambda key: key.startswith("diffusion_model.") and "lora_" in key,
        "gemma_lora": lambda key: key.startswith("text_encoder.model.model.language_model.")
        and "lora_" in key,
        "planner": lambda key: key.startswith("training_strategy.planner_tokens."),
        "visual_projection": lambda key: key.startswith(
            "training_strategy.visual_token_projection."
        ),
        "visual_full_encoder": lambda key: key.startswith(
            "training_strategy.visual_full_encoder."
        ),
        "connector": lambda key: key.startswith("embeddings_processor.video_connector."),
    }
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = dict(handle.metadata() or {})
    counts = {
        name: sum(1 for key in keys if predicate(key))
        for name, predicate in component_predicates.items()
    }
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"Stage 3 checkpoint is missing required components: {missing}")
    filename_step = checkpoint_step(path)
    metadata_step = metadata.get("global_step")
    if metadata_step is None:
        raise RuntimeError("Stage 3 checkpoint metadata is missing global_step")
    if int(metadata_step) != filename_step:
        raise RuntimeError(
            "Stage 3 checkpoint step mismatch: "
            f"filename={filename_step}, metadata={metadata_step}"
        )
    if metadata.get("training_phase") != "stage3":
        raise RuntimeError("Online inference requires checkpoint metadata training_phase=stage3")
    return {
        "checkpoint_step": filename_step,
        "checkpoint_metadata": metadata,
        "component_key_counts": counts,
        "checkpoint_key_count": len(keys),
    }


@dataclass
class OnlineInferenceRuntime:
    cfg: Any
    transformer: nn.Module
    embeddings_processor: nn.Module
    text_encoder: nn.Module
    vae_encoder: nn.Module
    vae_decoder: nn.Module | None
    strategy: MultiReferencePlannerStage2Strategy
    online_encoder: OnlineBatchEncoder
    negative_conditions: dict[str, torch.Tensor | None] | None
    negative_prompt: str | None
    config_path: Path
    checkpoint_path: Path
    ready_marker_path: Path | None
    checkpoint_audit: dict[str, Any]
    checkpoint_flags: dict[str, bool]
    device: torch.device
    dtype: torch.dtype
    stage1: Any
    stage2: Any


def load_online_inference_runtime(
    *,
    config_path: str | Path,
    checkpoint_path: Path,
    ready_marker_path: Path | None,
    ready_marker: dict[str, Any] | None,
    device: torch.device,
    dtype: torch.dtype,
    guidance_scale: float,
    negative_prompt: str | None,
    load_vae_decoder: bool = True,
) -> OnlineInferenceRuntime:
    """Load each model once while preserving the feature extractor needed by raw encoding."""
    stage1, stage2 = load_legacy_inference_helpers()
    config = Path(config_path).expanduser().resolve()
    cfg = stage2._load_config(config)
    if cfg.data.encoding_mode != "online" or cfg.data.online_encoding is None:
        raise ValueError("Raw online inference requires data.encoding_mode='online' and online_encoding")
    effective_negative_prompt = stage2._resolve_negative_prompt(
        cli_negative_prompt=negative_prompt,
        config_negative_prompt=cfg.validation.negative_prompt,
        guidance_scale=guidance_scale,
    )

    before = checkpoint_path.stat()
    checkpoint_hash = _sha256_file(checkpoint_path)
    audit = audit_checkpoint(checkpoint_path)
    if ready_marker is not None and checkpoint_hash != str(ready_marker["checkpoint_sha256"]):
        raise RuntimeError("Checkpoint SHA256 no longer matches its ready marker")

    transformer = load_transformer(cfg.model.model_path, device=device, dtype=dtype)
    transformer = stage2._setup_dit_lora(transformer, cfg)
    embeddings_processor = load_embeddings_processor(cfg.model.model_path, device=device, dtype=dtype)
    text_encoder = load_text_encoder(
        gemma_model_path=cfg.model.text_encoder_path,
        device=device,
        dtype=dtype,
        load_in_8bit=cfg.acceleration.load_text_encoder_in_8bit,
    )
    stage2._setup_gemma_lora(text_encoder, cfg)
    strategy = get_training_strategy(cfg.training_strategy)
    if not isinstance(strategy, MultiReferencePlannerStage2Strategy):
        raise TypeError(f"Expected MultiReferencePlannerStage2Strategy, got {type(strategy).__name__}")
    base_transformer = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
    strategy.attach_models(
        transformer=base_transformer,
        embeddings_processor=embeddings_processor,
        text_encoder=text_encoder,
    )
    checkpoint_flags = stage2._load_checkpoint_weights(
        checkpoint_path=checkpoint_path,
        transformer=transformer,
        embeddings_processor=embeddings_processor,
        text_encoder=text_encoder,
        strategy=strategy,
    )
    after = checkpoint_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Checkpoint changed while inference runtime was loading")

    stage2._disable_gradient_checkpointing(transformer, strategy)
    transformer.requires_grad_(False).eval()
    embeddings_processor.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    for module in strategy.get_trainable_modules().values():
        module.requires_grad_(False).eval()
    negative_conditions = None
    if guidance_scale > 1.0:
        negative_conditions = stage2._encode_negative_prompt_condition(
            text_encoder=text_encoder,
            embeddings_processor=embeddings_processor,
            strategy=strategy,
            negative_prompt=effective_negative_prompt,
            device=device,
            dtype=dtype,
        )
    feature_extractor = getattr(embeddings_processor, "feature_extractor", None)
    if feature_extractor is None:
        raise RuntimeError("Raw online inference requires embeddings_processor.feature_extractor")

    vae_encoder = load_video_vae_encoder(cfg.model.model_path, device=device, dtype=dtype)
    vae_encoder.requires_grad_(False).eval()
    online_encoder = OnlineBatchEncoder(
        config=cfg.data.online_encoding,
        model_path=cfg.model.model_path,
        text_encoder_path=cfg.model.text_encoder_path,
        vae_encoder=vae_encoder,
        text_encoder=text_encoder,
        embeddings_processor=embeddings_processor,
        device=device,
    )
    vae_decoder = None
    if load_vae_decoder:
        vae_decoder = load_video_vae_decoder(cfg.model.model_path, device=device, dtype=dtype)
        vae_decoder.requires_grad_(False).eval()

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    audit.update(
        {
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_size_bytes": before.st_size,
            "ready_marker_path": str(ready_marker_path) if ready_marker_path else None,
            "missing_keys": [],
            "unexpected_keys": [],
        }
    )
    return OnlineInferenceRuntime(
        cfg=cfg,
        transformer=transformer,
        embeddings_processor=embeddings_processor,
        text_encoder=text_encoder,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        strategy=strategy,
        online_encoder=online_encoder,
        negative_conditions=negative_conditions,
        negative_prompt=effective_negative_prompt,
        config_path=config,
        checkpoint_path=checkpoint_path,
        ready_marker_path=ready_marker_path,
        checkpoint_audit=audit,
        checkpoint_flags=checkpoint_flags,
        device=device,
        dtype=dtype,
        stage1=stage1,
        stage2=stage2,
    )
