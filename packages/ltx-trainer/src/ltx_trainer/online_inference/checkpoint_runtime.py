"""Stable Stage 3 checkpoint loading for raw online-sample inference."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import get_peft_model_state_dict
from safetensors import safe_open
from torch import Tensor, nn

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

_COMPONENT_PREFIXES = {
    "dit_lora": "diffusion_model.",
    "gemma_lora": "text_encoder.model.model.language_model.",
    "planner": "training_strategy.planner_tokens.",
    "visual_projection": "training_strategy.visual_token_projection.",
    "visual_full_encoder": "training_strategy.visual_full_encoder.",
    "connector": "embeddings_processor.video_connector.",
}


class CheckpointAuditError(RuntimeError):
    """A Stage 3 checkpoint does not exactly match its owned components."""

    def __init__(self, message: str, audit: dict[str, Any]) -> None:
        super().__init__(message)
        self.audit = audit


@dataclass(frozen=True)
class CheckpointComponentSpec:
    prefix: str
    expected_state: Mapping[str, Tensor]
    key_normalizer: Callable[[str], str] | None = None


@dataclass(frozen=True)
class CheckpointSnapshot:
    size_bytes: int
    mtime_ns: int
    sha256: str


def _unwrap_module(module: nn.Module) -> nn.Module:
    return getattr(module, "module", module)


def _normalize_peft_adapter_key(key: str) -> str:
    normalized = key
    while normalized.startswith("base_model.model."):
        normalized = normalized.removeprefix("base_model.model.")
    for adapter_key in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
        normalized = normalized.replace(f".{adapter_key}.default.", f".{adapter_key}.")
    return normalized


def _module_parameter_state(module: nn.Module) -> dict[str, Tensor]:
    return {name: parameter for name, parameter in _unwrap_module(module).named_parameters()}


def build_expected_checkpoint_components(
    *,
    transformer: nn.Module,
    embeddings_processor: nn.Module,
    strategy: MultiReferencePlannerStage2Strategy,
) -> dict[str, CheckpointComponentSpec]:
    """Build the exact checkpoint-owned key/shape contract from live modules."""
    strategy_modules = strategy.get_trainable_modules()
    required_strategy_modules = {
        "planner": "planner_tokens",
        "visual_projection": "visual_token_projection",
        "visual_full_encoder": "visual_full_encoder",
    }
    missing_modules = [
        module_name
        for module_name in required_strategy_modules.values()
        if module_name not in strategy_modules
    ]
    if missing_modules:
        raise RuntimeError(f"Stage 3 strategy is missing checkpoint-owned modules: {missing_modules}")

    language_model = strategy._get_language_model()
    return {
        "dit_lora": CheckpointComponentSpec(
            prefix=_COMPONENT_PREFIXES["dit_lora"],
            expected_state=get_peft_model_state_dict(_unwrap_module(transformer)),
            key_normalizer=_normalize_peft_adapter_key,
        ),
        "gemma_lora": CheckpointComponentSpec(
            prefix=_COMPONENT_PREFIXES["gemma_lora"],
            expected_state=get_peft_model_state_dict(_unwrap_module(language_model)),
            key_normalizer=_normalize_peft_adapter_key,
        ),
        "planner": CheckpointComponentSpec(
            prefix=_COMPONENT_PREFIXES["planner"],
            expected_state=_unwrap_module(strategy_modules["planner_tokens"]).state_dict(),
        ),
        "visual_projection": CheckpointComponentSpec(
            prefix=_COMPONENT_PREFIXES["visual_projection"],
            expected_state=_unwrap_module(
                strategy_modules["visual_token_projection"]
            ).state_dict(),
        ),
        "visual_full_encoder": CheckpointComponentSpec(
            prefix=_COMPONENT_PREFIXES["visual_full_encoder"],
            expected_state=_unwrap_module(strategy_modules["visual_full_encoder"]).state_dict(),
        ),
        "connector": CheckpointComponentSpec(
            prefix=_COMPONENT_PREFIXES["connector"],
            # Stage 3 saves trainable connector parameters, not frozen processor state.
            expected_state=_module_parameter_state(embeddings_processor.video_connector),
        ),
    }


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


def checkpoint_snapshot(path: Path) -> CheckpointSnapshot:
    stat_result = path.stat()
    return CheckpointSnapshot(
        size_bytes=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        sha256=_sha256_file(path),
    )


def assert_checkpoint_unchanged(path: Path, snapshot: CheckpointSnapshot) -> None:
    current = checkpoint_snapshot(path)
    if current != snapshot:
        raise RuntimeError(
            "Checkpoint changed while inference runtime was loading: "
            f"before={snapshot}, after={current}"
        )


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


def _checkpoint_structure(path: Path) -> tuple[list[str], dict[str, tuple[int, ...]], dict[str, str]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in keys}
        metadata = dict(handle.metadata() or {})
    return keys, shapes, metadata


def _validate_checkpoint_metadata(path: Path, metadata: Mapping[str, str]) -> int:
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
    return filename_step


def _indexed_shapes(
    *,
    keys: list[str],
    shapes: Mapping[str, tuple[int, ...]],
    spec: CheckpointComponentSpec,
) -> tuple[dict[str, tuple[str, tuple[int, ...]]], list[str]]:
    normalize = spec.key_normalizer or (lambda key: key)
    indexed: dict[str, tuple[str, tuple[int, ...]]] = {}
    duplicates: list[str] = []
    for full_key in keys:
        if not full_key.startswith(spec.prefix):
            continue
        local_key = full_key.removeprefix(spec.prefix)
        normalized = normalize(local_key)
        if normalized in indexed:
            duplicates.extend([indexed[normalized][0], full_key])
            continue
        indexed[normalized] = (full_key, shapes[full_key])
    if duplicates:
        duplicates = sorted(set(duplicates))
    return indexed, duplicates


def audit_checkpoint(
    path: Path,
    *,
    component_specs: Mapping[str, CheckpointComponentSpec] | None = None,
) -> dict[str, Any]:
    """Audit Stage 3 metadata and, when provided, every checkpoint-owned key."""
    keys, shapes, metadata = _checkpoint_structure(path)
    filename_step = _validate_checkpoint_metadata(path, metadata)
    counts = {
        name: sum(1 for key in keys if key.startswith(prefix))
        for name, prefix in _COMPONENT_PREFIXES.items()
    }
    required_missing_keys: list[str] = []
    unexpected_checkpoint_keys: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    expected_counts: dict[str, int] = {}
    component_complete: dict[str, bool] = {}

    if component_specs is None:
        required_missing_keys = [
            f"{_COMPONENT_PREFIXES[name]}*" for name, count in counts.items() if count == 0
        ]
        expected_counts = {name: 1 for name in _COMPONENT_PREFIXES}
        component_complete = {name: counts[name] > 0 for name in _COMPONENT_PREFIXES}
    else:
        if set(component_specs) != set(_COMPONENT_PREFIXES):
            raise ValueError(
                "component_specs must describe exactly the six Stage 3 components; "
                f"got {sorted(component_specs)}"
            )
        consumed_keys: set[str] = set()
        for component, spec in component_specs.items():
            normalize = spec.key_normalizer or (lambda key: key)
            expected_index: dict[str, tuple[str, tuple[int, ...]]] = {}
            for local_key, value in spec.expected_state.items():
                normalized = normalize(local_key)
                if normalized in expected_index:
                    raise RuntimeError(
                        f"Expected {component} state has duplicate normalized key {normalized!r}"
                    )
                expected_index[normalized] = (
                    f"{spec.prefix}{local_key}",
                    tuple(value.shape),
                )
            actual_index, duplicate_keys = _indexed_shapes(
                keys=keys,
                shapes=shapes,
                spec=spec,
            )
            consumed_keys.update(full_key for full_key, _shape in actual_index.values())
            expected_counts[component] = len(expected_index)
            missing_normalized = sorted(set(expected_index) - set(actual_index))
            unexpected_normalized = sorted(set(actual_index) - set(expected_index))
            required_missing_keys.extend(
                expected_index[normalized][0] for normalized in missing_normalized
            )
            unexpected_checkpoint_keys.extend(
                actual_index[normalized][0] for normalized in unexpected_normalized
            )
            unexpected_checkpoint_keys.extend(duplicate_keys)
            component_shape_mismatch = False
            for normalized in sorted(set(expected_index) & set(actual_index)):
                expected_key, expected_shape = expected_index[normalized]
                actual_key, actual_shape = actual_index[normalized]
                if expected_shape != actual_shape:
                    component_shape_mismatch = True
                    shape_mismatches.append(
                        {
                            "component": component,
                            "checkpoint_key": actual_key,
                            "expected_key": expected_key,
                            "checkpoint_shape": list(actual_shape),
                            "expected_shape": list(expected_shape),
                        }
                    )
            component_complete[component] = not (
                missing_normalized
                or unexpected_normalized
                or duplicate_keys
                or component_shape_mismatch
            )
        unexpected_checkpoint_keys.extend(sorted(set(keys) - consumed_keys))

    required_missing_keys = sorted(set(required_missing_keys))
    unexpected_checkpoint_keys = sorted(set(unexpected_checkpoint_keys))
    audit = {
        "checkpoint_step": filename_step,
        "checkpoint_metadata": metadata,
        "component_key_counts": counts,
        "component_expected_key_counts": expected_counts,
        "component_complete": component_complete,
        "required_missing_keys": required_missing_keys,
        "unexpected_checkpoint_keys": unexpected_checkpoint_keys,
        "checkpoint_shape_mismatches": shape_mismatches,
        # Base weights intentionally come from model.model_path/text_encoder_path and
        # are outside the checkpoint-owned comparison above.
        "ignored_base_model_missing_keys": [
            "diffusion_model.<non-LoRA base weights>",
            "text_encoder.<non-LoRA base weights>",
            "embeddings_processor.<non-video-connector base weights>",
        ],
        "ignored_base_model_policy": (
            "Base DiT, Gemma, and embeddings-processor weights are loaded from their "
            "configured base paths and are not required in a Stage 3 adapter checkpoint."
        ),
        "checkpoint_key_count": len(keys),
    }
    if required_missing_keys or unexpected_checkpoint_keys or shape_mismatches:
        raise CheckpointAuditError(
            "Stage 3 checkpoint-owned state mismatch: "
            f"required_missing={required_missing_keys[:20]}, "
            f"unexpected={unexpected_checkpoint_keys[:20]}, "
            f"shape_mismatches={shape_mismatches[:20]}",
            audit,
        )
    return audit


def _checkpoint_flags_from_audit(audit: Mapping[str, Any]) -> dict[str, bool]:
    complete = audit["component_complete"]
    return {
        "dit_lora_checkpoint_loaded": bool(complete["dit_lora"]),
        "gemma_lora_checkpoint_loaded": bool(complete["gemma_lora"]),
        "planner_checkpoint_loaded": bool(complete["planner"]),
        "visual_token_projection_checkpoint_loaded": bool(complete["visual_projection"]),
        "visual_full_encoder_checkpoint_loaded": bool(complete["visual_full_encoder"]),
        "connector_checkpoint_loaded": bool(complete["connector"]),
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

    initial_snapshot = checkpoint_snapshot(checkpoint_path)
    checkpoint_hash = initial_snapshot.sha256
    # Cheap structural/metadata validation before loading the large base models.
    audit_checkpoint(checkpoint_path)
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
    component_specs = build_expected_checkpoint_components(
        transformer=transformer,
        embeddings_processor=embeddings_processor,
        strategy=strategy,
    )
    audit = audit_checkpoint(checkpoint_path, component_specs=component_specs)
    stage2._load_checkpoint_weights(
        checkpoint_path=checkpoint_path,
        transformer=transformer,
        embeddings_processor=embeddings_processor,
        text_encoder=text_encoder,
        strategy=strategy,
    )
    checkpoint_flags = _checkpoint_flags_from_audit(audit)
    assert_checkpoint_unchanged(checkpoint_path, initial_snapshot)

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
            "checkpoint_size_bytes": initial_snapshot.size_bytes,
            "ready_marker_path": str(ready_marker_path) if ready_marker_path else None,
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
