"""Checkpoint loading for strict-no-GT semantic/video flow inference."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor, nn

from ltx_core.types import VideoLatentShape, VideoPixelShape
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.model_loader import (
    load_embeddings_processor,
    load_text_encoder,
    load_transformer,
    load_video_vae_decoder,
    load_video_vae_encoder,
)
from ltx_trainer.online_data.constants import IMAGE_TASK
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.training_strategies.semantic_flow import SemanticFlowStrategy

if TYPE_CHECKING:
    from ltx_trainer.online_data.online_batch_encoder import OnlineBatchEncoder


_STEP_PATTERN = re.compile(r"(?:^|_)step_(\d+)(?:\.|$)")
SEMANTIC_STRATEGY_CHECKPOINT_PREFIXES = (
    "training_strategy.semantic_query.",
    "training_strategy.semantic_encoder.",
    "training_strategy.semantic_reconstruction_decoder.",
)
SEMANTIC_TRANSFORMER_CHECKPOINT_PREFIXES = (
    "semantic_token_type_embedding.",
    "semantic_entity_embedding.",
    "semantic_position_adapter.",
    "semantic_norm_out.",
    "semantic_proj_out.",
)


class CheckpointAuditError(RuntimeError):
    """A checkpoint cannot be used as a complete semantic-flow checkpoint."""


@dataclass(frozen=True)
class CheckpointSnapshot:
    size_bytes: int
    mtime_ns: int
    sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_snapshot(path: Path) -> CheckpointSnapshot:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return CheckpointSnapshot(
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=_sha256_file(resolved),
    )


def assert_checkpoint_unchanged(path: Path, snapshot: CheckpointSnapshot) -> None:
    if checkpoint_snapshot(path) != snapshot:
        raise RuntimeError(f"Checkpoint changed while it was being loaded: {path}")


def checkpoint_step(path: Path) -> int:
    match = _STEP_PATTERN.search(path.name)
    return int(match.group(1)) if match else -1


def _load_ready_marker(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Ready marker must be a JSON object: {path}")
    return payload


def resolve_checkpoint(
    *,
    checkpoint: str | None,
    latest_ready_dir: str | None,
) -> tuple[Path, Path | None, dict[str, Any] | None]:
    """Resolve an explicit checkpoint or the newest valid ready marker."""
    if bool(checkpoint) == bool(latest_ready_dir):
        raise ValueError("Specify exactly one of checkpoint or latest_ready_dir")
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, None, None

    directory = Path(str(latest_ready_dir)).expanduser().resolve()
    candidates: list[tuple[int, Path, Path, dict[str, Any]]] = []
    for marker_path in directory.glob("checkpoint_step_*.ready.json"):
        payload = _load_ready_marker(marker_path)
        raw_checkpoint = payload.get("checkpoint_path") or payload.get("weights_path")
        if not raw_checkpoint:
            continue
        candidate = Path(str(raw_checkpoint)).expanduser()
        if not candidate.is_absolute():
            candidate = marker_path.parent / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            continue
        expected_sha = payload.get("checkpoint_sha256")
        if expected_sha and _sha256_file(candidate) != str(expected_sha):
            continue
        candidates.append((checkpoint_step(candidate), candidate, marker_path, payload))
    if not candidates:
        raise FileNotFoundError(f"No valid ready checkpoint found in {directory}")
    _, path, marker_path, payload = max(candidates, key=lambda item: item[0])
    return path, marker_path, payload


def audit_checkpoint(path: Path) -> dict[str, Any]:
    """Return a non-mutating structural summary of one safetensors checkpoint."""
    resolved = path.expanduser().resolve()
    try:
        with safe_open(resolved, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            metadata = dict(handle.metadata() or {})
            shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
    except Exception as exc:
        raise CheckpointAuditError(f"Invalid safetensors checkpoint {resolved}: {exc}") from exc
    semantic_prefixes = SEMANTIC_STRATEGY_CHECKPOINT_PREFIXES + SEMANTIC_TRANSFORMER_CHECKPOINT_PREFIXES
    missing = [prefix for prefix in semantic_prefixes if not any(key.startswith(prefix) for key in keys)]
    if missing:
        raise CheckpointAuditError(f"Checkpoint is missing semantic modules: {missing}")
    return {
        "checkpoint_path": str(resolved),
        "checkpoint_step": checkpoint_step(resolved),
        "checkpoint_key_count": len(keys),
        "checkpoint_keys": keys,
        "checkpoint_shapes": shapes,
        "metadata": metadata,
        "semantic_module_prefixes": list(SEMANTIC_STRATEGY_CHECKPOINT_PREFIXES),
        "semantic_transformer_prefixes": list(SEMANTIC_TRANSFORMER_CHECKPOINT_PREFIXES),
        "required_missing_keys": [],
        "unexpected_checkpoint_keys": [],
    }


def read_checkpoint_metadata(path: Path) -> dict[str, str]:
    """Read safetensors metadata without materializing checkpoint tensors."""
    resolved = path.expanduser().resolve()
    try:
        with safe_open(resolved, framework="pt", device="cpu") as handle:
            return dict(handle.metadata() or {})
    except Exception as exc:
        raise CheckpointAuditError(f"Invalid safetensors checkpoint {resolved}: {exc}") from exc


def checkpoint_contains_semantic_flow_modules(path: Path) -> bool:
    """Return whether a checkpoint header contains any semantic-flow module."""
    resolved = path.expanduser().resolve()
    prefixes = SEMANTIC_STRATEGY_CHECKPOINT_PREFIXES + SEMANTIC_TRANSFORMER_CHECKPOINT_PREFIXES
    try:
        with safe_open(resolved, framework="pt", device="cpu") as handle:
            return any(key.startswith(prefixes) for key in handle.keys())
    except Exception as exc:
        raise CheckpointAuditError(f"Invalid safetensors checkpoint {resolved}: {exc}") from exc


def validate_reference_rope_checkpoint_metadata(
    metadata: dict[str, str],
    *,
    expected_mode: str,
    allow_legacy: bool = False,
) -> None:
    required = {
        "reference_rope_layout_version": "2",
        "reference_rope_mode": expected_mode,
        "reference_rope_temporal_slots": (
            "fixed_after_target" if expected_mode == "appended_time_shifted_width" else "native_overlap"
        ),
        "reference_rope_spatial_shift": (
            "width_adjacent" if expected_mode == "appended_time_shifted_width" else "native_overlap"
        ),
        "semantic_rope_mode": "target_interpolated_8x8",
    }
    present = set(required) & set(metadata)
    if not present:
        if allow_legacy and expected_mode == "native_overlap":
            return
        if allow_legacy:
            raise CheckpointAuditError(
                "Legacy Reference RoPE checkpoints can only be loaded "
                "with reference_rope_mode='native_overlap'; "
                f"current mode is {expected_mode!r}"
            )
    missing = sorted(set(required) - set(metadata))
    if missing:
        raise CheckpointAuditError(
            "Semantic-flow checkpoint is missing Reference RoPE metadata: "
            f"{missing}. Use --allow-legacy-reference-rope only for an explicit legacy migration test."
        )
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise CheckpointAuditError(f"Semantic-flow Reference RoPE metadata mismatch: {mismatches}")


@dataclass
class OnlineInferenceRuntime:
    cfg: LtxTrainerConfig
    transformer: nn.Module
    embeddings_processor: nn.Module
    text_encoder: nn.Module
    vae_encoder: nn.Module
    vae_decoder: nn.Module | None
    strategy: SemanticFlowStrategy
    online_encoder: OnlineBatchEncoder
    config_path: Path
    checkpoint_path: Path
    ready_marker_path: Path | None
    checkpoint_audit: dict[str, Any]
    device: torch.device
    dtype: torch.dtype
    last_generation_geometry: dict[str, Any] = field(default_factory=dict)

    def connector_conditions(self, raw_conditions: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply the frozen LTX connector exactly once to Gemma prefix features."""
        video_features = raw_conditions["video_prompt_embeds"]
        audio_features = raw_conditions.get("audio_prompt_embeds")
        mask = raw_conditions["prompt_attention_mask"]
        additive = (mask.to(torch.int64) - 1).to(video_features.dtype).reshape(
            mask.shape[0],
            1,
            -1,
            mask.shape[-1],
        ) * torch.finfo(video_features.dtype).max
        with torch.inference_mode():
            video, audio, processed_mask = self.embeddings_processor.create_embeddings(
                video_features,
                audio_features,
                additive,
            )
        result = {
            "video_prompt_embeds": video,
            "prompt_attention_mask": processed_mask,
        }
        if audio is not None:
            result["audio_prompt_embeds"] = audio
        return result

    def generate_latents(
        self,
        encoded: dict[str, Any],
        *,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        num_inference_steps: int,
    ) -> tuple[Tensor, Tensor]:
        """Generate semantic and video latents without accepting any target-derived input."""
        self.last_generation_geometry = {}
        forbidden = {"target_pixels", "latents", "semantic_teacher_inputs", "evidence_tokens"}
        leaked = sorted(forbidden & encoded.keys())
        if leaked:
            raise RuntimeError(f"Strict-no-GT inference received forbidden keys: {leaked}")
        pixel_shape = VideoPixelShape(
            batch=1,
            frames=int(num_frames),
            height=int(height),
            width=int(width),
            fps=float(fps),
        )
        target_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
        task = str(encoded["task"])
        semantic_frames = 1 if task == IMAGE_TASK else max(
            1,
            round(int(num_frames) * self.strategy.config.anchor_frame_ratio),
        )
        state = self.strategy.prepare_inference_state(
            conditions=self.connector_conditions(encoded["conditions"]),
            reference_latents=encoded["reference_latents"],
            target_shape=target_shape,
            semantic_frame_count=semantic_frames,
            pixel_frame_count=int(num_frames),
            fps=float(fps),
            seed=seed,
        )
        target_start = state.sequence_offsets["semantic_end"]
        target_end = state.sequence_offsets["target_end"]
        self.last_generation_geometry = {
            "fps": float(fps),
            "target_latent_shape": list(state.target_shape.to_torch_shape()),
            "target_token_count": target_end - target_start,
            "target_position_count": int(state.modality.positions[:, :, target_start:target_end].shape[2]),
            "reference_rope_mode": self.strategy.config.reference_rope_mode,
        }
        return self.strategy.denoise_joint(
            transformer=self.transformer,
            state=state,
            num_inference_steps=num_inference_steps,
        )


def _load_config(path: Path) -> LtxTrainerConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Trainer config must be a mapping: {path}")
    return LtxTrainerConfig(**payload)


def load_online_inference_runtime(
    *,
    config_path: str | Path,
    checkpoint_path: Path,
    ready_marker_path: Path | None,
    ready_marker: dict[str, Any] | None,
    device: torch.device,
    dtype: torch.dtype,
    load_vae_decoder: bool = True,
    allow_legacy_reference_rope: bool = False,
) -> OnlineInferenceRuntime:
    """Load a complete full-DiT semantic checkpoint and frozen encoders once."""
    config = Path(config_path).expanduser().resolve()
    cfg = _load_config(config)
    if cfg.model.training_mode != "full":
        raise ValueError("semantic-flow inference requires model.training_mode='full'")
    if cfg.data.encoding_mode != "online" or cfg.data.online_encoding is None:
        raise ValueError("semantic-flow inference requires online encoding")
    strategy = get_training_strategy(cfg.training_strategy)
    if not isinstance(strategy, SemanticFlowStrategy):
        raise TypeError(f"Expected SemanticFlowStrategy, got {type(strategy).__name__}")

    snapshot = checkpoint_snapshot(checkpoint_path)
    audit = audit_checkpoint(checkpoint_path)
    validate_reference_rope_checkpoint_metadata(
        audit["metadata"],
        expected_mode=strategy.config.reference_rope_mode,
        allow_legacy=allow_legacy_reference_rope,
    )
    if ready_marker is not None:
        expected_sha = ready_marker.get("checkpoint_sha256")
        if expected_sha and snapshot.sha256 != str(expected_sha):
            raise RuntimeError("Checkpoint SHA256 no longer matches its ready marker")

    transformer = load_transformer(cfg.model.model_path, device="cpu", dtype=dtype).to(device=device)
    embeddings_processor = load_embeddings_processor(cfg.model.model_path, device=device, dtype=dtype)
    text_encoder = load_text_encoder(
        gemma_model_path=cfg.model.text_encoder_path,
        device=device,
        dtype=dtype,
        load_in_8bit=cfg.acceleration.load_text_encoder_in_8bit,
    )
    strategy.attach_models(
        transformer=transformer,
        embeddings_processor=embeddings_processor,
        text_encoder=text_encoder,
    )
    state = load_file(checkpoint_path, device="cpu")
    strategy.load_extra_checkpoint_state_dict(state)
    transformer_state = {
        key: value
        for key, value in state.items()
        if not key.startswith("training_strategy.")
        and not key.startswith("embeddings_processor.")
        and not key.startswith("text_encoder.")
    }
    try:
        transformer.load_state_dict(transformer_state, strict=True)
    except RuntimeError as exc:
        raise CheckpointAuditError(f"Full transformer checkpoint is incomplete: {exc}") from exc

    transformer.requires_grad_(False).eval()
    embeddings_processor.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    for module in strategy.get_trainable_modules().values():
        module.requires_grad_(False).eval()
    vae_encoder = load_video_vae_encoder(cfg.model.model_path, device=device, dtype=dtype)
    vae_encoder.requires_grad_(False).eval()
    from ltx_trainer.online_data.online_batch_encoder import OnlineBatchEncoder  # noqa: PLC0415

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
    assert_checkpoint_unchanged(checkpoint_path, snapshot)
    audit.update(
        {
            "checkpoint_sha256": snapshot.sha256,
            "checkpoint_size_bytes": snapshot.size_bytes,
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
        config_path=config,
        checkpoint_path=checkpoint_path,
        ready_marker_path=ready_marker_path,
        checkpoint_audit=audit,
        device=device,
        dtype=dtype,
    )


__all__ = [
    "CheckpointAuditError",
    "CheckpointSnapshot",
    "OnlineInferenceRuntime",
    "assert_checkpoint_unchanged",
    "audit_checkpoint",
    "checkpoint_contains_semantic_flow_modules",
    "checkpoint_snapshot",
    "checkpoint_step",
    "load_online_inference_runtime",
    "read_checkpoint_metadata",
    "resolve_checkpoint",
    "validate_reference_rope_checkpoint_metadata",
]
