"""Explicit trainability and checkpoint contract for the Phase 2 video bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

PHASE2_BRIDGE_CHECKPOINT_PREFIXES = (
    "embeddings_processor.feature_extractor.aggregate_embed.",
    "embeddings_processor.feature_extractor.video_aggregate_embed.",
    "embeddings_processor.video_connector.",
)


@dataclass(frozen=True)
class Phase2BridgeParameter:
    name: str
    parameter: nn.Parameter


def _unwrap(module: nn.Module) -> nn.Module:
    return getattr(module, "module", module)


def resolve_phase2_bridge_modules(embeddings_processor: nn.Module) -> dict[str, nn.Module]:
    """Resolve the exact modules between frozen Gemma states and video DiT context."""
    processor = _unwrap(embeddings_processor)
    feature_extractor = getattr(processor, "feature_extractor", None)
    video_connector = getattr(processor, "video_connector", None)
    if not isinstance(feature_extractor, nn.Module):
        raise RuntimeError("Phase 2 requires embeddings_processor.feature_extractor")
    if not isinstance(video_connector, nn.Module):
        raise RuntimeError("Phase 2 requires embeddings_processor.video_connector")
    if hasattr(feature_extractor, "video_aggregate_embed"):
        projection_name = "video_aggregate_embed"
        projection = feature_extractor.video_aggregate_embed
    elif hasattr(feature_extractor, "aggregate_embed"):
        projection_name = "aggregate_embed"
        projection = feature_extractor.aggregate_embed
    else:
        raise RuntimeError(
            "Phase 2 feature extractor is unsupported: expected video_aggregate_embed or aggregate_embed"
        )
    if not isinstance(projection, nn.Module):
        raise RuntimeError("Phase 2 feature extractor projection is not an nn.Module")
    return {
        f"feature_extractor.{projection_name}": projection,
        "video_connector": video_connector,
    }


def phase2_bridge_parameters(embeddings_processor: nn.Module) -> list[Phase2BridgeParameter]:
    """Return the explicit video-only bridge allowlist in stable parameter-name order."""
    modules = resolve_phase2_bridge_modules(embeddings_processor)
    parameters: list[Phase2BridgeParameter] = []
    projection_name = next(name for name in modules if name.startswith("feature_extractor."))
    for name, parameter in _unwrap(modules[projection_name]).named_parameters():
        parameters.append(
            Phase2BridgeParameter(
                name=f"{projection_name}.{name}",
                parameter=parameter,
            )
        )
    for name, parameter in _unwrap(modules["video_connector"]).named_parameters():
        parameters.append(
            Phase2BridgeParameter(
                name=f"video_connector.{name}",
                parameter=parameter,
            )
        )
    if not parameters:
        raise RuntimeError("Phase 2 conditioning bridge allowlist resolved to zero parameters")
    names = [item.name for item in parameters]
    if len(names) != len(set(names)):
        raise RuntimeError("Phase 2 conditioning bridge contains duplicate parameter names")
    parameter_ids = [id(item.parameter) for item in parameters]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("Phase 2 conditioning bridge contains overlapping parameters")
    return parameters


def configure_phase2_bridge_trainability(embeddings_processor: nn.Module) -> list[Phase2BridgeParameter]:
    """Freeze the processor, then enable only the audited video bridge allowlist."""
    processor = _unwrap(embeddings_processor)
    processor.requires_grad_(False)
    parameters = phase2_bridge_parameters(processor)
    for item in parameters:
        item.parameter.requires_grad_(True)
        if item.parameter.is_floating_point() and item.parameter.dtype != torch.float32:
            item.parameter.data = item.parameter.data.to(dtype=torch.float32)

    audio_connector = getattr(processor, "audio_connector", None)
    if isinstance(audio_connector, nn.Module) and any(
        parameter.requires_grad for parameter in audio_connector.parameters()
    ):
        raise RuntimeError("Phase 2 audio connector must remain frozen")
    scalar = [item.name for item in parameters if item.parameter.ndim == 0]
    if scalar:
        raise RuntimeError(f"Phase 2 bridge has FSDP-incompatible scalar parameters: {scalar}")
    return parameters


def expected_phase2_bridge_state(embeddings_processor: nn.Module) -> dict[str, nn.Parameter]:
    return {
        f"embeddings_processor.{item.name}": item.parameter for item in phase2_bridge_parameters(embeddings_processor)
    }


def validate_and_load_phase2_bridge_state(
    embeddings_processor: nn.Module,
    state_dict: dict[str, Tensor],
) -> int:
    """Strictly validate and copy every Phase 2 bridge parameter."""
    expected = expected_phase2_bridge_state(embeddings_processor)
    supplied = {key: value for key, value in state_dict.items() if key.startswith("embeddings_processor.")}
    missing = sorted(set(expected) - set(supplied))
    unexpected = sorted(set(supplied) - set(expected))
    shape_mismatches = [
        (key, tuple(supplied[key].shape), tuple(expected[key].shape))
        for key in sorted(set(expected) & set(supplied))
        if tuple(supplied[key].shape) != tuple(expected[key].shape)
    ]
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Phase 2 conditioning bridge checkpoint is incomplete: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, "
            f"shape_mismatches={shape_mismatches[:20]}"
        )
    with torch.no_grad():
        for key, parameter in expected.items():
            parameter.copy_(supplied[key].to(device=parameter.device, dtype=parameter.dtype))
    different = [
        key
        for key, parameter in expected.items()
        if not torch.equal(
            parameter.detach().cpu().to(dtype=supplied[key].dtype),
            supplied[key].detach().cpu(),
        )
    ]
    if different:
        raise RuntimeError(f"Phase 2 bridge tensors differ after loading: {different[:20]}")
    return len(expected)


def phase2_bridge_audit_rows(embeddings_processor: nn.Module) -> list[dict[str, Any]]:
    allowed = {item.name for item in phase2_bridge_parameters(embeddings_processor)}
    processor = _unwrap(embeddings_processor)
    rows: list[dict[str, Any]] = []
    for name, parameter in processor.named_parameters():
        rows.append(
            {
                "parameter_name": f"embeddings_processor.{name}",
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "requires_grad": bool(parameter.requires_grad),
                "owner_group": "bridge" if name in allowed else "frozen_other",
            }
        )
    return rows
