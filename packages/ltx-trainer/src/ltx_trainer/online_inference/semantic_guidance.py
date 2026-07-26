"""Semantic/video joint guidance primitives for strict-no-GT inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import torch
from torch import Tensor

from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)

GuidanceMode = Literal["positive_ref", "debiased_ref"]


@dataclass(frozen=True)
class SemanticGuidanceConfig:
    """Guidance controls shared by semantic and target-video tokens."""

    guidance_mode: GuidanceMode = "positive_ref"
    guidance_scale: float = 4.0
    ref_guidance_scale: float = 1.0
    guidance_rescale: float = 0.7
    stg_scale: float = 0.0
    stg_blocks: tuple[int, ...] = (28,)

    def __post_init__(self) -> None:
        if self.guidance_mode not in {"positive_ref", "debiased_ref"}:
            raise ValueError(
                "guidance_mode must be 'positive_ref' or 'debiased_ref', "
                f"got {self.guidance_mode!r}"
            )
        values = {
            "guidance_scale": self.guidance_scale,
            "ref_guidance_scale": self.ref_guidance_scale,
            "guidance_rescale": self.guidance_rescale,
            "stg_scale": self.stg_scale,
        }
        non_finite = [name for name, value in values.items() if not math.isfinite(float(value))]
        if non_finite:
            raise ValueError(f"Guidance scales must be finite: {non_finite}")
        if self.guidance_scale < 1.0:
            raise ValueError("guidance_scale must be >= 1.0")
        if self.ref_guidance_scale < 0.0:
            raise ValueError("ref_guidance_scale must be >= 0.0")
        if not 0.0 <= self.guidance_rescale <= 1.0:
            raise ValueError("guidance_rescale must be between 0.0 and 1.0")
        if self.stg_scale < 0.0:
            raise ValueError("stg_scale must be >= 0.0")
        if not self.stg_blocks:
            raise ValueError("stg_blocks must contain at least one zero-based block index")
        if any(block < 0 for block in self.stg_blocks):
            raise ValueError("stg_blocks must contain non-negative zero-based indices")
        if len(set(self.stg_blocks)) != len(self.stg_blocks):
            raise ValueError("stg_blocks must be deduplicated")

    @property
    def need_negative(self) -> bool:
        return self.guidance_scale != 1.0

    @property
    def need_reference(self) -> bool:
        return self.ref_guidance_scale != 0.0

    @property
    def need_stg(self) -> bool:
        return self.stg_scale != 0.0

    @property
    def need_control_pair(self) -> bool:
        return (
            self.guidance_mode == "debiased_ref"
            and (self.need_negative or self.need_reference)
        )

    @property
    def need_reference_comparison(self) -> bool:
        if self.guidance_mode == "positive_ref":
            return self.need_reference
        return self.need_control_pair

    @property
    def transformer_forwards_per_step(self) -> int:
        if self.guidance_mode == "positive_ref":
            return (
                1
                + int(self.need_negative)
                + int(self.need_reference)
                + int(self.need_stg)
            )
        return (
            1
            + int(self.need_negative)
            + 2 * int(self.need_control_pair)
            + int(self.need_stg)
        )

    @property
    def enabled_branches(self) -> tuple[str, ...]:
        branches = ["P"]
        if self.need_negative:
            branches.append("N")
        if self.guidance_mode == "positive_ref":
            if self.need_reference:
                branches.append("Q")
        elif self.need_control_pair:
            branches.extend(("R", "U"))
        if self.need_stg:
            branches.append("S")
        return tuple(branches)

    def metadata(self, *, negative_prompt: str | None) -> dict[str, Any]:
        metadata = {
            "guidance_mode": self.guidance_mode,
            "guidance_enabled": self.transformer_forwards_per_step > 1 or self.guidance_rescale > 0.0,
            "negative_prompt": negative_prompt if self.need_negative else None,
            "guidance_scale": self.guidance_scale,
            "ref_guidance_scale": self.ref_guidance_scale,
            "guidance_rescale": self.guidance_rescale,
            "stg_scale": self.stg_scale,
            "stg_blocks_zero_based": list(self.stg_blocks),
            "stg_layers_one_based": [block + 1 for block in self.stg_blocks],
            "transformer_forwards_per_step": self.transformer_forwards_per_step,
            "enabled_guidance_branches": list(self.enabled_branches),
            "joint_guided_span": "semantic_and_target",
            "rescale_statistics_segment": "target_video",
            "rescale_application_segment": "semantic_and_target",
        }
        if self.guidance_mode == "positive_ref":
            metadata.update(
                {
                    "cfg_formula": "N + cfg*(P-N), P/N share reference images and reference latents",
                    "ref_formula": "ref*(P-Q)",
                    "stg_formula": (
                        "stg*(P-S), S skips video self-attention at zero-based block "
                        + ",".join(str(block) for block in self.stg_blocks)
                    ),
                    "P_text": "positive",
                    "P_vlm_references": "present",
                    "P_reference_latents": "present",
                    "N_text": "negative",
                    "N_vlm_references": "present",
                    "N_reference_latents": "present",
                    "Q_text": "positive",
                    "Q_vlm_references": "absent",
                    "Q_reference_latents": "absent",
                    "Q_prompt": "positive",
                    "Q_reference_vlm_images": "absent",
                    "reference_guidance_training_match": "drop_reference_all",
                }
            )
        else:
            metadata.update(
                {
                    "cfg_formula": "P + (cfg-1)*((P-N)-(R-U))",
                    "semantic_delta_formula": "(P-N)-(R-U)",
                    "control_delta_formula": "R-U",
                    "ref_formula": "ref*(R-U)",
                    "stg_formula": "stg*(P-S)",
                    "control_main_effect_in_cfg": "subtracted",
                    "P_text": "positive",
                    "P_vlm_references": "present",
                    "P_reference_latents": "present",
                    "N_text": "negative",
                    "N_vlm_references": "absent",
                    "N_reference_latents": "absent",
                    "R_text": "empty",
                    "R_vlm_references": "present",
                    "R_reference_latents": "present",
                    "U_text": "drop_all_zero_conditions",
                    "U_vlm_references": "absent",
                    "U_reference_latents": "absent",
                    "reference_guidance_training_match": "drop_text_vs_drop_all",
                }
            )
        return metadata


@dataclass(frozen=True)
class SemanticGuidanceStateBundle:
    """Condition branches that share one canonical generated-token trajectory."""

    positive: Any
    negative: Any | None = None
    no_reference: Any | None = None
    empty_reference: Any | None = None
    empty_no_reference: Any | None = None


def parse_stg_blocks(value: str | Iterable[int]) -> tuple[int, ...]:
    """Parse stable, deduplicated zero-based STG block indices."""
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if not parts or any(not part for part in parts):
            raise ValueError("stg_blocks must be a comma-separated list of integers")
        try:
            candidates = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid stg_blocks value {value!r}") from exc
    else:
        candidates = [int(block) for block in value]
    blocks: list[int] = []
    for block in candidates:
        if block < 0:
            raise ValueError("STG block indices must be non-negative")
        if block not in blocks:
            blocks.append(block)
    if not blocks:
        raise ValueError("At least one STG block index is required")
    return tuple(blocks)


def validate_stg_blocks(blocks: tuple[int, ...], *, transformer_block_count: int) -> None:
    if transformer_block_count < 1:
        raise ValueError("transformer must expose at least one transformer block")
    invalid = [block for block in blocks if block >= transformer_block_count]
    if invalid:
        raise ValueError(
            f"STG block indices {invalid} are outside [0, {transformer_block_count})"
        )


def build_stg_perturbation(
    blocks: tuple[int, ...],
    *,
    batch_size: int,
) -> BatchedPerturbationConfig:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    perturbation = Perturbation(
        type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
        blocks=list(blocks),
    )
    return BatchedPerturbationConfig(
        [
            PerturbationConfig(perturbations=[perturbation])
            for _ in range(batch_size)
        ]
    )


def _sigma_view(sigma: Tensor, value: Tensor) -> Tensor:
    if sigma.ndim != 1 or sigma.shape[0] != value.shape[0]:
        raise ValueError(
            f"sigma must be [B] matching value batch, got {tuple(sigma.shape)} "
            f"for {tuple(value.shape)}"
        )
    if not torch.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError("sigma must be finite and strictly positive during guided denoising")
    return sigma.to(device=value.device, dtype=value.dtype).view(
        sigma.shape[0],
        *([1] * (value.ndim - 1)),
    )


def velocity_to_denoised(current: Tensor, velocity: Tensor, sigma: Tensor) -> Tensor:
    if current.shape != velocity.shape:
        raise ValueError(
            f"current and velocity shapes differ: {tuple(current.shape)} != {tuple(velocity.shape)}"
        )
    return current - _sigma_view(sigma, current) * velocity


def denoised_to_velocity(current: Tensor, denoised: Tensor, sigma: Tensor) -> Tensor:
    if current.shape != denoised.shape:
        raise ValueError(
            f"current and denoised shapes differ: {tuple(current.shape)} != {tuple(denoised.shape)}"
        )
    return (current - denoised) / _sigma_view(sigma, current)


def combine_guided_denoised(
    *,
    positive: Tensor,
    config: SemanticGuidanceConfig,
    negative: Tensor | None = None,
    no_reference: Tensor | None = None,
    empty_reference: Tensor | None = None,
    empty_no_reference: Tensor | None = None,
    stg: Tensor | None = None,
) -> Tensor:
    """Apply CFG, reference-latent guidance, and STG in denoised space."""
    expected = positive.shape
    supplied = {
        "negative": negative,
        "no_reference": no_reference,
        "empty_reference": empty_reference,
        "empty_no_reference": empty_no_reference,
        "stg": stg,
    }
    mismatched = {
        name: tuple(value.shape)
        for name, value in supplied.items()
        if value is not None and value.shape != expected
    }
    if mismatched:
        raise ValueError(f"Guidance branch shapes differ from positive {tuple(expected)}: {mismatched}")
    if config.guidance_mode == "positive_ref":
        if config.need_negative:
            if negative is None:
                raise ValueError("negative denoised prediction is required when CFG is enabled")
            guided = negative + config.guidance_scale * (positive - negative)
        else:
            guided = positive
        if config.need_reference:
            if no_reference is None:
                raise ValueError("Q denoised prediction is required for reference guidance")
            guided = guided + config.ref_guidance_scale * (positive - no_reference)
        if config.need_stg:
            if stg is None:
                raise ValueError("STG denoised prediction is required when STG is enabled")
            guided = guided + config.stg_scale * (positive - stg)
        return guided

    guided = positive
    control_delta = None
    if config.need_control_pair:
        if empty_reference is None or empty_no_reference is None:
            raise ValueError(
                "R and U denoised predictions are required "
                "for debiased reference guidance"
            )
        control_delta = empty_reference - empty_no_reference

    if config.need_negative:
        if negative is None:
            raise ValueError("negative denoised prediction is required when CFG is enabled")
        assert control_delta is not None
        semantic_delta = positive - negative - control_delta
        guided = guided + (config.guidance_scale - 1.0) * semantic_delta

    if config.need_reference:
        assert control_delta is not None
        guided = guided + config.ref_guidance_scale * control_delta

    if config.need_stg:
        if stg is None:
            raise ValueError("STG denoised prediction is required when STG is enabled")
        guided = guided + config.stg_scale * (positive - stg)
    return guided


def rescale_guided_denoised(
    *,
    positive_generated: Tensor,
    guided_generated: Tensor,
    semantic_token_count: int,
    guidance_rescale: float,
) -> tuple[Tensor, Tensor]:
    """Measure target-video std and apply one factor to semantic plus video."""
    if positive_generated.shape != guided_generated.shape:
        raise ValueError("positive and guided generated spans must have identical shapes")
    if not 0 <= semantic_token_count < positive_generated.shape[1]:
        raise ValueError("semantic_token_count must leave a non-empty target-video segment")
    if not 0.0 <= guidance_rescale <= 1.0:
        raise ValueError("guidance_rescale must be between 0.0 and 1.0")
    batch_size = positive_generated.shape[0]
    if guidance_rescale == 0.0:
        factor = torch.ones(batch_size, device=guided_generated.device, dtype=torch.float32)
        return guided_generated, factor
    positive_video = positive_generated[:, semantic_token_count:].float().reshape(batch_size, -1)
    guided_video = guided_generated[:, semantic_token_count:].float().reshape(batch_size, -1)
    std_positive = positive_video.std(dim=1, correction=0)
    std_guided = guided_video.std(dim=1, correction=0).clamp_min(1.0e-8)
    factor = guidance_rescale * (std_positive / std_guided) + (1.0 - guidance_rescale)
    scaled = guided_generated.float() * factor.view(
        batch_size,
        *([1] * (guided_generated.ndim - 1)),
    )
    return scaled.to(dtype=guided_generated.dtype), factor


__all__ = [
    "GuidanceMode",
    "SemanticGuidanceConfig",
    "SemanticGuidanceStateBundle",
    "build_stg_perturbation",
    "combine_guided_denoised",
    "denoised_to_velocity",
    "parse_stg_blocks",
    "rescale_guided_denoised",
    "validate_stg_blocks",
    "velocity_to_denoised",
]
