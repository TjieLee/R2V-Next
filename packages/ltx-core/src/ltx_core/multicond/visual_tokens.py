from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class VisualTokenBatch:
    tokens: Tensor
    mask: Tensor


class VisualPlannerTokens(nn.Module):
    """Learnable VLM planner placeholders with a fixed SigLIP-token count."""

    def __init__(
        self,
        *,
        base_tokens: Tensor,
        token_count: int,
        dim: int,
        source_dim: int | None = None,
    ) -> None:
        super().__init__()
        if base_tokens.ndim != 2:
            raise ValueError(f"base_tokens must be [N, D], got {tuple(base_tokens.shape)}")
        if token_count <= 0:
            raise ValueError("token_count must be positive")

        self.token_count = token_count
        self.dim = dim
        self.source_dim = source_dim or dim

        repeats = (token_count + base_tokens.shape[0] - 1) // base_tokens.shape[0]
        init_tokens = base_tokens.detach().float().repeat(repeats, 1)[:token_count]
        if init_tokens.shape[-1] != dim:
            self.input_projection = nn.Linear(init_tokens.shape[-1], dim)
        else:
            self.input_projection = nn.Identity()
        self.input_tokens = nn.Parameter(init_tokens, requires_grad=True)

        self.output_norm = nn.LayerNorm(self.source_dim)
        if self.source_dim != dim:
            self.output_projection = nn.Linear(self.source_dim, dim)
        else:
            self.output_projection = nn.Identity()

    def input_embeddings(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        tokens = self.input_projection(self.input_tokens.to(device=device, dtype=dtype))
        return tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def project_hidden(self, hidden: Tensor) -> Tensor:
        return self.output_projection(self.output_norm(hidden))


def extract_projected_visual_tokens(
    gemma_causal_lm: nn.Module,
    pixel_values: Tensor,
    *,
    image_counts: Tensor | None = None,
) -> VisualTokenBatch:
    """Run frozen Gemma/SigLIP vision tower + projector and return flattened tokens.

    The helper accepts the tensor layouts produced by ``Gemma3Processor`` both
    before and after DataLoader collation: ``[R,C,H,W]``, ``[B,R,C,H,W]`` or
    ``[B,1,R,C,H,W]``. Returned tokens are grouped per training sample as
    ``[B, R * T, D]`` and padded on the token axis when image counts differ.
    """

    gemma_model = getattr(gemma_causal_lm, "model", gemma_causal_lm)
    vision_tower = getattr(gemma_model, "vision_tower", None)
    projector = getattr(gemma_model, "multi_modal_projector", None)
    if vision_tower is None or projector is None:
        raise ValueError("Gemma model must expose vision_tower and multi_modal_projector to build GT visual tokens.")

    pixel_values, batch_size, images_per_sample = _normalize_pixel_values(pixel_values)
    flat_pixels = pixel_values.reshape(batch_size * images_per_sample, *pixel_values.shape[-3:])

    vision_outputs = vision_tower(pixel_values=flat_pixels)
    vision_hidden = _last_hidden_state(vision_outputs)
    projected = _call_projector(projector, vision_hidden)
    projected = _last_hidden_state(projected)

    if projected.ndim != 3:
        raise ValueError(f"Projected visual tokens must be [B*R,T,D], got {tuple(projected.shape)}")

    tokens_per_image = projected.shape[1]
    projected = projected.reshape(batch_size, images_per_sample * tokens_per_image, projected.shape[-1])

    if image_counts is None:
        mask = torch.ones(projected.shape[:2], dtype=torch.bool, device=projected.device)
        return VisualTokenBatch(tokens=projected, mask=mask)

    image_counts = image_counts.to(device=projected.device, dtype=torch.long).clamp(min=0, max=images_per_sample)
    token_counts = image_counts * tokens_per_image
    mask = torch.arange(projected.shape[1], device=projected.device).unsqueeze(0) < token_counts.unsqueeze(1)
    projected = projected * mask.unsqueeze(-1).to(dtype=projected.dtype)
    return VisualTokenBatch(tokens=projected, mask=mask)


def scatter_visual_tokens_into_embeddings(
    *,
    inputs_embeds: Tensor,
    input_ids: Tensor,
    visual_tokens: Tensor,
    image_token_index: int,
    visual_mask: Tensor | None = None,
    planner_placeholder_mask: Tensor | None = None,
) -> Tensor:
    """Replace real Gemma image-token positions with projected visual tokens."""

    image_mask = input_ids == image_token_index
    if planner_placeholder_mask is not None:
        image_mask = image_mask & ~planner_placeholder_mask.to(device=image_mask.device, dtype=torch.bool)

    if visual_mask is None:
        visual_mask = torch.ones(visual_tokens.shape[:2], dtype=torch.bool, device=visual_tokens.device)
    else:
        visual_mask = visual_mask.to(device=visual_tokens.device, dtype=torch.bool)

    expected = image_mask.sum(dim=1)
    available = visual_mask.sum(dim=1)
    if not torch.all(expected == available):
        raise ValueError(
            "Image token count does not match projected SigLIP token count: "
            f"expected mask counts {expected.tolist()}, valid visual tokens {available.tolist()}"
        )

    out = inputs_embeds.clone()
    for batch_index in range(out.shape[0]):
        valid_tokens = visual_tokens[batch_index, visual_mask[batch_index]].to(dtype=out.dtype)
        out[batch_index, image_mask[batch_index]] = valid_tokens
    return out


def _normalize_pixel_values(pixel_values: Tensor) -> tuple[Tensor, int, int]:
    if pixel_values.ndim == 6 and pixel_values.shape[1] == 1:
        pixel_values = pixel_values.squeeze(1)
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)
    if pixel_values.ndim != 5:
        raise ValueError(
            "pixel_values must be [R,C,H,W], [B,R,C,H,W] or [B,1,R,C,H,W], "
            f"got {tuple(pixel_values.shape)}"
        )
    return pixel_values, pixel_values.shape[0], pixel_values.shape[1]


def _last_hidden_state(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    hidden = getattr(value, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    raise ValueError(f"Cannot extract hidden states from {type(value).__name__}")


def _call_projector(projector: nn.Module, hidden: Tensor) -> Tensor:
    try:
        return projector(hidden)
    except TypeError:
        return projector(image_features=hidden)
