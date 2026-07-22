"""Gemma 3 full/sliding attention mask helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Gemma3AttentionMasks:
    full_attention: Tensor
    sliding_attention: Tensor

    def as_mapping(self) -> dict[str, Tensor]:
        return {
            "full_attention": self.full_attention,
            "sliding_attention": self.sliding_attention,
        }


def _to_additive_mask(visibility: Tensor, dtype: torch.dtype) -> Tensor:
    bias = torch.zeros_like(visibility, dtype=dtype)
    bias.masked_fill_(~visibility, torch.finfo(dtype).min)
    return bias.unsqueeze(1)


def build_native_sliding_visibility(
    *,
    sequence_length: int,
    sliding_window: int,
    device: torch.device,
) -> Tensor:
    if sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    positions = torch.arange(sequence_length, device=device)
    query = positions[:, None]
    key = positions[None, :]
    return (key <= query) & ((query - key) < int(sliding_window))


def build_gemma3_attention_masks(
    *,
    valid_token_mask: Tensor,
    image_token_mask: Tensor,
    custom_visibility: Tensor,
    sliding_window: int,
    dtype: torch.dtype,
) -> Gemma3AttentionMasks:
    """Build additive masks for Gemma 3 full and sliding attention layers."""
    if valid_token_mask.ndim != 2:
        raise ValueError("valid_token_mask must be [B,T]")
    if image_token_mask.shape != valid_token_mask.shape:
        raise ValueError("image_token_mask must match valid_token_mask")
    if custom_visibility.ndim != 3:
        raise ValueError("custom_visibility must be [B,T,T]")
    batch_size, sequence_length = valid_token_mask.shape
    if custom_visibility.shape != (batch_size, sequence_length, sequence_length):
        raise ValueError("custom_visibility must match valid_token_mask")
    valid = valid_token_mask.to(dtype=torch.bool)
    image = image_token_mask.to(device=valid.device, dtype=torch.bool) & valid
    visibility = custom_visibility.to(device=valid.device, dtype=torch.bool)
    visibility = visibility & valid[:, :, None] & valid[:, None, :]

    image_pair = image[:, :, None] & image[:, None, :]
    full_visibility = visibility
    positions = torch.arange(sequence_length, device=valid.device)
    query = positions[:, None]
    key = positions[None, :]
    local_window = (query - key).abs() < int(sliding_window)
    causal_local = (key <= query) & local_window
    custom_image_pair = visibility & image_pair
    native_sliding_visibility = causal_local.unsqueeze(0) | (custom_image_pair & local_window.unsqueeze(0))
    sliding_visibility = full_visibility & native_sliding_visibility
    return Gemma3AttentionMasks(
        full_attention=_to_additive_mask(full_visibility, dtype),
        sliding_attention=_to_additive_mask(sliding_visibility, dtype),
    )


def resolve_gemma3_sliding_window(language_model: object, *, default: int = 1024) -> int:
    module = getattr(language_model, "module", language_model)
    config = getattr(module, "config", None)
    value = getattr(config, "sliding_window", None)
    if value is None:
        return int(default)
    return int(value)
