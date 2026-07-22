"""Native Gemma vision-tower extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class VisualTokenBatch:
    tokens: Tensor
    mask: Tensor


def module_compute_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype] | None:
    """Return the first floating parameter/buffer device and dtype, including wrappers."""
    candidates: list[nn.Module] = []
    current = module
    seen: set[int] = set()
    while id(current) not in seen:
        candidates.append(current)
        seen.add(id(current))
        wrapped = getattr(current, "module", None)
        if not isinstance(wrapped, nn.Module):
            break
        current = wrapped
    for candidate in candidates:
        for parameter in candidate.parameters():
            if parameter.is_floating_point():
                return parameter.device, parameter.dtype
        for buffer in candidate.buffers():
            if buffer.is_floating_point():
                return buffer.device, buffer.dtype
    return None


def _unwrapped_module_name(module: nn.Module) -> str:
    current = module
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "module", None)
        if not isinstance(wrapped, nn.Module):
            break
        current = wrapped
    return type(current).__name__


def extract_projected_visual_tokens(
    gemma_causal_lm: nn.Module,
    pixel_values: Tensor,
    *,
    image_counts: Tensor | None = None,
    dtype_diagnostics: dict[str, str] | None = None,
) -> VisualTokenBatch:
    """Run frozen Gemma native vision tower/projector and group tokens per sample."""
    gemma_model = getattr(gemma_causal_lm, "model", gemma_causal_lm)
    vision_tower = getattr(gemma_model, "vision_tower", None)
    projector = getattr(gemma_model, "multi_modal_projector", None)
    if vision_tower is None or projector is None:
        raise ValueError("Gemma model must expose vision_tower and multi_modal_projector")

    pixel_values, batch_size, images_per_sample = _normalize_pixel_values(pixel_values)
    flat_pixels = pixel_values.reshape(batch_size * images_per_sample, *pixel_values.shape[-3:])
    vision_compute = module_compute_device_dtype(vision_tower)
    if vision_compute is not None:
        flat_pixels = flat_pixels.to(device=vision_compute[0], dtype=vision_compute[1])
    if dtype_diagnostics is not None:
        dtype_diagnostics["vision_module"] = _unwrapped_module_name(vision_tower)
        dtype_diagnostics["vision_input_dtype"] = str(flat_pixels.dtype)
        dtype_diagnostics["vision_input_device"] = str(flat_pixels.device)
    vision_hidden = _last_hidden_state(vision_tower(pixel_values=flat_pixels))
    if dtype_diagnostics is not None:
        dtype_diagnostics["vision_output_dtype"] = str(vision_hidden.dtype)
        dtype_diagnostics["vision_output_device"] = str(vision_hidden.device)

    projector_compute = module_compute_device_dtype(projector)
    if projector_compute is not None:
        vision_hidden = vision_hidden.to(device=projector_compute[0], dtype=projector_compute[1])
    if dtype_diagnostics is not None:
        dtype_diagnostics["projector_module"] = _unwrapped_module_name(projector)
        dtype_diagnostics["projector_input_dtype"] = str(vision_hidden.dtype)
        dtype_diagnostics["projector_input_device"] = str(vision_hidden.device)
    projected = _last_hidden_state(_call_projector(projector, vision_hidden))
    if dtype_diagnostics is not None:
        dtype_diagnostics["projector_output_dtype"] = str(projected.dtype)
        dtype_diagnostics["projector_output_device"] = str(projected.device)
    if projected.ndim != 3:
        raise ValueError(f"Projected visual tokens must be [B*R,T,D], got {tuple(projected.shape)}")

    tokens_per_image = projected.shape[1]
    projected = projected.reshape(batch_size, images_per_sample * tokens_per_image, projected.shape[-1])
    if image_counts is None:
        mask = torch.ones(projected.shape[:2], dtype=torch.bool, device=projected.device)
        return VisualTokenBatch(projected, mask)
    image_counts = image_counts.to(device=projected.device, dtype=torch.long).clamp(0, images_per_sample)
    token_counts = image_counts * tokens_per_image
    mask = torch.arange(projected.shape[1], device=projected.device).unsqueeze(0) < token_counts.unsqueeze(1)
    return VisualTokenBatch(projected * mask.unsqueeze(-1).to(projected.dtype), mask)


def scatter_visual_tokens_into_embeddings(
    *,
    inputs_embeds: Tensor,
    input_ids: Tensor,
    visual_tokens: Tensor,
    image_token_index: int,
    visual_mask: Tensor | None = None,
) -> Tensor:
    """Replace native Gemma image-token positions with projected visual tokens."""
    image_mask = input_ids == image_token_index
    if visual_mask is None:
        visual_mask = torch.ones(visual_tokens.shape[:2], dtype=torch.bool, device=visual_tokens.device)
    else:
        visual_mask = visual_mask.to(device=visual_tokens.device, dtype=torch.bool)
    expected = image_mask.sum(dim=1)
    available = visual_mask.sum(dim=1)
    if not torch.all(expected == available):
        raise ValueError(
            "Image token count does not match projected native visual token count: "
            f"expected={expected.tolist()}, available={available.tolist()}"
        )
    output = inputs_embeds.clone()
    for batch_index in range(output.shape[0]):
        valid = visual_tokens[batch_index, visual_mask[batch_index]].to(dtype=output.dtype)
        output[batch_index, image_mask[batch_index]] = valid
    return output


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
