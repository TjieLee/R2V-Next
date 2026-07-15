"""Fixed-capacity visual-token packing shared by Stage 1/2/3 online data."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.online_data.constants import (
    MAX_VLM_FRAMES,
    TOKENS_PER_FRAME,
    VISUAL_TOKEN_CAPACITY,
)


def pack_visual_tokens(
    tokens: Tensor,
    *,
    valid_frames: int,
    capacity: int = VISUAL_TOKEN_CAPACITY,
    tokens_per_frame: int = TOKENS_PER_FRAME,
) -> tuple[Tensor, Tensor]:
    """Pad real SigLIP tokens to capacity without repeating image features."""
    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [B,K,D], got {tuple(tokens.shape)}")
    if not 1 <= valid_frames <= MAX_VLM_FRAMES:
        raise ValueError(f"valid_frames must be in [1,{MAX_VLM_FRAMES}], got {valid_frames}")
    expected = valid_frames * tokens_per_frame
    if tokens.shape[1] != expected:
        raise ValueError(f"Expected {expected} real visual tokens, got {tokens.shape[1]}")
    if expected > capacity:
        raise ValueError(f"Real token count {expected} exceeds capacity={capacity}")

    packed = tokens.new_zeros(tokens.shape[0], capacity, tokens.shape[-1])
    packed[:, :expected] = tokens
    mask = torch.zeros(tokens.shape[0], capacity, dtype=torch.bool, device=tokens.device)
    mask[:, :expected] = True
    return packed, mask


def build_visual_metadata(
    *,
    batch_size: int,
    valid_frames: int,
    target_num_frames: int,
    target_fps: float,
    sampled_frame_indices: Tensor,
    device: torch.device | str,
) -> dict[str, Tensor | int]:
    if sampled_frame_indices.ndim == 1:
        sampled_frame_indices = sampled_frame_indices.unsqueeze(0)
    if sampled_frame_indices.shape != (batch_size, MAX_VLM_FRAMES):
        raise ValueError(
            f"sampled_frame_indices must be [{batch_size},{MAX_VLM_FRAMES}], "
            f"got {tuple(sampled_frame_indices.shape)}"
        )
    frame_mask = torch.zeros(batch_size, MAX_VLM_FRAMES, dtype=torch.bool, device=device)
    frame_mask[:, :valid_frames] = True
    return {
        "visual_token_capacity": VISUAL_TOKEN_CAPACITY,
        "tokens_per_frame": TOKENS_PER_FRAME,
        "vlm_frame_capacity": MAX_VLM_FRAMES,
        "num_valid_vlm_frames": torch.full((batch_size,), valid_frames, dtype=torch.long, device=device),
        "num_valid_visual_tokens": torch.full(
            (batch_size,), valid_frames * TOKENS_PER_FRAME, dtype=torch.long, device=device
        ),
        "sampled_frame_indices": sampled_frame_indices.to(device=device, dtype=torch.long),
        "sampled_frame_mask": frame_mask,
        "target_num_frames": torch.full((batch_size,), target_num_frames, dtype=torch.long, device=device),
        "target_fps": torch.full((batch_size,), target_fps, dtype=torch.float32, device=device),
        "target_modality_id": torch.full(
            (batch_size,), 0 if target_num_frames == 1 else 1, dtype=torch.long, device=device
        ),
    }


def planner_output_mask_from_visual_mask(visual_token_mask: Tensor) -> Tensor:
    if visual_token_mask.ndim != 2 or visual_token_mask.shape[1] != VISUAL_TOKEN_CAPACITY:
        raise ValueError(
            f"visual_token_mask must be [B,{VISUAL_TOKEN_CAPACITY}], got {tuple(visual_token_mask.shape)}"
        )
    return visual_token_mask.to(dtype=torch.bool).clone()
