from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class MultiReferencePack:
    """Packed video stream tensors for multi-reference conditioning."""

    latents: Tensor
    positions: Tensor
    timesteps: Tensor
    loss_mask: Tensor
    attention_mask: Tensor
    ref_token_mask: Tensor
    target_token_mask: Tensor


def build_multiref_sequence(
    *,
    ref_tokens: Tensor,
    ref_positions: Tensor,
    ref_valid_mask: Tensor,
    target_tokens: Tensor,
    target_positions: Tensor,
    target_timesteps: Tensor,
    target_loss_mask: Tensor,
    reference_time_stride: float = 1.0,
) -> MultiReferencePack:
    """Concatenate clean multi-reference tokens before noisy target tokens.

    Args:
        ref_tokens: Reference latents in ``[B, R, S_ref, C]``.
        ref_positions: Reference RoPE bounds in ``[B, R, 3, S_ref, 2]``.
        ref_valid_mask: Boolean mask in ``[B, R]``.
        target_tokens: Target latents in ``[B, S_target, C]``.
        target_positions: Target RoPE bounds in ``[B, 3, S_target, 2]``.
        target_timesteps: Per-token diffusion timesteps in ``[B, S_target]``.
        target_loss_mask: Target loss mask in ``[B, S_target]``.
        reference_time_stride: Negative temporal slot spacing. Ref 0 is shifted
            by ``-reference_time_stride``, ref 1 by ``-2 * reference_time_stride``.
    """

    if ref_tokens.ndim != 4:
        raise ValueError(f"ref_tokens must be [B, R, S, C], got {tuple(ref_tokens.shape)}")
    if ref_positions.ndim != 5:
        raise ValueError(f"ref_positions must be [B, R, 3, S, 2], got {tuple(ref_positions.shape)}")

    batch_size, num_refs, ref_seq_len, channels = ref_tokens.shape
    if target_tokens.shape[0] != batch_size or target_tokens.shape[-1] != channels:
        raise ValueError("Reference and target tokens must share batch size and channel dimension")
    if ref_valid_mask.shape != (batch_size, num_refs):
        raise ValueError(f"ref_valid_mask must be {(batch_size, num_refs)}, got {tuple(ref_valid_mask.shape)}")

    device = target_tokens.device
    dtype = target_tokens.dtype

    ref_positions = ref_positions.to(device=device).clone()
    offsets = torch.arange(1, num_refs + 1, device=device, dtype=ref_positions.dtype)
    offsets = offsets.view(1, num_refs, 1, 1) * reference_time_stride
    ref_positions[:, :, 0, :, :] = ref_positions[:, :, 0, :, :] - offsets

    ref_valid_mask = ref_valid_mask.to(device=device, dtype=torch.bool)
    ref_token_mask = ref_valid_mask[:, :, None].expand(batch_size, num_refs, ref_seq_len).reshape(batch_size, -1)
    target_token_mask = torch.ones(
        batch_size,
        target_tokens.shape[1],
        dtype=torch.bool,
        device=device,
    )

    ref_tokens = ref_tokens.to(device=device, dtype=dtype)
    ref_tokens = ref_tokens * ref_token_mask.reshape(batch_size, num_refs, ref_seq_len, 1).to(dtype)
    ref_tokens = ref_tokens.reshape(batch_size, num_refs * ref_seq_len, channels)
    ref_positions = ref_positions.permute(0, 2, 1, 3, 4).reshape(batch_size, 3, num_refs * ref_seq_len, 2)

    ref_timesteps = torch.zeros(batch_size, num_refs * ref_seq_len, device=device, dtype=target_timesteps.dtype)
    ref_loss_mask = torch.zeros(batch_size, num_refs * ref_seq_len, device=device, dtype=torch.bool)

    latents = torch.cat([ref_tokens, target_tokens], dim=1)
    positions = torch.cat([ref_positions, target_positions], dim=2)
    timesteps = torch.cat([ref_timesteps, target_timesteps], dim=1)
    loss_mask = torch.cat([ref_loss_mask, target_loss_mask], dim=1)

    valid_tokens = torch.cat([ref_token_mask, target_token_mask], dim=1)
    attention_mask = valid_tokens[:, None, :].to(dtype)

    return MultiReferencePack(
        latents=latents,
        positions=positions,
        timesteps=timesteps,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        ref_token_mask=ref_token_mask,
        target_token_mask=target_token_mask,
    )
