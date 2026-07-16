"""Dtype-safe VAE decoding for raw online inference."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange
from torch import Tensor, nn

from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig

_DEFAULT_TILING = TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=192, tile_overlap_in_pixels=64),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=48, tile_overlap_in_frames=24),
)


@dataclass(frozen=True)
class ModuleComputeSpec:
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class VaeDecodeDiagnostics:
    decoder_weight_dtype: str
    decoder_input_dtype: str
    decoder_output_dtype: str
    decoder_device: str


def _unwrap_module(module: nn.Module) -> nn.Module:
    current = module
    seen: set[int] = set()
    while isinstance(getattr(current, "module", None), nn.Module) and id(current) not in seen:
        seen.add(id(current))
        current = current.module
    return current


def module_compute_spec(module: nn.Module) -> ModuleComputeSpec:
    """Resolve compute device/dtype from the first floating parameter or buffer."""
    unwrapped = _unwrap_module(module)
    for value in unwrapped.parameters():
        if value.is_floating_point():
            return ModuleComputeSpec(device=value.device, dtype=value.dtype)
    for value in unwrapped.buffers():
        if value.is_floating_point():
            return ModuleComputeSpec(device=value.device, dtype=value.dtype)
    raise ValueError(f"Cannot infer floating compute dtype/device for {type(unwrapped).__name__}")


def validate_vae_decode_request(device: torch.device, dtype: torch.dtype) -> None:
    """Reject unsupported real decode combinations before model loading."""
    if dtype not in {torch.bfloat16, torch.float16, torch.float32}:
        raise ValueError(f"VAE decode dtype must be bfloat16, float16, or float32, got {dtype}")
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU VAE decode supports only float32")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA VAE decode was requested but CUDA is unavailable")
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bfloat16 VAE decode")
        return
    if device.type != "cpu":
        raise ValueError(f"Unsupported VAE decode device type: {device.type}")


def _autocast_context(spec: ModuleComputeSpec) -> Any:
    if spec.device.type == "cuda" and spec.dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type="cuda", dtype=spec.dtype)
    return nullcontext()


def decode_video_latents(
    *,
    vae_decoder: nn.Module,
    latents: Tensor,
    decode_tile: bool,
    tiling_config: TilingConfig = _DEFAULT_TILING,
) -> tuple[Tensor, VaeDecodeDiagnostics]:
    """Decode latents using the decoder's actual compute dtype and device."""
    decoder = _unwrap_module(vae_decoder)
    spec = module_compute_spec(decoder)
    aligned_latents = latents.to(device=spec.device, dtype=spec.dtype)
    with torch.inference_mode(), _autocast_context(spec):
        if decode_tile:
            chunks = list(decoder.tiled_decode(aligned_latents, tiling_config=tiling_config))
            if not chunks:
                raise RuntimeError("VAE tiled_decode returned no chunks")
            raw_video = torch.cat(chunks, dim=2)
        else:
            raw_video = decoder(aligned_latents)
    diagnostics = VaeDecodeDiagnostics(
        decoder_weight_dtype=str(spec.dtype),
        decoder_input_dtype=str(aligned_latents.dtype),
        decoder_output_dtype=str(raw_video.dtype),
        decoder_device=str(spec.device),
    )
    video = ((raw_video + 1.0) / 2.0).clamp(0.0, 1.0)
    return rearrange(video, "1 c f h w -> f c h w").float().cpu(), diagnostics
