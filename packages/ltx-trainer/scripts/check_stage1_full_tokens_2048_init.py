#!/usr/bin/env python3
"""Check initialization health of the Stage 1 full 2048-token visual branch."""

from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn

from ltx_core.model.transformer.rope import LTXRopeType
from ltx_trainer.training_strategies.multi_reference_video import (
    MultiReferenceVideoConfig,
    MultiReferenceVideoStrategy,
)


class _Connector(nn.Module):
    def __init__(self, *, dim: int, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.inner_dim = dim
        self.num_learnable_registers = 128
        self.anchor = nn.Parameter(torch.zeros(1, device=device, dtype=dtype), requires_grad=False)


class _EmbeddingsProcessor(nn.Module):
    def __init__(self, connector: nn.Module) -> None:
        super().__init__()
        self.video_connector = connector


class _TransformerGeometry(nn.Module):
    positional_embedding_theta = 10000.0
    positional_embedding_max_pos = [20, 2048, 2048]
    rope_type = LTXRopeType.SPLIT


def _make_positions(*, device: torch.device) -> Tensor:
    sampled_times = (torch.arange(8, device=device, dtype=torch.float32) / 6.0).unsqueeze(0)
    return MultiReferenceVideoStrategy._make_visual_positions(
        sampled_times,
        height=torch.tensor([384.0], device=device),
        width=torch.tensor([640.0], device=device),
        spatial_grid=16,
        dtype=torch.float32,
    )


def _rms(value: Tensor) -> float:
    return float(value.float().square().mean().sqrt().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if device.type == "cpu" and dtype == torch.bfloat16:
        raise ValueError("Use --dtype fp32 for the CPU smoke test")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    strategy = MultiReferenceVideoStrategy(
        MultiReferenceVideoConfig(
            visual_token_source_dim=3840,
            visual_token_target_dim=4096,
            visual_context_mode="full_tokens_3d_sa",
            visual_context_expected_tokens=2048,
            visual_context_max_tokens=2048,
            visual_full_sa_num_heads=32,
            visual_full_sa_depth=1,
            visual_full_sa_ffn_multiplier=2.0,
            visual_full_sa_dropout=0.0,
            visual_full_sa_residual_init_gain=0.1,
            visual_full_sa_use_middle_positions=True,
            visual_connector_enabled=False,
        )
    )
    strategy.attach_models(
        transformer=_TransformerGeometry(),
        embeddings_processor=_EmbeddingsProcessor(
            _Connector(dim=4096, device=device, dtype=dtype)
        ),
        text_encoder=None,
    )
    modules = strategy.get_trainable_modules()
    if set(modules) != {"visual_token_projection", "visual_full_encoder"}:
        raise RuntimeError(f"Unexpected full-token trainable modules: {sorted(modules)}")
    projection = modules["visual_token_projection"]
    encoder = modules["visual_full_encoder"]
    positions = _make_positions(device=device)
    mask = torch.ones(1, 2048, dtype=torch.bool, device=device)

    raw_a = torch.randn(1, 2048, 3840, device=device, dtype=dtype)
    raw_b = torch.randn_like(raw_a)
    with torch.no_grad():
        projected_a = projection(raw_a)
        context_a, _ = encoder(tokens=projected_a, token_positions=positions, token_mask=mask)
        zero_context, _ = encoder(
            tokens=projection(torch.zeros_like(raw_a)),
            token_positions=positions,
            token_mask=mask,
        )
        context_b, _ = encoder(
            tokens=projection(raw_b),
            token_positions=positions,
            token_mask=mask,
        )
        relative_delta = torch.linalg.vector_norm((context_a - context_b).float()) / torch.linalg.vector_norm(
            context_a.float()
        ).clamp(min=1.0e-8)

    print(f"visual_token_projection input shape: {list(raw_a.shape)}")
    print(f"visual_token_projection output shape: {list(projected_a.shape)}")
    print(f"visual_context shape: {list(context_a.shape)}")
    print(f"visual_context RMS: {_rms(context_a):.6f}")
    print(f"zero SigLIP visual_context RMS: {_rms(zero_context):.6f}")
    print(f"zero SigLIP visual_context max_abs: {float(zero_context.abs().max()):.6e}")
    print(f"different-sample relative_delta: {float(relative_delta):.6f}")
    del projected_a, context_a, zero_context, context_b, raw_b

    projection.zero_grad(set_to_none=True)
    encoder.zero_grad(set_to_none=True)
    projected = projection(raw_a)
    output, _ = encoder(tokens=projected, token_positions=positions, token_mask=mask)
    loss = (output * torch.randn_like(output)).mean()
    loss.backward()

    gradient_parameters = {"visual_token_projection.weight": projection.weight}
    gradient_parameters.update(
        {
            f"visual_full_encoder.{name}": parameter
            for name, parameter in encoder.named_parameters()
            if name.endswith("weight")
        }
    )
    for name, parameter in gradient_parameters.items():
        grad = parameter.grad
        if grad is None or not torch.isfinite(grad).all() or float(torch.linalg.vector_norm(grad)) <= 0.0:
            raise RuntimeError(f"Invalid first-step gradient for {name}")
        print(f"gradient norm {name}: {float(torch.linalg.vector_norm(grad.float())):.6e}")

    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"GPU peak memory: {peak_gib:.3f} GiB")


if __name__ == "__main__":
    main()
