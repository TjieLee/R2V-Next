import pytest
import torch
from torch import nn

from ltx_core.multicond.visual_tokens import Visual3DTokenEncoder


def _positions(batch_size: int, token_count: int, *, device: torch.device | None = None) -> torch.Tensor:
    device = device or torch.device("cpu")
    index = torch.arange(token_count, device=device, dtype=torch.float32)
    coords = torch.stack(
        [
            index / 8.0,
            torch.remainder(index, 16.0) + 0.5,
            torch.remainder(index * 3.0, 16.0) + 0.5,
        ],
        dim=0,
    )
    bounds = torch.stack([coords - 0.25, coords + 0.25], dim=-1)
    return bounds.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()


def _small_encoder() -> Visual3DTokenEncoder:
    return Visual3DTokenEncoder(
        dim=32,
        num_heads=4,
        depth=1,
        ffn_multiplier=2.0,
        dropout=0.0,
        residual_init_gain=0.1,
        positional_embedding_max_pos=[20, 64, 64],
        use_middle_positions=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Production-shape test requires CUDA SDPA")
def test_visual_3d_token_encoder_production_shape_is_full_2048_tokens() -> None:
    device = torch.device("cuda")
    encoder = Visual3DTokenEncoder(dim=4096, num_heads=32, depth=1).to(device=device, dtype=torch.bfloat16)
    tokens = torch.randn(2, 2048, 4096, device=device, dtype=torch.bfloat16)
    positions = _positions(2, 2048, device=device)
    mask = torch.ones(2, 2048, dtype=torch.bool, device=device)

    with torch.no_grad():
        output, output_mask = encoder(tokens=tokens, token_positions=positions, token_mask=mask)

    assert output.shape == (2, 2048, 4096)
    assert output_mask.shape == (2, 2048)
    assert torch.isfinite(output).all()


def test_visual_3d_token_encoder_zero_input_maps_to_zero() -> None:
    encoder = _small_encoder().eval()
    tokens = torch.zeros(2, 16, 32)
    mask = torch.ones(2, 16, dtype=torch.bool)

    output, output_mask = encoder(tokens=tokens, token_positions=_positions(2, 16), token_mask=mask)

    assert output_mask.dtype == torch.bool
    assert float(output.abs().max()) < 1.0e-5


def test_visual_3d_token_encoder_is_content_and_position_dependent() -> None:
    encoder = _small_encoder().eval()
    mask = torch.ones(1, 16, dtype=torch.bool)
    positions = _positions(1, 16)
    tokens_a = torch.randn(1, 16, 32)
    tokens_b = torch.randn(1, 16, 32)

    out_a, _ = encoder(tokens=tokens_a, token_positions=positions, token_mask=mask)
    out_b, _ = encoder(tokens=tokens_b, token_positions=positions, token_mask=mask)
    reversed_time = positions.clone()
    reversed_time[:, 0] = torch.flip(reversed_time[:, 0], dims=[1])
    out_reversed, _ = encoder(tokens=tokens_a, token_positions=reversed_time, token_mask=mask)

    relative_delta = torch.linalg.vector_norm(out_a - out_b) / torch.linalg.vector_norm(out_a).clamp(min=1.0e-8)
    assert float(relative_delta) > 1.0e-3
    assert not torch.allclose(out_a, out_reversed)


def test_visual_3d_token_encoder_is_permutation_equivariant() -> None:
    encoder = _small_encoder().eval()
    tokens = torch.randn(1, 16, 32)
    positions = _positions(1, 16)
    mask = torch.ones(1, 16, dtype=torch.bool)
    permutation = torch.randperm(16)

    output, _ = encoder(tokens=tokens, token_positions=positions, token_mask=mask)
    permuted_output, _ = encoder(
        tokens=tokens[:, permutation],
        token_positions=positions[:, :, permutation],
        token_mask=mask[:, permutation],
    )

    assert torch.allclose(permuted_output, output[:, permutation], atol=1.0e-5, rtol=1.0e-5)


def test_visual_3d_token_encoder_masks_keys_and_zeroes_invalid_queries() -> None:
    encoder = _small_encoder().eval()
    tokens = torch.randn(2, 16, 32)
    modified = tokens.clone()
    modified[:, 12:] = torch.randn_like(modified[:, 12:]) * 1000.0
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[:, 12:] = False
    positions = _positions(2, 16)

    output, output_mask = encoder(tokens=tokens, token_positions=positions, token_mask=mask)
    modified_output, _ = encoder(tokens=modified, token_positions=positions, token_mask=mask)

    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[:, 12:]) == 0
    assert torch.equal(output_mask, mask)
    assert torch.allclose(output[:, :12], modified_output[:, :12], atol=1.0e-5, rtol=1.0e-5)

    all_masked_output, _ = encoder(
        tokens=tokens,
        token_positions=positions,
        token_mask=torch.zeros_like(mask),
    )
    assert torch.isfinite(all_masked_output).all()
    assert torch.count_nonzero(all_masked_output) == 0


def test_visual_3d_token_encoder_core_parameters_receive_first_step_gradients() -> None:
    encoder = _small_encoder().train()
    tokens = torch.randn(2, 16, 32)
    output, _ = encoder(
        tokens=tokens,
        token_positions=_positions(2, 16),
        token_mask=torch.ones(2, 16, dtype=torch.bool),
    )

    loss = (output * torch.randn_like(output)).mean()
    loss.backward()

    required = {
        "input_norm.weight",
        "blocks.0.norm1.weight",
        "blocks.0.qkv.weight",
        "blocks.0.attn_out.weight",
        "blocks.0.norm2.weight",
        "blocks.0.ffn_fc1.weight",
        "blocks.0.ffn_fc2.weight",
        "output_norm.weight",
    }
    named_parameters = dict(encoder.named_parameters())
    assert required.issubset(named_parameters)
    for name in required:
        grad = named_parameters[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert float(torch.linalg.vector_norm(grad)) > 0.0, name

    assert all(module.bias is None for module in encoder.modules() if isinstance(module, nn.Linear))
    assert not any("query" in name or "gate" in name for name, _ in encoder.named_parameters())
