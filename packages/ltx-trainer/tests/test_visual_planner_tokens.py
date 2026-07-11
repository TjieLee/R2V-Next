import torch

from ltx_core.multicond.visual_tokens import VisualPlannerTokens


def _positions(batch_size: int, token_count: int, *, offset: float = 0.0) -> torch.Tensor:
    coords = torch.arange(token_count, dtype=torch.float32) + offset
    axes = torch.stack([coords, coords.remainder(2), coords.remainder(3)], dim=0)
    axes = axes.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.stack([axes, axes], dim=-1)


def _planner() -> VisualPlannerTokens:
    return VisualPlannerTokens(
        token_count=4,
        dim=8,
        source_dim=8,
        num_heads=2,
        zero_init_output=False,
        zero_init_ffn=False,
        use_learned_query_tokens=True,
        use_content_residual=True,
        use_3d_rope=True,
        residual_init_gain=0.1,
    )


def test_planner_content_dependent_at_initialization() -> None:
    planner = _planner()
    first = torch.randn(1, 4, 8)
    second = first + 0.5
    positions = _positions(1, 4)

    output_first = planner(planner_hidden=first, token_positions=positions)
    output_second = planner(planner_hidden=second, token_positions=positions)

    assert not torch.allclose(output_first, output_second)
    assert torch.allclose(planner.content_projection.weight, torch.eye(8))


def test_3d_positions_affect_planner_output() -> None:
    planner = _planner()
    hidden = torch.randn(1, 4, 8)

    first_positions = _positions(1, 4)
    second_positions = first_positions.clone()
    second_positions[:, 0] *= 2.5
    first = planner(planner_hidden=hidden, token_positions=first_positions)
    second = planner(planner_hidden=hidden, token_positions=second_positions)

    assert not torch.allclose(first, second)


def test_planner_first_backward_reaches_all_bridge_parameters() -> None:
    planner = _planner()
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    output = planner(planner_hidden=hidden, token_positions=_positions(1, 4))
    output.square().mean().backward()

    parameters = {
        "content": planner.content_projection.weight,
        "query": planner.query_projection.weight,
        "key": planner.key_projection.weight,
        "value": planner.value_projection.weight,
        "attention_output": planner.output_projection.weight,
        "ffn_input": planner.ffn_fc1.weight,
        "ffn_output": planner.ffn_fc2.weight,
        "learned_queries": planner.query_tokens,
    }
    for name, parameter in parameters.items():
        assert parameter is not None, name
        assert parameter.grad is not None, name
        assert bool(torch.any(parameter.grad != 0)), name


def test_planner_query_chunking_matches_unchunked_eval() -> None:
    full = _planner().eval()
    chunked = _planner().eval()
    chunked.load_state_dict(full.state_dict())
    chunked.query_chunk_size = 2
    hidden = torch.randn(1, 4, 8)
    positions = _positions(1, 4)

    assert torch.allclose(
        full(planner_hidden=hidden, token_positions=positions),
        chunked(planner_hidden=hidden, token_positions=positions),
        atol=1.0e-6,
        rtol=1.0e-5,
    )
