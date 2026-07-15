from __future__ import annotations

import torch

from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.multicond.visual_tokens import Visual3DTokenEncoder, VisualPlannerTokens
from ltx_trainer.training_strategies.multi_reference_planner_stage2 import (
    MultiReferencePlannerStage2Strategy,
)


def _positions(batch_size: int, tokens: int) -> torch.Tensor:
    return torch.zeros(batch_size, 3, tokens, 2)


def test_invalid_visual_tokens_do_not_affect_valid_3d_attention() -> None:
    torch.manual_seed(0)
    encoder = Visual3DTokenEncoder(
        dim=8,
        num_heads=2,
        depth=1,
        ffn_multiplier=2.0,
        rope_type=LTXRopeType.SPLIT,
    ).eval()
    mask = torch.tensor([[True, True, False, False]])
    tokens = torch.randn(1, 4, 8)
    changed = tokens.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 1000
    first, first_mask = encoder(tokens=tokens, token_positions=_positions(1, 4), token_mask=mask)
    second, second_mask = encoder(tokens=changed, token_positions=_positions(1, 4), token_mask=mask)
    torch.testing.assert_close(first[:, :2], second[:, :2])
    assert torch.count_nonzero(first[:, 2:]).item() == 0
    assert torch.equal(first_mask, second_mask)


def test_planner_invalid_queries_are_zero_and_masked_from_cross_attention() -> None:
    torch.manual_seed(1)
    planner = VisualPlannerTokens(
        token_count=4,
        dim=8,
        source_dim=8,
        num_heads=2,
        use_learned_query_tokens=True,
        use_3d_rope=True,
        rope_type=LTXRopeType.SPLIT,
    ).eval()
    hidden = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, False, False, False]])
    output = planner(
        planner_hidden=hidden,
        planner_mask=mask,
        token_positions=_positions(1, 4),
    )
    assert output.shape == (1, 4, 8)
    assert torch.count_nonzero(output[:, 1:]).item() == 0


def test_visual_mse_is_meaned_per_valid_token_and_feature() -> None:
    predicted = torch.zeros(2, 4, 8)
    target = torch.ones_like(predicted)
    mask = torch.tensor([[True, False, False, False], [True, True, True, True]])
    loss = MultiReferencePlannerStage2Strategy._compute_visual_alignment_loss(
        object(),
        predicted_tokens=predicted,
        gt_tokens=target,
        mask=mask,
    )
    torch.testing.assert_close(loss, torch.ones(2))
