from types import SimpleNamespace

import torch
from torch import nn

from ltx_trainer.training_strategies.multi_reference_planner_stage2 import (
    MultiReferencePlannerStage2Config,
    MultiReferencePlannerStage2Strategy,
)


class _VideoConnector(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.inner_dim = dim
        self.num_learnable_registers = 1
        self.weight = nn.Parameter(torch.zeros(1, dim))


class _EmbeddingsProcessor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.video_connector = _VideoConnector(dim)


class _CountingLmHead(nn.Module):
    def __init__(self, dim: int, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(dim, vocab_size, bias=False)
        self.seen_rows: list[int] = []

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.seen_rows.append(hidden.shape[0])
        return self.projection(hidden)


def _strategy() -> MultiReferencePlannerStage2Strategy:
    strategy = MultiReferencePlannerStage2Strategy(
        MultiReferencePlannerStage2Config(
            use_online_vlm=False,
            planner_token_count=8,
            planner_source_dim=8,
            planner_output_dim=8,
            planner_cross_attention_heads=2,
            visual_token_source_dim=8,
            visual_token_target_dim=12,
            visual_context_expected_tokens=8,
            visual_full_sa_num_heads=2,
            visual_full_sa_ffn_multiplier=1.0,
            cfg_dropout_enabled=False,
        )
    )
    strategy.attach_models(
        transformer=nn.Identity(),
        embeddings_processor=_EmbeddingsProcessor(12),
        text_encoder=None,
    )
    return strategy


def _batch(predicted: torch.Tensor) -> dict:
    return {
        "latents": {"height": torch.tensor([64]), "width": torch.tensor([96])},
        "gt_visual_tokens": {
            "visual_tokens": torch.randn_like(predicted),
            "visual_token_mask": torch.ones(predicted.shape[:2], dtype=torch.bool),
            "tokens_per_frame": torch.tensor([4]),
            "sampled_frame_indices": torch.tensor([[0, 6]]),
            "source_fps": torch.tensor([24.0]),
        },
        "planner_conditions": {
            "predicted_visual_tokens": predicted,
            "predicted_visual_token_mask": torch.ones(predicted.shape[:2], dtype=torch.bool),
        },
    }


def _conditions() -> dict[str, torch.Tensor]:
    return {
        "video_prompt_embeds": torch.randn(1, 3, 12),
        "audio_prompt_embeds": None,
        "prompt_attention_mask": torch.ones(1, 3, dtype=torch.long),
    }


def test_stage2_uses_postconnector_full_token_path() -> None:
    strategy = _strategy()
    predicted = torch.randn(1, 8, 8)
    batch = _batch(predicted)
    base = _conditions()

    preconnector = strategy.prepare_conditions(batch, base)
    assert preconnector["video_prompt_embeds"].shape == (1, 3, 12)
    assert batch["_planner_predicted_raw_tokens"].shape == (1, 8, 8)

    postconnector = strategy.postprocess_conditions_after_connector(batch, preconnector)
    assert postconnector["video_prompt_embeds"].shape == (1, 11, 12)
    assert postconnector["prompt_attention_mask"].shape == (1, 11)


def test_no_visual_connector_exists() -> None:
    strategy = _strategy()

    assert strategy.config.visual_connector_enabled is False
    assert strategy._visual_connector is None
    assert strategy._visual_gate is None
    assert strategy._visual_resampler is None
    assert strategy._visual_full_encoder is not None


def test_planner_position_order_matches_gt() -> None:
    times = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    positions = MultiReferencePlannerStage2Strategy._make_visual_positions(
        times,
        height=torch.tensor([16.0]),
        width=torch.tensor([16.0]),
        spatial_grid=16,
        dtype=torch.float32,
    )

    assert positions.shape == (1, 3, 2048, 2)
    assert positions[0, 0, 0, 0] == 0
    assert positions[0, 0, 255, 0] == 0
    assert positions[0, 0, 256, 0] == 1
    assert positions[0, 0, 2047, 0] == 7
    assert positions[0, 1, 255, 0] > positions[0, 1, 0, 0]
    assert positions[0, 2, 255, 0] > positions[0, 2, 0, 0]


def test_loss_contains_exactly_three_terms() -> None:
    strategy = _strategy()
    strategy.config.flow_loss_weight = 2.0
    strategy.config.siglip_loss_weight = 3.0
    strategy.config.ntp_loss_weight = 4.0
    total = strategy._combine_losses(
        torch.tensor([11.0]),
        torch.tensor([5.0]),
        torch.tensor([7.0]),
    )
    assert torch.equal(total, torch.tensor([65.0]))


def test_siglip_loss_is_raw_feature_mean_then_token_mean() -> None:
    strategy = _strategy()
    predicted = torch.zeros(1, 2, 8)
    target = torch.ones(1, 2, 8)
    mask = torch.tensor([[True, False]])

    loss = strategy._compute_visual_alignment_loss(
        predicted_tokens=predicted,
        gt_tokens=target,
        mask=mask,
    )

    assert torch.equal(loss, torch.ones(1))


def test_ntp_selective_lm_head_only_receives_valid_text_rows() -> None:
    strategy = _strategy()
    lm_head = _CountingLmHead(dim=8, vocab_size=32)
    strategy.text_encoder = SimpleNamespace(model=SimpleNamespace(lm_head=lm_head))
    strategy.config.ntp_logits_chunk_size = 2
    hidden = torch.randn(2, 6, 8)
    labels = torch.tensor(
        [
            [-100, 1, -100, 2, 3, -100],
            [-100, -100, -100, -100, -100, -100],
        ]
    )

    loss = strategy._compute_lm_loss(hidden, labels)

    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()
    assert loss[1] == 0
    assert lm_head.seen_rows == [2, 1]


def test_stage1_checkpoint_loads_projection_and_full_encoder_without_planner_keys() -> None:
    strategy = _strategy()
    state_dict = {}
    for name, module in {
        "visual_token_projection": strategy._visual_token_projection,
        "visual_full_encoder": strategy._visual_full_encoder,
    }.items():
        assert module is not None
        for key, value in module.state_dict().items():
            state_dict[f"training_strategy.{name}.{key}"] = torch.full_like(value, 0.25)

    strategy.load_extra_checkpoint_state_dict(state_dict)

    assert strategy._visual_token_projection is not None
    assert torch.allclose(
        strategy._visual_token_projection.weight,
        torch.full_like(strategy._visual_token_projection.weight, 0.25),
    )
    assert strategy.planner_tokens is not None
    assert not torch.allclose(
        strategy.planner_tokens.query_tokens,
        torch.full_like(strategy.planner_tokens.query_tokens, 0.25),
    )


def test_stage2_strategy_checkpoint_roundtrip() -> None:
    first = _strategy()
    state_dict = {}
    for name, module in first.get_trainable_modules().items():
        for key, value in module.state_dict().items():
            state_dict[f"training_strategy.{name}.{key}"] = value.detach().clone()

    second = _strategy()
    second.load_extra_checkpoint_state_dict(state_dict)

    for name, first_module in first.get_trainable_modules().items():
        second_module = second.get_trainable_modules()[name]
        for key, first_value in first_module.state_dict().items():
            assert torch.equal(first_value, second_module.state_dict()[key]), f"{name}.{key}"
