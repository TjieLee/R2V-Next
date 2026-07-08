import importlib.util
from pathlib import Path

import torch
from torch import nn

from ltx_core.multicond.cfg_sampler import CFGModeBatch, sample_cfg_modes
from ltx_core.multicond.visual_tokens import VisualPlannerTokens
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_trainer.training_strategies.multi_reference_planner_stage2 import (
    MultiReferencePlannerStage2Config,
    MultiReferencePlannerStage2Strategy,
)
from ltx_trainer.training_strategies.multi_reference_video import (
    MultiReferenceVideoConfig,
    MultiReferenceVideoStrategy,
)


_PRECOMPUTE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "precompute_planner_vlm_inputs.py"
_SPEC = importlib.util.spec_from_file_location("precompute_planner_vlm_inputs", _PRECOMPUTE_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
precompute_planner_vlm_inputs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(precompute_planner_vlm_inputs)


def _fixed_modes() -> CFGModeBatch:
    return CFGModeBatch(
        mode_id=torch.tensor([0, 1, 2, 3]),
        drop_text=torch.tensor([False, True, False, False]),
        drop_ref=torch.tensor([False, False, True, False]),
        drop_all=torch.tensor([False, False, False, True]),
        keep_full=torch.tensor([True, False, False, False]),
    )


def test_sample_cfg_modes_supports_four_modes_and_legacy_alias() -> None:
    cases = {
        "full": "keep_full",
        "drop_text": "drop_text",
        "drop_ref": "drop_ref",
        "drop_all": "drop_all",
        "null": "drop_all",
        "drop_planner": "drop_all",
    }
    for mode_name, field_name in cases.items():
        modes = sample_cfg_modes(4, probs={mode_name: 1.0})
        assert bool(getattr(modes, field_name).all())
        if field_name == "drop_all":
            assert bool(modes.drop_planner.all())


def test_drop_ref_and_drop_all_zero_visual_tokens() -> None:
    strategy = MultiReferenceVideoStrategy(MultiReferenceVideoConfig(cfg_dropout_enabled=True))
    batch = {"_cfg_modes": _fixed_modes()}
    visual_tokens = torch.ones(4, 2, 3)
    visual_mask = torch.ones(4, 2, dtype=torch.bool)

    dropped_tokens, dropped_mask = strategy._apply_cfg_planner_dropout(batch, visual_tokens, visual_mask)

    assert torch.equal(dropped_tokens[0], visual_tokens[0])
    assert torch.equal(dropped_tokens[1], visual_tokens[1])
    assert torch.equal(dropped_tokens[2], torch.zeros_like(visual_tokens[2]))
    assert torch.equal(dropped_tokens[3], torch.zeros_like(visual_tokens[3]))
    assert torch.equal(dropped_mask, visual_mask)


def test_stage2_visual_drop_mask_excludes_drop_ref_and_drop_all_from_mse() -> None:
    strategy = MultiReferencePlannerStage2Strategy(
        MultiReferencePlannerStage2Config(cfg_dropout_enabled=True, cfg_drop_planner_p=0.25)
    )
    batch = {"_cfg_modes": _fixed_modes()}

    drop_visual = strategy._cfg_drop_visual_mask(batch, batch_size=4, device=torch.device("cpu"))

    assert torch.equal(drop_visual, torch.tensor([False, False, True, True]))
    assert strategy._cfg_drop_all_probability() == 0.25


def test_inferred_text_token_mask_does_not_cover_planner_placeholders() -> None:
    image = GEMMA3_CONFIG_FOR_LTX.image_token_index
    boi = GEMMA3_CONFIG_FOR_LTX.boi_token_index
    eoi = GEMMA3_CONFIG_FOR_LTX.eoi_token_index
    input_ids = torch.tensor([[10, image, 11, boi, image, eoi, boi, image, image, eoi, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
    planner_placeholder_mask = torch.tensor(
        [[False, False, False, False, False, False, False, True, True, False, False]]
    )
    planner_boundary_mask = torch.tensor(
        [[False, False, False, False, False, False, True, False, False, True, False]]
    )
    planner_region_mask = planner_placeholder_mask | planner_boundary_mask

    text_token_mask = MultiReferencePlannerStage2Strategy._infer_vlm_text_token_mask(
        planner_data={"planner_region_mask": planner_region_mask},
        forward_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        device=torch.device("cpu"),
    )

    assert not bool((text_token_mask & planner_placeholder_mask).any())
    assert not bool((text_token_mask & planner_region_mask).any())
    assert torch.equal(
        text_token_mask,
        torch.tensor([[True, False, True, False, False, False, False, False, False, False, False]]),
    )


def test_planner_precompute_masks_separate_text_ref_and_planner_regions() -> None:
    image = GEMMA3_CONFIG_FOR_LTX.image_token_index
    boi = GEMMA3_CONFIG_FOR_LTX.boi_token_index
    eoi = GEMMA3_CONFIG_FOR_LTX.eoi_token_index
    tensor_data = {
        "input_ids": torch.tensor([10, boi, image, eoi, 11]),
        "attention_mask": torch.ones(5, dtype=torch.long),
    }

    out = precompute_planner_vlm_inputs._append_planner_placeholders(
        tensor_data,
        planner_token_count=2,
        max_length=10,
        source_max_length=6,
        placeholder_token_id=image,
        start_token_id=boi,
        end_token_id=eoi,
        pad_token_id=0,
    )

    ref_visual = out["ref_visual_token_mask"]
    ref_region = out["ref_image_region_mask"]
    text_mask = out["text_token_mask"]
    planner_placeholder = out["planner_placeholder_mask"]
    planner_boundary = out["planner_boundary_mask"]

    assert torch.equal(ref_visual, torch.tensor([False, False, True, False, False, False, False, False, False, False]))
    assert torch.equal(ref_region, torch.tensor([False, True, True, True, False, False, False, False, False, False]))
    assert torch.equal(out["gt_image_token_mask"], ref_region)
    assert bool(ref_visual[2])
    assert not bool((ref_visual & (out["input_ids"] == boi)).any())
    assert not bool((ref_visual & (out["input_ids"] == eoi)).any())
    assert not bool((text_mask & planner_placeholder).any())
    assert not bool((text_mask & planner_boundary).any())
    assert not bool((text_mask & ref_region).any())
    assert torch.equal(text_mask, torch.tensor([True, False, False, False, True, False, False, False, False, False]))


def test_planner_precompute_source_length_budget_and_overflow() -> None:
    assert precompute_planner_vlm_inputs._compute_source_max_length(max_length=4096, planner_token_count=2048) == 2046

    raised = False
    try:
        precompute_planner_vlm_inputs._compute_source_max_length(max_length=10, planner_token_count=8)
    except Exception:
        raised = True
    assert raised

    image = GEMMA3_CONFIG_FOR_LTX.image_token_index
    raised = False
    try:
        precompute_planner_vlm_inputs._append_planner_placeholders(
            {"input_ids": torch.arange(7), "attention_mask": torch.ones(7, dtype=torch.long)},
            planner_token_count=2,
            max_length=10,
            source_max_length=6,
            placeholder_token_id=image,
            start_token_id=GEMMA3_CONFIG_FOR_LTX.boi_token_index,
            end_token_id=GEMMA3_CONFIG_FOR_LTX.eoi_token_index,
            pad_token_id=0,
        )
    except ValueError:
        raised = True
    assert raised


def test_reference_image_token_integrity_check_rejects_mismatched_count() -> None:
    raised = False
    try:
        precompute_planner_vlm_inputs._validate_reference_image_token_count(
            {"ref_visual_token_count": torch.tensor(GEMMA3_CONFIG_FOR_LTX.mm_tokens_per_image - 1)},
            num_ref_images=1,
        )
    except ValueError as exc:
        raised = True
        assert "Reference image tokens were truncated or mismatched" in str(exc)
    assert raised


def test_stage2_dropout_masks_are_built_from_original_attention_for_old_inputs() -> None:
    image = GEMMA3_CONFIG_FOR_LTX.image_token_index
    boi = GEMMA3_CONFIG_FOR_LTX.boi_token_index
    eoi = GEMMA3_CONFIG_FOR_LTX.eoi_token_index
    input_ids = torch.tensor(
        [
            [10, boi, image, eoi, 11, boi, image, image, eoi, 0],
            [10, boi, image, eoi, 11, boi, image, image, eoi, 0],
        ]
    )
    original_attention = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
    planner_placeholder = torch.tensor(
        [
            [False, False, False, False, False, False, True, True, False, False],
            [False, False, False, False, False, False, True, True, False, False],
        ]
    )
    planner_boundary = torch.tensor(
        [
            [False, False, False, False, False, True, False, False, True, False],
            [False, False, False, False, False, True, False, False, True, False],
        ]
    )
    planner_data = {
        "planner_placeholder_mask": planner_placeholder,
        "planner_boundary_mask": planner_boundary,
    }
    forward_inputs = {"input_ids": input_ids, "attention_mask": original_attention}
    strategy = MultiReferencePlannerStage2Strategy(MultiReferencePlannerStage2Config())

    dropped_image, dropped_text = strategy._build_vlm_dropout_masks(
        planner_data=planner_data,
        forward_inputs=forward_inputs,
        drop_ref_mask=torch.tensor([False, True]),
        drop_text_mask=torch.tensor([False, True]),
    )
    assert dropped_image is not None
    assert dropped_text is not None
    assert torch.equal(dropped_image[1], torch.tensor([False, True, True, True, False, False, False, False, False, False]))
    assert torch.equal(dropped_text[1], torch.tensor([True, False, False, False, True, False, False, False, False, False]))
    assert not bool((dropped_text & planner_placeholder).any())
    assert not bool((dropped_text & planner_boundary).any())
    assert not bool((dropped_text & dropped_image).any())

    dropped_forward = strategy._apply_vlm_condition_dropout(
        forward_inputs=forward_inputs,
        dropped_image_token_mask=dropped_image,
        dropped_text_token_mask=dropped_text,
    )
    assert torch.equal(dropped_forward["attention_mask"][0], original_attention[0])
    assert torch.equal(dropped_forward["attention_mask"][1], torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 0]))

    dummy_embeds = torch.ones(2, input_ids.shape[1], 3)
    zeroed_embeds = dummy_embeds.masked_fill(dropped_text.unsqueeze(-1), 0)
    assert torch.equal(zeroed_embeds[1, 0], torch.zeros(3))
    assert torch.equal(zeroed_embeds[1, 4], torch.zeros(3))
    assert torch.equal(zeroed_embeds[1, 6], torch.ones(3))


def test_visual_planner_slot_encodings_default_break_symmetry_but_keep_zero_init_adapters() -> None:
    planner = VisualPlannerTokens(
        token_count=4,
        dim=8,
        source_dim=10,
        num_heads=2,
        slot_init_std=1e-4,
        slot_init_seed=0,
    )

    assert not torch.equal(planner.query_slot_encoding, torch.zeros_like(planner.query_slot_encoding))
    assert not torch.equal(planner.kv_slot_encoding, torch.zeros_like(planner.kv_slot_encoding))
    assert torch.equal(planner.query_type_encoding, torch.zeros_like(planner.query_type_encoding))
    assert torch.equal(planner.kv_type_encoding, torch.zeros_like(planner.kv_type_encoding))
    assert torch.equal(planner.output_projection.weight, torch.zeros_like(planner.output_projection.weight))
    assert torch.equal(planner.output_projection.bias, torch.zeros_like(planner.output_projection.bias))
    assert torch.equal(planner.ffn_fc2.weight, torch.zeros_like(planner.ffn_fc2.weight))
    assert torch.equal(planner.ffn_fc2.bias, torch.zeros_like(planner.ffn_fc2.bias))


def test_visual_planner_slot_init_std_zero_recovers_old_zero_slot_behavior() -> None:
    planner = VisualPlannerTokens(
        token_count=4,
        dim=8,
        source_dim=10,
        num_heads=2,
        slot_init_std=0.0,
    )

    assert torch.equal(planner.query_slot_encoding, torch.zeros_like(planner.query_slot_encoding))
    assert torch.equal(planner.kv_slot_encoding, torch.zeros_like(planner.kv_slot_encoding))


def test_visual_planner_slot_init_seed_controls_determinism() -> None:
    first = VisualPlannerTokens(
        token_count=4,
        dim=8,
        source_dim=10,
        num_heads=2,
        slot_init_std=1e-4,
        slot_init_seed=123,
    )
    second = VisualPlannerTokens(
        token_count=4,
        dim=8,
        source_dim=10,
        num_heads=2,
        slot_init_std=1e-4,
        slot_init_seed=123,
    )
    different = VisualPlannerTokens(
        token_count=4,
        dim=8,
        source_dim=10,
        num_heads=2,
        slot_init_std=1e-4,
        slot_init_seed=124,
    )

    assert torch.equal(first.query_slot_encoding, second.query_slot_encoding)
    assert torch.equal(first.kv_slot_encoding, second.kv_slot_encoding)
    assert not torch.equal(first.query_slot_encoding, different.query_slot_encoding)
    assert not torch.equal(first.kv_slot_encoding, different.kv_slot_encoding)


def test_stage2_config_parses_planner_slot_init_fields_and_defaults() -> None:
    default_config = MultiReferencePlannerStage2Config()
    assert default_config.planner_slot_init_std == 1e-4
    assert default_config.planner_slot_init_seed == 0

    explicit_config = MultiReferencePlannerStage2Config(planner_slot_init_std=0.0, planner_slot_init_seed=None)
    assert explicit_config.planner_slot_init_std == 0.0
    assert explicit_config.planner_slot_init_seed is None


class _FakeVideoConnector(nn.Module):
    def __init__(self, dim: int = 4096):
        super().__init__()
        self.inner_dim = dim
        self.num_learnable_registers = 1
        self.learnable_registers = nn.Parameter(torch.zeros(1, dim))


class _FakeEmbeddingsProcessor(nn.Module):
    def __init__(self, dim: int = 4096):
        super().__init__()
        self.video_connector = _FakeVideoConnector(dim)


def _projection_conditions(batch_size: int = 2, seq_len: int = 3, dim: int = 4096) -> dict[str, torch.Tensor]:
    return {
        "video_prompt_embeds": torch.randn(batch_size, seq_len, dim),
        "prompt_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
    }


def _projection_gt_tokens(batch_size: int = 2, token_count: int = 4, source_dim: int = 3840) -> dict[str, torch.Tensor]:
    return {
        "visual_tokens": torch.randn(batch_size, token_count, source_dim),
        "visual_token_mask": torch.ones(batch_size, token_count, dtype=torch.bool),
    }


def test_stage1_projects_gt_visual_tokens_before_append() -> None:
    strategy = MultiReferenceVideoStrategy(
        MultiReferenceVideoConfig(visual_token_source_dim=3840, visual_token_target_dim=4096)
    )
    strategy.attach_models(
        transformer=nn.Identity(),
        embeddings_processor=_FakeEmbeddingsProcessor(4096),
        text_encoder=None,
    )
    conditions = _projection_conditions(batch_size=2, seq_len=3, dim=4096)
    batch = {"gt_visual_tokens": _projection_gt_tokens(batch_size=2, token_count=4, source_dim=3840)}

    out = strategy.prepare_conditions(batch, conditions)

    assert out["video_prompt_embeds"].shape == (2, 7, 4096)
    assert out["prompt_attention_mask"].shape == (2, 7)
    assert bool(out["prompt_attention_mask"][:, -4:].all())


def test_visual_token_projection_module_registered_and_identity_pad_initialized() -> None:
    strategy = MultiReferenceVideoStrategy(
        MultiReferenceVideoConfig(visual_token_source_dim=3840, visual_token_target_dim=4096)
    )
    strategy.attach_models(
        transformer=nn.Identity(),
        embeddings_processor=_FakeEmbeddingsProcessor(4096),
        text_encoder=None,
    )

    modules = strategy.get_trainable_modules()

    assert "visual_token_projection" in modules
    projection = modules["visual_token_projection"]
    assert projection.weight.shape == (4096, 3840)
    assert torch.allclose(projection.weight[:3840, :3840], torch.eye(3840, dtype=projection.weight.dtype))
    assert torch.equal(projection.weight[3840:, :], torch.zeros_like(projection.weight[3840:, :]))


def test_stage2_trainable_modules_include_visual_projection_and_planner_tokens() -> None:
    strategy = MultiReferencePlannerStage2Strategy(
        MultiReferencePlannerStage2Config(
            use_online_vlm=False,
            planner_token_count=4,
            planner_source_dim=3840,
            visual_token_source_dim=3840,
            visual_token_target_dim=4096,
        )
    )
    strategy.attach_models(
        transformer=nn.Identity(),
        embeddings_processor=_FakeEmbeddingsProcessor(4096),
        text_encoder=None,
    )

    modules = strategy.get_trainable_modules()

    assert "visual_token_projection" in modules
    assert "planner_tokens" in modules


def test_stage2_mse_uses_projected_gt_visual_tokens() -> None:
    strategy = MultiReferencePlannerStage2Strategy(
        MultiReferencePlannerStage2Config(
            use_online_vlm=False,
            planner_token_count=4,
            planner_mse_weight=1.0,
            visual_token_source_dim=3840,
            visual_token_target_dim=4096,
        )
    )
    strategy.attach_models(
        transformer=nn.Identity(),
        embeddings_processor=_FakeEmbeddingsProcessor(4096),
        text_encoder=None,
    )
    conditions = _projection_conditions(batch_size=1, seq_len=2, dim=4096)
    batch = {
        "gt_visual_tokens": _projection_gt_tokens(batch_size=1, token_count=4, source_dim=3840),
        "planner_conditions": {
            "predicted_visual_tokens": torch.zeros(1, 4, 4096),
            "predicted_visual_token_mask": torch.ones(1, 4, dtype=torch.bool),
        },
    }

    out = strategy.prepare_conditions(batch, conditions)

    assert out["video_prompt_embeds"].shape == (1, 6, 4096)
    assert strategy._last_planner_mse_loss is not None
    assert strategy._last_planner_mse_loss.shape == (1,)


def test_visual_token_dim_mismatch_without_projection_raises_clear_error() -> None:
    strategy = MultiReferenceVideoStrategy(MultiReferenceVideoConfig())
    strategy.attach_models(
        transformer=nn.Identity(),
        embeddings_processor=_FakeEmbeddingsProcessor(4096),
        text_encoder=None,
    )
    conditions = _projection_conditions(batch_size=1, seq_len=2, dim=4096)
    batch = {"gt_visual_tokens": _projection_gt_tokens(batch_size=1, token_count=4, source_dim=3840)}

    raised = False
    try:
        strategy.prepare_conditions(batch, conditions)
    except ValueError as exc:
        raised = True
        assert "visual_token_source_dim" in str(exc)
        assert "visual_token_target_dim" in str(exc)
    assert raised
