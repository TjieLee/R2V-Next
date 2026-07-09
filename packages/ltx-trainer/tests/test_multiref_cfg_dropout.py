import importlib.util
from pathlib import Path

import torch
from torch import nn

from ltx_core.multicond.cfg_sampler import CFGModeBatch, sample_cfg_modes
from ltx_core.multicond.visual_tokens import Visual3DResampler, VisualPlannerTokens
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

_STAGE1_INFER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "infer_multiref_stage1_overfit.py"
_STAGE1_SPEC = importlib.util.spec_from_file_location("infer_multiref_stage1_overfit", _STAGE1_INFER_SCRIPT)
assert _STAGE1_SPEC is not None and _STAGE1_SPEC.loader is not None
infer_multiref_stage1_overfit = importlib.util.module_from_spec(_STAGE1_SPEC)
_STAGE1_SPEC.loader.exec_module(infer_multiref_stage1_overfit)


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

    def forward(self, hidden_states: torch.Tensor, additive_attention_mask: torch.Tensor):
        return hidden_states, additive_attention_mask


class _FakeEmbeddingsProcessor(nn.Module):
    def __init__(self, dim: int = 4096):
        super().__init__()
        self.video_connector = _FakeVideoConnector(dim)


def _projection_conditions(batch_size: int = 2, seq_len: int = 3, dim: int = 4096) -> dict[str, torch.Tensor]:
    return {
        "video_prompt_embeds": torch.randn(batch_size, seq_len, dim),
        "prompt_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
    }


def _projection_gt_tokens(
    batch_size: int = 2,
    token_count: int = 4,
    source_dim: int = 3840,
    tokens_per_frame: int = 4,
) -> dict[str, torch.Tensor]:
    frame_count = token_count // tokens_per_frame
    return {
        "visual_tokens": torch.randn(batch_size, token_count, source_dim),
        "visual_token_mask": torch.ones(batch_size, token_count, dtype=torch.bool),
        "tokens_per_frame": torch.tensor(tokens_per_frame),
        "sampled_frame_indices": torch.arange(frame_count).repeat(batch_size, 1),
        "source_fps": torch.ones(batch_size),
    }


def test_stage1_visual_tokens_append_after_connector_not_before() -> None:
    strategy = MultiReferenceVideoStrategy(
        MultiReferenceVideoConfig(
            visual_token_source_dim=64,
            visual_token_target_dim=64,
            visual_context_spatial_grid=2,
            visual_context_max_tokens=16,
            visual_resampler_num_heads=8,
            visual_connector_enabled=False,
        )
    )
    strategy.attach_models(
        transformer=nn.Identity(),
        embeddings_processor=_FakeEmbeddingsProcessor(64),
        text_encoder=None,
    )
    conditions = _projection_conditions(batch_size=2, seq_len=3, dim=64)
    batch = {
        "gt_visual_tokens": _projection_gt_tokens(batch_size=2, token_count=4, source_dim=64),
        "latents": {"height": torch.tensor([8, 8]), "width": torch.tensor([8, 8])},
    }

    pre_connector = strategy.prepare_conditions(batch, conditions)
    post_connector = strategy.postprocess_conditions_after_connector(batch, pre_connector)

    assert pre_connector["video_prompt_embeds"].shape == (2, 3, 4096)
    assert post_connector["video_prompt_embeds"].shape == (2, 7, 64)
    assert post_connector["prompt_attention_mask"].shape == (2, 7)
    assert bool(post_connector["prompt_attention_mask"][:, -4:].all())


def test_visual_3d_resampler_shape_and_mask() -> None:
    resampler = Visual3DResampler(
        dim=64,
        max_query_tokens=128,
        num_heads=8,
        depth=1,
        ffn_multiplier=0.5,
    )
    tokens = torch.randn(2, 2048, 64)
    token_positions = torch.zeros(2, 3, 2048, 2)
    token_mask = torch.ones(2, 2048, dtype=torch.bool)
    query_positions = torch.zeros(2, 3, 32, 2)

    out, mask = resampler(
        tokens=tokens,
        token_positions=token_positions,
        token_mask=token_mask,
        query_positions=query_positions,
    )

    assert out.shape == (2, 32, 64)
    assert mask.shape == (2, 32)
    assert torch.isfinite(out).all()


def test_stage1_visual_position_builders_are_monotonic_and_in_range() -> None:
    strategy = MultiReferenceVideoStrategy(
        MultiReferenceVideoConfig(
            visual_token_source_dim=4,
            visual_token_target_dim=4,
            visual_context_spatial_grid=2,
        )
    )
    visual_data = _projection_gt_tokens(batch_size=1, token_count=8, source_dim=4, tokens_per_frame=4)
    visual_data["sampled_frame_indices"] = torch.tensor([[0, 6]])
    visual_data["source_fps"] = torch.tensor([6.0])
    latents_data = {"height": torch.tensor([8]), "width": torch.tensor([16])}

    token_positions = strategy._build_visual_token_positions(
        visual_data,
        latents_data,
        token_count=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    query_positions = strategy._build_visual_query_positions(
        visual_data,
        latents_data,
        output_grid=2,
        frame_stride=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert token_positions.shape == (1, 3, 8, 2)
    assert query_positions.shape == (1, 3, 8, 2)
    assert torch.all(token_positions[:, 0, 4:, 0] >= token_positions[:, 0, :4, 0])
    assert float(query_positions[:, 1, :, 0].max()) <= 8.0
    assert float(query_positions[:, 2, :, 0].max()) <= 16.0


def test_visual_token_projection_module_registered_and_identity_pad_initialized() -> None:
    strategy = MultiReferenceVideoStrategy(
        MultiReferenceVideoConfig(
            visual_token_source_dim=3840,
            visual_token_target_dim=4096,
            visual_branch_enabled=False,
        )
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


def test_stage2_mse_uses_raw_siglip_tokens_then_appends_projected_tokens() -> None:
    strategy = MultiReferencePlannerStage2Strategy(
        MultiReferencePlannerStage2Config(
            use_online_vlm=False,
            planner_token_count=4,
            planner_mse_weight=1.0,
            planner_source_dim=3840,
            planner_output_dim=3840,
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
    gt_tokens = _projection_gt_tokens(batch_size=1, token_count=4, source_dim=3840)
    predicted_raw = torch.zeros(1, 4, 3840)
    batch = {
        "gt_visual_tokens": gt_tokens,
        "planner_conditions": {
            "predicted_visual_tokens": predicted_raw,
            "predicted_visual_token_mask": torch.ones(1, 4, dtype=torch.bool),
        },
    }

    out = strategy.prepare_conditions(batch, conditions)

    assert out["video_prompt_embeds"].shape == (1, 6, 4096)
    assert out["video_prompt_embeds"].shape[-1] == 4096
    assert strategy._last_planner_mse_loss is not None
    expected_loss = torch.mean((predicted_raw - gt_tokens["visual_tokens"]) ** 2, dim=[1, 2])
    assert torch.allclose(strategy._last_planner_mse_loss, expected_loss)


def test_visual_token_dim_mismatch_without_projection_raises_clear_error() -> None:
    strategy = MultiReferenceVideoStrategy(
        MultiReferenceVideoConfig(
            visual_resampler_num_heads=1,
            visual_context_spatial_grid=2,
            visual_context_max_tokens=4,
            visual_connector_enabled=False,
        )
    )
    strategy.attach_models(
        transformer=nn.Identity(),
        embeddings_processor=_FakeEmbeddingsProcessor(5),
        text_encoder=None,
    )
    conditions = _projection_conditions(batch_size=1, seq_len=2, dim=5)
    batch = {
        "gt_visual_tokens": _projection_gt_tokens(batch_size=1, token_count=4, source_dim=3),
        "latents": {"height": torch.tensor([2]), "width": torch.tensor([2])},
    }

    raised = False
    try:
        pre_connector = strategy.prepare_conditions(batch, conditions)
        strategy.postprocess_conditions_after_connector(batch, pre_connector)
    except ValueError as exc:
        raised = True
        assert "visual_token_source_dim" in str(exc)
        assert "visual_token_target_dim" in str(exc)
    assert raised


def test_stage1_infer_manifest_readers_support_json_jsonl_and_csv(tmp_path) -> None:
    json_path = tmp_path / "samples.json"
    json_path.write_text('[{"video": "a.mp4"}, {"video": "b.mp4"}]', encoding="utf-8")
    assert [row["video"] for row in infer_multiref_stage1_overfit._read_manifest(json_path)] == ["a.mp4", "b.mp4"]

    jsonl_path = tmp_path / "samples.jsonl"
    jsonl_path.write_text('{"video": "c.mp4"}\n{"video": "d.mp4"}\n', encoding="utf-8")
    assert [row["video"] for row in infer_multiref_stage1_overfit._read_manifest(jsonl_path)] == ["c.mp4", "d.mp4"]

    csv_path = tmp_path / "samples.csv"
    csv_path.write_text("video,caption\ne.mp4,hello\n", encoding="utf-8")
    assert infer_multiref_stage1_overfit._read_manifest(csv_path)[0]["video"] == "e.mp4"


def test_stage1_infer_precomputed_relative_path_matches_absolute_video_path() -> None:
    manifest_root = Path("/tmp/manifest_root")
    video_path = Path("/mnt/workspace/public/dataset/phantom_data/part_000/demo/demo.mp4")
    rel_path = infer_multiref_stage1_overfit._output_relative(video_path, manifest_root).with_suffix(".pt")

    assert rel_path == Path("mnt/workspace/public/dataset/phantom_data/part_000/demo/demo.pt")


def test_stage1_infer_batch_construction_and_postconnector_visual_shape() -> None:
    latents = {
        "latents": torch.randn(128, 1, 2, 2),
        "num_frames": 1,
        "height": 2,
        "width": 2,
        "fps": 6.0,
    }
    conditions = {
        "video_prompt_embeds": torch.randn(2, 5),
        "prompt_attention_mask": torch.ones(2, dtype=torch.bool),
    }
    multi_reference_latents = {
        "latents": torch.randn(1, 128, 1, 2, 2),
        "ref_valid_mask": torch.ones(1, dtype=torch.bool),
        "num_refs": 1,
    }
    gt_visual_tokens = {
        "visual_tokens": torch.randn(4, 3),
        "visual_token_mask": torch.ones(4, dtype=torch.bool),
        "tokens_per_frame": torch.tensor(4),
        "sampled_frame_indices": torch.tensor([0]),
        "source_fps": torch.tensor(1.0),
    }
    batch = infer_multiref_stage1_overfit._build_single_sample_batch(
        latents=latents,
        conditions=conditions,
        multi_reference_latents=multi_reference_latents,
        gt_visual_tokens=gt_visual_tokens,
    )

    assert batch["latents"]["latents"].shape == (1, 128, 1, 2, 2)
    assert batch["conditions"]["video_prompt_embeds"].shape == (1, 2, 5)
    assert batch["gt_visual_tokens"]["visual_tokens"].shape == (1, 4, 3)

    strategy = MultiReferenceVideoStrategy(
        MultiReferenceVideoConfig(
            visual_token_source_dim=3,
            visual_token_target_dim=5,
            visual_context_spatial_grid=2,
            visual_context_max_tokens=4,
            visual_resampler_num_heads=1,
            visual_connector_enabled=False,
        )
    )
    strategy.attach_models(
        transformer=nn.Identity(),
        embeddings_processor=_FakeEmbeddingsProcessor(5),
        text_encoder=None,
    )
    pre_connector = strategy.prepare_conditions(batch, batch["conditions"])
    out = strategy.postprocess_conditions_after_connector(batch, pre_connector)

    assert pre_connector["video_prompt_embeds"].shape == (1, 2, 5)
    assert out["video_prompt_embeds"].shape == (1, 6, 5)
    assert out["prompt_attention_mask"].shape == (1, 6)
    assert bool(out["prompt_attention_mask"][:, -4:].all())


def test_stage1_infer_velocity_to_denoised_uses_per_token_timestep_broadcast() -> None:
    latent = torch.randn(1, 2880, 128)
    velocity = torch.randn(1, 2880, 128)
    timesteps = torch.linspace(0, 1, 2880).reshape(1, 2880)

    denoised = infer_multiref_stage1_overfit._velocity_to_denoised(latent, velocity, timesteps)
    expected = latent.to(torch.float32) - velocity.to(torch.float32) * timesteps.unsqueeze(-1)

    assert denoised.shape == (1, 2880, 128)
    assert torch.allclose(denoised, expected.to(latent.dtype))


def test_stage1_infer_old_timestep_broadcast_shape_would_fail() -> None:
    latent = torch.randn(1, 2880, 128)
    velocity = torch.randn(1, 2880, 128)
    timesteps = torch.linspace(0, 1, 2880).reshape(1, 2880)

    raised = False
    try:
        _ = latent - velocity * timesteps
    except RuntimeError as exc:
        raised = True
        assert "must match" in str(exc)
    assert raised

def test_visual_planner_learned_query_mode_outputs_raw_siglip_dim() -> None:
    planner = VisualPlannerTokens(
        token_count=8,
        dim=3840,
        source_dim=3840,
        num_heads=16,
        ffn_multiplier=0.01,
        use_learned_query_tokens=True,
        query_init_std=1e-4,
    )
    planner_hidden = torch.randn(2, 8, 3840)

    out = planner(planner_hidden=planner_hidden, query_registers=None)

    assert out.shape == (2, 8, 3840)
    assert planner.query_tokens is not None
    assert planner.query_tokens.shape == (8, 3840)


def test_visual_planner_legacy_external_query_mode_still_outputs_connector_dim() -> None:
    planner = VisualPlannerTokens(
        token_count=8,
        dim=4096,
        source_dim=3840,
        num_heads=16,
        ffn_multiplier=0.01,
        use_learned_query_tokens=False,
    )
    planner_hidden = torch.randn(2, 8, 3840)
    query_registers = torch.randn(128, 4096)

    out = planner(planner_hidden=planner_hidden, query_registers=query_registers)

    assert out.shape == (2, 8, 4096)


def test_stage1_train_text_connector_flag_requests_embeddings_processor_training() -> None:
    strategy = MultiReferenceVideoStrategy(MultiReferenceVideoConfig(train_text_connector=True))

    assert strategy.train_embeddings_processor() is True


def test_stage2_default_uses_learned_3840_query_tokens_not_connector_registers() -> None:
    strategy = MultiReferencePlannerStage2Strategy(
        MultiReferencePlannerStage2Config(
            use_online_vlm=False,
            planner_token_count=4,
            planner_source_dim=3840,
            planner_output_dim=3840,
            visual_token_source_dim=3840,
            visual_token_target_dim=4096,
        )
    )
    strategy.attach_models(
        transformer=nn.Identity(),
        embeddings_processor=_FakeEmbeddingsProcessor(4096),
        text_encoder=None,
    )

    assert strategy._planner_query_registers is None
    assert strategy.planner_tokens is not None
    assert strategy.planner_tokens.dim == 3840
    assert strategy.planner_tokens.query_tokens is not None


def test_stage2_connector_register_query_dim_mismatch_raises() -> None:
    strategy = MultiReferencePlannerStage2Strategy(
        MultiReferencePlannerStage2Config(
            use_online_vlm=False,
            planner_token_count=4,
            planner_source_dim=3840,
            planner_output_dim=3840,
            use_connector_register_queries=True,
            visual_token_source_dim=3840,
            visual_token_target_dim=4096,
        )
    )

    raised = False
    try:
        strategy.attach_models(
            transformer=nn.Identity(),
            embeddings_processor=_FakeEmbeddingsProcessor(4096),
            text_encoder=None,
        )
    except ValueError as exc:
        raised = True
        assert "connector dim 4096 != planner_output_dim 3840" in str(exc)
    assert raised


def test_stage2_offline_predicted_tokens_reject_projected_4096_dim() -> None:
    strategy = MultiReferencePlannerStage2Strategy(
        MultiReferencePlannerStage2Config(
            use_online_vlm=False,
            planner_token_count=4,
            planner_source_dim=3840,
            planner_output_dim=3840,
            visual_token_source_dim=3840,
            visual_token_target_dim=4096,
        )
    )
    gt_tokens = torch.randn(1, 4, 3840)
    planner_data = {
        "predicted_visual_tokens": torch.randn(1, 4, 4096),
        "predicted_visual_token_mask": torch.ones(1, 4, dtype=torch.bool),
    }

    raised = False
    try:
        strategy._load_offline_predicted_tokens(planner_data, gt_tokens)
    except ValueError as exc:
        raised = True
        assert "raw SigLIP-space tokens" in str(exc)
        assert "got dim=4096" in str(exc)
    assert raised


def test_stage1_infer_cfg_combine_formula_matches_ltx_guidance() -> None:
    pos = torch.tensor([[[2.0, 4.0]]])
    neg = torch.tensor([[[1.0, 3.0]]])

    assert torch.equal(infer_multiref_stage1_overfit._combine_cfg_denoised(pos, neg, 1.0), pos)
    assert torch.equal(infer_multiref_stage1_overfit._combine_cfg_denoised(pos, neg, 2.0), 2 * pos - neg)


def test_stage1_infer_cfg_stg_combine_formula_matches_expected_guidance() -> None:
    pos = torch.tensor([[[2.0, 4.0]]])
    neg = torch.tensor([[[1.0, 3.0]]])
    stg = torch.tensor([[[0.5, 2.0]]])

    actual = infer_multiref_stage1_overfit._combine_cfg_stg_denoised(
        denoised_pos=pos,
        denoised_neg=neg,
        denoised_stg=stg,
        guidance_scale=1.5,
        stg_scale=0.4,
    )
    expected = neg + 1.5 * (pos - neg) + 0.4 * (pos - stg)

    assert torch.allclose(actual, expected)


def test_stage1_infer_cfg_stg_combine_handles_disabled_branches() -> None:
    pos = torch.tensor([[[2.0, 4.0]]])
    neg = torch.tensor([[[1.0, 3.0]]])
    stg = torch.tensor([[[0.5, 2.0]]])

    no_guidance = infer_multiref_stage1_overfit._combine_cfg_stg_denoised(
        denoised_pos=pos,
        denoised_neg=None,
        denoised_stg=None,
        guidance_scale=1.0,
        stg_scale=0.0,
    )
    stg_only = infer_multiref_stage1_overfit._combine_cfg_stg_denoised(
        denoised_pos=pos,
        denoised_neg=None,
        denoised_stg=stg,
        guidance_scale=1.0,
        stg_scale=0.5,
    )
    cfg_only = infer_multiref_stage1_overfit._combine_cfg_stg_denoised(
        denoised_pos=pos,
        denoised_neg=neg,
        denoised_stg=None,
        guidance_scale=1.5,
        stg_scale=0.0,
    )

    assert torch.equal(no_guidance, pos)
    assert torch.allclose(stg_only, pos + 0.5 * (pos - stg))
    assert torch.allclose(cfg_only, neg + 1.5 * (pos - neg))


def test_stage1_infer_cfg_enabled_only_when_guidance_scale_not_one() -> None:
    assert infer_multiref_stage1_overfit._cfg_enabled(1.0) is False
    assert infer_multiref_stage1_overfit._cfg_enabled(1.2) is True


def test_stage1_infer_stg_helpers_parse_blocks_and_config() -> None:
    assert infer_multiref_stage1_overfit._stg_enabled(0.0) is False
    assert infer_multiref_stage1_overfit._stg_enabled(0.5) is True
    assert infer_multiref_stage1_overfit._parse_stg_blocks(None) == [29]
    assert infer_multiref_stage1_overfit._parse_stg_blocks("29, 30") == [29, 30]
    assert infer_multiref_stage1_overfit._parse_stg_blocks("none") is None

    cfg = infer_multiref_stage1_overfit._build_stg_perturbation_config([29])
    assert len(cfg.perturbations) == 1
    assert cfg.perturbations[0].perturbations[0].blocks == [29]


def test_stage1_infer_negative_ref_valid_mask_can_keep_or_drop_references() -> None:
    ref_mask = torch.tensor([[True, False, True]])

    kept = infer_multiref_stage1_overfit._negative_ref_valid_mask(ref_mask, drop_ref_latents=False)
    dropped = infer_multiref_stage1_overfit._negative_ref_valid_mask(ref_mask, drop_ref_latents=True)

    assert torch.equal(kept, ref_mask)
    assert torch.equal(dropped, torch.zeros_like(ref_mask))



def test_stage1_infer_condition_mode_detail_tracks_ablation_and_guidance() -> None:
    assert (
        infer_multiref_stage1_overfit._condition_mode_detail(
            "full_siglip",
            guidance_scale=1.0,
            stg_scale=0.0,
        )
        == "stage1_teacher_gt_siglip_full_condition"
    )
    assert (
        infer_multiref_stage1_overfit._condition_mode_detail(
            "text_only_no_siglip",
            guidance_scale=1.0,
            stg_scale=0.0,
        )
        == "stage1_text_only_no_siglip_with_reference_latents"
    )
    assert (
        infer_multiref_stage1_overfit._condition_mode_detail(
            "text_only_no_siglip",
            guidance_scale=1.5,
            stg_scale=0.5,
        )
        == "stage1_text_only_no_siglip_cfg_stg"
    )


def test_stage1_text_only_batch_omits_vlm_and_gt_siglip_entries() -> None:
    batch = infer_multiref_stage1_overfit._build_single_sample_batch(
        latents={"latents": torch.zeros(2, 1, 1, 1)},
        multi_reference_latents={"latents": torch.zeros(1, 2, 1, 1, 1)},
    )

    assert "latents" in batch
    assert "multi_ref_latents" in batch
    assert "conditions" not in batch
    assert "gt_visual_tokens" not in batch


def test_stage1_text_only_precompute_loading_does_not_require_siglip_or_vlm(tmp_path: Path) -> None:
    precomputed_root = tmp_path / ".precomputed"
    rel_path = Path("clip.pt")
    latents_path = precomputed_root / "latents" / rel_path
    ref_path = precomputed_root / "multi_reference_latents" / rel_path
    latents_path.parent.mkdir(parents=True)
    ref_path.parent.mkdir(parents=True)
    torch.save({"latents": torch.zeros(2, 1, 1, 1)}, latents_path)
    torch.save({"latents": torch.zeros(1, 2, 1, 1, 1)}, ref_path)

    loaded_rel_path, precomputed = infer_multiref_stage1_overfit._load_sample_precomputed(
        row={"video": "clip.mp4"},
        manifest_root=tmp_path,
        precomputed_root=precomputed_root,
        video_column="video",
        condition_mode="text_only_no_siglip",
    )

    assert loaded_rel_path == rel_path
    assert set(precomputed) == {"latents", "multi_reference_latents"}
