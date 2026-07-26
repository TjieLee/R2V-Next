from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ltx_core.guidance.perturbations import PerturbationType
from ltx_core.types import VideoLatentShape
from ltx_trainer.online_data.online_batch_encoder import (
    OnlineBatchEncoder,
    _zero_condition_tensors,
)
from ltx_trainer.online_inference.checkpoint_runtime import OnlineInferenceRuntime
from ltx_trainer.online_inference.semantic_guidance import (
    GuidanceMode,
    SemanticGuidanceConfig,
    SemanticGuidanceStateBundle,
    build_stg_perturbation,
    combine_guided_denoised,
    parse_stg_blocks,
    rescale_guided_denoised,
)
from ltx_trainer.training_strategies.semantic_flow import (
    ENTITY_GLOBAL,
    SemanticFlowConfig,
    SemanticFlowStrategy,
)


def test_guidance_formula_and_forward_counts() -> None:
    positive = torch.tensor([[[5.0, 7.0]]])
    negative = torch.tensor([[[1.0, 2.0]]])
    no_reference = torch.tensor([[[3.0, 3.0]]])
    stg = torch.tensor([[[2.0, 6.0]]])
    config = SemanticGuidanceConfig(
        guidance_scale=4.0,
        ref_guidance_scale=1.0,
        guidance_rescale=0.0,
        stg_scale=0.5,
    )
    actual = combine_guided_denoised(
        positive=positive,
        negative=negative,
        no_reference=no_reference,
        stg=stg,
        config=config,
    )
    expected = negative + 4.0 * (positive - negative) + (positive - no_reference)
    expected = expected + 0.5 * (positive - stg)
    assert torch.equal(actual, expected)
    assert config.transformer_forwards_per_step == 4
    assert config.enabled_branches == ("P", "N", "Q", "S")
    assert SemanticGuidanceConfig(stg_scale=0.0).transformer_forwards_per_step == 3
    metadata = SemanticGuidanceConfig().metadata(negative_prompt="negative")
    assert metadata["guidance_mode"] == "positive_ref"
    assert metadata["enabled_guidance_branches"] == ["P", "N", "Q"]
    assert metadata["transformer_forwards_per_step"] == 3
    assert metadata["ref_formula"] == "ref*(P-Q)"
    assert metadata["reference_guidance_training_match"] == "drop_reference_all"
    assert metadata["Q_prompt"] == "positive"
    assert metadata["Q_reference_vlm_images"] == "absent"
    assert metadata["Q_reference_latents"] == "absent"
    assert "R" not in metadata["enabled_guidance_branches"]
    assert "U" not in metadata["enabled_guidance_branches"]
    assert (
        SemanticGuidanceConfig(
            guidance_scale=1.0,
            ref_guidance_scale=0.0,
            guidance_rescale=0.0,
        ).transformer_forwards_per_step
        == 1
    )


def test_debiased_reference_guidance_formula_counts_and_metadata() -> None:
    positive = torch.tensor([[[5.0, 7.0]]])
    negative = torch.tensor([[[1.0, 2.0]]])
    empty_reference = torch.tensor([[[4.0, 6.0]]])
    empty_no_reference = torch.tensor([[[2.0, 1.0]]])
    stg = torch.tensor([[[2.0, 6.0]]])
    config = SemanticGuidanceConfig(
        guidance_mode="debiased_ref",
        guidance_scale=4.0,
        ref_guidance_scale=1.5,
        guidance_rescale=0.0,
        stg_scale=0.5,
    )
    actual = combine_guided_denoised(
        positive=positive,
        negative=negative,
        empty_reference=empty_reference,
        empty_no_reference=empty_no_reference,
        stg=stg,
        config=config,
    )
    control_delta = empty_reference - empty_no_reference
    semantic_delta = positive - negative - control_delta
    expected = positive + 3.0 * semantic_delta
    expected = expected + 1.5 * control_delta
    expected = expected + 0.5 * (positive - stg)
    assert torch.equal(actual, expected)
    assert config.transformer_forwards_per_step == 5
    assert config.enabled_branches == ("P", "N", "R", "U", "S")
    metadata = config.metadata(negative_prompt="negative")
    assert metadata["guidance_mode"] == "debiased_ref"
    assert metadata["cfg_formula"] == "P + (cfg-1)*((P-N)-(R-U))"
    assert metadata["semantic_delta_formula"] == "(P-N)-(R-U)"
    assert metadata["control_delta_formula"] == "R-U"
    assert metadata["ref_formula"] == "ref*(R-U)"
    assert metadata["stg_formula"] == "stg*(P-S)"
    assert metadata["N_vlm_references"] == "absent"
    assert metadata["R_text"] == "empty"
    assert metadata["R_reference_latents"] == "present"
    assert metadata["U_text"] == "drop_all_zero_conditions"
    assert metadata["U_reference_latents"] == "absent"
    assert metadata["control_main_effect_in_cfg"] == "subtracted"
    assert metadata["reference_guidance_training_match"] == "drop_text_vs_drop_all"


@pytest.mark.parametrize("guidance_scale", [2.0, 4.0, 8.0])
def test_debiased_cfg_does_not_repeat_control_main_effect(
    guidance_scale: float,
) -> None:
    empty_no_reference = torch.tensor([[[1.0, 2.0]]])
    empty_reference = torch.tensor([[[3.0, 5.0]]])
    negative = torch.tensor([[[4.0, 6.0]]])
    control_delta = empty_reference - empty_no_reference
    positive = negative + control_delta
    config = SemanticGuidanceConfig(
        guidance_mode="debiased_ref",
        guidance_scale=guidance_scale,
        ref_guidance_scale=0.0,
        guidance_rescale=0.0,
    )
    actual = combine_guided_denoised(
        positive=positive,
        negative=negative,
        empty_reference=empty_reference,
        empty_no_reference=empty_no_reference,
        config=config,
    )
    assert torch.equal(actual, positive)
    assert config.enabled_branches == ("P", "N", "R", "U")
    assert config.transformer_forwards_per_step == 4


def test_guidance_mode_defaults_and_validation() -> None:
    assert SemanticGuidanceConfig().guidance_mode == "positive_ref"
    with pytest.raises(ValueError, match="guidance_mode"):
        SemanticGuidanceConfig(guidance_mode="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="guidance_mode"):
        SemanticGuidanceConfig(guidance_mode="multimodal_ref")  # type: ignore[arg-type]


def test_zero_condition_tensors_preserves_source_and_non_tensors() -> None:
    source = {
        "embeds": torch.ones(1, 2, 8),
        "mask": torch.ones(1, 2, dtype=torch.bool),
        "metadata": "unchanged",
    }

    zeroed = _zero_condition_tensors(source)

    assert zeroed is not source
    assert torch.count_nonzero(zeroed["embeds"]) == 0
    assert torch.count_nonzero(zeroed["mask"]) == 0
    assert zeroed["metadata"] == "unchanged"
    assert torch.count_nonzero(source["embeds"]) > 0
    assert torch.count_nonzero(source["mask"]) > 0


def test_training_drop_all_reuses_zero_condition_helper() -> None:
    source_path = (
        Path(__file__).parents[1]
        / "src"
        / "ltx_trainer"
        / "online_data"
        / "online_batch_encoder.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    encoder_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "OnlineBatchEncoder"
    )
    encode_for_strategy = next(
        node
        for node in encoder_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "encode_for_strategy"
    )
    drop_all_blocks = [
        node
        for node in ast.walk(encode_for_strategy)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "condition_mode"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "drop_all"
    ]
    assert len(drop_all_blocks) == 1
    helper_calls = [
        node
        for statement in drop_all_blocks[0].body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_zero_condition_tensors"
    ]
    assert len(helper_calls) == 1
    assert ast.unparse(helper_calls[0]) == "_zero_condition_tensors(conditions)"


@pytest.mark.parametrize(
    "script_name",
    [
        "infer_external_r2v_eval.py",
        "infer_multitask_online_train_samples.py",
    ],
)
def test_cli_guidance_mode_defaults_to_positive_ref(script_name: str) -> None:
    script = Path(__file__).parents[1] / "scripts" / script_name
    module = ast.parse(script.read_text(encoding="utf-8"))
    main = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    defaults = dict(
        zip(
            (argument.arg for argument in main.args.args[-len(main.args.defaults) :]),
            main.args.defaults,
            strict=True,
        )
    )
    option = defaults["guidance_mode"]
    assert isinstance(option, ast.Call)
    assert isinstance(option.args[0], ast.Constant)
    assert option.args[0].value == "positive_ref"
    assert any(
        isinstance(argument, ast.Constant) and argument.value == "--guidance-mode"
        for argument in option.args
    )
    source = script.read_text(encoding="utf-8")
    assert "positive_ref or debiased_ref" in source
    assert "multimodal_ref" not in source


@pytest.mark.parametrize(
    ("config", "kwargs", "expected"),
    [
        (
            SemanticGuidanceConfig(
                guidance_scale=3.0,
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
            ),
            {"negative": torch.tensor([[[1.0]]])},
            torch.tensor([[[4.0]]]),
        ),
        (
            SemanticGuidanceConfig(
                guidance_scale=1.0,
                ref_guidance_scale=2.0,
                guidance_rescale=0.0,
            ),
            {"no_reference": torch.tensor([[[1.0]]])},
            torch.tensor([[[4.0]]]),
        ),
        (
            SemanticGuidanceConfig(
                guidance_scale=1.0,
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
                stg_scale=0.5,
            ),
            {"stg": torch.tensor([[[0.0]]])},
            torch.tensor([[[3.0]]]),
        ),
        (
            SemanticGuidanceConfig(
                guidance_scale=1.0,
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
            ),
            {},
            torch.tensor([[[2.0]]]),
        ),
    ],
)
def test_individual_guidance_components(
    config: SemanticGuidanceConfig,
    kwargs: dict[str, torch.Tensor],
    expected: torch.Tensor,
) -> None:
    assert torch.equal(
        combine_guided_denoised(
            positive=torch.tensor([[[2.0]]]),
            config=config,
            **kwargs,
        ),
        expected,
    )


def test_rescale_uses_video_statistics_and_applies_one_factor_to_joint_span() -> None:
    positive = torch.tensor([[[100.0], [200.0], [1.0], [3.0]]])
    guided = torch.tensor([[[10.0], [20.0], [2.0], [10.0]]])
    scaled, factor = rescale_guided_denoised(
        positive_generated=positive,
        guided_generated=guided,
        semantic_token_count=2,
        guidance_rescale=0.7,
    )
    expected_factor = 0.7 * (1.0 / 4.0) + 0.3
    torch.testing.assert_close(factor, torch.tensor([expected_factor]))
    torch.testing.assert_close(scaled, guided * expected_factor)
    unchanged, disabled_factor = rescale_guided_denoised(
        positive_generated=positive,
        guided_generated=torch.ones_like(guided),
        semantic_token_count=2,
        guidance_rescale=0.0,
    )
    assert torch.equal(unchanged, torch.ones_like(guided))
    assert torch.equal(disabled_factor, torch.ones(1))


def test_rescale_near_zero_guided_std_is_finite() -> None:
    scaled, factor = rescale_guided_denoised(
        positive_generated=torch.tensor([[[1.0], [1.0], [0.0], [2.0]]]),
        guided_generated=torch.ones(1, 4, 1),
        semantic_token_count=2,
        guidance_rescale=0.7,
    )
    assert torch.isfinite(scaled).all()
    assert torch.isfinite(factor).all()


def test_stg_parse_metadata_and_perturbation() -> None:
    assert parse_stg_blocks("28, 28,3") == (28, 3)
    config = SemanticGuidanceConfig(stg_blocks=parse_stg_blocks("28"))
    metadata = config.metadata(negative_prompt="bad")
    assert metadata["stg_blocks_zero_based"] == [28]
    assert metadata["stg_layers_one_based"] == [29]
    perturbation = build_stg_perturbation((28,), batch_size=1)
    assert perturbation.perturbations[0].is_perturbed(
        PerturbationType.SKIP_VIDEO_SELF_ATTN,
        28,
    )
    with pytest.raises(ValueError):
        parse_stg_blocks("-1")


def _state_bundle(
    config: SemanticGuidanceConfig,
) -> tuple[SemanticFlowStrategy, SemanticGuidanceStateBundle]:
    strategy = SemanticFlowStrategy(SemanticFlowConfig())
    strategy._semantic_dim = 128
    references = {
        "latents": torch.ones(1, 1, 128, 1, 2, 2),
        "ref_valid_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    target_shape = VideoLatentShape(
        batch=1,
        channels=128,
        frames=1,
        height=2,
        width=2,
    )

    def prepare(context_value: float, refs: dict[str, torch.Tensor], noise=None):
        kwargs = {}
        if noise is not None:
            kwargs = {"semantic_noise": noise[0], "target_noise": noise[1]}
        return strategy.prepare_inference_state(
            conditions={
                "video_prompt_embeds": torch.full((1, 2, 8), context_value),
                "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
            },
            reference_latents=refs,
            target_shape=target_shape,
            semantic_frame_count=1,
            pixel_frame_count=1,
            fps=1.0,
            seed=11,
            **kwargs,
        )

    positive = prepare(1.0, references)
    offsets = positive.sequence_offsets
    noise = (
        positive.modality.latent[:, offsets["reference_end"] : offsets["semantic_end"]],
        positive.modality.latent[:, offsets["semantic_end"] : offsets["target_end"]],
    )
    no_refs = dict(references)
    no_refs["ref_valid_mask"] = torch.zeros_like(references["ref_valid_mask"])
    negative_references = references if config.guidance_mode == "positive_ref" else no_refs
    negative = prepare(-1.0, negative_references, noise) if config.need_negative else None
    no_reference = None
    empty_reference = None
    empty_no_reference = None
    if config.need_reference and config.guidance_mode == "positive_ref":
        no_reference = prepare(0.0, no_refs, noise)
    elif config.need_control_pair:
        empty_reference = prepare(0.0, references, noise)
        empty_no_reference = prepare(0.0, no_refs, noise)
    return strategy, SemanticGuidanceStateBundle(
        positive=positive,
        negative=negative,
        no_reference=no_reference,
        empty_reference=empty_reference,
        empty_no_reference=empty_no_reference,
    )


class _CountingBranchTransformer(nn.Module):
    def __init__(self, reference_end: int) -> None:
        super().__init__()
        self.reference_end = reference_end
        self.transformer_blocks = nn.ModuleList([nn.Identity() for _ in range(29)])
        self.calls: list[dict[str, object]] = []

    def forward(self, *, video, audio, perturbations):
        del audio
        context = float(video.context.mean().item())
        reference_nonzero = bool(torch.count_nonzero(video.latent[:, : self.reference_end]))
        if perturbations is not None:
            value = 5.0
        elif context > 0:
            value = 1.0
        elif context < 0:
            value = 2.0
        elif reference_nonzero:
            value = 3.0
        else:
            value = 4.0
        self.calls.append(
            {
                "generated": video.latent[:, self.reference_end :].clone(),
                "reference_nonzero": reference_nonzero,
                "perturbed": perturbations is not None,
            }
        )
        return torch.full_like(video.latent, value), None


@pytest.mark.parametrize(
    ("config", "forward_count"),
    [
        (SemanticGuidanceConfig(guidance_rescale=0.0), 3),
        (
            SemanticGuidanceConfig(
                guidance_scale=1.0,
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
            ),
            1,
        ),
        (
            SemanticGuidanceConfig(
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
            ),
            2,
        ),
        (
            SemanticGuidanceConfig(
                guidance_scale=1.0,
                guidance_rescale=0.0,
            ),
            2,
        ),
        (SemanticGuidanceConfig(guidance_rescale=0.0, stg_scale=0.5), 4),
        (
            SemanticGuidanceConfig(
                guidance_mode="debiased_ref",
                guidance_rescale=0.0,
            ),
            4,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="debiased_ref",
                guidance_scale=1.0,
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
            ),
            1,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="debiased_ref",
                guidance_scale=1.0,
                guidance_rescale=0.0,
            ),
            3,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="debiased_ref",
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
            ),
            4,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="debiased_ref",
                guidance_rescale=0.0,
                stg_scale=0.5,
            ),
            5,
        ),
    ],
)
def test_joint_guidance_forward_count_and_shared_trajectory(
    config: SemanticGuidanceConfig,
    forward_count: int,
) -> None:
    strategy, states = _state_bundle(config)
    reference_end = states.positive.sequence_offsets["reference_end"]
    transformer = _CountingBranchTransformer(reference_end)
    semantic, video = strategy.denoise_joint_guided(
        transformer=transformer,
        states=states,
        guidance=config,
        num_inference_steps=1,
    )
    assert len(transformer.calls) == forward_count
    assert semantic.shape == (1, 64, 128)
    assert video.shape == (1, 128, 1, 2, 2)
    assert all(
        torch.equal(call["generated"], transformer.calls[0]["generated"])
        for call in transformer.calls
    )
    if config.need_reference_comparison:
        assert any(call["reference_nonzero"] is False for call in transformer.calls)
    if config.need_stg:
        assert sum(call["perturbed"] is True for call in transformer.calls) == 1


def test_positive_ref_guided_denoise_is_bitwise_legacy_formula() -> None:
    config = SemanticGuidanceConfig(guidance_rescale=0.0)
    strategy, states = _state_bundle(config)
    offsets = states.positive.sequence_offsets
    ref_end = offsets["reference_end"]
    semantic_end = offsets["semantic_end"]
    target_end = offsets["target_end"]
    current = states.positive.modality.latent[:, ref_end:target_end]
    transformer = _CountingBranchTransformer(ref_end)
    semantic, video = strategy.denoise_joint_guided(
        transformer=transformer,
        states=states,
        guidance=config,
        num_inference_steps=1,
    )
    positive = current - torch.full_like(current, 1.0)
    negative = current - torch.full_like(current, 2.0)
    no_reference = current - torch.full_like(current, 4.0)
    guided = negative + 4.0 * (positive - negative)
    guided = guided + (positive - no_reference)
    velocity = current - guided
    expected = current + torch.tensor(-1.0, dtype=current.dtype) * velocity
    semantic_length = semantic_end - ref_end
    expected_video = strategy._video_patchifier.unpatchify(
        expected[:, semantic_length:],
        states.positive.target_shape,
    )
    assert torch.equal(semantic, expected[:, :semantic_length])
    assert torch.equal(video, expected_video)
    assert [call["reference_nonzero"] for call in transformer.calls] == [True, True, False]


def test_branch_validation_rejects_non_shared_noise() -> None:
    config = SemanticGuidanceConfig(guidance_rescale=0.0)
    strategy, states = _state_bundle(config)
    assert states.negative is not None
    changed = replace(
        states.negative,
        modality=replace(
            states.negative.modality,
            latent=states.negative.modality.latent + 1.0,
        ),
    )
    with pytest.raises(ValueError, match="generated noise"):
        strategy.validate_guidance_state_bundle(
            replace(states, negative=changed),
            config,
        )


def test_runtime_builds_isolated_branches_from_one_reference_and_noise_set() -> None:
    strategy = SemanticFlowStrategy(SemanticFlowConfig(max_ref_images_per_sample=1))
    strategy._semantic_dim = 128
    runtime = object.__new__(OnlineInferenceRuntime)
    runtime.strategy = strategy
    runtime.transformer = _CountingBranchTransformer(reference_end=4)
    runtime.connector_conditions = lambda conditions: conditions  # type: ignore[method-assign]
    condition = lambda value: {
        "video_prompt_embeds": torch.full((1, 2, 8), value),
        "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    encoded = {
        "task": "r2v",
        "positive_conditions": condition(1.0),
        "negative_conditions": condition(-1.0),
        "no_reference_conditions": condition(0.0),
        "reference_latents": {
            "latents": torch.ones(1, 1, 128, 1, 2, 2),
            "ref_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        },
    }
    guidance = SemanticGuidanceConfig(guidance_rescale=0.0)
    states = runtime.prepare_guidance_states(
        encoded,
        width=64,
        height=64,
        num_frames=9,
        fps=24.0,
        seed=9,
        guidance=guidance,
        negative_prompt="negative",
    )
    assert states.negative is not None
    assert states.no_reference is not None
    offsets = states.positive.sequence_offsets
    ref_end = offsets["reference_end"]
    generated = states.positive.modality.latent[:, ref_end:]
    for branch in (states.negative, states.no_reference):
        assert torch.equal(branch.modality.latent[:, ref_end:], generated)
        assert branch.sequence_offsets == offsets
        assert torch.equal(
            branch.modality.entity_ids[:, ref_end:],
            states.positive.modality.entity_ids[:, ref_end:],
        )
    assert torch.equal(
        states.positive.modality.latent[:, :ref_end],
        states.negative.modality.latent[:, :ref_end],
    )
    assert torch.count_nonzero(states.no_reference.modality.latent[:, :ref_end]) == 0
    assert torch.all(
        states.no_reference.modality.entity_ids[:, :ref_end] == ENTITY_GLOBAL
    )
    assert not torch.equal(
        states.no_reference.modality.entity_ids[:, :ref_end],
        states.positive.modality.entity_ids[:, :ref_end],
    )
    assert not states.no_reference.modality.attention_mask[:, ref_end:, :ref_end].any()


def test_runtime_builds_debiased_reference_states_from_one_noise_set() -> None:
    strategy = SemanticFlowStrategy(SemanticFlowConfig(max_ref_images_per_sample=1))
    strategy._semantic_dim = 128
    runtime = object.__new__(OnlineInferenceRuntime)
    runtime.strategy = strategy
    runtime.transformer = _CountingBranchTransformer(reference_end=4)
    runtime.connector_conditions = lambda conditions: conditions  # type: ignore[method-assign]

    def condition(value: float) -> dict[str, torch.Tensor]:
        return {
            "video_prompt_embeds": torch.full((1, 2, 8), value),
            "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
        }

    encoded = {
        "task": "r2v",
        "positive_conditions": condition(1.0),
        "negative_conditions": condition(-1.0),
        "empty_reference_conditions": condition(0.0),
        "empty_no_reference_conditions": condition(0.0),
        "reference_latents": {
            "latents": torch.ones(1, 1, 128, 1, 2, 2),
            "ref_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        },
    }
    guidance = SemanticGuidanceConfig(
        guidance_mode="debiased_ref",
        guidance_rescale=0.0,
    )
    states = runtime.prepare_guidance_states(
        encoded,
        width=64,
        height=64,
        num_frames=9,
        fps=24.0,
        seed=9,
        guidance=guidance,
        negative_prompt="negative",
    )
    assert states.negative is not None
    assert states.no_reference is None
    assert states.empty_reference is not None
    assert states.empty_no_reference is not None
    offsets = states.positive.sequence_offsets
    ref_end = offsets["reference_end"]
    generated = states.positive.modality.latent[:, ref_end:]
    for branch in (
        states.negative,
        states.empty_reference,
        states.empty_no_reference,
    ):
        assert torch.equal(branch.modality.latent[:, ref_end:], generated)
        assert branch.sequence_offsets == offsets
    assert torch.equal(
        states.positive.modality.latent[:, :ref_end],
        states.empty_reference.modality.latent[:, :ref_end],
    )
    for branch in (states.negative, states.empty_no_reference):
        assert torch.count_nonzero(branch.modality.latent[:, :ref_end]) == 0
        assert not branch.modality.attention_mask[:, ref_end:, :ref_end].any()
    assert runtime.last_generation_geometry["guidance_mode"] == "debiased_ref"
    assert runtime.last_generation_geometry["guidance_branch_count"] == 4
    assert runtime.last_generation_geometry["transformer_forwards_per_step"] == 4


@pytest.mark.parametrize("guidance_mode", ["positive_ref", "debiased_ref"])
def test_all_guidance_disabled_is_bitwise_legacy_denoise(guidance_mode: GuidanceMode) -> None:
    config = SemanticGuidanceConfig(
        guidance_mode=guidance_mode,
        guidance_scale=1.0,
        ref_guidance_scale=0.0,
        guidance_rescale=0.0,
        stg_scale=0.0,
    )
    strategy, states = _state_bundle(config)
    reference_end = states.positive.sequence_offsets["reference_end"]
    legacy_transformer = _CountingBranchTransformer(reference_end)
    guided_transformer = _CountingBranchTransformer(reference_end)
    legacy = strategy.denoise_joint(
        transformer=legacy_transformer,
        state=states.positive,
        num_inference_steps=3,
    )
    guided = strategy.denoise_joint_guided(
        transformer=guided_transformer,
        states=states,
        guidance=config,
        num_inference_steps=3,
    )
    assert torch.equal(legacy[0], guided[0])
    assert torch.equal(legacy[1], guided[1])


def test_guidance_condition_bundle_encodes_references_once_and_each_prompt_once() -> None:
    encoder = object.__new__(OnlineBatchEncoder)
    encoder.config = SimpleNamespace(
        width=64,
        height=64,
        image_num_frames=1,
        video_num_frames=9,
        image_fps=1.0,
        video_fps=24.0,
        max_ref_images=4,
    )
    vae_calls: list[list[list[torch.Tensor]]] = []
    prefix_calls: list[tuple[str, object]] = []
    shared_images = [object()]
    encoder._reference_images = lambda references: shared_images  # type: ignore[method-assign]

    def encode_latents(references, **_kwargs):
        vae_calls.append(references)
        return {
            "latents": torch.ones(1, 1, 128, 1, 2, 2),
            "ref_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        }

    def encode_prefix(*, caption, reference_images, **_kwargs):
        prefix_calls.append((caption, reference_images))
        return (
            {
                "video_prompt_embeds": torch.full(
                    (1, 2, 8),
                    float(len(prefix_calls)),
                ),
                "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
            },
            {},
        )

    encoder._encode_reference_latents = encode_latents  # type: ignore[method-assign]
    encoder._encode_prefix = encode_prefix  # type: ignore[method-assign]
    bundle = encoder.encode_inference_guidance_bundle_from_references(
        task="r2v",
        positive_prompt="positive",
        negative_prompt="negative",
        need_negative=True,
        need_no_reference=True,
        reference_pixels_vae=[torch.zeros(1)],
        reference_images_vlm=[torch.zeros(1)],
        width=64,
        height=64,
        num_frames=9,
        fps=24.0,
    )
    assert len(vae_calls) == 1
    assert [caption for caption, _ in prefix_calls] == [
        "positive",
        "negative",
        "positive",
    ]
    assert prefix_calls[0][1] is shared_images
    assert prefix_calls[1][1] is shared_images
    assert prefix_calls[2][1] == []
    assert bundle["negative_conditions"] is not None
    assert bundle["no_reference_conditions"] is not None
    assert "no_prompt_conditions" not in bundle
    assert bundle["strict_no_gt_checks"]["uses_target_latents"] is False


def test_debiased_guidance_bundle_encodes_exact_branch_sequence() -> None:
    encoder = object.__new__(OnlineBatchEncoder)
    encoder.config = SimpleNamespace(
        width=64,
        height=64,
        image_num_frames=1,
        video_num_frames=9,
        image_fps=1.0,
        video_fps=24.0,
        max_ref_images=4,
    )
    vae_calls: list[object] = []
    prefix_calls: list[tuple[str, object, str]] = []
    shared_images = [object()]

    def reference_images(_references: object) -> list[object]:
        return shared_images

    def encode_latents(references: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        vae_calls.append(references)
        return {
            "latents": torch.ones(1, 1, 128, 1, 2, 2),
            "ref_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        }

    def encode_prefix(
        *,
        caption: str,
        reference_images: object,
        sample_key: str,
        **_kwargs: object,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        prefix_calls.append((caption, reference_images, sample_key))
        return (
            {
                "video_prompt_embeds": torch.ones(1, 2, 8),
                "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
            },
            {},
        )

    encoder._reference_images = reference_images  # type: ignore[method-assign]
    encoder._encode_reference_latents = encode_latents  # type: ignore[method-assign]
    encoder._encode_prefix = encode_prefix  # type: ignore[method-assign]
    guidance = SemanticGuidanceConfig(
        guidance_mode="debiased_ref",
        guidance_scale=4.0,
        ref_guidance_scale=0.0,
        guidance_rescale=0.0,
    )
    bundle = encoder.encode_inference_guidance_bundle_from_references(
        task="r2v",
        positive_prompt="positive",
        negative_prompt="negative",
        need_negative=guidance.need_negative,
        need_no_reference=guidance.need_reference_comparison,
        reference_pixels_vae=[torch.zeros(1)],
        reference_images_vlm=[torch.zeros(1)],
        width=64,
        height=64,
        num_frames=9,
        fps=24.0,
        guidance_mode=guidance.guidance_mode,
    )
    assert len(vae_calls) == 1
    assert [(caption, key) for caption, _, key in prefix_calls] == [
        ("positive", "inference-positive"),
        ("negative", "inference-negative-no-reference"),
        ("", "inference-empty-reference"),
        ("", "inference-empty-no-reference"),
    ]
    assert [images is shared_images for _, images, _ in prefix_calls] == [True, False, True, False]
    assert "no_reference_conditions" not in bundle
    r_conditions = bundle["empty_reference_conditions"]
    u_conditions = bundle["empty_no_reference_conditions"]
    assert r_conditions
    assert u_conditions
    assert any(
        isinstance(value, torch.Tensor)
        and torch.count_nonzero(value).item() > 0
        for value in r_conditions.values()
    )
    for key, value in u_conditions.items():
        if isinstance(value, torch.Tensor):
            assert torch.count_nonzero(value).item() == 0, key
