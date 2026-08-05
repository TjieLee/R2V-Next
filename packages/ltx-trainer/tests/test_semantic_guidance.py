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


def test_latent_reference_guidance_formula_counts_and_metadata() -> None:
    positive = torch.tensor([[[5.0, 7.0]]])
    negative = torch.tensor([[[1.0, 2.0]]])
    no_latent_reference = torch.tensor([[[3.0, 3.0]]])
    stg = torch.tensor([[[2.0, 6.0]]])
    config = SemanticGuidanceConfig(
        guidance_mode="latent_ref",
        guidance_scale=4.0,
        ref_guidance_scale=1.0,
        guidance_rescale=0.0,
        stg_scale=0.5,
    )
    actual = combine_guided_denoised(
        positive=positive,
        negative=negative,
        no_latent_reference=no_latent_reference,
        stg=stg,
        config=config,
    )
    expected = negative + 4.0 * (positive - negative)
    expected = expected + (positive - no_latent_reference)
    expected = expected + 0.5 * (positive - stg)
    assert torch.equal(actual, expected)
    assert config.transformer_forwards_per_step == 4
    assert config.enabled_branches == ("P", "N", "QL", "S")
    metadata = SemanticGuidanceConfig(
        guidance_mode="latent_ref"
    ).metadata(negative_prompt="negative")
    assert metadata["guidance_mode"] == "latent_ref"
    assert metadata["enabled_guidance_branches"] == ["P", "N", "QL"]
    assert metadata["transformer_forwards_per_step"] == 3
    assert metadata["cfg_formula"] == (
        "N + cfg*(P-N), P/N share VLM references and reference latents"
    )
    assert metadata["ref_formula"] == "ref*(P-Q_latent)"
    assert metadata["P_vlm_references"] == "present"
    assert metadata["P_reference_latents"] == "present"
    assert metadata["N_vlm_references"] == "present"
    assert metadata["N_reference_latents"] == "present"
    assert metadata["Q_latent_text"] == "positive"
    assert metadata["Q_latent_vlm_references"] == "present"
    assert metadata["Q_latent_reference_latents"] == "absent"
    assert (
        metadata["reference_guidance_target"]
        == "dit_reference_latent_effect"
    )


@pytest.mark.parametrize(
    ("guidance_mode", "comparison_name", "comparison_kwargs"),
    [
        (
            "negative_no_vlm_positive_ref",
            "Q",
            {"no_reference": torch.tensor([[[3.0, 3.0]]])},
        ),
        (
            "negative_no_vlm_latent_ref",
            "QL",
            {"no_latent_reference": torch.tensor([[[3.0, 3.0]]])},
        ),
    ],
)
def test_no_vlm_negative_guidance_formulas_counts_and_metadata(
    guidance_mode: GuidanceMode,
    comparison_name: str,
    comparison_kwargs: dict[str, torch.Tensor],
) -> None:
    positive = torch.tensor([[[5.0, 7.0]]])
    negative_no_vlm = torch.tensor([[[1.0, 2.0]]])
    comparison = next(iter(comparison_kwargs.values()))
    stg = torch.tensor([[[2.0, 6.0]]])
    config = SemanticGuidanceConfig(
        guidance_mode=guidance_mode,
        guidance_scale=4.0,
        ref_guidance_scale=1.5,
        guidance_rescale=0.0,
        stg_scale=0.5,
    )
    actual = combine_guided_denoised(
        positive=positive,
        negative=negative_no_vlm,
        stg=stg,
        config=config,
        **comparison_kwargs,
    )
    expected = negative_no_vlm + 4.0 * (positive - negative_no_vlm)
    expected = expected + 1.5 * (positive - comparison)
    expected = expected + 0.5 * (positive - stg)
    assert torch.equal(actual, expected)
    assert config.transformer_forwards_per_step == 4
    assert config.enabled_branches == ("P", "N_I0", comparison_name, "S")

    metadata = config.metadata(negative_prompt="negative")
    assert metadata["guidance_mode"] == guidance_mode
    assert metadata["cfg_formula"] == "N_I0 + cfg*(P-N_I0)"
    assert metadata["ref_formula"] == f"ref*(P-{comparison_name})"
    assert metadata["stg_formula"] == "stg*(P-S)"
    assert metadata["N_I0_text"] == "negative"
    assert metadata["N_I0_vlm_references"] == "absent"
    assert metadata["N_I0_reference_latents"] == "present"
    assert metadata["negative_branch_condition_axes"] == "T_negative_I0_L1"
    assert metadata["reference_comparison_branch"] == comparison_name
    if comparison_name == "Q":
        assert metadata["Q_text"] == "positive"
        assert metadata["Q_vlm_references"] == "absent"
        assert metadata["Q_reference_latents"] == "absent"
        assert (
            metadata["reference_guidance_target"]
            == "vlm_and_latent_reference_effect"
        )
    else:
        assert metadata["QL_text"] == "positive"
        assert metadata["QL_vlm_references"] == "present"
        assert metadata["QL_reference_latents"] == "absent"
        assert (
            metadata["reference_guidance_target"]
            == "dit_reference_latent_effect"
        )


def test_standard_negative_latent_guidance_formula_counts_and_metadata() -> None:
    positive = torch.tensor([[[5.0, 7.0]]])
    negative_drop_all = torch.tensor([[[1.0, 2.0]]])
    no_latent_reference = torch.tensor([[[3.0, 3.0]]])
    stg = torch.tensor([[[2.0, 6.0]]])
    config = SemanticGuidanceConfig(
        guidance_mode="standard_negative_latent_ref",
        guidance_scale=4.0,
        ref_guidance_scale=1.5,
        guidance_rescale=0.0,
        stg_scale=0.5,
    )

    actual = combine_guided_denoised(
        positive=positive,
        negative=negative_drop_all,
        no_latent_reference=no_latent_reference,
        stg=stg,
        config=config,
    )
    expected = negative_drop_all + 4.0 * (positive - negative_drop_all)
    expected = expected + 1.5 * (positive - no_latent_reference)
    expected = expected + 0.5 * (positive - stg)

    assert torch.equal(actual, expected)
    assert config.enabled_branches == ("P", "N0", "QL", "S")
    assert config.transformer_forwards_per_step == 4
    metadata = config.metadata(negative_prompt="negative")
    assert metadata["guidance_mode"] == "standard_negative_latent_ref"
    assert metadata["cfg_formula"] == "N0 + cfg*(P-N0)"
    assert metadata["ref_formula"] == "ref*(P-QL)"
    assert metadata["stg_formula"] == "stg*(P-S)"
    assert metadata["negative_branch_condition_axes"] == "T_negative_I0_L0"
    assert metadata["negative_branch_vlm_references"] == "absent"
    assert metadata["negative_branch_reference_latents"] == "absent"
    assert metadata["reference_comparison_branch"] == "QL"
    assert metadata["reference_guidance_target"] == "dit_reference_latent_effect"
    assert metadata["N0_text"] == "negative"
    assert metadata["N0_vlm_references"] == "absent"
    assert metadata["N0_reference_latents"] == "absent"
    assert metadata["QL_text"] == "positive"
    assert metadata["QL_vlm_references"] == "present"
    assert metadata["QL_reference_latents"] == "absent"

    legacy = SemanticGuidanceConfig(
        guidance_mode="negative_no_vlm_latent_ref"
    ).metadata(negative_prompt="negative")
    assert legacy["negative_branch_condition_axes"] == "T_negative_I0_L1"
    assert legacy["negative_branch_vlm_references"] == "absent"
    assert legacy["negative_branch_reference_latents"] == "present"
    assert legacy["N_I0_reference_latents"] == "present"


def test_factorized_til_guidance_formula_counts_and_metadata() -> None:
    positive = torch.tensor([[[11.0, 13.0]]])
    negative_drop_all = torch.tensor([[[1.0, 2.0]]])
    text_only = torch.tensor([[[3.0, 5.0]]])
    vlm_only = torch.tensor([[[7.0, 8.0]]])
    stg = torch.tensor([[[9.0, 10.0]]])
    config = SemanticGuidanceConfig(
        guidance_mode="factorized_til_guidance",
        guidance_scale=4.0,
        vlm_guidance_scale=2.0,
        ref_guidance_scale=2.0,
        guidance_rescale=0.0,
        stg_scale=1.0,
    )

    actual = combine_guided_denoised(
        positive=positive,
        negative=negative_drop_all,
        no_reference=text_only,
        no_latent_reference=vlm_only,
        stg=stg,
        config=config,
    )
    expected = negative_drop_all
    expected = expected + 4.0 * (text_only - negative_drop_all)
    expected = expected + 2.0 * (vlm_only - text_only)
    expected = expected + 2.0 * (positive - vlm_only)
    expected = expected + 1.0 * (positive - stg)

    assert torch.equal(actual, expected)
    assert config.enabled_branches == ("P", "N0", "T", "I", "S")
    assert config.transformer_forwards_per_step == 5
    metadata = config.metadata(negative_prompt="negative")
    assert metadata["guidance_mode"] == "factorized_til_guidance"
    assert metadata["guidance_formula"] == (
        "N0 + text*(T-N0) + vlm*(I-T) + latent*(P-I) + stg*(P-S)"
    )
    assert metadata["text_guidance_scale"] == 4.0
    assert metadata["vlm_guidance_scale"] == 2.0
    assert metadata["latent_guidance_scale"] == 2.0
    assert metadata["stg_scale"] == 1.0
    assert metadata["condition_factorization"] == "T_I_L_incremental_v1"
    assert metadata["negative_branch"] == "N0"
    assert metadata["text_only_branch"] == "T"
    assert metadata["vlm_branch"] == "I"
    assert metadata["full_positive_branch"] == "P"
    assert metadata["stg_branch"] == "S"
    assert metadata["N0_condition_axes"] == "T_negative_I0_L0"
    assert metadata["T_condition_axes"] == "T_positive_I0_L0"
    assert metadata["I_condition_axes"] == "T_positive_I1_L0"
    assert metadata["P_condition_axes"] == "T_positive_I1_L1"
    assert metadata["enabled_guidance_branches"] == ["P", "N0", "T", "I", "S"]
    assert metadata["transformer_forwards_per_step"] == 5


def test_factorized_til_identity_point_telescopes_to_positive() -> None:
    positive = torch.tensor([[[8.0, 16.0]]])
    negative_drop_all = torch.tensor([[[1.0, 2.0]]])
    text_only = torch.tensor([[[3.0, 4.0]]])
    vlm_only = torch.tensor([[[5.0, 7.0]]])
    config = SemanticGuidanceConfig(
        guidance_mode="factorized_til_guidance",
        guidance_scale=1.0,
        vlm_guidance_scale=1.0,
        ref_guidance_scale=1.0,
        guidance_rescale=0.0,
        stg_scale=0.0,
    )

    actual = combine_guided_denoised(
        positive=positive,
        negative=negative_drop_all,
        no_reference=text_only,
        no_latent_reference=vlm_only,
        config=config,
    )

    assert torch.equal(actual, positive)
    assert config.enabled_branches == ("P", "N0", "T", "I")
    assert config.transformer_forwards_per_step == 4


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
    for guidance_mode in (
        "latent_ref",
        "negative_no_vlm_positive_ref",
        "negative_no_vlm_latent_ref",
        "standard_negative_latent_ref",
        "factorized_til_guidance",
    ):
        assert (
            SemanticGuidanceConfig(guidance_mode=guidance_mode).guidance_mode
            == guidance_mode
        )
    with pytest.raises(ValueError, match="guidance_mode"):
        SemanticGuidanceConfig(guidance_mode="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="guidance_mode"):
        SemanticGuidanceConfig(guidance_mode="multimodal_ref")  # type: ignore[arg-type]


def test_vlm_scale_preserves_legacy_positional_constructor_order() -> None:
    config = SemanticGuidanceConfig(
        "positive_ref",
        3.0,
        2.0,
        0.4,
        0.5,
        (3,),
    )

    assert config.guidance_scale == 3.0
    assert config.ref_guidance_scale == 2.0
    assert config.guidance_rescale == 0.4
    assert config.stg_scale == 0.5
    assert config.stg_blocks == (3,)
    assert config.vlm_guidance_scale == 1.0


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
    assert "positive_ref, debiased_ref, latent_ref" in source
    assert "negative_no_vlm_positive_ref" in source
    assert "negative_no_vlm_latent_ref" in source
    assert "standard_negative_latent_ref" in source
    assert "factorized_til_guidance" in source
    assert "--vlm-guidance-scale" in source
    assert "vlm_guidance_scale=vlm_guidance_scale" in source
    assert "guidance.metadata(" in source
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
                guidance_mode="latent_ref",
                guidance_scale=1.0,
                ref_guidance_scale=2.0,
                guidance_rescale=0.0,
            ),
            {"no_latent_reference": torch.tensor([[[1.0]]])},
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


def test_latent_reference_scale_zero_does_not_require_ql() -> None:
    config = SemanticGuidanceConfig(
        guidance_mode="latent_ref",
        guidance_scale=3.0,
        ref_guidance_scale=0.0,
        guidance_rescale=0.0,
        stg_scale=0.5,
    )
    actual = combine_guided_denoised(
        positive=torch.tensor([[[2.0]]]),
        negative=torch.tensor([[[1.0]]]),
        stg=torch.tensor([[[0.0]]]),
        config=config,
    )
    assert torch.equal(actual, torch.tensor([[[5.0]]]))
    assert config.enabled_branches == ("P", "N", "S")
    assert config.transformer_forwards_per_step == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"guidance_scale": 0.99},
        {"guidance_scale": float("nan")},
        {"vlm_guidance_scale": -0.1},
        {"vlm_guidance_scale": float("inf")},
        {"ref_guidance_scale": -0.1},
        {"ref_guidance_scale": float("nan")},
        {"stg_scale": float("inf")},
    ],
)
def test_factorized_til_guidance_rejects_invalid_scales(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        SemanticGuidanceConfig(
            guidance_mode="factorized_til_guidance",
            **kwargs,
        )


def test_legacy_guidance_rejects_non_default_vlm_scale() -> None:
    with pytest.raises(ValueError, match="must remain 1.0"):
        SemanticGuidanceConfig(
            guidance_mode="positive_ref",
            vlm_guidance_scale=2.0,
        )


@pytest.mark.parametrize(
    ("guidance_scale", "ref_scale", "stg_scale", "expected_branches"),
    [
        (1.0, 0.0, 0.0, ("P",)),
        (4.0, 0.0, 0.0, ("P", "N0")),
        (1.0, 2.0, 0.0, ("P", "QL")),
        (1.0, 0.0, 1.0, ("P", "S")),
    ],
)
def test_standard_negative_latent_guidance_skips_disabled_branches(
    guidance_scale: float,
    ref_scale: float,
    stg_scale: float,
    expected_branches: tuple[str, ...],
) -> None:
    config = SemanticGuidanceConfig(
        guidance_mode="standard_negative_latent_ref",
        guidance_scale=guidance_scale,
        ref_guidance_scale=ref_scale,
        guidance_rescale=0.0,
        stg_scale=stg_scale,
    )
    assert config.enabled_branches == expected_branches
    assert config.transformer_forwards_per_step == len(expected_branches)


def test_no_vlm_negative_modes_are_bitwise_equal_when_reference_is_disabled() -> None:
    common = {
        "guidance_scale": 3.0,
        "ref_guidance_scale": 0.0,
        "guidance_rescale": 0.0,
        "stg_scale": 0.5,
    }
    q_config = SemanticGuidanceConfig(
        guidance_mode="negative_no_vlm_positive_ref",
        **common,
    )
    ql_config = SemanticGuidanceConfig(
        guidance_mode="negative_no_vlm_latent_ref",
        **common,
    )
    kwargs = {
        "positive": torch.tensor([[[2.0, 4.0]]]),
        "negative": torch.tensor([[[1.0, -1.0]]]),
        "stg": torch.tensor([[[0.0, 2.0]]]),
    }
    q_guided = combine_guided_denoised(config=q_config, **kwargs)
    ql_guided = combine_guided_denoised(config=ql_config, **kwargs)
    assert torch.equal(q_guided, ql_guided)
    assert q_config.enabled_branches == ql_config.enabled_branches == (
        "P",
        "N_I0",
        "S",
    )
    assert q_config.transformer_forwards_per_step == 3
    assert ql_config.transformer_forwards_per_step == 3


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
    negative_references = (
        no_refs
        if config.guidance_mode
        in {
            "debiased_ref",
            "standard_negative_latent_ref",
            "factorized_til_guidance",
        }
        else references
    )
    negative = prepare(-1.0, negative_references, noise) if config.need_negative else None
    no_reference = None
    no_latent_reference = None
    empty_reference = None
    empty_no_reference = None
    if config.uses_factorized_til_guidance:
        no_reference = prepare(0.0, no_refs, noise)
        no_latent_reference = prepare(1.0, no_refs, noise)
    elif config.need_reference and config.uses_q_reference_comparison:
        no_reference = prepare(0.0, no_refs, noise)
    elif config.need_reference and config.uses_ql_reference_comparison:
        no_latent_reference = prepare(1.0, no_refs, noise)
    elif config.need_control_pair:
        empty_reference = prepare(0.0, references, noise)
        empty_no_reference = prepare(0.0, no_refs, noise)
    return strategy, SemanticGuidanceStateBundle(
        positive=positive,
        negative=negative,
        no_reference=no_reference,
        no_latent_reference=no_latent_reference,
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
                "context": context,
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
                guidance_mode="latent_ref",
                guidance_rescale=0.0,
            ),
            3,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="latent_ref",
                guidance_scale=1.0,
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
            ),
            1,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="latent_ref",
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
            ),
            2,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="latent_ref",
                guidance_scale=1.0,
                guidance_rescale=0.0,
            ),
            2,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="latent_ref",
                guidance_rescale=0.0,
                stg_scale=0.5,
            ),
            4,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="negative_no_vlm_positive_ref",
                guidance_scale=1.0,
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
            ),
            1,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="negative_no_vlm_positive_ref",
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
            ),
            2,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="negative_no_vlm_positive_ref",
                guidance_rescale=0.0,
            ),
            3,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="negative_no_vlm_positive_ref",
                guidance_rescale=0.0,
                stg_scale=0.5,
            ),
            4,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="negative_no_vlm_latent_ref",
                guidance_scale=1.0,
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
            ),
            1,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="negative_no_vlm_latent_ref",
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
            ),
            2,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="negative_no_vlm_latent_ref",
                guidance_rescale=0.0,
            ),
            3,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="negative_no_vlm_latent_ref",
                guidance_rescale=0.0,
                stg_scale=0.5,
            ),
            4,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="standard_negative_latent_ref",
                guidance_scale=1.0,
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
            ),
            1,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="standard_negative_latent_ref",
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
            ),
            2,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="standard_negative_latent_ref",
                guidance_rescale=0.0,
            ),
            3,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="standard_negative_latent_ref",
                guidance_rescale=0.0,
                stg_scale=0.5,
            ),
            4,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="factorized_til_guidance",
                guidance_rescale=0.0,
            ),
            4,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="factorized_til_guidance",
                guidance_scale=1.0,
                vlm_guidance_scale=0.0,
                ref_guidance_scale=0.0,
                guidance_rescale=0.0,
            ),
            4,
        ),
        (
            SemanticGuidanceConfig(
                guidance_mode="factorized_til_guidance",
                guidance_rescale=0.0,
                stg_scale=0.5,
            ),
            5,
        ),
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


@pytest.mark.parametrize(
    ("guidance_mode", "expected_contexts", "expected_references"),
    [
        ("positive_ref", [1.0, -1.0, 0.0], [True, True, False]),
        ("latent_ref", [1.0, -1.0, 1.0], [True, True, False]),
        (
            "debiased_ref",
            [1.0, -1.0, 0.0, 0.0],
            [True, False, True, False],
        ),
    ],
)
def test_existing_guidance_branch_order_is_unchanged(
    guidance_mode: GuidanceMode,
    expected_contexts: list[float],
    expected_references: list[bool],
) -> None:
    config = SemanticGuidanceConfig(
        guidance_mode=guidance_mode,
        guidance_rescale=0.0,
    )
    strategy, states = _state_bundle(config)
    reference_end = states.positive.sequence_offsets["reference_end"]
    transformer = _CountingBranchTransformer(reference_end)
    strategy.denoise_joint_guided(
        transformer=transformer,
        states=states,
        guidance=config,
        num_inference_steps=1,
    )
    assert [call["context"] for call in transformer.calls] == expected_contexts
    assert [
        call["reference_nonzero"] for call in transformer.calls
    ] == expected_references


@pytest.mark.parametrize(
    ("guidance_mode", "expected_contexts", "expected_references"),
    [
        (
            "negative_no_vlm_positive_ref",
            [1.0, -1.0, 0.0, 1.0],
            [True, True, False, True],
        ),
        (
            "negative_no_vlm_latent_ref",
            [1.0, -1.0, 1.0, 1.0],
            [True, True, False, True],
        ),
        (
            "standard_negative_latent_ref",
            [1.0, -1.0, 1.0, 1.0],
            [True, False, False, True],
        ),
    ],
)
def test_no_vlm_negative_guidance_branch_order(
    guidance_mode: GuidanceMode,
    expected_contexts: list[float],
    expected_references: list[bool],
) -> None:
    config = SemanticGuidanceConfig(
        guidance_mode=guidance_mode,
        guidance_rescale=0.0,
        stg_scale=0.5,
    )
    strategy, states = _state_bundle(config)
    reference_end = states.positive.sequence_offsets["reference_end"]
    transformer = _CountingBranchTransformer(reference_end)
    strategy.denoise_joint_guided(
        transformer=transformer,
        states=states,
        guidance=config,
        num_inference_steps=1,
    )
    assert [call["context"] for call in transformer.calls] == expected_contexts
    assert [
        call["reference_nonzero"] for call in transformer.calls
    ] == expected_references
    assert [call["perturbed"] for call in transformer.calls] == [
        False,
        False,
        False,
        True,
    ]


def test_factorized_til_guidance_branch_order_and_reference_activation() -> None:
    config = SemanticGuidanceConfig(
        guidance_mode="factorized_til_guidance",
        guidance_scale=4.0,
        vlm_guidance_scale=2.0,
        ref_guidance_scale=2.0,
        guidance_rescale=0.0,
        stg_scale=1.0,
    )
    strategy, states = _state_bundle(config)
    reference_end = states.positive.sequence_offsets["reference_end"]
    transformer = _CountingBranchTransformer(reference_end)

    strategy.denoise_joint_guided(
        transformer=transformer,
        states=states,
        guidance=config,
        num_inference_steps=1,
    )

    assert [call["context"] for call in transformer.calls] == [
        1.0,
        -1.0,
        0.0,
        1.0,
        1.0,
    ]
    assert [call["reference_nonzero"] for call in transformer.calls] == [
        True,
        False,
        False,
        False,
        True,
    ]
    assert [call["perturbed"] for call in transformer.calls] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert all(
        torch.equal(call["generated"], transformer.calls[0]["generated"])
        for call in transformer.calls
    )


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


@pytest.mark.parametrize(
    "guidance_mode",
    ["negative_no_vlm_positive_ref", "negative_no_vlm_latent_ref"],
)
def test_no_vlm_negative_validation_requires_active_reference_latents(
    guidance_mode: GuidanceMode,
) -> None:
    config = SemanticGuidanceConfig(
        guidance_mode=guidance_mode,
        guidance_rescale=0.0,
    )
    strategy, states = _state_bundle(config)
    assert states.negative is not None
    ref_end = states.positive.sequence_offsets["reference_end"]
    inactive_negative = replace(
        states.negative,
        modality=replace(
            states.negative.modality,
            latent=torch.cat(
                [
                    torch.zeros_like(
                        states.negative.modality.latent[:, :ref_end]
                    ),
                    states.negative.modality.latent[:, ref_end:],
                ],
                dim=1,
            ),
        ),
    )
    with pytest.raises(ValueError, match="share reference latents"):
        strategy.validate_guidance_state_bundle(
            replace(states, negative=inactive_negative),
            config,
        )


def test_standard_negative_validation_requires_inactive_reference_latents() -> None:
    config = SemanticGuidanceConfig(
        guidance_mode="standard_negative_latent_ref",
        guidance_rescale=0.0,
    )
    strategy, states = _state_bundle(config)
    assert states.negative is not None
    ref_end = states.positive.sequence_offsets["reference_end"]
    assert torch.count_nonzero(states.negative.modality.latent[:, :ref_end]) == 0
    assert not states.negative.modality.attention_mask[:, ref_end:, :ref_end].any()

    active_negative = replace(
        states.negative,
        modality=replace(
            states.negative.modality,
            latent=torch.cat(
                [
                    states.positive.modality.latent[:, :ref_end],
                    states.negative.modality.latent[:, ref_end:],
                ],
                dim=1,
            ),
        ),
    )
    with pytest.raises(ValueError, match="N0 reference tokens must be zero"):
        strategy.validate_guidance_state_bundle(
            replace(states, negative=active_negative),
            config,
        )


def test_latent_validation_requires_shared_vlm_and_inactive_reference() -> None:
    config = SemanticGuidanceConfig(
        guidance_mode="latent_ref",
        guidance_rescale=0.0,
    )
    strategy, states = _state_bundle(config)
    assert states.no_latent_reference is not None
    q_latent = states.no_latent_reference
    different_context = replace(
        q_latent,
        modality=replace(
            q_latent.modality,
            context=q_latent.modality.context + 1.0,
        ),
    )
    with pytest.raises(ValueError, match="same VLM condition"):
        strategy.validate_guidance_state_bundle(
            replace(states, no_latent_reference=different_context),
            config,
        )

    ref_end = states.positive.sequence_offsets["reference_end"]
    active_reference = replace(
        q_latent,
        modality=replace(
            q_latent.modality,
            latent=torch.cat(
                [
                    states.positive.modality.latent[:, :ref_end],
                    q_latent.modality.latent[:, ref_end:],
                ],
                dim=1,
            ),
        ),
    )
    with pytest.raises(ValueError, match="QL reference tokens must be zero"):
        strategy.validate_guidance_state_bundle(
            replace(states, no_latent_reference=active_reference),
            config,
        )


def test_runtime_builds_isolated_branches_from_one_reference_and_noise_set() -> None:
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
    assert states.no_latent_reference is None
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


def test_runtime_builds_latent_reference_state_from_positive_condition() -> None:
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
        "reference_latents": {
            "latents": torch.ones(1, 1, 128, 1, 2, 2),
            "ref_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        },
    }
    guidance = SemanticGuidanceConfig(
        guidance_mode="latent_ref",
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
    assert states.no_latent_reference is not None
    assert states.empty_reference is None
    assert states.empty_no_reference is None
    q_latent = states.no_latent_reference
    offsets = states.positive.sequence_offsets
    ref_end = offsets["reference_end"]
    assert q_latent.sequence_offsets == offsets
    assert q_latent.target_shape == states.positive.target_shape
    assert torch.equal(
        q_latent.modality.latent[:, ref_end:],
        states.positive.modality.latent[:, ref_end:],
    )
    for name in (
        "positions",
        "token_type_ids",
        "timesteps",
        "semantic_position_bounds",
    ):
        assert torch.equal(
            getattr(q_latent.modality, name),
            getattr(states.positive.modality, name),
        )
    assert q_latent.modality.context is states.positive.modality.context
    assert q_latent.modality.context_mask is states.positive.modality.context_mask
    assert torch.equal(
        states.negative.modality.latent[:, :ref_end],
        states.positive.modality.latent[:, :ref_end],
    )
    assert torch.count_nonzero(q_latent.modality.latent[:, :ref_end]) == 0
    assert not q_latent.modality.attention_mask[:, ref_end:, :ref_end].any()
    assert runtime.last_generation_geometry["guidance_mode"] == "latent_ref"
    assert runtime.last_generation_geometry["enabled_guidance_branches"] == [
        "P",
        "N",
        "QL",
    ]
    assert runtime.last_generation_geometry["transformer_forwards_per_step"] == 3


@pytest.mark.parametrize(
    ("guidance_mode", "comparison_branch"),
    [
        ("negative_no_vlm_positive_ref", "Q"),
        ("negative_no_vlm_latent_ref", "QL"),
    ],
)
def test_runtime_builds_no_vlm_negative_with_active_reference_latents(
    guidance_mode: GuidanceMode,
    comparison_branch: str,
) -> None:
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
        "negative_no_vlm_conditions": condition(-2.0),
        "no_reference_conditions": condition(0.0),
        "reference_latents": {
            "latents": torch.ones(1, 1, 128, 1, 2, 2),
            "ref_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        },
    }
    guidance = SemanticGuidanceConfig(
        guidance_mode=guidance_mode,
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
        negative_prompt="real negative prompt",
    )
    assert states.negative is not None
    offsets = states.positive.sequence_offsets
    ref_end = offsets["reference_end"]
    assert states.negative.sequence_offsets == offsets
    assert states.negative.target_shape == states.positive.target_shape
    assert torch.equal(
        states.negative.modality.latent[:, :ref_end],
        states.positive.modality.latent[:, :ref_end],
    )
    assert torch.count_nonzero(
        states.negative.modality.latent[:, :ref_end]
    ).item()
    assert torch.equal(
        states.negative.modality.latent[:, ref_end:],
        states.positive.modality.latent[:, ref_end:],
    )
    assert torch.equal(
        states.negative.modality.entity_ids,
        states.positive.modality.entity_ids,
    )
    assert torch.equal(
        states.negative.modality.attention_mask,
        states.positive.modality.attention_mask,
    )
    assert states.negative.modality.attention_mask[:, ref_end:, :ref_end].any()
    assert torch.all(states.negative.modality.context == -2.0)
    assert not torch.equal(
        states.negative.modality.context,
        states.positive.modality.context,
    )

    if comparison_branch == "Q":
        assert states.no_reference is not None
        assert states.no_latent_reference is None
        comparison = states.no_reference
        assert not torch.equal(
            comparison.modality.context,
            states.positive.modality.context,
        )
    else:
        assert states.no_reference is None
        assert states.no_latent_reference is not None
        comparison = states.no_latent_reference
        assert comparison.modality.context is states.positive.modality.context
        assert (
            comparison.modality.context_mask
            is states.positive.modality.context_mask
        )
    assert torch.count_nonzero(comparison.modality.latent[:, :ref_end]) == 0
    assert not comparison.modality.attention_mask[:, ref_end:, :ref_end].any()
    assert torch.equal(
        comparison.modality.latent[:, ref_end:],
        states.positive.modality.latent[:, ref_end:],
    )

    geometry = runtime.last_generation_geometry
    assert geometry["guidance_mode"] == guidance_mode
    assert geometry["enabled_guidance_branches"] == [
        "P",
        "N_I0",
        comparison_branch,
    ]
    assert geometry["negative_branch_condition_axes"] == "T_negative_I0_L1"
    assert geometry["guidance_branch_generated_noise_identical"] is True


def test_runtime_builds_standard_negative_drop_all_and_ql() -> None:
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
        "negative_no_vlm_conditions": condition(-2.0),
        "reference_latents": {
            "latents": torch.ones(1, 1, 128, 1, 2, 2),
            "ref_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        },
    }
    guidance = SemanticGuidanceConfig(
        guidance_mode="standard_negative_latent_ref",
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
        negative_prompt="real negative prompt",
    )

    assert states.negative is not None
    assert states.no_reference is None
    assert states.no_latent_reference is not None
    negative = states.negative.modality
    ql = states.no_latent_reference.modality
    positive = states.positive.modality
    ref_end = states.positive.sequence_offsets["reference_end"]
    generated = positive.latent[:, ref_end:]

    for branch in (negative, ql):
        assert torch.equal(branch.latent[:, ref_end:], generated)
        assert torch.count_nonzero(branch.latent[:, :ref_end]) == 0
        assert not branch.attention_mask[:, ref_end:, :ref_end].any()
        for name in (
            "positions",
            "token_type_ids",
            "timesteps",
            "semantic_position_bounds",
        ):
            assert torch.equal(getattr(branch, name), getattr(positive, name))
    assert torch.all(negative.context == -2.0)
    assert ql.context is positive.context
    assert ql.context_mask is positive.context_mask

    geometry = runtime.last_generation_geometry
    assert geometry["enabled_guidance_branches"] == ["P", "N0", "QL"]
    assert geometry["negative_branch_condition_axes"] == "T_negative_I0_L0"
    assert geometry["negative_branch_vlm_references"] == "absent"
    assert geometry["negative_branch_reference_latents"] == "absent"
    assert geometry["reference_comparison_branch"] == "QL"
    assert geometry["guidance_branch_generated_noise_identical"] is True


def test_runtime_builds_factorized_til_states_from_one_trajectory() -> None:
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
        "negative_no_vlm_conditions": condition(-2.0),
        "no_reference_conditions": condition(0.25),
        "reference_latents": {
            "latents": torch.ones(1, 1, 128, 1, 2, 2),
            "ref_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        },
    }
    guidance = SemanticGuidanceConfig(
        guidance_mode="factorized_til_guidance",
        guidance_scale=4.0,
        vlm_guidance_scale=2.0,
        ref_guidance_scale=2.0,
        guidance_rescale=0.0,
        stg_scale=1.0,
    )
    states = runtime.prepare_guidance_states(
        encoded,
        width=64,
        height=64,
        num_frames=9,
        fps=24.0,
        seed=9,
        guidance=guidance,
        negative_prompt="real negative prompt",
    )

    assert states.negative is not None
    assert states.no_reference is not None
    assert states.no_latent_reference is not None
    positive = states.positive.modality
    negative = states.negative.modality
    text_only = states.no_reference.modality
    vlm_only = states.no_latent_reference.modality
    ref_end = states.positive.sequence_offsets["reference_end"]
    generated = positive.latent[:, ref_end:]

    assert torch.count_nonzero(positive.latent[:, :ref_end]) > 0
    for branch in (negative, text_only, vlm_only):
        assert torch.equal(branch.latent[:, ref_end:], generated)
        assert torch.count_nonzero(branch.latent[:, :ref_end]) == 0
        assert not branch.attention_mask[:, ref_end:, :ref_end].any()
        for name in (
            "positions",
            "token_type_ids",
            "timesteps",
            "semantic_position_bounds",
        ):
            assert torch.equal(getattr(branch, name), getattr(positive, name))
        assert torch.equal(
            branch.entity_ids[:, ref_end:],
            positive.entity_ids[:, ref_end:],
        )
        assert torch.equal(
            branch.attention_mask[:, ref_end:, ref_end:],
            positive.attention_mask[:, ref_end:, ref_end:],
        )
    assert torch.all(negative.context == -2.0)
    assert torch.all(text_only.context == 0.25)
    assert vlm_only.context is positive.context
    assert vlm_only.context_mask is positive.context_mask

    geometry = runtime.last_generation_geometry
    assert geometry["enabled_guidance_branches"] == ["P", "N0", "T", "I", "S"]
    assert geometry["transformer_forwards_per_step"] == 5
    assert geometry["condition_factorization"] == "T_I_L_incremental_v1"
    assert geometry["guidance_branch_generated_noise_identical"] is True


def test_factorized_til_validation_rejects_mismatched_i_context() -> None:
    config = SemanticGuidanceConfig(
        guidance_mode="factorized_til_guidance",
        guidance_rescale=0.0,
    )
    strategy, states = _state_bundle(config)
    assert states.no_latent_reference is not None
    changed = replace(
        states.no_latent_reference,
        modality=replace(
            states.no_latent_reference.modality,
            context=states.no_latent_reference.modality.context + 1.0,
        ),
    )

    with pytest.raises(ValueError, match="P and I must share"):
        strategy.validate_guidance_state_bundle(
            replace(states, no_latent_reference=changed),
            config,
        )


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
    assert states.no_latent_reference is None
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


@pytest.mark.parametrize(
    "guidance_mode",
    [
        "positive_ref",
        "debiased_ref",
        "latent_ref",
        "negative_no_vlm_positive_ref",
        "negative_no_vlm_latent_ref",
        "standard_negative_latent_ref",
    ],
)
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


def test_latent_guidance_bundle_reuses_positive_vlm_reference_condition() -> None:
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
    encoder._reference_images = lambda _references: shared_images  # type: ignore[method-assign]

    def encode_latents(
        references: object,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
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

    encoder._encode_reference_latents = encode_latents  # type: ignore[method-assign]
    encoder._encode_prefix = encode_prefix  # type: ignore[method-assign]
    guidance = SemanticGuidanceConfig(guidance_mode="latent_ref")
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
        ("negative", "inference-negative"),
    ]
    assert all(images is shared_images for _, images, _ in prefix_calls)
    assert bundle["positive_conditions"] is not None
    assert bundle["negative_conditions"] is not None
    assert "no_reference_conditions" not in bundle
    assert "empty_reference_conditions" not in bundle
    assert "empty_no_reference_conditions" not in bundle


@pytest.mark.parametrize(
    ("guidance_mode", "expected_calls"),
    [
        (
            "negative_no_vlm_positive_ref",
            [
                ("positive", True, "r2v", "inference-positive"),
                (
                    "keep this negative",
                    False,
                    "r2v",
                    "inference-negative-no-vlm",
                ),
                ("positive", False, "r2v", "inference-no-reference"),
            ],
        ),
        (
            "negative_no_vlm_latent_ref",
            [
                ("positive", True, "r2v", "inference-positive"),
                (
                    "keep this negative",
                    False,
                    "r2v",
                    "inference-negative-no-vlm",
                ),
            ],
        ),
        (
            "standard_negative_latent_ref",
            [
                ("positive", True, "r2v", "inference-positive"),
                (
                    "keep this negative",
                    False,
                    "r2v",
                    "inference-negative-no-vlm",
                ),
            ],
        ),
        (
            "factorized_til_guidance",
            [
                ("positive", True, "r2v", "inference-positive"),
                (
                    "keep this negative",
                    False,
                    "r2v",
                    "inference-negative-no-vlm",
                ),
                ("positive", False, "r2v", "inference-no-reference"),
            ],
        ),
    ],
)
def test_no_vlm_negative_bundle_encodes_real_negative_without_images(
    guidance_mode: GuidanceMode,
    expected_calls: list[tuple[str, bool, str, str]],
) -> None:
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
    shared_images = [object()]
    prefix_calls: list[tuple[str, object, str, str]] = []
    encoder._reference_images = lambda _references: shared_images  # type: ignore[method-assign]
    encoder._encode_reference_latents = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "latents": torch.ones(1, 1, 128, 1, 2, 2),
        "ref_valid_mask": torch.ones(1, 1, dtype=torch.bool),
    }

    def encode_prefix(
        *,
        caption: str,
        reference_images: object,
        task: str,
        sample_key: str,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        prefix_calls.append((caption, reference_images, task, sample_key))
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

    encoder._encode_prefix = encode_prefix  # type: ignore[method-assign]
    guidance = SemanticGuidanceConfig(guidance_mode=guidance_mode)
    bundle = encoder.encode_inference_guidance_bundle_from_references(
        task="r2v",
        positive_prompt="positive",
        negative_prompt="keep this negative",
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
    normalized_calls = [
        (
            caption,
            images is shared_images,
            task,
            sample_key,
        )
        for caption, images, task, sample_key in prefix_calls
    ]
    assert normalized_calls == expected_calls
    assert bundle["negative_conditions"] is None
    assert bundle["negative_no_vlm_conditions"] is not None
    if guidance_mode == "factorized_til_guidance":
        assert bundle["no_reference_conditions"] is not None
    assert bundle["strict_no_gt_checks"] == {
        "target_path_passed_to_condition_encoder": False,
        "target_path_passed_to_denoiser": False,
        "uses_target_latents": False,
        "uses_gt_teacher_evidence": False,
        "semantic_initialization": "noise",
    }
    forbidden = {
        "target_pixels",
        "latents",
        "semantic_teacher_inputs",
        "evidence_tokens",
    }
    assert not forbidden.intersection(bundle)


@pytest.mark.parametrize(
    "guidance_mode",
    [
        "negative_no_vlm_positive_ref",
        "negative_no_vlm_latent_ref",
        "standard_negative_latent_ref",
    ],
)
def test_no_vlm_negative_bundle_skips_inactive_conditions(
    guidance_mode: GuidanceMode,
) -> None:
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
    shared_images = [object()]
    prefix_calls: list[tuple[str, object]] = []
    encoder._reference_images = lambda _references: shared_images  # type: ignore[method-assign]
    encoder._encode_reference_latents = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "latents": torch.ones(1, 1, 128, 1, 2, 2),
        "ref_valid_mask": torch.ones(1, 1, dtype=torch.bool),
    }

    def encode_prefix(
        *,
        caption: str,
        reference_images: object,
        **_kwargs: object,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        prefix_calls.append((caption, reference_images))
        return (
            {
                "video_prompt_embeds": torch.ones(1, 2, 8),
                "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
            },
            {},
        )

    encoder._encode_prefix = encode_prefix  # type: ignore[method-assign]
    bundle = encoder.encode_inference_guidance_bundle_from_references(
        task="r2v",
        positive_prompt="positive",
        negative_prompt=None,
        need_negative=False,
        need_no_reference=False,
        reference_pixels_vae=[torch.zeros(1)],
        reference_images_vlm=[torch.zeros(1)],
        width=64,
        height=64,
        num_frames=9,
        fps=24.0,
        guidance_mode=guidance_mode,
    )
    assert prefix_calls == [("positive", shared_images)]
    assert bundle["negative_conditions"] is None
    assert bundle["negative_no_vlm_conditions"] is None
    if guidance_mode == "negative_no_vlm_positive_ref":
        assert bundle["no_reference_conditions"] is None
    else:
        assert "no_reference_conditions" not in bundle


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
