import torch

from ltx_core.multicond.cfg_sampler import CFGModeBatch, sample_cfg_modes
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_trainer.training_strategies.multi_reference_planner_stage2 import (
    MultiReferencePlannerStage2Config,
    MultiReferencePlannerStage2Strategy,
)
from ltx_trainer.training_strategies.multi_reference_video import (
    MultiReferenceVideoConfig,
    MultiReferenceVideoStrategy,
)


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
