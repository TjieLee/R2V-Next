import importlib.util
import inspect
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from ltx_trainer.training_strategies.multi_reference_planner_stage2 import (
    MultiReferencePlannerStage2Strategy,
)
from ltx_trainer.training_strategies.multi_reference_video import MultiReferenceVideoStrategy


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "infer_multiref_stage2_overfit.py"
_SPEC = importlib.util.spec_from_file_location("infer_multiref_stage2_overfit_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
infer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = infer
_SPEC.loader.exec_module(infer)


def _valid_checkpoint_state() -> dict[str, torch.Tensor]:
    return {
        "diffusion_model.block.lora_A.default.weight": torch.ones(1),
        "text_encoder.model.model.language_model.block.lora_A.default.weight": torch.ones(1),
        "training_strategy.planner_tokens.query_tokens": torch.ones(1),
        "training_strategy.visual_token_projection.weight": torch.ones(1),
        "training_strategy.visual_full_encoder.input_norm.weight": torch.ones(1),
        "embeddings_processor.video_connector.weight": torch.ones(1),
    }


def test_stage2_config_loads_and_stage1_config_is_rejected() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"

    cfg = infer._load_config(configs / "multiref_stage2_overfit100.yaml")

    assert cfg.training_strategy.name == "multi_reference_planner_stage2"
    assert cfg.training_strategy.cfg_dropout_enabled is False
    assert cfg.training_strategy.gemma_gradient_checkpointing is False
    with pytest.raises(ValueError, match="multi_reference_planner_stage2"):
        infer._load_config(configs / "multiref_stage1_overfit100.yaml")


@pytest.mark.parametrize(
    ("prefix", "message"),
    [
        ("text_encoder.model.model.language_model.", "Gemma"),
        ("training_strategy.planner_tokens.", "planner_tokens"),
        ("training_strategy.visual_token_projection.", "visual_token_projection"),
        ("training_strategy.visual_full_encoder.", "visual_full_encoder"),
    ],
)
def test_checkpoint_missing_required_stage2_component_fails(prefix: str, message: str) -> None:
    state = {key: value for key, value in _valid_checkpoint_state().items() if not key.startswith(prefix)}

    with pytest.raises(RuntimeError, match=message):
        infer._validate_stage2_checkpoint_state(state)


class _FakeProcessor:
    def __init__(self, events: list[str], target_dim: int) -> None:
        self.events = events
        self.target_dim = target_dim

    def create_embeddings(self, video_features, audio_features, attention_mask):
        del attention_mask
        self.events.append("text_connector")
        if video_features.shape[-1] < self.target_dim:
            video_features = F.pad(video_features, (0, self.target_dim - video_features.shape[-1]))
        binary_mask = torch.ones(video_features.shape[:2], dtype=torch.long, device=video_features.device)
        return video_features[..., : self.target_dim], audio_features, binary_mask


class _FakeVisualEncoder:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __call__(self, *, tokens, token_positions, token_mask):
        del token_positions
        self.events.append("visual_encoder")
        return tokens, token_mask


class _InferenceHarness:
    def __init__(self, *, token_count: int = 8, raw_dim: int = 6, target_dim: int = 8) -> None:
        self.config = SimpleNamespace(
            use_online_vlm=True,
            planner_token_count=token_count,
        )
        self.token_count = token_count
        self.raw_dim = raw_dim
        self.target_dim = target_dim
        self.events: list[str] = []
        self.planner_tokens = object()
        self.text_encoder = object()
        self._inference_embeddings_processor = _FakeProcessor(self.events, target_dim)
        self._visual_full_encoder = _FakeVisualEncoder(self.events)

    def _build_visual_token_positions(self, visual_data, latents_data, **kwargs):
        del latents_data, kwargs
        sampled = visual_data["sampled_frame_indices"].float().sum()
        return torch.full((1, 3, self.token_count, 2), sampled.item(), dtype=torch.float32)

    def _run_online_vlm_inference(
        self,
        planner_data,
        *,
        device,
        token_positions,
        drop_reference_images=False,
    ):
        del token_positions
        self.events.append("vlm_planner_no_ref" if drop_reference_images else "vlm_planner")
        value = planner_data["input_ids"].float().sum()
        pixel_values = planner_data.get("pixel_values")
        if isinstance(pixel_values, torch.Tensor) and not drop_reference_images:
            value = value + pixel_values.float().sum()
        tokens = torch.full(
            (1, self.token_count, self.raw_dim),
            value.item(),
            device=device,
            dtype=torch.float16,
        )
        return tokens, torch.ones(1, self.token_count, device=device, dtype=torch.bool)

    def _assert_token_count(self, name, tokens, mask):
        del name
        assert tokens.shape[:2] == mask.shape == (1, self.token_count)

    def _resolved_planner_source_dim(self):
        return self.raw_dim

    def _resolved_planner_output_dim(self, **kwargs):
        del kwargs
        return self.raw_dim

    @staticmethod
    def _assert_visual_token_dim(name, tokens, expected_dim):
        del name
        assert tokens.shape[-1] == expected_dim

    @staticmethod
    def _pad_conditions_to_connector_multiple(conditions):
        return conditions

    def _project_visual_tokens(self, tokens, *, target_dim):
        self.events.append("visual_projection")
        if tokens.shape[-1] < target_dim:
            tokens = F.pad(tokens, (0, target_dim - tokens.shape[-1]))
        return tokens[..., :target_dim]

    @staticmethod
    def _validate_full_visual_token_layout(visual_data, *, token_count):
        del visual_data, token_count

    def _append_postconnector_visual_context(self, conditions, visual_context, visual_mask):
        self.events.append("append_visual")
        return MultiReferenceVideoStrategy._append_postconnector_visual_context(
            self,
            conditions,
            visual_context,
            visual_mask,
        )

    def prepare_inference_conditions(self, **kwargs):
        return MultiReferencePlannerStage2Strategy.prepare_inference_conditions(self, **kwargs)

    def prepare_inference_condition_parts(self, **kwargs):
        return MultiReferencePlannerStage2Strategy.prepare_inference_condition_parts(self, **kwargs)

    def append_shared_inference_visual_context(self, conditions, **kwargs):
        return MultiReferencePlannerStage2Strategy.append_shared_inference_visual_context(
            self,
            conditions,
            **kwargs,
        )


def _prepare_with_harness(
    harness: _InferenceHarness,
    *,
    input_ids: torch.Tensor | None = None,
    pixel_values: torch.Tensor | None = None,
    visual_metadata: dict | None = None,
    condition_value: float = 0.0,
    drop_reference_images: bool = False,
):
    conditions = {
        "video_prompt_embeds": torch.full(
            (1, 2, harness.raw_dim),
            condition_value,
            dtype=torch.float16,
        ),
        "audio_prompt_embeds": None,
        "prompt_attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    planner_inputs = {"input_ids": input_ids if input_ids is not None else torch.tensor([[1, 2]])}
    if pixel_values is not None:
        planner_inputs["pixel_values"] = pixel_values
    if visual_metadata is None:
        visual_metadata = {
            "tokens_per_frame": torch.tensor([max(harness.token_count, 1)]),
            "sampled_frame_indices": torch.tensor([[0]]),
            "source_fps": torch.tensor([24.0]),
        }
    return MultiReferencePlannerStage2Strategy.prepare_inference_conditions(
        harness,
        conditions=conditions,
        planner_vlm_inputs=planner_inputs,
        latents_metadata={"height": torch.tensor([64]), "width": torch.tensor([96])},
        visual_position_metadata=visual_metadata,
        drop_reference_images=drop_reference_images,
    )


def _prepare_guidance_with_harness(
    harness: _InferenceHarness,
    *,
    ref_guidance_mode: infer.RefGuidanceMode,
    guidance_scale: float = 2.0,
    ref_guidance_scale: float = 2.0,
):
    positive_conditions = {
        "video_prompt_embeds": torch.full((1, 2, harness.raw_dim), 1.0, dtype=torch.float16),
        "audio_prompt_embeds": None,
        "prompt_attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    text_conditions = {
        "video_prompt_embeds": torch.full((1, 2, harness.raw_dim), 5.0, dtype=torch.float16),
        "audio_prompt_embeds": None,
        "prompt_attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    negative_conditions = {
        "video_prompt_embeds": torch.full((1, 3, harness.target_dim), -3.0, dtype=torch.float16),
        "audio_prompt_embeds": None,
        "prompt_attention_mask": None,
    }
    return infer._prepare_guidance_condition_bundle(
        strategy=harness,
        conditions=positive_conditions,
        text_conditions=text_conditions,
        planner_vlm_inputs={
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.ones(1, 1, 1),
        },
        latents_metadata={"height": torch.tensor([64]), "width": torch.tensor([96])},
        visual_position_metadata={
            "tokens_per_frame": torch.tensor([max(harness.token_count, 1)]),
            "sampled_frame_indices": torch.tensor([[0]]),
            "source_fps": torch.tensor([24.0]),
        },
        negative_text_conditions=negative_conditions,
        guidance_scale=guidance_scale,
        ref_guidance_scale=ref_guidance_scale,
        ref_guidance_mode=ref_guidance_mode,
    )


def test_stage2_inference_shapes_and_postconnector_append_order() -> None:
    harness = _InferenceHarness(token_count=2048, raw_dim=3840, target_dim=4096)

    conditions, diagnostics = _prepare_with_harness(harness)

    assert diagnostics["planner_raw_shape"] == [1, 2048, 3840]
    assert diagnostics["planner_projected_shape"] == [1, 2048, 4096]
    assert diagnostics["visual_context_shape"] == [1, 2048, 4096]
    assert conditions["video_prompt_embeds"].shape == (1, 2050, 4096)
    assert harness.events == [
        "vlm_planner",
        "text_connector",
        "visual_projection",
        "visual_encoder",
        "append_visual",
    ]


def test_gt_visual_token_values_do_not_change_stage2_generation_conditions() -> None:
    first_metadata = infer._extract_gt_position_metadata(
        {
            "tokens_per_frame": torch.tensor(8),
            "sampled_frame_indices": torch.tensor([0]),
            "source_fps": torch.tensor(24.0),
            "visual_tokens": torch.zeros(8, 6),
        }
    )
    second_metadata = infer._extract_gt_position_metadata(
        {
            "tokens_per_frame": torch.tensor(8),
            "sampled_frame_indices": torch.tensor([0]),
            "source_fps": torch.tensor(24.0),
            "visual_tokens": torch.full((8, 6), 1000.0),
        }
    )
    first_harness = _InferenceHarness()
    second_harness = _InferenceHarness()

    first_conditions, first_diagnostics = _prepare_with_harness(
        first_harness,
        visual_metadata=first_metadata,
    )
    second_conditions, second_diagnostics = _prepare_with_harness(
        second_harness,
        visual_metadata=second_metadata,
    )

    assert torch.equal(first_diagnostics["predicted_visual_tokens"], second_diagnostics["predicted_visual_tokens"])
    assert torch.equal(first_conditions["video_prompt_embeds"], second_conditions["video_prompt_embeds"])


def test_stage2_inference_rejects_raw_gt_visual_tokens() -> None:
    with pytest.raises(ValueError, match="remove raw 'visual_tokens'"):
        _prepare_with_harness(
            _InferenceHarness(),
            visual_metadata={
                "tokens_per_frame": torch.tensor([8]),
                "sampled_frame_indices": torch.tensor([[0]]),
                "source_fps": torch.tensor([24.0]),
                "visual_tokens": torch.zeros(1, 8, 6),
            },
        )


def test_gt_position_metadata_explicitly_excludes_visual_tokens() -> None:
    metadata = infer._extract_gt_position_metadata(
        {
            "tokens_per_frame": torch.tensor(256),
            "sampled_frame_indices": torch.arange(8),
            "source_fps": torch.tensor(24.0),
            "visual_tokens": torch.randn(2048, 4),
        }
    )

    assert set(metadata) == {"tokens_per_frame", "sampled_frame_indices", "source_fps"}
    assert "visual_tokens" not in metadata


def test_caption_and_reference_inputs_change_planner_prediction() -> None:
    caption_a = _prepare_with_harness(_InferenceHarness(), input_ids=torch.tensor([[1, 2]]))[1]
    caption_b = _prepare_with_harness(_InferenceHarness(), input_ids=torch.tensor([[1, 3]]))[1]
    ref_a = _prepare_with_harness(_InferenceHarness(), pixel_values=torch.zeros(1, 1, 1))[1]
    ref_b = _prepare_with_harness(_InferenceHarness(), pixel_values=torch.ones(1, 1, 1))[1]

    assert not torch.equal(caption_a["predicted_visual_tokens"], caption_b["predicted_visual_tokens"])
    assert not torch.equal(ref_a["predicted_visual_tokens"], ref_b["predicted_visual_tokens"])


def test_no_ref_planner_reruns_without_pixels_and_uses_text_only_context() -> None:
    harness = _InferenceHarness()
    pixel_values = torch.ones(1, 2, 3)

    _full_conditions, full_diagnostics = _prepare_with_harness(
        harness,
        pixel_values=pixel_values,
        condition_value=1.0,
    )
    no_ref_conditions, no_ref_diagnostics = _prepare_with_harness(
        harness,
        pixel_values=pixel_values,
        condition_value=5.0,
        drop_reference_images=True,
    )

    assert harness.events.count("vlm_planner") == 1
    assert harness.events.count("vlm_planner_no_ref") == 1
    assert no_ref_diagnostics["drop_reference_images"] is True
    assert not torch.equal(
        full_diagnostics["predicted_visual_tokens"],
        no_ref_diagnostics["predicted_visual_tokens"],
    )
    text_prefix = no_ref_conditions["video_prompt_embeds"][:, :2]
    assert torch.all(text_prefix[..., : harness.raw_dim] == 5.0)
    assert torch.all(text_prefix[..., harness.raw_dim :] == 0.0)


def test_no_ref_planner_data_removes_pixels_and_zeros_reference_count() -> None:
    planner_data = {
        "input_ids": torch.ones(2, 5, dtype=torch.long),
        "pixel_values": torch.randn(3, 3, 8, 8),
        "num_ref_images": torch.tensor([1, 2]),
    }

    no_ref_data = MultiReferencePlannerStage2Strategy._prepare_inference_planner_data(
        planner_data,
        drop_reference_images=True,
        device=torch.device("cpu"),
    )

    assert "pixel_values" not in no_ref_data
    assert no_ref_data["num_ref_images"].tolist() == [0, 0]
    assert "pixel_values" in planner_data
    assert planner_data["num_ref_images"].tolist() == [1, 2]


def test_ref_guidance_mode_defaults_to_synchronized() -> None:
    option = inspect.signature(infer.main).parameters["ref_guidance_mode"].default

    assert infer._DEFAULT_REF_GUIDANCE_MODE == "synchronized"
    assert option.default == "synchronized"


def test_synchronized_ref_guidance_preserves_legacy_condition_construction() -> None:
    harness = _InferenceHarness()
    bundle = _prepare_guidance_with_harness(
        harness,
        ref_guidance_mode="synchronized",
    )

    assert harness.events.count("vlm_planner") == 1
    assert harness.events.count("vlm_planner_no_ref") == 1
    assert bundle.planner_forward_count == 2
    assert bundle.negative_conditions is not None
    assert bundle.negative_conditions["video_prompt_embeds"].shape == (1, 3, harness.target_dim)
    assert bundle.no_ref_conditions is not None
    assert bundle.shared_visual_context is None
    assert bundle.shared_visual_mask is None


def test_shared_planner_is_run_once_and_reused_by_positive_negative_and_no_ref() -> None:
    harness = _InferenceHarness()
    bundle = _prepare_guidance_with_harness(
        harness,
        ref_guidance_mode="shared_planner_latent_only",
    )

    assert harness.events.count("vlm_planner") == 1
    assert harness.events.count("vlm_planner_no_ref") == 0
    assert bundle.planner_forward_count == 1
    assert bundle.shared_visual_context is not None
    assert bundle.shared_visual_mask is not None
    assert bundle.negative_conditions is not None
    positive_context = bundle.positive_conditions["video_prompt_embeds"]
    negative_context = bundle.negative_conditions["video_prompt_embeds"]
    positive_mask = bundle.positive_conditions["prompt_attention_mask"]
    negative_mask = bundle.negative_conditions["prompt_attention_mask"]
    visual_token_count = bundle.shared_visual_context.shape[1]

    assert torch.equal(positive_context[:, -visual_token_count:], bundle.shared_visual_context)
    assert torch.equal(negative_context[:, -visual_token_count:], bundle.shared_visual_context)
    assert torch.equal(positive_context[:, -visual_token_count:], negative_context[:, -visual_token_count:])
    assert torch.equal(positive_mask[:, -visual_token_count:], bundle.shared_visual_mask.long())
    assert torch.equal(negative_mask[:, -visual_token_count:], bundle.shared_visual_mask.long())
    assert torch.all(positive_context[:, :2, : harness.raw_dim] == 5.0)
    assert torch.all(positive_context[:, :2, harness.raw_dim :] == 0.0)
    assert not torch.equal(positive_context[:, :2], negative_context[:, :2])
    assert negative_context.shape[1] == 3 + visual_token_count
    assert bundle.no_ref_conditions is None


def test_renamed_full_vlm_shared_mode_preserves_previous_positive_prefix() -> None:
    harness = _InferenceHarness()
    bundle = _prepare_guidance_with_harness(
        harness,
        ref_guidance_mode="shared_planner_full_vlm_latent_only",
    )

    positive_context = bundle.positive_conditions["video_prompt_embeds"]

    assert harness.events.count("vlm_planner") == 1
    assert torch.all(positive_context[:, :2, : harness.raw_dim] == 1.0)
    assert torch.all(positive_context[:, :2, harness.raw_dim :] == 0.0)


@pytest.mark.parametrize(
    "ref_guidance_mode",
    [
        "synchronized",
        "shared_planner_latent_only",
        "shared_planner_full_vlm_latent_only",
    ],
)
def test_zero_ref_guidance_does_not_build_no_ref_condition(
    ref_guidance_mode: infer.RefGuidanceMode,
) -> None:
    harness = _InferenceHarness()
    bundle = _prepare_guidance_with_harness(
        harness,
        ref_guidance_mode=ref_guidance_mode,
        ref_guidance_scale=0.0,
    )

    assert bundle.no_ref_conditions is None
    assert harness.events.count("vlm_planner_no_ref") == 0


def test_cfg_scale_one_does_not_build_shared_negative_condition() -> None:
    harness = _InferenceHarness()
    bundle = _prepare_guidance_with_harness(
        harness,
        ref_guidance_mode="shared_planner_latent_only",
        guidance_scale=1.0,
    )

    assert bundle.negative_conditions is None
    assert bundle.planner_forward_count == 1


def test_ref_guidance_metadata_records_both_modes() -> None:
    synchronized = infer._guidance_metadata(
        ref_guidance_mode="synchronized",
        planner_forward_count=2,
        guidance_enabled=True,
        ref_guidance_enabled=True,
    )
    shared = infer._guidance_metadata(
        ref_guidance_mode="shared_planner_latent_only",
        planner_forward_count=1,
        guidance_enabled=True,
        ref_guidance_enabled=True,
    )

    assert synchronized["ref_guidance_mode"] == "synchronized"
    assert synchronized["negative_uses_shared_planner"] is False
    assert synchronized["no_ref_uses_shared_planner"] is False
    assert shared == {
        "ref_guidance_mode": "shared_planner_latent_only",
        "planner_forward_count": 1,
        "positive_uses_text_only_prefix": True,
        "negative_uses_shared_planner": True,
        "no_ref_uses_shared_planner": True,
        "no_ref_branch_is_synchronized": False,
        "ref_guidance_formula": (
            "positive_text_shared_planner_with_refs - "
            "positive_text_shared_planner_without_ref_latents"
        ),
        "guidance_formula": (
            "N_shared_planner_with_refs + "
            "cfg*(P_positive_text_shared_planner_with_refs-N_shared_planner_with_refs) + "
            "ref*(P_positive_text_shared_planner_with_refs-"
            "P_positive_text_shared_planner_without_ref_latents) + "
            "stg*(P_positive_text_shared_planner_with_refs-S_stg_of_P)"
        ),
    }


def test_shared_planner_metadata_flags_follow_enabled_guidance_branches() -> None:
    metadata = infer._guidance_metadata(
        ref_guidance_mode="shared_planner_latent_only",
        planner_forward_count=1,
        guidance_enabled=False,
        ref_guidance_enabled=False,
    )

    assert metadata["negative_uses_shared_planner"] is False
    assert metadata["no_ref_uses_shared_planner"] is False


def test_shared_multidirectional_guidance_rescale_runs_after_all_deltas() -> None:
    positive = torch.tensor([[[1.0, 4.0], [2.0, 8.0]]])
    negative = torch.tensor([[[0.0, 1.0], [1.0, 2.0]]])
    no_ref = torch.tensor([[[0.5, 2.0], [1.5, 3.0]]])
    stg = torch.tensor([[[0.25, 1.0], [0.5, 2.0]]])
    unscaled = infer.stage1._combine_multidirectional_denoised(
        denoised_pos=positive,
        denoised_neg=negative,
        denoised_no_ref=no_ref,
        denoised_siglip_isolated=None,
        denoised_siglip_null=None,
        denoised_stg=stg,
        guidance_scale=2.0,
        ref_guidance_scale=1.5,
        siglip_guidance_scale=0.0,
        stg_scale=0.75,
        guidance_rescale=0.0,
        target_seq_len=2,
    )
    rescaled = infer.stage1._combine_multidirectional_denoised(
        denoised_pos=positive,
        denoised_neg=negative,
        denoised_no_ref=no_ref,
        denoised_siglip_isolated=None,
        denoised_siglip_null=None,
        denoised_stg=stg,
        guidance_scale=2.0,
        ref_guidance_scale=1.5,
        siglip_guidance_scale=0.0,
        stg_scale=0.75,
        guidance_rescale=0.7,
        target_seq_len=2,
    )
    factor = 0.7 * (positive.float().std() / unscaled.float().std().clamp(min=1.0e-8)) + 0.3

    assert torch.allclose(rescaled, unscaled * factor.to(dtype=unscaled.dtype))


def _run_fake_shared_denoise(
    monkeypatch: pytest.MonkeyPatch,
    *,
    guidance_scale: float,
    ref_guidance_scale: float,
    stg_scale: float,
) -> tuple[list[dict[str, object]], torch.Tensor]:
    calls: list[dict[str, object]] = []

    class FakePatchifier:
        @staticmethod
        def patchify(latents):
            return latents.flatten(2).transpose(1, 2)

        @staticmethod
        def unpatchify(tokens, output_shape):
            del output_shape
            return tokens

    class FakeStrategy:
        config = SimpleNamespace(max_ref_images_per_sample=None, reference_time_stride=1.0)
        _video_patchifier = FakePatchifier()

        @staticmethod
        def _normalize_reference_latents(latents):
            return latents

        @staticmethod
        def _get_reference_valid_mask(ref_data, ref_latents):
            del ref_data
            return torch.ones(ref_latents.shape[:2], dtype=torch.bool)

        @staticmethod
        def _get_video_positions(**kwargs):
            return torch.zeros(kwargs["batch_size"], 3, 1, 2)

        @staticmethod
        def _scale_reference_positions(positions, *args):
            del args
            return positions

        @staticmethod
        def _first_scalar(value, *, default):
            del value
            return default

    class FakeTransformer(torch.nn.Module):
        def forward(self, *, video, audio, perturbations):
            del audio
            calls.append(
                {
                    "context": video.context,
                    "context_mask": video.context_mask,
                    "ref_valid_mask": video.attention_mask.clone(),
                    "latent": video.latent.clone(),
                    "timesteps": video.timesteps.clone(),
                    "positions": video.positions.clone(),
                    "is_stg": perturbations is not None,
                }
            )
            return torch.zeros_like(video.latent), None

    class FakeScheduler:
        @staticmethod
        def execute(*, steps):
            del steps
            return torch.tensor([1.0, 0.0])

    class FakeStepper:
        @staticmethod
        def step(latent, denoised, sigmas, step_idx):
            del denoised, sigmas, step_idx
            return latent

    def fake_build_multiref_sequence(
        *,
        ref_tokens,
        ref_positions,
        ref_valid_mask,
        target_tokens,
        target_positions,
        target_timesteps,
        target_loss_mask,
        reference_time_stride,
    ):
        del ref_positions, target_positions, target_loss_mask, reference_time_stride
        flat_ref_tokens = ref_tokens.reshape(ref_tokens.shape[0], -1, ref_tokens.shape[-1])
        latents = torch.cat([flat_ref_tokens, target_tokens], dim=1)
        timesteps = torch.cat(
            [
                torch.zeros(latents.shape[0], flat_ref_tokens.shape[1]),
                target_timesteps,
            ],
            dim=1,
        )
        positions = torch.zeros(latents.shape[0], 3, latents.shape[1], 2)
        return SimpleNamespace(
            latents=latents,
            timesteps=timesteps,
            positions=positions,
            attention_mask=ref_valid_mask,
        )

    monkeypatch.setattr(infer.stage1, "LTX2Scheduler", FakeScheduler)
    monkeypatch.setattr(infer.stage1, "EulerDiffusionStep", FakeStepper)
    monkeypatch.setattr(infer.stage1, "build_multiref_sequence", fake_build_multiref_sequence)
    positive_context = torch.ones(1, 4, 3)
    positive_mask = torch.ones(1, 4, dtype=torch.long)
    negative_context = torch.full((1, 5, 3), -1.0)
    output = infer.stage1._denoise_stage1(
        transformer=FakeTransformer(),
        strategy=FakeStrategy(),
        batch={
            "latents": {
                "latents": torch.zeros(1, 2, 1, 1, 1),
                "num_frames": torch.tensor([1]),
                "height": torch.tensor([1]),
                "width": torch.tensor([1]),
                "fps": torch.tensor([24.0]),
            },
            "multi_ref_latents": {
                "latents": torch.ones(1, 1, 2, 1, 1, 1),
                "fps": torch.tensor([1.0]),
            },
        },
        positive_conditions={
            "video_prompt_embeds": positive_context,
            "prompt_attention_mask": positive_mask,
        },
        negative_conditions={
            "video_prompt_embeds": negative_context,
            "prompt_attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        if guidance_scale != 1.0
        else None,
        no_ref_conditions=None,
        guidance_scale=guidance_scale,
        cfg_drop_ref_latents_in_negative=False,
        ref_guidance_scale=ref_guidance_scale,
        siglip_guidance_scale=0.0,
        guidance_rescale=0.7,
        stg_scale=stg_scale,
        stg_blocks=[28],
        num_inference_steps=1,
        seed=42,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return calls, output


def test_shared_mode_four_dit_branches_reuse_context_and_drop_only_ref_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, output = _run_fake_shared_denoise(
        monkeypatch,
        guidance_scale=2.0,
        ref_guidance_scale=2.0,
        stg_scale=1.0,
    )

    assert len(calls) == 4
    positive, negative, no_ref, stg = calls
    assert negative["context"] is not positive["context"]
    assert no_ref["context"] is positive["context"]
    assert no_ref["context_mask"] is positive["context_mask"]
    assert stg["context"] is positive["context"]
    assert torch.equal(positive["ref_valid_mask"], torch.ones(1, 1, dtype=torch.bool))
    assert torch.equal(negative["ref_valid_mask"], positive["ref_valid_mask"])
    assert torch.equal(no_ref["ref_valid_mask"], torch.zeros(1, 1, dtype=torch.bool))
    for field in ("latent", "timesteps", "positions"):
        assert torch.equal(no_ref[field], positive[field])
    assert stg["is_stg"] is True
    assert torch.isfinite(output).all()


def test_zero_ref_scale_skips_no_ref_dit_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, _ = _run_fake_shared_denoise(
        monkeypatch,
        guidance_scale=2.0,
        ref_guidance_scale=0.0,
        stg_scale=1.0,
    )

    assert len(calls) == 3
    assert all(torch.any(call["ref_valid_mask"]) for call in calls)


def test_cfg_scale_one_skips_negative_dit_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, _ = _run_fake_shared_denoise(
        monkeypatch,
        guidance_scale=1.0,
        ref_guidance_scale=2.0,
        stg_scale=1.0,
    )

    assert len(calls) == 3
    assert calls[1]["context"] is calls[0]["context"]
    assert torch.equal(calls[1]["ref_valid_mask"], torch.zeros(1, 1, dtype=torch.bool))


def test_gt_metrics_are_diagnostic_only() -> None:
    conditions, diagnostics = _prepare_with_harness(_InferenceHarness())
    condition_before = conditions["video_prompt_embeds"].clone()
    metrics = infer._compute_planner_metrics(
        predicted_tokens=diagnostics["predicted_visual_tokens"],
        predicted_mask=diagnostics["predicted_visual_token_mask"],
        gt_data={"visual_tokens": torch.ones(8, 6)},
        visual_token_key="visual_tokens",
    )

    assert "planner_siglip_mse" in metrics
    assert torch.equal(conditions["video_prompt_embeds"], condition_before)


def test_predicted_token_artifact_roundtrip(tmp_path: Path) -> None:
    _conditions, diagnostics = _prepare_with_harness(_InferenceHarness())

    output = infer._save_predicted_tokens(
        output_dir=tmp_path,
        rel_path=Path("part_000/sample.pt"),
        inference_diagnostics=diagnostics,
        planner_metrics={"planner_siglip_mse": 1.25},
    )
    loaded = torch.load(output, map_location="cpu", weights_only=True)

    assert torch.equal(loaded["predicted_visual_tokens"], diagnostics["predicted_visual_tokens"][0])
    assert loaded["predicted_visual_tokens"].shape == (8, 6)
    assert loaded["token_positions"].shape == (3, 8, 2)
    assert loaded["diagnostics"]["planner_siglip_mse"] == 1.25


def test_stage2_shard_selection_and_skip_existing(tmp_path: Path) -> None:
    assert infer.stage1._select_sample_indices(
        row_count=10,
        start_index=0,
        end_index=None,
        shard_index=2,
        num_shards=3,
    ) == [2, 5, 8]
    video = infer.stage1._expected_generated_path(tmp_path, 5, infer._CONDITION_MODE)
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    (video.parent / "metadata.json").write_text(
        json.dumps(
            {
                "sample_index": 5,
                "condition_mode": infer._CONDITION_MODE,
                "generated": str(video),
            }
        ),
        encoding="utf-8",
    )
    run_sample = Mock()

    summary = infer.stage1._run_selected_samples(
        selected_indices=[5],
        output_dir=tmp_path,
        condition_mode=infer._CONDITION_MODE,
        checkpoint_path=tmp_path / "checkpoint.safetensors",
        shard_index=0,
        num_shards=1,
        skip_existing=True,
        continue_on_error=True,
        gc_interval=0,
        device=torch.device("cpu"),
        run_sample=run_sample,
        write_summary=True,
    )

    assert summary["num_skipped"] == 1
    assert run_sample.call_count == 0


def test_ref_guidance_modes_use_distinct_output_condition_modes() -> None:
    synchronized = infer._condition_mode_for_ref_guidance("synchronized")
    shared_text_only = infer._condition_mode_for_ref_guidance("shared_planner_latent_only")
    shared_full_vlm = infer._condition_mode_for_ref_guidance("shared_planner_full_vlm_latent_only")

    assert synchronized == infer._CONDITION_MODE == "stage2_planner"
    assert len({synchronized, shared_text_only, shared_full_vlm}) == 3
    assert infer.stage1._expected_generated_path(Path("outputs"), 0, synchronized) != (
        infer.stage1._expected_generated_path(Path("outputs"), 0, shared_text_only)
    )


def test_strict_no_gt_loader_never_builds_or_loads_gt_path(monkeypatch, tmp_path: Path) -> None:
    loaded_paths: list[Path] = []

    def load_pt(path: Path) -> dict:
        loaded_paths.append(path)
        return {}

    monkeypatch.setattr(infer.stage1, "_load_pt_file", load_pt)
    _rel_path, precomputed = infer._load_sample_precomputed(
        row={"video": "part_000/sample.mp4"},
        manifest_root=tmp_path / "manifest",
        precomputed_root=tmp_path / ".precomputed",
        video_column="video",
        need_gt=False,
        strict_no_gt=True,
    )

    assert set(precomputed) == {
        "latents",
        "multi_reference_latents",
        "conditions",
        "text_conditions",
        "planner_vlm_inputs",
    }
    assert len(loaded_paths) == 5
    assert all("gt_siglip_tokens" not in path.parts for path in loaded_paths)


def test_strict_no_gt_rejects_gt_metadata_and_metrics() -> None:
    with pytest.raises(infer.typer.BadParameter, match="position-source uniform_target"):
        infer._validate_strict_no_gt(
            strict_no_gt=True,
            position_source="gt_metadata",
            compute_gt_siglip_metrics=False,
        )
    with pytest.raises(infer.typer.BadParameter, match="no-compute-gt-siglip-metrics"):
        infer._validate_strict_no_gt(
            strict_no_gt=True,
            position_source="uniform_target",
            compute_gt_siglip_metrics=True,
        )


def test_uniform_positions_and_guidance_validation_are_deterministic() -> None:
    latents = {"num_frames": torch.tensor([97]), "fps": torch.tensor([24.0])}

    first = infer._uniform_position_metadata(latents)
    second = infer._uniform_position_metadata(latents)

    assert torch.equal(first["sampled_frame_indices"], second["sampled_frame_indices"])
    assert first["sampled_frame_indices"].tolist() == [[0, 14, 27, 41, 55, 69, 82, 96]]
    infer._validate_guidance(
        guidance_scale=2.0,
        ref_guidance_scale=2.0,
        siglip_guidance_scale=0.0,
        stg_scale=1.0,
    )
    with pytest.raises(infer.typer.BadParameter, match="siglip-guidance-scale 0.0"):
        infer._validate_guidance(
            guidance_scale=2.0,
            ref_guidance_scale=2.0,
            siglip_guidance_scale=1.0,
            stg_scale=1.0,
        )


def test_negative_prompt_resolution_prefers_cli_and_rejects_empty_cfg_prompt() -> None:
    assert infer._resolve_negative_prompt(
        cli_negative_prompt="cli negative",
        config_negative_prompt="config negative",
        guidance_scale=2.0,
    ) == "cli negative"
    assert infer._resolve_negative_prompt(
        cli_negative_prompt=None,
        config_negative_prompt="config negative",
        guidance_scale=2.0,
    ) == "config negative"
    assert infer._resolve_negative_prompt(
        cli_negative_prompt="  ",
        config_negative_prompt="config negative",
        guidance_scale=2.0,
    ) == "config negative"
    with pytest.raises(ValueError, match="non-empty negative prompt"):
        infer._resolve_negative_prompt(
            cli_negative_prompt="  ",
            config_negative_prompt="",
            guidance_scale=2.0,
        )


def test_negative_condition_uses_prompt_once_and_has_no_planner_visual_context(monkeypatch) -> None:
    events: list[str] = []

    class LanguageModel:
        def disable_adapter(self):
            events.append("disable_adapter")
            return nullcontext()

    class TextEncoder:
        def encode(self, prompts):
            events.append(f"encode:{prompts[0]}")
            return [(torch.ones(1, 3, 4), torch.ones(1, 3, dtype=torch.long))]

    class Processor:
        def process_hidden_states(self, hidden_states, attention_mask):
            del attention_mask
            events.append("text_connector")
            return SimpleNamespace(video_encoding=hidden_states, audio_encoding=None)

    strategy = SimpleNamespace(_get_language_model=lambda: LanguageModel())
    monkeypatch.setattr(infer, "_autocast_context", lambda device, dtype: nullcontext())
    conditions = infer._encode_negative_prompt_condition(
        text_encoder=TextEncoder(),
        embeddings_processor=Processor(),
        strategy=strategy,
        negative_prompt="held-out negative",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert events == ["disable_adapter", "encode:held-out negative", "text_connector"]
    assert conditions["video_prompt_embeds"].shape == (1, 3, 4)
    assert conditions["prompt_attention_mask"] is None


class _RuntimeFakeProcessor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = object()
        self.process_calls = 0

    def process_hidden_states(self, hidden_states, attention_mask):
        del attention_mask
        assert self.feature_extractor is not None
        self.process_calls += 1
        return SimpleNamespace(video_encoding=hidden_states, audio_encoding=None)


class _RuntimeFakeTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encode_calls = 0

    def encode(self, prompts):
        self.encode_calls += 1
        assert prompts == ["negative"]
        return [(torch.ones(1, 3, 4), torch.ones(1, 3, dtype=torch.long))]


class _RuntimeFakeLanguageModel:
    @staticmethod
    def disable_adapter():
        return nullcontext()


class _RuntimeFakeStrategy:
    def attach_models(self, **kwargs) -> None:
        del kwargs

    @staticmethod
    def get_trainable_modules() -> dict:
        return {}

    @staticmethod
    def _get_language_model():
        return _RuntimeFakeLanguageModel()


def _install_runtime_fakes(monkeypatch, processor: _RuntimeFakeProcessor) -> _RuntimeFakeTextEncoder:
    text_encoder = _RuntimeFakeTextEncoder()
    strategy = _RuntimeFakeStrategy()
    cfg = SimpleNamespace(
        model=SimpleNamespace(model_path="model", text_encoder_path="gemma"),
        acceleration=SimpleNamespace(load_text_encoder_in_8bit=False),
        training_strategy=object(),
        validation=SimpleNamespace(negative_prompt="negative"),
    )
    monkeypatch.setattr(infer, "MultiReferencePlannerStage2Strategy", _RuntimeFakeStrategy)
    monkeypatch.setattr(infer, "_load_config", Mock(return_value=cfg))
    monkeypatch.setattr(infer, "load_transformer", Mock(return_value=torch.nn.Identity()))
    monkeypatch.setattr(infer, "_setup_dit_lora", lambda transformer, config: transformer)
    monkeypatch.setattr(infer, "load_embeddings_processor", Mock(return_value=processor))
    monkeypatch.setattr(infer, "load_text_encoder", Mock(return_value=text_encoder))
    monkeypatch.setattr(infer, "_setup_gemma_lora", lambda encoder, config: None)
    monkeypatch.setattr(infer, "get_training_strategy", Mock(return_value=strategy))
    monkeypatch.setattr(infer, "_load_checkpoint_weights", Mock(return_value={}))
    monkeypatch.setattr(infer, "_disable_gradient_checkpointing", lambda transformer, value: None)
    monkeypatch.setattr(infer, "load_video_vae_decoder", Mock(return_value=torch.nn.Identity()))
    monkeypatch.setattr(infer, "_autocast_context", lambda device, dtype: nullcontext())
    return text_encoder


def test_negative_prompt_is_encoded_before_feature_extractor_release(monkeypatch, tmp_path: Path) -> None:
    processor = _RuntimeFakeProcessor()
    text_encoder = _install_runtime_fakes(monkeypatch, processor)

    runtime = infer._load_inference_runtime(
        config_path=tmp_path / "config.yaml",
        checkpoint_path=tmp_path / "checkpoint.safetensors",
        device=torch.device("cpu"),
        dtype=torch.float32,
        guidance_scale=2.0,
        negative_prompt=None,
    )

    assert processor.process_calls == 1
    assert text_encoder.encode_calls == 1
    assert runtime.negative_conditions is not None
    assert processor.feature_extractor is None


def test_cfg_disabled_releases_feature_extractor_without_encoding_negative(monkeypatch, tmp_path: Path) -> None:
    processor = _RuntimeFakeProcessor()
    text_encoder = _install_runtime_fakes(monkeypatch, processor)
    encode_negative = Mock(side_effect=AssertionError("negative prompt must not be encoded"))
    monkeypatch.setattr(infer, "_encode_negative_prompt_condition", encode_negative)

    runtime = infer._load_inference_runtime(
        config_path=tmp_path / "config.yaml",
        checkpoint_path=tmp_path / "checkpoint.safetensors",
        device=torch.device("cpu"),
        dtype=torch.float32,
        guidance_scale=1.0,
        negative_prompt=None,
    )

    assert encode_negative.call_count == 0
    assert text_encoder.encode_calls == 0
    assert runtime.negative_conditions is None
    assert processor.feature_extractor is None


def test_held_out_guidance_formula_is_numerically_correct() -> None:
    full = torch.tensor([[[10.0]]])
    negative = torch.tensor([[[2.0]]])
    no_ref = torch.tensor([[[4.0]]])
    stg = torch.tensor([[[7.0]]])

    guided = infer.stage1._combine_multidirectional_denoised(
        denoised_pos=full,
        denoised_neg=negative,
        denoised_no_ref=no_ref,
        denoised_siglip_isolated=None,
        denoised_siglip_null=None,
        denoised_stg=stg,
        guidance_scale=2.0,
        ref_guidance_scale=2.0,
        siglip_guidance_scale=0.0,
        stg_scale=1.0,
        guidance_rescale=0.0,
        target_seq_len=1,
    )

    assert guided.item() == 33.0
