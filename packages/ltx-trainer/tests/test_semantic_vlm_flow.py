from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from accelerate import DistributedType
from safetensors.torch import save_file
from torch import nn

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.modality import Modality, SemanticVideoPrediction
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.multicond.semantic_tokens import EVIDENCE_TOKENS_PER_FRAME
from ltx_core.types import VideoLatentShape
from ltx_trainer.online_data.constants import IMAGE_TASK
from ltx_trainer.online_inference.checkpoint_runtime import (
    SEMANTIC_VLM_STRATEGY_CHECKPOINT_PREFIXES,
    SEMANTIC_VLM_TRANSFORMER_CHECKPOINT_PREFIXES,
    CheckpointAuditError,
    audit_checkpoint,
)
from ltx_trainer.online_inference.semantic_guidance import rescale_guided_denoised_branches
from ltx_trainer.trainer import LtxvTrainer, _enforce_semantic_flow_fsdp_runtime_safety
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.semantic_flow import (
    ENTITY_GLOBAL,
    TYPE_REFERENCE,
    TYPE_SEMANTIC,
    TYPE_TARGET,
)
from ltx_trainer.training_strategies.semantic_vlm_flow import (
    SemanticVLMFlowConfig,
    SemanticVLMFlowStrategy,
)


class _TinyLanguageModel(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, hidden_dim)
        self.config = SimpleNamespace(sliding_window=1024)
        self.forward_calls = 0

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(self, *, inputs_embeds: torch.Tensor, output_hidden_states: bool, **_kwargs) -> SimpleNamespace:
        assert output_hidden_states is False
        self.forward_calls += 1
        return SimpleNamespace(last_hidden_state=inputs_embeds + 2.0)


class _TinyTextEncoder(nn.Module):
    def __init__(self, language_model: nn.Module) -> None:
        super().__init__()
        body = nn.Module()
        body.language_model = language_model
        wrapper = nn.Module()
        wrapper.model = body
        self.model = wrapper


class _FixedTimestepSampler:
    @staticmethod
    def sample_for(samples: torch.Tensor) -> torch.Tensor:
        return torch.full((samples.shape[0],), 0.25, device=samples.device, dtype=samples.dtype)


def _strategy(*, gemma_dim: int = 7, video_dim: int = 3) -> SemanticVLMFlowStrategy:
    strategy = SemanticVLMFlowStrategy(SemanticVLMFlowConfig())
    strategy._gemma_dim = gemma_dim
    strategy._video_dim = video_dim
    strategy._dit_hidden_dim = 5
    return strategy


def _training_batch(*, gemma_dim: int = 7, video_dim: int = 3) -> dict[str, object]:
    return {
        "semantic_teacher_inputs": {
            "normalized_timestamps": torch.zeros(1, 1),
        },
        "latents": {
            "latents": torch.randn(1, video_dim, 1, 1, 1),
            "num_frames": torch.tensor([1]),
            "height": torch.tensor([1]),
            "width": torch.tensor([1]),
            "fps": torch.tensor([1.0]),
        },
        "reference_latents": {
            "latents": torch.randn(1, 1, video_dim, 1, 1, 1),
            "ref_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        },
        "conditions": {
            "video_prompt_embeds": torch.randn(1, 2, 5),
            "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
        },
        "task": [IMAGE_TASK],
        "condition_mode": ["til_111"],
    }


def _prepare_inputs(
    *,
    semantic_clean: torch.Tensor,
    seed: int,
    strategy: SemanticVLMFlowStrategy | None = None,
) -> ModelInputs:
    strategy = strategy or _strategy(gemma_dim=semantic_clean.shape[-1])
    strategy.build_semantic_teacher_outputs = lambda _inputs: {  # type: ignore[method-assign]
        "teacher_hidden": semantic_clean
    }
    torch.manual_seed(seed)
    return strategy.prepare_training_inputs(
        _training_batch(gemma_dim=semantic_clean.shape[-1]),
        _FixedTimestepSampler(),
    )


def _transformer_args(x: torch.Tensor) -> TransformerArgs:
    batch, tokens, hidden = x.shape
    return TransformerArgs(
        x=x,
        context=torch.zeros(batch, 1, hidden),
        context_mask=torch.ones(batch, 1),
        timesteps=torch.zeros(batch, tokens, hidden),
        embedded_timestep=torch.zeros(batch, tokens, hidden),
        positional_embeddings=(torch.zeros(1), torch.zeros(1)),
        cross_positional_embeddings=None,
        cross_scale_shift_timestep=None,
        cross_gate_timestep=None,
        enabled=True,
    )


def _bare_vlm_model(*, gemma_dim: int = 7, video_dim: int = 3, hidden_dim: int = 5) -> LTXModel:
    model = LTXModel.__new__(LTXModel)
    nn.Module.__init__(model)
    model.inner_dim = hidden_dim
    model.patchify_proj = nn.Linear(video_dim, hidden_dim)
    model.proj_out = nn.Linear(hidden_dim, video_dim)
    model.semantic_token_type_embedding = None
    model.semantic_entity_embedding = None
    model.semantic_position_adapter = None
    model.semantic_norm_out = None
    model.semantic_proj_out = None
    model.semantic_token_type_id = None
    model.reference_token_type_id = None
    model.semantic_input_proj = None
    model.semantic_output_proj = None
    model.reference_type_embedding = None
    model.reference_slot_embedding = None
    model.semantic_type_embedding = None
    model.enable_semantic_vlm_flow(
        semantic_dim=gemma_dim,
        num_reference_slots=4,
        semantic_token_type_id=TYPE_SEMANTIC,
        reference_token_type_id=TYPE_REFERENCE,
    )
    return model


def test_teacher_hidden_is_the_detached_semantic_clean_state() -> None:
    gemma_dim = 7
    language_model = _TinyLanguageModel(gemma_dim)
    strategy = _strategy(gemma_dim=gemma_dim)
    strategy._text_encoder = _TinyTextEncoder(language_model)
    evidence = torch.randn(1, 1, EVIDENCE_TOKENS_PER_FRAME, gemma_dim, requires_grad=True)
    prefix = torch.randn(1, 2, gemma_dim, requires_grad=True)

    output = strategy.build_semantic_teacher_outputs(
        {
            "prefix_inputs_embeds": prefix,
            "prefix_attention_mask": torch.ones(1, 2, dtype=torch.bool),
            "prefix_image_token_mask": torch.zeros(1, 2, dtype=torch.bool),
            "evidence_tokens": evidence,
        }
    )["teacher_hidden"]

    assert output.shape == (1, 1, EVIDENCE_TOKENS_PER_FRAME, gemma_dim)
    assert not output.requires_grad
    torch.testing.assert_close(output, evidence.detach() + 2.0)
    assert language_model.forward_calls == 1


def test_semantic_noise_and_target_remain_in_frozen_gemma_space() -> None:
    clean = torch.randn(1, 1, EVIDENCE_TOKENS_PER_FRAME, 7)
    inputs = _prepare_inputs(semantic_clean=clean, seed=17)
    assert inputs.video is not None
    noisy = inputs.video.semantic_latent
    target = inputs.semantic_targets
    assert noisy is not None and target is not None
    assert noisy.shape == target.shape == (1, EVIDENCE_TOKENS_PER_FRAME, 7)
    clean_flat = clean.flatten(1, 2)
    epsilon = target + clean_flat
    torch.testing.assert_close(noisy, 0.75 * clean_flat + 0.25 * epsilon)
    assert not clean.requires_grad
    assert not target.requires_grad


def test_input_projection_weights_cannot_change_clean_noise_or_flow_target() -> None:
    clean = torch.randn(1, 1, EVIDENCE_TOKENS_PER_FRAME, 7)
    strategy = _strategy()
    transformer = _bare_vlm_model()
    strategy._transformer = transformer
    before = _prepare_inputs(semantic_clean=clean, seed=23, strategy=strategy)
    with torch.no_grad():
        transformer.semantic_input_proj.weight.normal_(mean=100.0, std=5.0)
        transformer.semantic_input_proj.bias.fill_(-50.0)
    after = _prepare_inputs(semantic_clean=clean, seed=23, strategy=strategy)

    assert before.video is not None and after.video is not None
    torch.testing.assert_close(before.video.semantic_latent, after.video.semantic_latent, rtol=0.0, atol=0.0)
    torch.testing.assert_close(before.semantic_targets, after.semantic_targets, rtol=0.0, atol=0.0)


class _ProjectedCapture:
    def __init__(self) -> None:
        self.projected: torch.Tensor | None = None

    def prepare_projected(self, _video: Modality, projected: torch.Tensor, _audio: Modality | None) -> TransformerArgs:
        self.projected = projected
        return _transformer_args(projected)


def _separated_modality(*, gemma_dim: int = 7, video_dim: int = 3) -> Modality:
    reference = torch.randn(1, 2, video_dim)
    semantic = torch.randn(1, 3, gemma_dim)
    target = torch.randn(1, 4, video_dim)
    total = reference.shape[1] + semantic.shape[1] + target.shape[1]
    return Modality(
        latent=target,
        reference_latent=reference,
        semantic_latent=semantic,
        sigma=torch.ones(1),
        timesteps=torch.ones(1, total),
        positions=torch.zeros(1, 3, total, 2),
        context=torch.zeros(1, 1, 5),
        context_mask=torch.ones(1, 1, dtype=torch.bool),
        token_type_ids=torch.tensor([[TYPE_REFERENCE] * 2 + [TYPE_SEMANTIC] * 3 + [TYPE_TARGET] * 4]),
        entity_ids=torch.tensor([[1, 1] + [ENTITY_GLOBAL] * 7]),
    )


def test_branch_specific_projection_precedes_hidden_sequence_concat() -> None:
    model = _bare_vlm_model()
    capture = _ProjectedCapture()
    model.video_args_preprocessor = capture
    modality = _separated_modality()

    model._prepare_video_args(modality, None)

    expected = torch.cat(
        [
            model.patchify_proj(modality.reference_latent),
            model.semantic_input_proj(modality.semantic_latent),
            model.patchify_proj(modality.latent),
        ],
        dim=1,
    )
    assert capture.projected is not None
    torch.testing.assert_close(capture.projected, expected)
    assert capture.projected.shape == (1, 9, model.inner_dim)


def test_separated_forward_returns_gemma_and_native_video_widths() -> None:
    model = _bare_vlm_model()
    modality = _separated_modality()
    model.video_args_preprocessor = _ProjectedCapture()
    model.model_type = SimpleNamespace(
        is_video_enabled=lambda: True,
        is_audio_enabled=lambda: False,
    )
    model.scale_shift_table = nn.Parameter(torch.zeros(2, model.inner_dim))
    model.norm_out = nn.Identity()
    model._process_transformer_blocks = (  # type: ignore[method-assign]
        lambda *, video, audio, perturbations: (video, audio)
    )
    model._process_output = (  # type: ignore[method-assign]
        lambda _table, _norm, projection, hidden, _timestep: projection(hidden)
    )

    output, audio = model(
        video=modality,
        audio=None,
        perturbations=BatchedPerturbationConfig.empty(1),
    )

    assert isinstance(output, SemanticVideoPrediction)
    assert output.semantic.shape == (1, 3, 7)
    assert output.video.shape == (1, 4, 3)
    assert audio is None


def test_semantic_flow_backward_reaches_both_semantic_projections() -> None:
    model = _bare_vlm_model()
    modality = _separated_modality()
    model.video_args_preprocessor = _ProjectedCapture()
    model.model_type = SimpleNamespace(
        is_video_enabled=lambda: True,
        is_audio_enabled=lambda: False,
    )
    model.scale_shift_table = nn.Parameter(torch.zeros(2, model.inner_dim))
    model.norm_out = nn.Identity()
    model._process_transformer_blocks = (  # type: ignore[method-assign]
        lambda *, video, audio, perturbations: (video, audio)
    )
    model._process_output = (  # type: ignore[method-assign]
        lambda _table, _norm, projection, hidden, _timestep: projection(hidden)
    )
    output, _ = model(
        video=modality,
        audio=None,
        perturbations=BatchedPerturbationConfig.empty(1),
    )
    assert isinstance(output, SemanticVideoPrediction)
    (output.semantic.square().mean() + output.video.square().mean()).backward()
    assert model.semantic_input_proj.weight.grad is not None
    assert torch.count_nonzero(model.semantic_input_proj.weight.grad) > 0
    assert model.semantic_output_proj.weight.grad is not None
    assert torch.count_nonzero(model.semantic_output_proj.weight.grad) > 0
    assert model.patchify_proj.weight.grad is not None
    assert torch.count_nonzero(model.patchify_proj.weight.grad) > 0


def test_type_and_slot_embeddings_are_zero_initialized_and_applied_only_to_owned_spans() -> None:
    model = _bare_vlm_model(hidden_dim=5)
    assert torch.count_nonzero(model.reference_type_embedding.weight) == 0
    assert torch.count_nonzero(model.reference_slot_embedding.weight) == 0
    assert torch.count_nonzero(model.semantic_type_embedding.weight) == 0
    with torch.no_grad():
        model.reference_type_embedding.weight.fill_(2.0)
        model.reference_slot_embedding.weight[0].fill_(3.0)
        model.semantic_type_embedding.weight.fill_(5.0)
    before = torch.randn(1, 3, 5)
    modality = Modality(
        latent=torch.zeros(1, 1, 3),
        sigma=torch.ones(1),
        timesteps=torch.ones(1, 3),
        positions=torch.zeros(1, 3, 3, 2),
        context=torch.zeros(1, 1, 5),
        token_type_ids=torch.tensor([[TYPE_REFERENCE, TYPE_SEMANTIC, TYPE_TARGET]]),
        entity_ids=torch.tensor([[1, ENTITY_GLOBAL, ENTITY_GLOBAL]]),
    )
    after = model.apply_video_token_metadata(_transformer_args(before), modality).x
    torch.testing.assert_close(after[:, 0] - before[:, 0], torch.full((1, 5), 5.0))
    torch.testing.assert_close(after[:, 1] - before[:, 1], torch.full((1, 5), 5.0))
    torch.testing.assert_close(after[:, 2], before[:, 2], rtol=0.0, atol=0.0)


def test_reference_rope_is_negative_temporal_and_spatially_outside_target() -> None:
    strategy = _strategy()
    target_latents = torch.zeros(1, 3, 1, 2, 3)
    target_positions = strategy._get_video_positions(
        num_frames=1,
        height=2,
        width=3,
        batch_size=1,
        fps=1.0,
        device=torch.device("cpu"),
    )
    _tokens, reference_positions, _valid, _entities = strategy._reference_sequence(
        {
            "latents": torch.zeros(1, 2, 3, 1, 2, 3),
            "ref_valid_mask": torch.ones(1, 2, dtype=torch.bool),
        },
        target_latents=target_latents,
        target_positions=target_positions,
    )
    assert reference_positions[:, 0, :, 1].amax() <= 0
    assert reference_positions[:, 0, :, 0].amin() < 0
    assert reference_positions[:, 1, :, 0].amin() >= target_positions[:, 1, :, 1].amax()
    assert reference_positions[:, 2, :, 0].amin() >= target_positions[:, 2, :, 1].amax()


def test_semantic_rope_covers_target_spatial_domain_and_anchor_times() -> None:
    strategy = _strategy()
    target = strategy._get_video_positions(
        num_frames=3,
        height=2,
        width=3,
        batch_size=1,
        fps=1.0,
        device=torch.device("cpu"),
    )
    positions, _bounds = strategy._semantic_positions(target, torch.tensor([[0.0, 0.5, 1.0]]))
    assert positions.shape == (1, 3, 3 * EVIDENCE_TOKENS_PER_FRAME, 2)
    for axis in (1, 2):
        torch.testing.assert_close(positions[:, axis, :, 0].amin(), target[:, axis, :, 0].amin())
        torch.testing.assert_close(positions[:, axis, :, 1].amax(), target[:, axis, :, 1].amax())
    first = positions[:, 0, :EVIDENCE_TOKENS_PER_FRAME]
    middle = positions[:, 0, EVIDENCE_TOKENS_PER_FRAME : 2 * EVIDENCE_TOKENS_PER_FRAME]
    last = positions[:, 0, 2 * EVIDENCE_TOKENS_PER_FRAME :]
    assert first[..., 0].unique().item() == pytest.approx(target[:, 0, :, 0].amin().item())
    assert middle[..., 0].unique().item() == pytest.approx(
        ((target[:, 0, :, 0].amin() + target[:, 0, :, 1].amax()) / 2).item()
    )
    assert last[..., 0].unique().item() == pytest.approx(target[:, 0, :, 1].amax().item())


def test_loss_is_independent_branch_mean_then_weighted_sum() -> None:
    strategy = SemanticVLMFlowStrategy(
        SemanticVLMFlowConfig(video_flow_weight=2.0, semantic_flow_weight=3.0)
    )
    video_target = torch.zeros(1, 2, 3)
    semantic_target = torch.zeros(1, 4, 7)
    inputs = ModelInputs(
        video=None,
        audio=None,
        video_targets=video_target,
        audio_targets=None,
        video_loss_mask=torch.ones(1, 2, dtype=torch.bool),
        audio_loss_mask=None,
        semantic_targets=semantic_target,
        semantic_loss_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    prediction = SemanticVideoPrediction(
        semantic=torch.full_like(semantic_target, 2.0),
        video=torch.full_like(video_target, 1.0),
    )
    loss = strategy.compute_loss(prediction, None, inputs)
    torch.testing.assert_close(loss, torch.tensor([2.0 * 1.0 + 3.0 * 4.0]))
    assert set(strategy.get_last_training_metrics()) == {
        "train/loss_video_flow",
        "train/loss_semantic_flow",
        "train/loss_video_flow_weighted",
        "train/loss_semantic_flow_weighted",
    }


class _ConstantVelocityTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[torch.Tensor] = []

    def forward(self, *, video: Modality, audio: None, perturbations: object) -> tuple[SemanticVideoPrediction, None]:
        del audio, perturbations
        assert video.reference_latent is not None and video.semantic_latent is not None
        self.references.append(video.reference_latent.clone())
        return (
            SemanticVideoPrediction(
                semantic=torch.full_like(video.semantic_latent, 2.0),
                video=torch.full_like(video.latent, 3.0),
            ),
            None,
        )


def test_inference_integrates_separate_states_and_never_updates_reference() -> None:
    strategy = _strategy(gemma_dim=7, video_dim=3)
    state = strategy.prepare_inference_state(
        conditions={
            "video_prompt_embeds": torch.zeros(1, 2, 5),
            "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
        },
        reference_latents={
            "latents": torch.randn(1, 1, 3, 1, 1, 1),
            "ref_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        },
        target_shape=VideoLatentShape(batch=1, channels=3, frames=1, height=1, width=1),
        semantic_frame_count=1,
        pixel_frame_count=1,
        fps=1.0,
        seed=9,
    )
    semantic_initial = state.modality.semantic_latent.clone()
    video_initial = state.modality.latent.clone()
    reference_initial = state.modality.reference_latent.clone()
    transformer = _ConstantVelocityTransformer()
    semantic_final, video_final = strategy.denoise_joint(
        transformer=transformer,
        state=state,
        num_inference_steps=2,
    )
    torch.testing.assert_close(semantic_final, semantic_initial - 2.0)
    torch.testing.assert_close(strategy._video_patchifier.patchify(video_final), video_initial - 3.0)
    assert len(transformer.references) == 2
    for reference in transformer.references:
        torch.testing.assert_close(reference, reference_initial, rtol=0.0, atol=0.0)


def test_guidance_rescale_uses_video_statistics_for_both_branches() -> None:
    positive_semantic = torch.tensor([[[1.0, 3.0, 5.0]]])
    guided_semantic = positive_semantic * 4.0
    positive_video = torch.tensor([[[1.0, 2.0, 3.0]]])
    guided_video = positive_video * 2.0
    semantic, video, factor = rescale_guided_denoised_branches(
        positive_semantic=positive_semantic,
        guided_semantic=guided_semantic,
        positive_video=positive_video,
        guided_video=guided_video,
        guidance_rescale=1.0,
    )
    torch.testing.assert_close(factor, torch.tensor([0.5]))
    torch.testing.assert_close(video, positive_video)
    torch.testing.assert_close(semantic, guided_semantic * 0.5)


def test_checkpoint_metadata_and_required_prefixes(tmp_path: Path) -> None:
    strategy = _strategy()
    metadata = strategy.get_checkpoint_metadata()
    assert metadata["architecture"] == "semantic_vlm_joint_flow_v1"
    assert metadata["semantic_state_space"] == "frozen_gemma_last_hidden_state"
    assert metadata["semantic_noise_space"] == "frozen_gemma_hidden"
    assert metadata["reference_rope_spatial_shift"] == "height_width_adjacent"
    state = {
        f"{prefix}weight": torch.ones(1)
        for prefix in (
            *SEMANTIC_VLM_STRATEGY_CHECKPOINT_PREFIXES,
            *SEMANTIC_VLM_TRANSFORMER_CHECKPOINT_PREFIXES,
        )
    }
    path = tmp_path / "model_weights_step_00001.safetensors"
    save_file(state, path, metadata={key: str(value) for key, value in metadata.items()})
    audit = audit_checkpoint(path)
    assert audit["metadata"]["architecture"] == "semantic_vlm_joint_flow_v1"


def test_legacy_checkpoint_is_explicitly_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model_weights_step_00001.safetensors"
    save_file({"weight": torch.ones(1)}, path, metadata={"architecture": "semantic_repae_v1"})
    with pytest.raises(CheckpointAuditError, match="cannot initialize semantic_vlm_joint_flow_v1"):
        audit_checkpoint(path)


def test_production_config_contains_only_joint_flow_weights() -> None:
    path = Path(__file__).parents[1] / "configs" / "semantic_vlm_flow_multitask_480p121_full.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    strategy = payload["training_strategy"]
    optimization = payload["optimization"]
    assert strategy["name"] == "semantic_vlm_flow"
    assert strategy["reference_rope_mode"] == "negative_adjacent_shifted_hw"
    assert strategy["video_flow_weight"] == pytest.approx(1.0)
    assert strategy["semantic_flow_weight"] == pytest.approx(1.0)
    assert optimization["semantic_learning_rate"] > 0
    assert set(strategy) == {
        "name",
        "max_ref_images_per_sample",
        "reference_rope_mode",
        "required_fsdp_world_size",
        "semantic_anchor_count",
        "semantic_grid_size",
        "semantic_tokens_per_frame",
        "vlm_prefix_max_length",
        "vlm_teacher_max_length",
        "video_flow_weight",
        "semantic_flow_weight",
        "condition_probabilities",
    }


class _TinyTrainableTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(5, 5)
        self.semantic_input_proj = nn.Linear(7, 5)
        self.semantic_output_proj = nn.Linear(5, 7)
        self.semantic_norm_out = nn.RMSNorm(5)
        self.reference_type_embedding = nn.Embedding(1, 5)
        self.reference_slot_embedding = nn.Embedding(4, 5)
        self.semantic_type_embedding = nn.Embedding(1, 5)


def test_optimizer_groups_cover_new_modules_without_a_projector_group() -> None:
    trainer = LtxvTrainer.__new__(LtxvTrainer)
    trainer._transformer = nn.Module()
    trainer._transformer._fsdp_wrapped_module = _TinyTrainableTransformer()
    bridge_projection = nn.Linear(5, 5)
    bridge_connector = nn.Linear(5, 5)
    trainer._embeddings_processor_trainable_modules = {
        "feature_extractor.video_aggregate_embed": bridge_projection,
        "video_connector": bridge_connector,
    }
    trainer._training_strategy = SimpleNamespace(get_trainable_modules=lambda: {})
    trainer._trainable_params = trainer._deduplicate_parameters(
        [
            *trainer._transformer.parameters(),
            *bridge_projection.parameters(),
            *bridge_connector.parameters(),
        ]
    )
    groups = trainer._semantic_vlm_flow_trainable_parameter_groups({})
    assert set(groups) == {"dit", "conditioning_bridge", "semantic_flow"}
    assert {id(parameter) for parameter in groups["semantic_flow"]} == {
        id(parameter)
        for name, parameter in trainer._transformer.named_parameters()
        if name.removeprefix("_fsdp_wrapped_module.").startswith(
            (
                "semantic_input_proj.",
                "semantic_output_proj.",
                "semantic_norm_out.",
                "reference_type_embedding.",
                "reference_slot_embedding.",
                "semantic_type_embedding.",
            )
        )
    }


def _full_shard_accelerator(num_processes: int) -> SimpleNamespace:
    plugin = SimpleNamespace(
        fsdp_version=1,
        sharding_strategy=SimpleNamespace(name="FULL_SHARD"),
        state_dict_type=SimpleNamespace(name="FULL_STATE_DICT"),
    )
    return SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        num_processes=num_processes,
        state=SimpleNamespace(fsdp_plugin=plugin),
    )


def test_semantic_vlm_flow_requires_exactly_eight_full_shard_processes() -> None:
    config = SimpleNamespace(
        training_strategy=SemanticVLMFlowConfig(),
        model=SimpleNamespace(training_mode="full"),
    )
    _enforce_semantic_flow_fsdp_runtime_safety(config, _full_shard_accelerator(8))
    with pytest.raises(RuntimeError, match="requires exactly 8 FSDP processes"):
        _enforce_semantic_flow_fsdp_runtime_safety(config, _full_shard_accelerator(7))
