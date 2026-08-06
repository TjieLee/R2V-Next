from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from accelerate import DistributedType
from safetensors.torch import load_file, save_file
from torch import nn

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.transformer_args import TransformerArgs, TransformerArgsPreprocessor
from ltx_core.multicond.semantic_tokens import (
    EVIDENCE_TOKENS_PER_FRAME,
    SemanticInputProjection,
    SemanticRepaProjector,
    semantic_repa_loss,
)
from ltx_core.types import VideoLatentShape
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.online_data import path_safety
from ltx_trainer.online_data.adapters.base import CanonicalR2VSource
from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_data.manifest import prepare_canonical_r2v_record
from ltx_trainer.online_data.online_batch_encoder import (
    _build_messages,
    load_semantic_system_prompts,
)
from ltx_trainer.online_inference.checkpoint_runtime import (
    CheckpointAuditError,
    REPAE_STRATEGY_CHECKPOINT_PREFIXES,
    REPAE_TRANSFORMER_CHECKPOINT_PREFIXES,
    audit_checkpoint,
    read_checkpoint_metadata,
)
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.training_strategies.semantic_flow import (
    TYPE_REFERENCE,
    TYPE_SEMANTIC,
    TYPE_TARGET,
    SemanticFlowStrategy,
    build_compact_valid_token_mask,
)
from ltx_trainer.training_strategies.semantic_flow_bridge import (
    configure_phase2_bridge_trainability,
    phase2_bridge_parameters,
    validate_and_load_phase2_bridge_state,
)
from ltx_trainer.training_strategies.semantic_repae import SemanticRepaEConfig, SemanticRepaEStrategy
from ltx_trainer.trainer import LtxvTrainer, _enforce_semantic_flow_fsdp_runtime_safety


def _transformer_args(x: torch.Tensor) -> TransformerArgs:
    batch_size, token_count, hidden_dim = x.shape
    return TransformerArgs(
        x=x,
        context=torch.zeros(batch_size, 1, hidden_dim),
        context_mask=torch.ones(batch_size, 1),
        timesteps=torch.zeros(batch_size, token_count, hidden_dim),
        embedded_timestep=torch.zeros(batch_size, token_count, hidden_dim),
        positional_embeddings=(torch.zeros(1), torch.zeros(1)),
        cross_positional_embeddings=None,
        cross_scale_shift_timestep=None,
        cross_gate_timestep=None,
        enabled=True,
    )


def _modality(token_type_ids: torch.Tensor, entity_ids: torch.Tensor, hidden_dim: int) -> Modality:
    batch_size, token_count = token_type_ids.shape
    return Modality(
        latent=torch.zeros(batch_size, token_count, hidden_dim),
        sigma=torch.ones(batch_size),
        timesteps=torch.ones(batch_size, token_count),
        positions=torch.zeros(batch_size, 3, token_count, 2),
        context=torch.zeros(batch_size, 1, hidden_dim),
        token_type_ids=token_type_ids,
        entity_ids=entity_ids,
    )


def _bare_metadata_model(hidden_dim: int = 3) -> LTXModel:
    model = LTXModel.__new__(LTXModel)
    nn.Module.__init__(model)
    model.semantic_repae_reference_type_embedding = nn.Embedding(1, hidden_dim)
    model.semantic_repae_reference_slot_embedding = nn.Embedding(4, hidden_dim)
    model.semantic_repae_semantic_type_embedding = nn.Embedding(1, hidden_dim)
    model.semantic_token_type_id = TYPE_SEMANTIC
    model.reference_token_type_id = TYPE_REFERENCE
    with torch.no_grad():
        model.semantic_repae_reference_type_embedding.weight.fill_(10.0)
        model.semantic_repae_reference_slot_embedding.weight.copy_(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [2.0, 4.0, 6.0]])
        )
        model.semantic_repae_semantic_type_embedding.weight.fill_(20.0)
    return model


def _initializable_metadata_model(hidden_dim: int = 3, semantic_dim: int = 4) -> LTXModel:
    model = LTXModel.__new__(LTXModel)
    nn.Module.__init__(model)
    model.inner_dim = hidden_dim
    model.patchify_proj = nn.Linear(semantic_dim, hidden_dim)
    model.semantic_token_type_embedding = None
    model.semantic_repae_reference_type_embedding = None
    model.semantic_repae_reference_slot_embedding = None
    model.semantic_repae_semantic_type_embedding = None
    model.semantic_norm_out = None
    model.semantic_proj_out = None
    model.semantic_token_type_id = None
    model.reference_token_type_id = None
    return model


def test_repae_new_metadata_and_velocity_head_are_zero_initialized() -> None:
    model = _initializable_metadata_model()
    model.enable_semantic_repae_conditioning(
        semantic_dim=4,
        num_reference_slots=4,
        semantic_token_type_id=TYPE_SEMANTIC,
        reference_token_type_id=TYPE_REFERENCE,
    )

    assert torch.count_nonzero(model.semantic_repae_reference_type_embedding.weight).item() == 0
    assert torch.count_nonzero(model.semantic_repae_reference_slot_embedding.weight).item() == 0
    assert torch.count_nonzero(model.semantic_repae_semantic_type_embedding.weight).item() == 0
    assert torch.count_nonzero(model.semantic_proj_out.weight).item() == 0
    assert torch.count_nonzero(model.semantic_proj_out.bias).item() == 0

    before = torch.randn(1, 3, 3)
    token_types = torch.tensor([[TYPE_REFERENCE, TYPE_SEMANTIC, TYPE_TARGET]])
    after = model.apply_video_token_metadata(
        _transformer_args(before),
        _modality(token_types, torch.tensor([[1, 0, 0]]), hidden_dim=3),
    ).x
    torch.testing.assert_close(after[:, 2], before[:, 2], rtol=0.0, atol=0.0)


def test_repae_metadata_changes_reference_and_semantic_but_not_target() -> None:
    model = _bare_metadata_model()
    token_types = torch.tensor([[TYPE_REFERENCE, TYPE_REFERENCE, TYPE_SEMANTIC, TYPE_TARGET]])
    entity_ids = torch.tensor([[1, 2, 0, 0]])
    before = torch.randn(1, 4, 3)
    after = model.apply_video_token_metadata(
        _transformer_args(before),
        _modality(token_types, entity_ids, hidden_dim=3),
    ).x

    torch.testing.assert_close(after[:, 3], before[:, 3], rtol=0.0, atol=0.0)
    torch.testing.assert_close(after[:, 2] - before[:, 2], torch.full((1, 3), 20.0))
    torch.testing.assert_close(
        after[:, 0] - before[:, 0],
        torch.tensor([[11.0, 12.0, 13.0]]),
    )
    torch.testing.assert_close(
        after[:, 1] - before[:, 1],
        torch.tensor([[14.0, 15.0, 16.0]]),
    )


class _AddBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, video: TransformerArgs, audio: None) -> tuple[TransformerArgs, None]:
        del audio
        return replace(video, x=video.x + self.projection(video.x)), None


class _IdentityBlockInputProcessor:
    def __call__(self, args: TransformerArgs, *_args, **_kwargs) -> TransformerArgs:
        return args


def test_repae_capture_keeps_only_requested_semantic_span_and_gradients() -> None:
    model = LTXModel.__new__(LTXModel)
    nn.Module.__init__(model)
    model.transformer_blocks = nn.ModuleList([_AddBlock(4), _AddBlock(4)])
    model.block_input_processor = _IdentityBlockInputProcessor()
    model._enable_gradient_checkpointing = False
    model._semantic_repae_capture_request = None
    model._semantic_repae_captured_hidden = None
    model.configure_semantic_repae_capture(block_index=0, semantic_start=2, semantic_end=5)
    inputs = _transformer_args(torch.randn(1, 7, 4, requires_grad=True))
    output, _ = model._process_transformer_blocks(
        inputs,
        None,
        BatchedPerturbationConfig.empty(1),
    )
    captured = model.consume_semantic_repae_capture()

    assert captured.shape == (1, 3, 4)
    assert output is not None
    (captured.square().mean() + output.x.square().mean()).backward()
    assert all(block.projection.weight.grad is not None for block in model.transformer_blocks)


def test_repae_capture_survives_non_reentrant_gradient_checkpointing() -> None:
    model = LTXModel.__new__(LTXModel)
    nn.Module.__init__(model)
    model.transformer_blocks = nn.ModuleList([_AddBlock(4), _AddBlock(4), _AddBlock(4)])
    model.block_input_processor = _IdentityBlockInputProcessor()
    model._enable_gradient_checkpointing = True
    model._semantic_repae_capture_request = None
    model._semantic_repae_captured_hidden = None
    model.configure_semantic_repae_capture(block_index=1, semantic_start=2, semantic_end=5)
    inputs = _transformer_args(torch.randn(1, 7, 4, requires_grad=True))

    with torch.set_grad_enabled(True):
        model._process_transformer_blocks(
            inputs,
            None,
            BatchedPerturbationConfig.empty(1),
        )
        captured = model.consume_semantic_repae_capture()
        assert captured.shape == (1, 3, 4)
        assert captured.requires_grad
        captured.square().mean().backward()

    assert model.transformer_blocks[0].projection.weight.grad is not None
    assert model.transformer_blocks[1].projection.weight.grad is not None
    assert model._semantic_repae_capture_request is None
    assert model._semantic_repae_captured_hidden is None

    model._process_transformer_blocks(
        _transformer_args(torch.randn(1, 7, 4, requires_grad=True)),
        None,
        BatchedPerturbationConfig.empty(1),
    )
    with pytest.raises(RuntimeError, match="capture is missing"):
        model.consume_semantic_repae_capture()


def test_repae_reference_rope_and_worst_case_sequence_geometry() -> None:
    strategy = SemanticRepaEStrategy(SemanticRepaEConfig())
    target_latents = torch.zeros(1, 128, 16, 15, 26)
    target_positions = strategy._get_video_positions(
        num_frames=16,
        height=15,
        width=26,
        batch_size=1,
        fps=24.0,
        device=torch.device("cpu"),
    )
    reference_latents = {
        "latents": torch.zeros(1, 4, 128, 1, 15, 26),
        "ref_valid_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    ref_tokens, ref_positions, ref_valid, _entities = strategy._reference_sequence(
        reference_latents,
        target_latents=target_latents,
        target_positions=target_positions,
    )
    semantic_positions, _bounds = strategy._semantic_positions(
        target_positions,
        torch.tensor([[0.0, 17 / 120, 34 / 120, 51 / 120, 69 / 120, 86 / 120, 103 / 120, 1.0]]),
    )

    assert ref_tokens.shape[1] == 4 * 390
    assert ref_valid.all()
    torch.testing.assert_close(
        ref_positions[:, 0, :, 0],
        torch.full_like(ref_positions[:, 0, :, 0], -1.0 / 24.0),
    )
    assert torch.equal(ref_positions[:, 0, :, 1], torch.zeros_like(ref_positions[:, 0, :, 1]))
    assert ref_positions[:, 1, :, 0].amin() >= target_positions[:, 1, :, 1].amax()
    assert ref_positions[:, 2, :, 0].amin() >= target_positions[:, 2, :, 1].amax()
    per_reference = ref_tokens.shape[1] // 4
    for reference_index in range(1, 4):
        torch.testing.assert_close(
            ref_positions[:, :, :per_reference],
            ref_positions[:, :, reference_index * per_reference : (reference_index + 1) * per_reference],
        )
    assert semantic_positions.shape[2] == 8 * 256
    torch.testing.assert_close(
        semantic_positions[:, 0, :, 0],
        semantic_positions[:, 0, :, 1],
    )
    assert target_positions.shape[2] == 6240
    assert ref_tokens.shape[1] + semantic_positions.shape[2] + target_positions.shape[2] == 9848
    assert strategy.semantic_frame_count_for_task(task="i2i", pixel_frame_count=1) * 256 == 256

    image_target = torch.zeros(1, 128, 1, 15, 26)
    image_positions = strategy._get_video_positions(
        num_frames=1,
        height=15,
        width=26,
        batch_size=1,
        fps=1.0,
        device=torch.device("cpu"),
    )
    _tokens, image_ref_positions, _valid, _entities = strategy._reference_sequence(
        reference_latents,
        target_latents=image_target,
        target_positions=image_positions,
    )
    torch.testing.assert_close(
        image_ref_positions[:, 0, :, 0],
        torch.full_like(image_ref_positions[:, 0, :, 0], -1.0),
    )
    assert torch.equal(
        image_ref_positions[:, 0, :, 1],
        torch.zeros_like(image_ref_positions[:, 0, :, 1]),
    )

    mixed_target_positions = target_positions.expand(2, -1, -1, -1).clone()
    mixed_target_positions[1, 0] *= 24.0
    _tokens, mixed_ref_positions, _valid, _entities = strategy._reference_sequence(
        {
            "latents": reference_latents["latents"].expand(2, -1, -1, -1, -1, -1).clone(),
            "ref_valid_mask": torch.ones(2, 4, dtype=torch.bool),
        },
        target_latents=target_latents.expand(2, -1, -1, -1, -1).clone(),
        target_positions=mixed_target_positions,
    )
    torch.testing.assert_close(
        mixed_ref_positions[0, 0, :, 0],
        torch.full_like(mixed_ref_positions[0, 0, :, 0], -1.0 / 24.0),
    )
    torch.testing.assert_close(
        mixed_ref_positions[1, 0, :, 0],
        torch.full_like(mixed_ref_positions[1, 0, :, 0], -1.0),
    )


def test_repae_inference_uses_training_sequence_offsets_positions_and_types() -> None:
    strategy = SemanticRepaEStrategy(SemanticRepaEConfig())
    strategy._semantic_dim = 128
    reference_latents = {
        "latents": torch.zeros(1, 4, 128, 1, 15, 26),
        "ref_valid_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    state = strategy.prepare_inference_state(
        conditions={
            "video_prompt_embeds": torch.zeros(1, 3, 32),
            "prompt_attention_mask": torch.ones(1, 3),
        },
        reference_latents=reference_latents,
        target_shape=VideoLatentShape(batch=1, channels=128, frames=16, height=15, width=26),
        semantic_frame_count=8,
        pixel_frame_count=121,
        fps=24.0,
        seed=7,
    )

    assert state.sequence_offsets == {
        "reference_end": 1560,
        "semantic_end": 3608,
        "target_end": 9848,
    }
    token_types = state.modality.token_type_ids
    assert token_types is not None
    assert torch.equal(token_types[:, :1560], torch.full((1, 1560), TYPE_REFERENCE))
    assert torch.equal(token_types[:, 1560:3608], torch.full((1, 2048), TYPE_SEMANTIC))
    assert torch.equal(token_types[:, 3608:], torch.full((1, 6240), TYPE_TARGET))
    assert state.modality.positions.shape == (1, 3, 9848, 2)
    assert state.modality.attention_mask is None


def test_repae_compact_attention_mask_masks_only_invalid_reference_keys() -> None:
    assert build_compact_valid_token_mask(torch.ones(2, 5, dtype=torch.bool)) is None
    valid = torch.tensor([[True, False, True], [True, True, False]])
    compact = build_compact_valid_token_mask(valid)
    assert compact is not None
    assert compact.shape == (2, 1, 3)
    assert torch.equal(compact[:, 0], valid)
    additive = TransformerArgsPreprocessor._prepare_self_attention_mask(
        object(),
        compact,
        torch.float32,
    )
    assert additive.shape == (2, 1, 1, 3)
    assert torch.equal(additive[:, 0, 0] == 0, valid)

    strategy = SemanticRepaEStrategy(SemanticRepaEConfig())
    strategy._semantic_dim = 128
    state = strategy.prepare_inference_state(
        conditions={
            "video_prompt_embeds": torch.zeros(1, 3, 32),
            "prompt_attention_mask": torch.ones(1, 3),
        },
        reference_latents={
            "latents": torch.zeros(1, 4, 128, 1, 15, 26),
            "ref_valid_mask": torch.tensor([[True, False, True, False]]),
        },
        target_shape=VideoLatentShape(batch=1, channels=128, frames=16, height=15, width=26),
        semantic_frame_count=8,
        pixel_frame_count=121,
        fps=24.0,
        seed=7,
    )
    mask = state.modality.attention_mask
    assert mask is not None
    assert mask.shape == (1, 1, state.sequence_offsets["target_end"])
    tokens_per_reference = state.sequence_offsets["reference_end"] // 4
    assert mask[:, :, :tokens_per_reference].all()
    assert not mask[:, :, tokens_per_reference : 2 * tokens_per_reference].any()
    assert mask[:, :, 2 * tokens_per_reference : 3 * tokens_per_reference].all()
    assert not mask[:, :, 3 * tokens_per_reference : 4 * tokens_per_reference].any()
    assert mask[:, :, state.sequence_offsets["reference_end"] :].all()

    training_source = inspect.getsource(SemanticRepaEStrategy.prepare_training_inputs)
    inference_source = inspect.getsource(SemanticFlowStrategy.prepare_inference_state)
    assert "build_compact_valid_token_mask" in training_source
    assert "build_compact_valid_token_mask" in inference_source
    assert "[:, :, None]" not in training_source
    assert "[:, :, None]" not in inference_source


def test_repae_teacher_projection_and_both_projectors_receive_gradients() -> None:
    teacher_parameter = nn.Parameter(torch.ones(1), requires_grad=False)
    teacher_hidden = torch.randn(2, 8, EVIDENCE_TOKENS_PER_FRAME, 12) * teacher_parameter
    semantic_projection = SemanticInputProjection(12, 8, hidden_dim=16)
    semantic_repa = SemanticRepaProjector(8, 12, hidden_dim=16)
    dit_repa = SemanticRepaProjector(10, 12, hidden_dim=16)
    semantic_clean = semantic_projection(teacher_hidden)
    dit_hidden = torch.randn(2, 8, EVIDENCE_TOKENS_PER_FRAME, 10, requires_grad=True)
    loss = semantic_repa_loss(semantic_repa(semantic_clean), teacher_hidden.detach()).mean()
    loss = loss + semantic_repa_loss(dit_repa(dit_hidden), teacher_hidden.detach()).mean()
    loss.backward()

    assert not teacher_hidden.requires_grad
    assert teacher_parameter.grad is None
    assert any(parameter.grad is not None for parameter in semantic_projection.parameters())
    assert any(parameter.grad is not None for parameter in semantic_repa.parameters())
    assert any(parameter.grad is not None for parameter in dit_repa.parameters())
    assert dit_hidden.grad is not None


class _TinyGemmaLanguageModel(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, hidden_dim)
        self.scale = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(sliding_window=1024)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        output_hidden_states: bool,
        **_kwargs,
    ) -> SimpleNamespace:
        assert output_hidden_states is False
        return SimpleNamespace(last_hidden_state=inputs_embeds * self.scale)


class _TinyTextEncoder(nn.Module):
    def __init__(self, language_model: nn.Module) -> None:
        super().__init__()
        body = nn.Module()
        body.language_model = language_model
        wrapper = nn.Module()
        wrapper.model = body
        self.model = wrapper


class _TinyRepaETransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patchify_proj = nn.Linear(128, 16)
        self.proj_out = nn.Linear(16, 128)
        self.inner_dim = 16
        self.transformer_blocks = nn.ModuleList([nn.Identity() for _ in range(16)])
        self.enabled = False

    def enable_semantic_repae_conditioning(self, **_kwargs) -> None:
        self.enabled = True


def test_repae_strategy_runs_gemma_teacher_under_no_grad_without_checkpoint_wrapping() -> None:
    language_model = _TinyGemmaLanguageModel(hidden_dim=12)
    text_encoder = _TinyTextEncoder(language_model)
    transformer = _TinyRepaETransformer()
    strategy = SemanticRepaEStrategy(
        SemanticRepaEConfig(semantic_hidden_dim=16, repa_hidden_dim=16)
    )
    strategy.attach_models(
        transformer=transformer,
        embeddings_processor=nn.Identity(),
        text_encoder=text_encoder,
    )
    teacher = strategy.build_semantic_teacher_outputs(
        {
            "prefix_inputs_embeds": torch.randn(1, 2, 12),
            "prefix_attention_mask": torch.ones(1, 2, dtype=torch.bool),
            "prefix_image_token_mask": torch.zeros(1, 2, dtype=torch.bool),
            "evidence_tokens": torch.randn(1, 1, EVIDENCE_TOKENS_PER_FRAME, 12),
        }
    )
    loss = teacher["semantic_projection_repa_prediction"].square().mean()
    loss.backward()

    assert transformer.enabled
    assert not teacher["teacher_hidden"].requires_grad
    assert teacher["semantic_clean"].requires_grad
    assert all(parameter.grad is None for parameter in language_model.parameters())
    assert any(
        parameter.grad is not None
        for parameter in strategy.get_trainable_modules()["semantic_input_projection"].parameters()
    )
    assert not hasattr(language_model, "_semantic_flow_non_reentrant_checkpoint")


def test_repae_manifest_explicit_anchor_count_preserves_ratio_fallback() -> None:
    canonical = CanonicalR2VSource(
        dataset_name="fixture",
        adapter_name="opens2v",
        source_record_id="row-0",
        video_path="/read-only/video.mp4",
        caption="caption",
        reference_paths=["/read-only/ref.jpg"],
        crop_xyxy=None,
        clip_start_frame=0,
        clip_end_frame=121,
        metadata={},
    )
    header = {"fps": 24.0, "frame_count": 121, "width": 832, "height": 480}
    explicit = prepare_canonical_r2v_record(
        canonical,
        manifest_seed=42,
        semantic_anchor_count=8,
        video_header=header,
        target_path_validated=True,
    )
    legacy = prepare_canonical_r2v_record(
        canonical,
        manifest_seed=42,
        anchor_frame_ratio=0.10,
        video_header=header,
        target_path_validated=True,
    )

    assert explicit.record["semantic_anchor_target_indices"] == [0, 17, 34, 51, 69, 86, 103, 120]
    assert len(legacy.record["semantic_anchor_target_indices"]) == 12


def test_repae_strategy_is_independent_and_registered() -> None:
    config = SemanticRepaEConfig()
    strategy = get_training_strategy(config)
    assert isinstance(strategy, SemanticRepaEStrategy)
    assert not hasattr(config, "training_phase")
    assert not hasattr(config, "parent_checkpoint_step")
    assert not hasattr(config, "target_fps")
    assert config.required_fsdp_world_size == 8
    metadata = strategy.get_checkpoint_metadata()
    assert metadata["architecture"] == "semantic_repae_v1"
    assert metadata["required_fsdp_world_size"] == 8
    with pytest.raises(ValueError, match="required_fsdp_world_size"):
        SemanticRepaEConfig(required_fsdp_world_size=4)  # type: ignore[arg-type]


def test_repae_full_dit_requires_fsdp_full_shard_runtime() -> None:
    config = SimpleNamespace(
        training_strategy=SemanticRepaEConfig(),
        model=SimpleNamespace(training_mode="full"),
    )
    accelerator = SimpleNamespace(distributed_type=DistributedType.NO)
    with pytest.raises(RuntimeError, match="semantic REPA-E.*FSDP FULL_SHARD"):
        _enforce_semantic_flow_fsdp_runtime_safety(config, accelerator)


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


@pytest.mark.parametrize("num_processes", [1, 2, 4, 7, 16])
def test_repae_full_dit_rejects_any_non_eight_process_world_size(
    num_processes: int,
) -> None:
    config = SimpleNamespace(
        training_strategy=SemanticRepaEConfig(),
        model=SimpleNamespace(training_mode="full"),
    )
    with pytest.raises(
        RuntimeError,
        match=rf"requires exactly 8 FSDP processes; configured num_processes={num_processes}",
    ):
        _enforce_semantic_flow_fsdp_runtime_safety(
            config,
            _full_shard_accelerator(num_processes),
        )


def test_repae_full_dit_accepts_eight_processes_without_restricting_semantic_flow() -> None:
    repae_config = SimpleNamespace(
        training_strategy=SemanticRepaEConfig(),
        model=SimpleNamespace(training_mode="full"),
    )
    _enforce_semantic_flow_fsdp_runtime_safety(
        repae_config,
        _full_shard_accelerator(8),
    )

    semantic_flow_config = SimpleNamespace(
        training_strategy=SimpleNamespace(name="semantic_flow"),
        model=SimpleNamespace(training_mode="full"),
    )
    _enforce_semantic_flow_fsdp_runtime_safety(
        semantic_flow_config,
        _full_shard_accelerator(4),
    )


class _TinyCheckpointTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.semantic_repae_reference_type_embedding = nn.Embedding(1, 4)
        self.semantic_repae_reference_slot_embedding = nn.Embedding(4, 4)
        self.semantic_repae_semantic_type_embedding = nn.Embedding(1, 4)
        self.semantic_norm_out = nn.RMSNorm(4, elementwise_affine=True)
        self.semantic_proj_out = nn.Linear(4, 4)


class _TinyFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_aggregate_embed = nn.Linear(4, 4)
        self.frozen_projection = nn.Linear(4, 4)


class _TinyEmbeddingsProcessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = _TinyFeatureExtractor()
        self.video_connector = nn.Linear(4, 4)
        self.audio_connector = nn.Linear(4, 4)


def _tiny_checkpoint_strategy() -> SemanticRepaEStrategy:
    strategy = SemanticRepaEStrategy(
        SemanticRepaEConfig(semantic_hidden_dim=8, repa_hidden_dim=8)
    )
    strategy._semantic_dim = 4
    strategy._gemma_dim = 6
    strategy._dit_hidden_dim = 4
    strategy._semantic_input_projection = SemanticInputProjection(6, 4, hidden_dim=8)
    strategy._semantic_repa_projector = SemanticRepaProjector(4, 6, hidden_dim=8)
    strategy._dit_repa_projector = SemanticRepaProjector(4, 6, hidden_dim=8)
    return strategy


def _semantic_repae_checkpoint_fixture() -> tuple[
    dict[str, torch.Tensor],
    dict[str, str],
    SemanticRepaEStrategy,
    _TinyCheckpointTransformer,
    _TinyEmbeddingsProcessor,
]:
    strategy = _tiny_checkpoint_strategy()
    transformer = _TinyCheckpointTransformer()
    processor = _TinyEmbeddingsProcessor()
    state = {key: value.detach().clone() for key, value in transformer.state_dict().items()}
    accelerator = SimpleNamespace(get_state_dict=lambda module: module.state_dict())
    state.update(strategy.get_extra_checkpoint_state_dict(accelerator))
    for item in phase2_bridge_parameters(processor):
        state[f"embeddings_processor.{item.name}"] = item.parameter.detach().clone()
    metadata = {key: str(value) for key, value in strategy.get_checkpoint_metadata().items()}
    return state, metadata, strategy, transformer, processor


def test_repae_production_config_parses_with_correct_worktree_and_no_augmentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path(__file__).parents[1] / "configs" / "semantic_repae_multitask_480p121_full.yaml"
    raw_text = config_path.read_text(encoding="utf-8")
    stale_worktree = "R2V-Next-semantic-repae-" + "codex"
    assert stale_worktree not in raw_text
    assert "/mnt/workspace/litengjie/R2V-Next-semantic-repae/" in raw_text
    assert "training_phase" not in raw_text
    assert "parent_checkpoint_step" not in raw_text
    source_data_config = yaml.safe_load(
        (config_path.parent / "multitask_online_480p121_opens2v_noaug.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert source_data_config["online_augmentation"]["enabled"] is False
    source_sampling = source_data_config["online_sampling"]
    assert source_sampling["image_ratio"] == pytest.approx(0.15)
    assert source_sampling["video_ratio"] == pytest.approx(0.85)
    assert source_sampling["image_ratio"] + source_sampling["video_ratio"] == pytest.approx(1.0)

    payload = yaml.safe_load(raw_text)
    data_config = tmp_path / "multitask.yaml"
    manifest = tmp_path / "manifest.jsonl"
    model = tmp_path / "model.safetensors"
    text_encoder = tmp_path / "text_encoder"
    data_config.write_text("online_augmentation:\n  enabled: false\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    model.write_bytes(b"fixture")
    text_encoder.mkdir()
    payload["model"]["model_path"] = str(model)
    payload["model"]["text_encoder_path"] = str(text_encoder)
    payload["data"]["train_data_config"] = str(data_config)
    payload["data"]["manifest_path"] = str(manifest)
    payload["data"]["online_encoding"]["runtime_reject_log_dir"] = str(tmp_path / "rejects")
    payload["output_dir"] = str(tmp_path / "output")
    monkeypatch.setattr(path_safety, "_ALLOWED_WRITE_ROOT", tmp_path)

    parsed = LtxTrainerConfig(**payload)
    assert isinstance(parsed.training_strategy, SemanticRepaEConfig)
    assert parsed.training_strategy.required_fsdp_world_size == 8
    assert parsed.training_strategy.semantic_anchor_count == 8
    assert parsed.model.training_mode == "full"
    assert parsed.data.online_encoding is not None
    assert parsed.data.online_encoding.augmentation.enabled is False
    assert parsed.data.online_encoding.image_ratio == pytest.approx(0.15)
    assert parsed.data.online_encoding.video_ratio == pytest.approx(0.85)
    assert (
        parsed.data.online_encoding.image_ratio
        + parsed.data.online_encoding.video_ratio
    ) == pytest.approx(1.0)
    assert parsed.data.online_encoding.image_ratio == pytest.approx(
        source_sampling["image_ratio"]
    )
    assert parsed.data.online_encoding.video_ratio == pytest.approx(
        source_sampling["video_ratio"]
    )
    assert parsed.optimization.learning_rate == pytest.approx(5.0e-6)
    assert parsed.optimization.bridge_learning_rate == pytest.approx(3.0e-6)
    assert parsed.optimization.semantic_learning_rate == pytest.approx(1.0e-5)
    assert parsed.optimization.repa_learning_rate == pytest.approx(1.0e-5)


def test_repae_task_specific_system_prompts_and_messages_are_distinct() -> None:
    prompts = load_semantic_system_prompts()
    assert set(prompts) == {IMAGE_TASK, VIDEO_TASK}
    i2i_prompt = prompts[IMAGE_TASK]
    r2v_prompt = prompts[VIDEO_TASK]
    assert i2i_prompt != r2v_prompt

    i2i_lower = i2i_prompt.lower()
    assert "image editing" in i2i_lower
    assert "single edited image" in i2i_lower
    assert "not as video frames" in i2i_lower
    assert "temporal motion" in i2i_lower
    assert "camera movement" in i2i_lower
    assert "video actions" in i2i_lower
    assert "left-right layout" in i2i_lower
    assert "background content" in i2i_lower
    assert "video action" in r2v_prompt.lower()

    i2i_messages = _build_messages(
        i2i_prompt,
        "replace the shirt color",
        1,
        task=IMAGE_TASK,
    )
    r2v_messages = _build_messages(
        r2v_prompt,
        "the subject walks forward",
        1,
        task=VIDEO_TASK,
    )
    assert i2i_messages[0]["content"] == i2i_prompt
    assert i2i_messages[0]["content"] != r2v_prompt
    assert r2v_messages[0]["content"] == r2v_prompt
    i2i_user_text = i2i_messages[1]["content"][-1]["text"]
    r2v_user_text = r2v_messages[1]["content"][-1]["text"]
    assert i2i_user_text == "Image editing instruction: replace the shirt color"
    assert r2v_user_text == "User Raw Input Prompt: the subject walks forward."


def test_repae_checkpoint_round_trip_covers_all_trainable_components(tmp_path: Path) -> None:
    state, metadata, _source_strategy, _source_transformer, _source_processor = (
        _semantic_repae_checkpoint_fixture()
    )
    checkpoint = tmp_path / "model_weights_step_00001.safetensors"
    save_file(state, checkpoint, metadata=metadata)

    audit = audit_checkpoint(checkpoint)
    assert audit["required_missing_keys"] == []
    assert read_checkpoint_metadata(checkpoint) == metadata
    assert metadata["architecture"] == "semantic_repae_v1"
    assert metadata["training_regime"] == "single_stage_full"
    assert metadata["reference_rope_mode"] == "negative_adjacent_shifted_hw"
    assert metadata["semantic_anchor_count"] == "8"
    assert metadata["semantic_tokens_per_frame"] == "256"

    loaded = load_file(checkpoint, device="cpu")
    target_strategy = _tiny_checkpoint_strategy()
    target_transformer = _TinyCheckpointTransformer()
    target_processor = _TinyEmbeddingsProcessor()
    target_strategy.load_extra_checkpoint_state_dict(loaded, checkpoint_metadata=metadata)
    validate_and_load_phase2_bridge_state(target_processor, loaded)
    transformer_state = {
        key: value
        for key, value in loaded.items()
        if not key.startswith("training_strategy.")
        and not key.startswith("embeddings_processor.")
    }
    target_transformer.load_state_dict(transformer_state, strict=True)

    for name, module in target_strategy.get_trainable_modules().items():
        prefix = f"training_strategy.{name}."
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, loaded[f"{prefix}{key}"])
    for key, value in target_transformer.state_dict().items():
        torch.testing.assert_close(value, loaded[key])
    for item in phase2_bridge_parameters(target_processor):
        torch.testing.assert_close(item.parameter, loaded[f"embeddings_processor.{item.name}"])


def test_repae_checkpoint_audit_fails_when_any_required_prefix_is_missing(tmp_path: Path) -> None:
    state, metadata, _strategy, _transformer, _processor = _semantic_repae_checkpoint_fixture()
    required = REPAE_STRATEGY_CHECKPOINT_PREFIXES + REPAE_TRANSFORMER_CHECKPOINT_PREFIXES
    for index, prefix in enumerate(required):
        incomplete = {key: value for key, value in state.items() if not key.startswith(prefix)}
        checkpoint = tmp_path / f"missing_{index}_step_00001.safetensors"
        save_file(incomplete, checkpoint, metadata=metadata)
        with pytest.raises(CheckpointAuditError, match="missing semantic modules"):
            audit_checkpoint(checkpoint)

    missing_strategy = {
        key: value
        for key, value in state.items()
        if not key.startswith("training_strategy.semantic_input_projection.")
    }
    with pytest.raises(RuntimeError, match="missing strategy modules"):
        _tiny_checkpoint_strategy().load_extra_checkpoint_state_dict(
            missing_strategy,
            checkpoint_metadata=metadata,
        )


def test_repae_rejects_semantic_flow_checkpoint_and_is_runtime_compatible() -> None:
    strategy = _tiny_checkpoint_strategy()
    with pytest.raises(RuntimeError, match="Unsupported semantic REPA-E checkpoint architecture"):
        strategy.load_extra_checkpoint_state_dict(
            {},
            checkpoint_metadata={"architecture": "semantic_flow_v2"},
        )
    assert isinstance(strategy, SemanticFlowStrategy)


def test_repae_trainability_audit_assigns_every_parameter_to_exactly_one_group() -> None:
    trainer = object.__new__(LtxvTrainer)
    trainer._training_strategy = _tiny_checkpoint_strategy()
    trainer._transformer = _TinyCheckpointTransformer()
    trainer._embeddings_processor = _TinyEmbeddingsProcessor()
    trainer._text_encoder = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    trainer._online_vae_encoder = nn.Linear(4, 4)
    trainer._text_encoder.requires_grad_(False)
    trainer._online_vae_encoder.requires_grad_(False)
    configure_phase2_bridge_trainability(trainer._embeddings_processor)
    trainer._embeddings_processor_trainable_modules = (
        trainer._training_strategy.get_embeddings_processor_trainable_modules(
            trainer._embeddings_processor
        )
    )
    strategy_modules = trainer._training_strategy.get_trainable_modules()
    trainer._trainable_params = LtxvTrainer._deduplicate_parameters(
        [
            *[parameter for parameter in trainer._transformer.parameters() if parameter.requires_grad],
            *[
                parameter
                for module in trainer._embeddings_processor_trainable_modules.values()
                for parameter in module.parameters()
                if parameter.requires_grad
            ],
            *[
                parameter
                for module in strategy_modules.values()
                for parameter in module.parameters()
                if parameter.requires_grad
            ],
        ]
    )

    trainer._validate_semantic_repae_trainability(strategy_modules)
    groups = trainer._semantic_repae_trainable_parameter_groups(strategy_modules)
    assert list(groups) == ["dit", "conditioning_bridge", "semantic_projection", "repa_projectors"]
    grouped_ids = [id(parameter) for values in groups.values() for parameter in values]
    assert len(grouped_ids) == len(set(grouped_ids)) == len(trainer._trainable_params)
    transformer_by_name = dict(trainer._transformer.named_parameters())
    dit_ids = {id(parameter) for parameter in groups["dit"]}
    for name in (
        "semantic_repae_reference_type_embedding.weight",
        "semantic_repae_reference_slot_embedding.weight",
        "semantic_repae_semantic_type_embedding.weight",
        "semantic_norm_out.weight",
        "semantic_proj_out.weight",
        "semantic_proj_out.bias",
    ):
        assert id(transformer_by_name[name]) in dit_ids
    assert all(parameter.requires_grad is False for parameter in trainer._text_encoder.parameters())
    assert all(parameter.requires_grad is False for parameter in trainer._online_vae_encoder.parameters())
    assert all(
        parameter.requires_grad is False
        for parameter in trainer._embeddings_processor.audio_connector.parameters()
    )
    trainer._config = SimpleNamespace(
        optimization=SimpleNamespace(
            learning_rate=5.0e-6,
            bridge_learning_rate=3.0e-6,
            semantic_learning_rate=1.0e-5,
            repa_learning_rate=2.0e-5,
            optimizer_type="adamw",
        )
    )
    trainer._create_scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(  # type: ignore[method-assign]
        optimizer,
        lr_lambda=lambda _step: 1.0,
    )
    trainer._accelerator = SimpleNamespace(prepare=lambda *values: values)
    trainer._init_optimizer()
    assert [group["name"] for group in trainer._optimizer.param_groups] == [
        "dit",
        "conditioning_bridge",
        "semantic_projection",
        "repa_projectors",
    ]
    assert [group["lr"] for group in trainer._optimizer.param_groups] == pytest.approx(
        [5.0e-6, 3.0e-6, 1.0e-5, 2.0e-5]
    )
