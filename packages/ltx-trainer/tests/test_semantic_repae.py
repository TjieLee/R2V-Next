from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from accelerate import DistributedType
from torch import nn

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.multicond.semantic_tokens import (
    EVIDENCE_TOKENS_PER_FRAME,
    SemanticInputProjection,
    SemanticRepaProjector,
    semantic_repa_loss,
)
from ltx_core.types import VideoLatentShape
from ltx_trainer.online_data.adapters.base import CanonicalR2VSource
from ltx_trainer.online_data.manifest import prepare_canonical_r2v_record
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.training_strategies.semantic_flow import TYPE_REFERENCE, TYPE_SEMANTIC, TYPE_TARGET
from ltx_trainer.training_strategies.semantic_repae import SemanticRepaEConfig, SemanticRepaEStrategy
from ltx_trainer.trainer import _enforce_semantic_flow_fsdp_runtime_safety


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
    assert (ref_positions[:, 0, :, 0] <= 0).all()
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

    def forward(self, *, inputs_embeds: torch.Tensor, **_kwargs) -> SimpleNamespace:
        return SimpleNamespace(hidden_states=(inputs_embeds * self.scale,))


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
    assert strategy.get_checkpoint_metadata()["architecture"] == "semantic_repae_v1"


def test_repae_full_dit_requires_fsdp_full_shard_runtime() -> None:
    config = SimpleNamespace(
        training_strategy=SimpleNamespace(name="semantic_repae"),
        model=SimpleNamespace(training_mode="full"),
    )
    accelerator = SimpleNamespace(distributed_type=DistributedType.NO)
    with pytest.raises(RuntimeError, match="semantic REPA-E.*FSDP FULL_SHARD"):
        _enforce_semantic_flow_fsdp_runtime_safety(config, accelerator)
