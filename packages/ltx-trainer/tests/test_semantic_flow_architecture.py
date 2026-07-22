from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from accelerate import DistributedType
from safetensors.torch import save_file
from torch import nn

from ltx_core.multicond.semantic_tokens import (
    EVIDENCE_TOKENS_PER_FRAME,
    SEMANTIC_TOKENS_PER_FRAME,
    SemanticEncoder,
    SemanticKeepMaskSample,
    SemanticQueryInitializer,
    SemanticReconstructionDecoder,
    build_multimodal_prefix_attention_mask,
    build_semantic_teacher_attention_mask,
    gather_local_evidence,
    sample_semantic_keep_mask,
    sample_semantic_keep_mask_with_stats,
)
from ltx_core.types import VideoLatentShape
from ltx_trainer.online_inference.checkpoint_runtime import (
    CheckpointAuditError,
    audit_checkpoint,
    resolve_checkpoint,
)
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.trainer import _enforce_semantic_flow_fsdp_runtime_safety
import ltx_trainer.training_strategies.semantic_flow as semantic_flow_module
from ltx_trainer.training_strategies.semantic_flow import SemanticFlowConfig, SemanticFlowStrategy


def test_local_queries_and_reconstruction_targets_preserve_spatial_neighborhoods() -> None:
    evidence = torch.arange(EVIDENCE_TOKENS_PER_FRAME, dtype=torch.float32).reshape(1, 1, -1, 1)
    grouped = gather_local_evidence(evidence)
    assert grouped.shape == (1, 1, SEMANTIC_TOKENS_PER_FRAME, 4, 1)
    assert grouped[0, 0, 0, :, 0].tolist() == [0.0, 1.0, 16.0, 17.0]
    assert grouped[0, 0, 63, :, 0].tolist() == [238.0, 239.0, 254.0, 255.0]

    initializer = SemanticQueryInitializer(gemma_dim=4)
    queries = initializer(torch.randn(2, 3, 256, 4), torch.tensor([[0.0, 0.5, 1.0]]).expand(2, -1))
    assert queries.shape == (2, 3, 64, 4)


def test_teacher_mask_blocks_prefix_from_gt_and_queries_from_nonlocal_evidence() -> None:
    prefix = torch.ones(1, 4, dtype=torch.bool)
    mask = build_semantic_teacher_attention_mask(prefix, frame_count=1)
    evidence_start = 4
    query_start = evidence_start + 256
    first_query = query_start
    assert not mask[0, :4, evidence_start:].any()
    assert mask[0, evidence_start : evidence_start + 256, :4].all()
    visible_evidence = torch.nonzero(
        mask[0, first_query, evidence_start : evidence_start + 256],
        as_tuple=False,
    ).flatten()
    assert visible_evidence.tolist() == [0, 1, 16, 17]
    assert mask[0, first_query, first_query]
    assert not mask[0, first_query, first_query + 1 :].any()


class _TinyMaskedAttentionLanguageModel(nn.Module):
    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
        use_cache: bool,
    ):
        assert output_hidden_states and return_dict and not use_cache
        visible = attention_mask[:, 0] == 0
        source = inputs_embeds + position_ids.to(dtype=inputs_embeds.dtype).unsqueeze(-1) * 0.01
        scores = torch.matmul(source, source.transpose(-1, -2)) / math.sqrt(source.shape[-1])
        scores = scores.masked_fill(~visible, -1.0e9)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(visible.any(dim=-1, keepdim=True), weights, torch.zeros_like(weights))
        hidden = torch.matmul(weights, source)
        return SimpleNamespace(hidden_states=(hidden,))


def _to_additive_attention_mask(visible: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    bias = torch.zeros_like(visible, dtype=dtype)
    bias.masked_fill_(~visible, torch.finfo(dtype).min)
    return bias.unsqueeze(1)


def test_prefix_only_hidden_matches_teacher_prefix_hidden_with_reference_regions_and_padding() -> None:
    generator = torch.Generator().manual_seed(11)
    prefix_embeddings = torch.randn(1, 12, 8, generator=generator)
    suffix_embeddings = torch.randn(
        1,
        EVIDENCE_TOKENS_PER_FRAME + SEMANTIC_TOKENS_PER_FRAME,
        8,
        generator=generator,
    )
    prefix_attention = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    reference_regions = torch.zeros_like(prefix_attention)
    reference_regions[:, 2:5] = True
    reference_regions[:, 6:9] = True

    prefix_visible = build_multimodal_prefix_attention_mask(
        prefix_attention,
        reference_region_mask=reference_regions,
    )
    assert prefix_visible[0, 2, 4]
    assert prefix_visible[0, 4, 2]
    assert prefix_visible[0, 6, 8]
    assert prefix_visible[0, 8, 6]
    assert not prefix_visible[0, 2, 6]
    assert prefix_visible[0, 6, 2]
    assert not prefix_visible[0, 10].any()
    assert not prefix_visible[0, :, 10].any()

    teacher_visible = build_semantic_teacher_attention_mask(
        prefix_attention,
        frame_count=1,
        reference_region_mask=reference_regions,
    )
    assert torch.equal(teacher_visible[:, : prefix_embeddings.shape[1], : prefix_embeddings.shape[1]], prefix_visible)
    assert not teacher_visible[:, : prefix_embeddings.shape[1], prefix_embeddings.shape[1] :].any()

    language_model = _TinyMaskedAttentionLanguageModel()
    prefix_positions = torch.arange(prefix_embeddings.shape[1]).unsqueeze(0)
    prefix_only_hidden = language_model(
        inputs_embeds=prefix_embeddings,
        attention_mask=_to_additive_attention_mask(prefix_visible, prefix_embeddings.dtype),
        position_ids=prefix_positions,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    ).hidden_states[-1]

    teacher_embeddings = torch.cat([prefix_embeddings, suffix_embeddings], dim=1)
    teacher_positions = torch.arange(teacher_embeddings.shape[1]).unsqueeze(0)
    teacher_hidden = language_model(
        inputs_embeds=teacher_embeddings,
        attention_mask=_to_additive_attention_mask(teacher_visible, teacher_embeddings.dtype),
        position_ids=teacher_positions,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    ).hidden_states[-1]
    teacher_prefix_hidden = teacher_hidden[:, : prefix_embeddings.shape[1]]

    torch.testing.assert_close(prefix_only_hidden, teacher_prefix_hidden, rtol=0.0, atol=0.0)


def test_semantic_dropout_uses_exact_counts_between_48_and_64_tokens_per_frame() -> None:
    generator = torch.Generator().manual_seed(7)
    sample = sample_semantic_keep_mask_with_stats(
        torch.zeros(8, 12, 64, 16),
        maximum_drop_rate=0.25,
        minimum_tokens_per_frame=48,
        generator=generator,
    )
    mask = sample.keep_mask
    assert mask.shape == (8, 12, 64)
    per_frame_kept = mask.sum(dim=-1)
    assert (per_frame_kept >= 48).all()
    assert (per_frame_kept <= 64).all()
    assert torch.equal(per_frame_kept, per_frame_kept[:, :1].expand_as(per_frame_kept))
    assert torch.equal(sample.drop_count_per_frame, 64 - per_frame_kept[:, 0])
    assert torch.equal(sample.requested_drop_rate, sample.drop_count_per_frame.float() / 64.0)
    wrapper_mask = sample_semantic_keep_mask(
        torch.zeros(2, 3, 64, 4),
        maximum_drop_rate=0.25,
        minimum_tokens_per_frame=48,
        generator=torch.Generator().manual_seed(9),
    )
    assert wrapper_mask.shape == (2, 3, 64)


def test_semantic_flow_reports_actual_kept_prefix_and_latent_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = SemanticFlowStrategy(SemanticFlowConfig())
    strategy._semantic_dim = 4
    strategy._gemma_dim = 4
    strategy._query_initializer = SemanticQueryInitializer(4)
    strategy._semantic_encoder = SemanticEncoder(4, 4)
    strategy._reconstruction_decoder = SemanticReconstructionDecoder(4, 4)

    semantic_clean = torch.ones(1, 2, SEMANTIC_TOKENS_PER_FRAME, 4)
    teacher = {
        "semantic_clean": semantic_clean,
        "reconstruction_prediction": torch.zeros(1, 2, SEMANTIC_TOKENS_PER_FRAME, 4, 4),
        "reconstruction_target": torch.zeros(1, 2, SEMANTIC_TOKENS_PER_FRAME, 4, 4),
    }
    strategy.build_semantic_teacher_outputs = lambda _teacher_inputs: teacher  # type: ignore[method-assign]
    keep_mask = torch.zeros(1, 2, SEMANTIC_TOKENS_PER_FRAME, dtype=torch.bool)
    keep_mask[:, :, :40] = True
    keep_sample = SemanticKeepMaskSample(
        keep_mask=keep_mask,
        requested_drop_rate=torch.tensor([24.0 / 64.0]),
        drop_count_per_frame=torch.tensor([24]),
    )
    monkeypatch.setattr(
        semantic_flow_module,
        "sample_semantic_keep_mask_with_stats",
        lambda *args, **kwargs: keep_sample,
    )

    batch = {
        "semantic_teacher_inputs": {
            "prefix_attention_mask": torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool),
            "normalized_timestamps": torch.tensor([[0.0, 1.0]]),
        },
        "latents": {
            "latents": torch.zeros(1, 4, 1, 1, 1),
            "num_frames": torch.tensor([1]),
            "height": torch.tensor([1]),
            "width": torch.tensor([1]),
            "fps": torch.tensor([24.0]),
        },
        "reference_latents": {
            "latents": torch.zeros(1, 1, 4, 1, 1, 1),
            "ref_valid_mask": torch.tensor([[True]]),
        },
        "conditions": {
            "video_prompt_embeds": torch.zeros(1, 3, 4),
            "prompt_attention_mask": torch.ones(1, 3, dtype=torch.bool),
        },
    }
    sampler = SimpleNamespace(sample_for=lambda target_tokens: torch.full((target_tokens.shape[0],), 0.5))

    inputs = strategy.prepare_training_inputs(batch, sampler)
    metrics = strategy.get_last_training_metrics()
    assert float(metrics["train/semantic_requested_drop_rate"]) == pytest.approx(24.0 / 64.0)
    assert float(metrics["train/semantic_drop_count_per_frame"]) == 24.0
    assert float(metrics["train/semantic_token_count_before_dropout"]) == 128.0
    assert float(metrics["train/semantic_token_count_kept"]) == 80.0
    assert float(metrics["train/semantic_keep_ratio"]) == pytest.approx(80.0 / 128.0)
    assert float(metrics["train/prefix_token_count"]) == 3.0
    assert float(metrics["train/anchor_frame_count"]) == 2.0
    assert float(metrics["train/semantic_latent_rms"]) == pytest.approx(1.0)
    assert float(metrics["train/query_position_gate"]) == pytest.approx(0.0)
    assert float(metrics["train/semantic_global_scale"]) == pytest.approx(1.0)

    strategy.compute_loss(torch.zeros_like(inputs.video.latent), None, inputs)
    metrics = strategy.get_last_training_metrics()
    assert "train/loss_semantic_flow" in metrics
    assert float(metrics["train/semantic_token_count_kept"]) == 80.0


def test_production_semantic_flow_config_uses_opens2v_only_litengjie_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    training_path = root / "configs" / "semantic_flow_multitask_480p121.yaml"
    data_path = root / "configs" / "multitask_online_480p121_data.yaml"
    opens2v_path = root / "configs" / "multitask_online_480p121_opens2v.yaml"

    training_text = training_path.read_text(encoding="utf-8")
    data_text = data_path.read_text(encoding="utf-8")
    opens2v_text = opens2v_path.read_text(encoding="utf-8")
    assert "/path/to/" not in training_text

    training_config = yaml.safe_load(training_text)
    data_config = yaml.safe_load(data_text)
    opens2v_config = yaml.safe_load(opens2v_text)
    parsed = LtxTrainerConfig.model_validate(training_config)

    assert parsed.model.model_path == "/mnt/workspace/litengjie/LTX-2/models/LTX-2.3/ltx-2.3-22b-dev.safetensors"
    assert (
        parsed.model.text_encoder_path
        == "/mnt/workspace/litengjie/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized"
    )
    assert parsed.output_dir.startswith("/mnt/workspace/litengjie/")
    assert parsed.data.manifest_path.startswith("/mnt/workspace/litengjie/")
    assert parsed.data.online_encoding is not None
    assert parsed.data.online_encoding.runtime_reject_log_dir.startswith("/mnt/workspace/litengjie/")
    assert parsed.checkpoints.save_training_state == "minimal"
    assert parsed.text_encoder_lora.enabled is False

    for config in (data_config, opens2v_config):
        dataset_names = {dataset["name"] for dataset in config["datasets"]}
        dataset_types = {dataset["dataset_type"] for dataset in config["datasets"]}
        assert "r2v_phantom" not in dataset_names
        assert "PhantomDataset" not in dataset_types
        assert config["online_sampling"]["video_source_ratios"] == {"r2v_opens2v": 1.0}

    lora_config = dict(training_config)
    lora_config["model"] = {**training_config["model"], "training_mode": "lora"}
    with pytest.raises(ValueError, match="semantic_flow requires full DiT training"):
        LtxTrainerConfig.model_validate(lora_config)

    gemma_lora_config = dict(training_config)
    gemma_lora_config["text_encoder_lora"] = {"enabled": True}
    with pytest.raises(ValueError, match="forbids text-encoder LoRA"):
        LtxTrainerConfig.model_validate(gemma_lora_config)

    strategy = SemanticFlowStrategy(SemanticFlowConfig())
    assert strategy.train_embeddings_processor() is False


def _fake_trainer_config(*, strategy_name: str = "semantic_flow", training_mode: str = "full") -> SimpleNamespace:
    return SimpleNamespace(
        training_strategy=SimpleNamespace(name=strategy_name),
        model=SimpleNamespace(training_mode=training_mode),
    )


def _fake_accelerator(distributed_type: DistributedType, *, sharding_strategy: str | None = None) -> SimpleNamespace:
    fsdp_plugin = SimpleNamespace(
        sharding_strategy=None if sharding_strategy is None else SimpleNamespace(name=sharding_strategy)
    )
    return SimpleNamespace(distributed_type=distributed_type, state=SimpleNamespace(fsdp_plugin=fsdp_plugin))


def test_semantic_flow_full_training_requires_fsdp_full_shard() -> None:
    config = _fake_trainer_config()
    _enforce_semantic_flow_fsdp_runtime_safety(
        config,
        _fake_accelerator(DistributedType.FSDP, sharding_strategy="FULL_SHARD"),
    )

    with pytest.raises(RuntimeError, match="requires Accelerate FSDP FULL_SHARD"):
        _enforce_semantic_flow_fsdp_runtime_safety(config, _fake_accelerator(DistributedType.MULTI_GPU))
    with pytest.raises(RuntimeError, match="requires Accelerate FSDP FULL_SHARD"):
        _enforce_semantic_flow_fsdp_runtime_safety(config, _fake_accelerator(DistributedType.NO))
    with pytest.raises(RuntimeError, match="Configured FSDP sharding strategy"):
        _enforce_semantic_flow_fsdp_runtime_safety(
            config,
            _fake_accelerator(DistributedType.FSDP, sharding_strategy="SHARD_GRAD_OP"),
        )

    _enforce_semantic_flow_fsdp_runtime_safety(
        _fake_trainer_config(strategy_name="text_to_video", training_mode="lora"),
        _fake_accelerator(DistributedType.NO),
    )


class _VelocityRecorder(nn.Module):
    def __init__(self, reference_end: int) -> None:
        super().__init__()
        self.reference_end = reference_end
        self.reference_inputs: list[torch.Tensor] = []
        self.timesteps: list[torch.Tensor] = []

    def forward(self, *, video, audio, perturbations):
        del audio, perturbations
        self.reference_inputs.append(video.latent[:, : self.reference_end].clone())
        self.timesteps.append(video.timesteps.clone())
        return torch.ones_like(video.latent), None


def test_joint_ode_keeps_references_clean_and_shares_semantic_video_sigma() -> None:
    strategy = SemanticFlowStrategy(SemanticFlowConfig())
    strategy._semantic_dim = 128
    references = {
        "latents": torch.randn(1, 2, 128, 1, 2, 2),
        "ref_valid_mask": torch.tensor([[True, True]]),
    }
    state = strategy.prepare_inference_state(
        conditions={
            "video_prompt_embeds": torch.zeros(1, 2, 8),
            "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
        },
        reference_latents=references,
        target_shape=VideoLatentShape(batch=1, channels=128, frames=1, height=2, width=2),
        semantic_frame_count=1,
        seed=13,
    )
    reference_end = state.sequence_offsets["reference_end"]
    semantic_end = state.sequence_offsets["semantic_end"]
    recorder = _VelocityRecorder(reference_end)
    semantic, video = strategy.denoise_joint(
        transformer=recorder,
        state=state,
        num_inference_steps=3,
    )
    assert semantic.shape == (1, 64, 128)
    assert video.shape == (1, 128, 1, 2, 2)
    assert all(torch.equal(value, recorder.reference_inputs[0]) for value in recorder.reference_inputs)
    for timesteps in recorder.timesteps:
        assert torch.count_nonzero(timesteps[:, :reference_end]) == 0
        semantic_sigma = timesteps[:, reference_end:semantic_end]
        video_sigma = timesteps[:, semantic_end:]
        assert torch.equal(semantic_sigma[:, :1], video_sigma[:, :1])


def _semantic_checkpoint(path: Path, *, complete: bool = True) -> None:
    tensors = {
        "block.weight": torch.ones(1),
        "training_strategy.semantic_query.weight": torch.ones(1),
        "training_strategy.semantic_encoder.weight": torch.ones(1),
    }
    if complete:
        tensors["training_strategy.semantic_reconstruction_decoder.weight"] = torch.ones(1)
    save_file(tensors, path, metadata={"architecture": "semantic_flow_v1"})


def test_checkpoint_audit_and_ready_resolution_require_semantic_modules(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_weights_step_00025.safetensors"
    _semantic_checkpoint(checkpoint)
    audit = audit_checkpoint(checkpoint)
    assert audit["checkpoint_step"] == 25
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    marker = tmp_path / "checkpoint_step_00025.ready.json"
    marker.write_text(
        json.dumps({"checkpoint_path": str(checkpoint), "checkpoint_sha256": digest}),
        encoding="utf-8",
    )
    resolved, marker_path, _ = resolve_checkpoint(checkpoint=None, latest_ready_dir=str(tmp_path))
    assert resolved == checkpoint.resolve()
    assert marker_path == marker

    incomplete = tmp_path / "model_weights_step_00026.safetensors"
    _semantic_checkpoint(incomplete, complete=False)
    with pytest.raises(CheckpointAuditError, match="missing semantic modules"):
        audit_checkpoint(incomplete)
