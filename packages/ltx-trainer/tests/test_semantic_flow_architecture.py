from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file
from torch import nn

from ltx_core.multicond.semantic_tokens import (
    EVIDENCE_TOKENS_PER_FRAME,
    SEMANTIC_TOKENS_PER_FRAME,
    SemanticQueryInitializer,
    build_semantic_teacher_attention_mask,
    gather_local_evidence,
    sample_semantic_keep_mask,
)
from ltx_core.types import VideoLatentShape
from ltx_trainer.online_inference.checkpoint_runtime import (
    CheckpointAuditError,
    audit_checkpoint,
    resolve_checkpoint,
)
from ltx_trainer.config import LtxTrainerConfig
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


def test_semantic_dropout_keeps_at_least_56_tokens_per_frame() -> None:
    generator = torch.Generator().manual_seed(7)
    mask = sample_semantic_keep_mask(
        torch.zeros(8, 12, 64, 16),
        maximum_drop_rate=0.2,
        minimum_tokens_per_frame=56,
        generator=generator,
    )
    assert mask.shape == (8, 12, 64)
    assert (mask.sum(dim=-1) >= 56).all()


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
