from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.utils.checkpoint
import transformers
import yaml
from accelerate import DistributedType
from safetensors.torch import save_file
from torch import nn

import ltx_trainer.training_strategies.semantic_flow as semantic_flow_module
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
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.online_inference.checkpoint_runtime import (
    CheckpointAuditError,
    audit_checkpoint,
    resolve_checkpoint,
)
from ltx_trainer.online_inference.runtime_lock import (
    SemanticFlowRuntimeLockError,
    build_semantic_flow_runtime_lock,
    write_or_validate_semantic_flow_runtime_lock,
)
from ltx_trainer.online_inference.smoke_marker import (
    SemanticFlowSmokeMarkerError,
    validate_semantic_flow_smoke_marker,
    write_semantic_flow_smoke_marker,
)
from ltx_trainer.training_strategies.semantic_flow import SemanticFlowConfig, SemanticFlowStrategy
from ltx_trainer.trainer import _enforce_semantic_flow_fsdp_runtime_safety


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
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(use_cache=True)
        self.layers = nn.ModuleList([nn.Identity()])

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | dict[str, torch.Tensor],
        position_ids: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
        use_cache: bool,
    ):
        assert output_hidden_states and return_dict and not use_cache
        if isinstance(attention_mask, dict):
            attention_mask = attention_mask["full_attention"]
        visible = attention_mask[:, 0] == 0
        source = inputs_embeds * self.scale + position_ids.to(dtype=inputs_embeds.dtype).unsqueeze(-1) * 0.01
        source = self.layers[0](source)
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
    image_regions = torch.zeros_like(prefix_attention)
    image_regions[:, 2:5] = True
    image_regions[:, 6:9] = True

    prefix_visible = build_multimodal_prefix_attention_mask(
        prefix_attention,
        image_token_mask=image_regions,
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
        image_token_mask=image_regions,
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


def _has_nonzero_grad(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in module.parameters()
    )


def test_frozen_gemma_teacher_backpropagates_only_to_semantic_modules() -> None:
    language_model = _TinyMaskedAttentionLanguageModel()
    language_model.requires_grad_(False)
    strategy = SemanticFlowStrategy(SemanticFlowConfig())
    strategy._text_encoder = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(language_model=language_model))
    )
    strategy._query_initializer = SemanticQueryInitializer(gemma_dim=8)
    strategy._semantic_encoder = SemanticEncoder(gemma_dim=8, semantic_dim=4, hidden_dim=8)
    strategy._reconstruction_decoder = SemanticReconstructionDecoder(semantic_dim=4, gemma_dim=8, hidden_dim=8)

    mode = strategy._enable_frozen_teacher_gradient_checkpointing(language_model)
    assert mode == "manual_non_reentrant"
    assert language_model.config.use_cache is False
    assert strategy.teacher_checkpointed_layer_count == 1

    prefix_embeddings = torch.randn(1, 6, 8, requires_grad=True)
    evidence = torch.randn(1, 1, EVIDENCE_TOKENS_PER_FRAME, 8, requires_grad=True)
    teacher = strategy.build_semantic_teacher_outputs(
        {
            "prefix_inputs_embeds": prefix_embeddings,
            "prefix_attention_mask": torch.ones(1, 6, dtype=torch.bool),
            "prefix_image_token_mask": torch.zeros(1, 6, dtype=torch.bool),
            "evidence_tokens": evidence,
            "normalized_timestamps": torch.tensor([[0.25]]),
        }
    )
    loss = teacher["semantic_clean"].pow(2).mean() + teacher["reconstruction_prediction"].pow(2).mean()
    loss.backward()

    assert _has_nonzero_grad(strategy._query_initializer)
    assert _has_nonzero_grad(strategy._semantic_encoder)
    assert _has_nonzero_grad(strategy._reconstruction_decoder)
    assert strategy.teacher_checkpoint_forward_calls > 0
    assert language_model.training is False
    assert all(parameter.grad is None for parameter in language_model.parameters())
    assert prefix_embeddings.grad is None
    assert evidence.grad is None
    assert teacher["reconstruction_target"].requires_grad is False


def test_frozen_gemma_checkpointing_falls_back_to_manual_layers() -> None:
    class _ManualCheckpointLanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(use_cache=True)
            self.layers = nn.ModuleList([nn.Linear(2, 2)])

    model = _ManualCheckpointLanguageModel()
    strategy = SemanticFlowStrategy(SemanticFlowConfig())
    mode = strategy._enable_frozen_teacher_gradient_checkpointing(model)

    assert mode == "manual_non_reentrant"
    assert model.config.use_cache is False
    assert getattr(model.layers[0], "_semantic_flow_non_reentrant_checkpoint")
    assert getattr(model.layers[0], "_semantic_flow_original_forward") is not None
    assert strategy._enable_frozen_teacher_gradient_checkpointing(model) == "manual_non_reentrant"
    assert strategy.teacher_checkpointed_layer_count == 1


def test_eval_tiny_gemma_teacher_executes_manual_checkpointing_and_backpropagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = transformers.Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        sliding_window=1024,
        layer_types=["full_attention", "sliding_attention"],
        use_cache=False,
    )
    language_model = transformers.Gemma3TextModel(config).eval().requires_grad_(False)
    strategy = SemanticFlowStrategy(SemanticFlowConfig())
    strategy._text_encoder = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(language_model=language_model))
    )
    strategy._query_initializer = SemanticQueryInitializer(gemma_dim=config.hidden_size)
    strategy._semantic_encoder = SemanticEncoder(
        gemma_dim=config.hidden_size,
        semantic_dim=8,
        hidden_dim=16,
    )
    strategy._reconstruction_decoder = SemanticReconstructionDecoder(
        semantic_dim=8,
        gemma_dim=config.hidden_size,
        hidden_dim=16,
    )

    checkpoint_calls = 0
    real_checkpoint = torch.utils.checkpoint.checkpoint

    def counted_checkpoint(*args, **kwargs) -> Any:
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return real_checkpoint(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counted_checkpoint)
    assert strategy._enable_frozen_teacher_gradient_checkpointing(language_model) == "manual_non_reentrant"

    teacher = strategy.build_semantic_teacher_outputs(
        {
            "prefix_inputs_embeds": torch.randn(1, 6, config.hidden_size),
            "prefix_attention_mask": torch.ones(1, 6, dtype=torch.bool),
            "prefix_image_token_mask": torch.zeros(1, 6, dtype=torch.bool),
            "evidence_tokens": torch.randn(1, 1, EVIDENCE_TOKENS_PER_FRAME, config.hidden_size),
            "normalized_timestamps": torch.tensor([[0.25]]),
        }
    )
    loss = teacher["semantic_clean"].pow(2).mean() + teacher["reconstruction_prediction"].pow(2).mean()
    loss.backward()

    assert checkpoint_calls > 0
    assert strategy.teacher_checkpoint_forward_calls == checkpoint_calls
    assert strategy.teacher_checkpointed_layer_count == 2
    assert language_model.training is False
    assert all(not parameter.requires_grad for parameter in language_model.parameters())
    assert all(parameter.grad is None for parameter in language_model.parameters())
    assert _has_nonzero_grad(strategy._query_initializer)
    assert _has_nonzero_grad(strategy._semantic_encoder)
    assert _has_nonzero_grad(strategy._reconstruction_decoder)


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


def test_production_semantic_flow_config_uses_opens2v_only_litengjie_paths(tmp_path: Path) -> None:
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

    assert training_config["model"]["model_path"] == (
        "/mnt/workspace/litengjie/LTX-2/models/LTX-2.3/ltx-2.3-22b-dev.safetensors"
    )
    assert (
        training_config["model"]["text_encoder_path"]
        == "/mnt/workspace/litengjie/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized"
    )
    assert training_config["output_dir"].startswith("/mnt/workspace/litengjie/")
    assert training_config["data"]["manifest_path"].startswith("/mnt/workspace/litengjie/")
    assert training_config["data"]["online_encoding"]["runtime_reject_log_dir"].startswith(
        "/mnt/workspace/litengjie/"
    )

    model_path = tmp_path / "ltx-2.3-22b-dev.safetensors"
    model_path.write_bytes(b"fake safetensors placeholder")
    gemma_dir = tmp_path / "gemma-3-12b-it"
    gemma_dir.mkdir()
    (gemma_dir / "config.json").write_text(json.dumps({"sliding_window": 1024}), encoding="utf-8")
    train_data_path = tmp_path / "multitask_online_480p121_opens2v.yaml"
    train_data_path.write_text(yaml.safe_dump(opens2v_config), encoding="utf-8")
    manifest_path = tmp_path / "train_i2i_opens2v.jsonl"
    manifest_path.write_text("", encoding="utf-8")

    schema_config = copy.deepcopy(training_config)
    schema_config["model"]["model_path"] = str(model_path)
    schema_config["model"]["text_encoder_path"] = str(gemma_dir)
    schema_config["data"]["train_data_config"] = str(train_data_path)
    schema_config["data"]["manifest_path"] = str(manifest_path)
    parsed = LtxTrainerConfig.model_validate(schema_config)

    assert parsed.output_dir.startswith("/mnt/workspace/litengjie/")
    assert parsed.data.manifest_path == str(manifest_path.resolve())
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

    lora_config = copy.deepcopy(schema_config)
    lora_config["model"] = {**schema_config["model"], "training_mode": "lora"}
    with pytest.raises(ValueError, match="semantic_flow requires full DiT training"):
        LtxTrainerConfig.model_validate(lora_config)

    gemma_lora_config = copy.deepcopy(schema_config)
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


def _fake_accelerator(
    distributed_type: DistributedType,
    *,
    fsdp_version: int | str | None = 1,
    sharding_strategy: str | None = "FULL_SHARD",
    state_dict_type: str | None = "FULL_STATE_DICT",
) -> SimpleNamespace:
    fsdp_plugin = SimpleNamespace(
        fsdp_version=fsdp_version,
        sharding_strategy=None if sharding_strategy is None else SimpleNamespace(name=sharding_strategy),
        state_dict_type=None if state_dict_type is None else SimpleNamespace(name=state_dict_type),
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
    with pytest.raises(RuntimeError, match="currently supports only FSDP1"):
        _enforce_semantic_flow_fsdp_runtime_safety(
            config,
            _fake_accelerator(DistributedType.FSDP, fsdp_version=2),
        )
    with pytest.raises(RuntimeError, match="Unable to identify FSDP sharding strategy"):
        _enforce_semantic_flow_fsdp_runtime_safety(
            config,
            _fake_accelerator(DistributedType.FSDP, sharding_strategy=None),
        )
    with pytest.raises(RuntimeError, match="requires FSDP FULL_STATE_DICT"):
        _enforce_semantic_flow_fsdp_runtime_safety(
            config,
            _fake_accelerator(DistributedType.FSDP, state_dict_type="SHARDED_STATE_DICT"),
        )

    _enforce_semantic_flow_fsdp_runtime_safety(
        _fake_trainer_config(strategy_name="text_to_video", training_mode="lora"),
        _fake_accelerator(DistributedType.NO),
    )


@pytest.mark.parametrize(
    ("filename", "expected_processes"),
    [
        ("accelerate_semantic_flow_fsdp_smoke_2gpu.yaml", 2),
        ("accelerate_semantic_flow_fsdp_train_8gpu.yaml", 8),
    ],
)
def test_semantic_flow_fsdp_accelerate_configs_are_full_shard(
    filename: str,
    expected_processes: int,
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / filename
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fsdp_config = config["fsdp_config"]
    assert config["compute_environment"] == "LOCAL_MACHINE"
    assert config["distributed_type"] == "FSDP"
    assert config["mixed_precision"] == "bf16"
    assert config["num_processes"] == expected_processes
    assert fsdp_config["fsdp_version"] == 1
    assert fsdp_config["fsdp_sharding_strategy"] == "FULL_SHARD"
    assert fsdp_config["fsdp_state_dict_type"] == "FULL_STATE_DICT"
    assert fsdp_config["fsdp_auto_wrap_policy"] == "TRANSFORMER_BASED_WRAP"
    assert fsdp_config["fsdp_transformer_layer_cls_to_wrap"] == "BasicAVTransformerBlock"
    assert fsdp_config["fsdp_use_orig_params"] is True
    assert fsdp_config["fsdp_sync_module_states"] is True
    assert fsdp_config["fsdp_cpu_ram_efficient_loading"] is False


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
        "semantic_token_type_embedding.weight": torch.ones(3, 4),
        "semantic_entity_embedding.weight": torch.ones(5, 4),
        "semantic_position_adapter.0.weight": torch.ones(4, 6),
        "semantic_position_adapter.0.bias": torch.ones(4),
        "semantic_norm_out.weight": torch.ones(4),
        "semantic_proj_out.weight": torch.ones(2, 4),
        "semantic_proj_out.bias": torch.ones(2),
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

    missing_transformer = tmp_path / "model_weights_step_00027.safetensors"
    tensors = {
        "block.weight": torch.ones(1),
        "semantic_token_type_embedding.weight": torch.ones(3, 4),
        "semantic_entity_embedding.weight": torch.ones(5, 4),
        "semantic_position_adapter.0.weight": torch.ones(4, 6),
        "semantic_position_adapter.0.bias": torch.ones(4),
        "semantic_norm_out.weight": torch.ones(4),
        "training_strategy.semantic_query.weight": torch.ones(1),
        "training_strategy.semantic_encoder.weight": torch.ones(1),
        "training_strategy.semantic_reconstruction_decoder.weight": torch.ones(1),
    }
    save_file(tensors, missing_transformer, metadata={"architecture": "semantic_flow_v1"})
    with pytest.raises(CheckpointAuditError, match="semantic_proj_out"):
        audit_checkpoint(missing_transformer)


def test_semantic_flow_smoke_marker_binds_code_configs_checkpoints_and_real_inference(tmp_path: Path) -> None:
    code_commit = "a" * 40
    training_config = tmp_path / "train.yaml"
    accelerate_config = tmp_path / "accelerate.yaml"
    i2i_checkpoint = tmp_path / "i2i.safetensors"
    r2v_checkpoint = tmp_path / "r2v.safetensors"
    runtime_audit = tmp_path / "runtime_audit.json"
    for path, content in (
        (training_config, "training: true\n"),
        (accelerate_config, "num_processes: 8\n"),
        (i2i_checkpoint, "i2i checkpoint\n"),
        (r2v_checkpoint, "r2v checkpoint\n"),
        (runtime_audit, "{}\n"),
    ):
        path.write_text(content, encoding="utf-8")

    sample_dir = tmp_path / "i2i-output"
    sample_dir.mkdir()
    (sample_dir / "generated.png").write_bytes(b"png")
    (sample_dir / "success.json").write_text('{"status":"success"}\n', encoding="utf-8")
    result = {
        "status": "success",
        "sample_dir": str(sample_dir),
        "task": "i2i",
        "dry_run": False,
        "num_inference_steps": 2,
        "reference_velocity": 0.0,
        "shared_semantic_video_sigma": True,
        "semantic_latent_finite": True,
        "video_latent_finite": True,
        "decoded_image_finite": True,
        "strict_no_gt_checks": {"target_path_passed_to_condition_encoder": False},
        "output_shape": [1, 3, 64, 64],
        "code_commit": code_commit,
        "checkpoint": str(i2i_checkpoint),
        "checkpoint_sha256": hashlib.sha256(i2i_checkpoint.read_bytes()).hexdigest(),
    }
    metadata = {key: value for key, value in result.items() if key not in {"status", "sample_dir"}}
    (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    inference_summary = tmp_path / "run_summary.json"
    inference_summary.write_text(
        json.dumps(
            {
                "strict_no_gt": True,
                "success_count": 1,
                "failure_count": 0,
                "results": [result],
            }
        ),
        encoding="utf-8",
    )

    marker_path = tmp_path / "semantic_flow_smoke_success.json"
    marker = write_semantic_flow_smoke_marker(
        marker_path,
        code_commit=code_commit,
        training_config_path=training_config,
        accelerate_config_path=accelerate_config,
        i2i_checkpoint_path=i2i_checkpoint,
        r2v_checkpoint_path=r2v_checkpoint,
        runtime_audit_path=runtime_audit,
        inference_summary_path=inference_summary,
    )
    assert marker["non_dry_run_inference_passed"] is True
    assert validate_semantic_flow_smoke_marker(
        marker_path,
        code_commit=code_commit,
        training_config_path=training_config,
        accelerate_config_path=accelerate_config,
    ) == marker

    training_config.write_text("training: changed\n", encoding="utf-8")
    with pytest.raises(SemanticFlowSmokeMarkerError, match="training_config SHA256 changed"):
        validate_semantic_flow_smoke_marker(
            marker_path,
            code_commit=code_commit,
            training_config_path=training_config,
            accelerate_config_path=accelerate_config,
        )


def test_semantic_flow_runtime_lock_requires_exact_environment_and_fsdp_roundtrip(tmp_path: Path) -> None:
    report = {
        "ready": True,
        "python": {"runtime_version": "3.11.9"},
        "packages": {
            name: {"installed": True, "version": version}
            for name, version in {
                "torch": "2.7.1",
                "transformers": "4.53.1",
                "accelerate": "1.12.0",
                "safetensors": "0.5.3",
            }.items()
        },
        "torch_runtime": {"cuda_version": "12.8", "nccl_version": [2, 26, 2]},
        "gemma": {"sliding_window": 1024},
        "capabilities": {
            "accelerate_fsdp_plugin": {
                "passed": True,
                "fsdp_version": 1,
                "sharding_strategy": "FULL_SHARD",
                "state_dict_type": "FULL_STATE_DICT",
            }
        },
    }
    fsdp_result = {"world_size": 2, "max_abs_tensor_diff_after_reload": 0.0}
    current = build_semantic_flow_runtime_lock(report, fsdp_result)
    lock_path = tmp_path / "runtime_lock.json"
    assert write_or_validate_semantic_flow_runtime_lock(lock_path, current, refresh=True) == "refreshed"
    assert write_or_validate_semantic_flow_runtime_lock(lock_path, current, refresh=False) == "validated"

    changed = dict(current)
    changed["transformers"] = "different"
    with pytest.raises(SemanticFlowRuntimeLockError, match="Runtime differs from lock"):
        write_or_validate_semantic_flow_runtime_lock(lock_path, changed, refresh=False)

    with pytest.raises(SemanticFlowRuntimeLockError, match="exactly two processes"):
        build_semantic_flow_runtime_lock(report, {"world_size": 8, "max_abs_tensor_diff_after_reload": 0.0})


def test_strategy_checkpoint_state_uses_precollected_full_states() -> None:
    strategy = SemanticFlowStrategy(SemanticFlowConfig())
    modules = {
        "semantic_query": nn.Linear(2, 2, bias=False),
        "semantic_encoder": nn.Linear(2, 3, bias=False),
        "semantic_reconstruction_decoder": nn.Linear(3, 2, bias=False),
    }
    strategy.set_trainable_modules(modules)
    precollected = {
        name: {key: value.detach().clone() + 1.0 for key, value in module.state_dict().items()}
        for name, module in modules.items()
    }

    class _NoCollectAccelerator:
        def get_state_dict(self, module: nn.Module) -> dict[str, torch.Tensor]:
            del module
            raise AssertionError("strategy state should already be precollected")

    state = strategy.get_extra_checkpoint_state_dict(
        _NoCollectAccelerator(),
        precollected_states=precollected,
    )

    assert set(state) == {
        "training_strategy.semantic_query.weight",
        "training_strategy.semantic_encoder.weight",
        "training_strategy.semantic_reconstruction_decoder.weight",
    }
    assert torch.equal(state["training_strategy.semantic_query.weight"], precollected["semantic_query"]["weight"])


def test_semantic_strategy_load_rejects_missing_or_mismatched_modules() -> None:
    strategy = SemanticFlowStrategy(SemanticFlowConfig())
    modules = {
        "semantic_query": nn.Linear(2, 2, bias=False),
        "semantic_encoder": nn.Linear(2, 3, bias=False),
        "semantic_reconstruction_decoder": nn.Linear(3, 2, bias=False),
    }
    strategy.set_trainable_modules(modules)
    full_state = {
        f"training_strategy.{name}.{key}": value.detach().clone()
        for name, module in modules.items()
        for key, value in module.state_dict().items()
    }

    missing = dict(full_state)
    missing.pop("training_strategy.semantic_encoder.weight")
    with pytest.raises(RuntimeError, match="semantic_encoder"):
        strategy.load_extra_checkpoint_state_dict(missing)

    mismatched = dict(full_state)
    mismatched["training_strategy.semantic_query.weight"] = torch.ones(3, 3)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        strategy.load_extra_checkpoint_state_dict(mismatched)

    strategy.load_extra_checkpoint_state_dict(full_state)
