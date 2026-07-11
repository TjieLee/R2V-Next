import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn
from typer.testing import CliRunner

from ltx_trainer.training_strategies.multi_reference_video import (
    MultiReferenceVideoConfig,
    MultiReferenceVideoStrategy,
)


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "infer_multiref_stage1_overfit.py"
_SPEC = importlib.util.spec_from_file_location("infer_multiref_stage1_persistent", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
infer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = infer
_SPEC.loader.exec_module(infer)


class _Connector(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.inner_dim = dim
        self.num_learnable_registers = 1
        self.weight = nn.Parameter(torch.zeros(1, dim))


class _Processor(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.video_connector = _Connector(dim)
        self.feature_extractor = nn.Identity()

    def create_embeddings(self, video_features, audio_features, attention_mask):
        return video_features, audio_features, attention_mask


def _fake_config(strategy_config: MultiReferenceVideoConfig):
    return SimpleNamespace(
        model=SimpleNamespace(model_path="model", training_mode="full"),
        training_strategy=strategy_config,
        validation=SimpleNamespace(negative_prompt="negative"),
    )


def _checkpoint_flags() -> dict[str, bool]:
    return {
        "connector_checkpoint_loaded": False,
        "visual_token_projection_checkpoint_loaded": True,
        "visual_full_encoder_checkpoint_loaded": False,
    }


def test_batch_runtime_loads_models_only_once(monkeypatch, tmp_path) -> None:
    strategy_config = MultiReferenceVideoConfig(
        visual_branch_enabled=False,
        visual_token_source_dim=6,
        visual_token_target_dim=8,
    )
    strategy = MultiReferenceVideoStrategy(strategy_config)
    cfg = _fake_config(strategy_config)
    load_transformer = Mock(return_value=nn.Identity())
    load_processor = Mock(return_value=_Processor(8))
    load_vae = Mock(return_value=nn.Identity())
    load_checkpoint = Mock(return_value=_checkpoint_flags())
    monkeypatch.setattr(infer, "_load_config", Mock(return_value=cfg))
    monkeypatch.setattr(infer, "load_transformer", load_transformer)
    monkeypatch.setattr(infer, "load_embeddings_processor", load_processor)
    monkeypatch.setattr(infer, "load_video_vae_decoder", load_vae)
    monkeypatch.setattr(infer, "get_training_strategy", Mock(return_value=strategy))
    monkeypatch.setattr(infer, "_load_checkpoint_weights", load_checkpoint)

    runtime = infer._load_inference_runtime(
        config_path=tmp_path / "config.yaml",
        checkpoint_path=tmp_path / "checkpoint.safetensors",
        device=torch.device("cpu"),
        dtype=torch.float32,
        guidance_scale=1.0,
        negative_prompt=None,
    )
    processed: list[int] = []
    infer._run_selected_samples(
        selected_indices=[0, 1, 2],
        output_dir=tmp_path / "output",
        condition_mode="full_siglip",
        checkpoint_path=runtime.checkpoint_path,
        shard_index=0,
        num_shards=1,
        skip_existing=False,
        continue_on_error=False,
        gc_interval=20,
        device=torch.device("cpu"),
        run_sample=lambda index: processed.append(index) or tmp_path / f"{index}.mp4",
        write_summary=True,
    )

    assert processed == [0, 1, 2]
    assert load_transformer.call_count == 1
    assert load_processor.call_count == 1
    assert load_vae.call_count == 1
    assert load_checkpoint.call_count == 1


def test_negative_prompt_is_encoded_once(monkeypatch, tmp_path) -> None:
    strategy = MultiReferenceVideoStrategy(MultiReferenceVideoConfig(visual_branch_enabled=False))
    cfg = _fake_config(strategy.config)
    encode_negative = Mock(
        return_value={
            "video_prompt_embeds": torch.zeros(1, 2, 8),
            "audio_prompt_embeds": None,
            "prompt_attention_mask": None,
        }
    )
    monkeypatch.setattr(infer, "_load_config", Mock(return_value=cfg))
    monkeypatch.setattr(
        infer,
        "_load_models_and_strategy",
        Mock(return_value=(nn.Identity(), _Processor(8), nn.Identity(), strategy, _checkpoint_flags())),
    )
    monkeypatch.setattr(infer, "_encode_negative_prompt_condition", encode_negative)

    runtime = infer._load_inference_runtime(
        config_path=tmp_path / "config.yaml",
        checkpoint_path=tmp_path / "checkpoint.safetensors",
        device=torch.device("cpu"),
        dtype=torch.float32,
        guidance_scale=2.0,
        negative_prompt="fixed negative",
    )
    infer._run_selected_samples(
        selected_indices=[0, 1, 2],
        output_dir=tmp_path / "output",
        condition_mode="full_siglip",
        checkpoint_path=runtime.checkpoint_path,
        shard_index=0,
        num_shards=1,
        skip_existing=False,
        continue_on_error=False,
        gc_interval=20,
        device=torch.device("cpu"),
        run_sample=lambda index: tmp_path / f"{index}.mp4",
        write_summary=True,
    )

    assert encode_negative.call_count == 1
    assert runtime.negative_conditions is encode_negative.return_value


def test_text_only_precomputed_does_not_load_text_encoder(monkeypatch) -> None:
    load_text_encoder = Mock(side_effect=AssertionError("Gemma must not be loaded"))
    monkeypatch.setattr(infer, "load_text_encoder", load_text_encoder)
    processor = _Processor(8)
    strategy = SimpleNamespace(prepare_conditions=lambda batch, conditions: conditions)
    batch = {
        "conditions": {
            "video_prompt_embeds": torch.randn(1, 3, 8),
            "prompt_attention_mask": torch.ones(1, 3, dtype=torch.bool),
        }
    }

    output = infer._prepare_precomputed_text_condition(
        strategy=strategy,
        embeddings_processor=processor,
        batch=batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert output["video_prompt_embeds"].shape == (1, 3, 8)
    assert load_text_encoder.call_count == 0


def test_shard_selection_uses_global_manifest_indices() -> None:
    assert infer._select_sample_indices(
        row_count=10, start_index=0, end_index=None, shard_index=0, num_shards=3
    ) == [0, 3, 6, 9]
    assert infer._select_sample_indices(
        row_count=10, start_index=0, end_index=None, shard_index=1, num_shards=3
    ) == [1, 4, 7]
    assert infer._select_sample_indices(
        row_count=10, start_index=0, end_index=None, shard_index=2, num_shards=3
    ) == [2, 5, 8]


def test_skip_existing_does_not_run_sample(tmp_path) -> None:
    existing = infer._expected_generated_path(tmp_path, 4, "full_siglip")
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"video")
    run_sample = Mock()

    summary = infer._run_selected_samples(
        selected_indices=[4],
        output_dir=tmp_path,
        condition_mode="full_siglip",
        checkpoint_path=tmp_path / "checkpoint.safetensors",
        shard_index=0,
        num_shards=1,
        skip_existing=True,
        continue_on_error=True,
        gc_interval=20,
        device=torch.device("cpu"),
        run_sample=run_sample,
        write_summary=True,
    )

    assert run_sample.call_count == 0
    assert summary["num_skipped"] == 1


def test_continue_on_error_processes_later_samples_and_writes_failure(tmp_path) -> None:
    processed: list[int] = []

    def run_sample(index: int) -> Path:
        processed.append(index)
        if index == 1:
            raise RuntimeError("broken sample")
        return tmp_path / f"{index}.mp4"

    summary = infer._run_selected_samples(
        selected_indices=[0, 1, 2],
        output_dir=tmp_path,
        condition_mode="full_siglip",
        checkpoint_path=tmp_path / "checkpoint.safetensors",
        shard_index=2,
        num_shards=3,
        skip_existing=False,
        continue_on_error=True,
        gc_interval=20,
        device=torch.device("cpu"),
        run_sample=run_sample,
        write_summary=True,
    )

    failure = json.loads((tmp_path / "failures_shard_2.jsonl").read_text().strip())
    assert processed == [0, 1, 2]
    assert summary["num_success"] == 2
    assert summary["num_failed"] == 1
    assert failure["sample_index"] == 1
    assert failure["error_type"] == "RuntimeError"


def test_legacy_single_sample_cli_still_processes_one_sample(monkeypatch, tmp_path) -> None:
    rows = [{"video": "a.mp4"}, {"video": "b.mp4"}, {"video": "c.mp4"}]
    runtime = SimpleNamespace()
    processed: list[int] = []
    monkeypatch.setattr(infer, "_read_manifest", Mock(return_value=rows))
    monkeypatch.setattr(infer, "_load_inference_runtime", Mock(return_value=runtime))

    def run_one_sample(**kwargs) -> Path:
        index = kwargs["sample_index"]
        processed.append(index)
        path = infer._expected_generated_path(kwargs["output_dir"], index, kwargs["condition_mode"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
        return path

    monkeypatch.setattr(infer, "_run_one_sample", run_one_sample)
    result = CliRunner().invoke(
        infer.app,
        [
            "--config",
            "config.yaml",
            "--checkpoint",
            "checkpoint.safetensors",
            "--manifest",
            "manifest.json",
            "--precomputed-root",
            "precomputed",
            "--output-dir",
            str(tmp_path),
            "--sample-index",
            "1",
            "--no-skip-existing",
        ],
    )

    assert result.exit_code == 0, result.output
    assert processed == [1]


def test_output_paths_match_legacy_names(tmp_path) -> None:
    assert infer._expected_generated_path(tmp_path, 3, "full_siglip") == (
        tmp_path / "sample_3" / "generated_full_siglip.mp4"
    )
    assert infer._expected_generated_path(tmp_path, 3, "text_only_no_siglip") == (
        tmp_path / "sample_3" / "generated_text_only_no_siglip.mp4"
    )
