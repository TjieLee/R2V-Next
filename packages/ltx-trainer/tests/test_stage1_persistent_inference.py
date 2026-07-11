import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
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


def _write_complete_sample(output_dir: Path, sample_index: int, condition_mode: str) -> Path:
    video_path = infer._expected_generated_path(output_dir, sample_index, condition_mode)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"video")
    infer._metadata_path_for_video(video_path).write_text(
        json.dumps(
            {
                "sample_index": sample_index,
                "condition_mode": condition_mode,
                "generated": str(video_path),
            }
        ),
        encoding="utf-8",
    )
    return video_path


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
        "_load_transformer_processor_and_strategy",
        Mock(return_value=(nn.Identity(), _Processor(8), strategy, _checkpoint_flags())),
    )
    monkeypatch.setattr(infer, "load_video_vae_decoder", Mock(return_value=nn.Identity()))
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
    _write_complete_sample(tmp_path, 4, "full_siglip")
    run_sample = Mock()

    assert infer._sample_is_complete(
        output_dir=tmp_path,
        sample_index=4,
        condition_mode="full_siglip",
    )

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


def test_sample_with_only_nonempty_mp4_is_not_complete_and_is_rerun(tmp_path) -> None:
    video_path = infer._expected_generated_path(tmp_path, 4, "full_siglip")
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video")
    run_sample = Mock(return_value=video_path)

    assert not infer._sample_is_complete(
        output_dir=tmp_path,
        sample_index=4,
        condition_mode="full_siglip",
    )
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

    assert run_sample.call_count == 1
    assert summary["num_success"] == 1


def test_corrupt_or_mismatched_metadata_is_not_complete(tmp_path) -> None:
    video_path = infer._expected_generated_path(tmp_path, 7, "full_siglip")
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video")
    metadata_path = infer._metadata_path_for_video(video_path)
    metadata_path.write_text("not-json", encoding="utf-8")
    assert not infer._sample_is_complete(
        output_dir=tmp_path, sample_index=7, condition_mode="full_siglip"
    )

    valid = {
        "sample_index": 7,
        "condition_mode": "full_siglip",
        "generated": str(video_path),
    }
    for key, wrong_value in (
        ("sample_index", 8),
        ("condition_mode", "text_only_no_siglip"),
        ("generated", str(video_path.with_name("wrong.mp4"))),
    ):
        metadata = dict(valid)
        metadata[key] = wrong_value
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        assert not infer._sample_is_complete(
            output_dir=tmp_path, sample_index=7, condition_mode="full_siglip"
        )


def test_temporary_outputs_are_not_complete(tmp_path) -> None:
    final_video = infer._expected_generated_path(tmp_path, 2, "full_siglip")
    final_video.parent.mkdir(parents=True)
    infer._temporary_output_path(final_video).write_bytes(b"partial")
    infer._temporary_metadata_path(final_video).write_text("{}", encoding="utf-8")

    assert not infer._sample_is_complete(
        output_dir=tmp_path, sample_index=2, condition_mode="full_siglip"
    )


def test_atomic_sample_output_commit(monkeypatch, tmp_path) -> None:
    final_video = infer._expected_generated_path(tmp_path, 3, "full_siglip")
    metadata = {
        "sample_index": 3,
        "condition_mode": "full_siglip",
        "generated": str(final_video),
    }

    def save_video(*, output_path, **kwargs) -> None:
        del kwargs
        output_path.write_bytes(b"video")

    monkeypatch.setattr(infer, "save_video", save_video)
    infer._save_sample_outputs_atomically(
        video_tensor=torch.zeros(1),
        final_video=final_video,
        metadata=metadata,
        fps=24.0,
    )

    assert final_video.is_file()
    assert infer._metadata_path_for_video(final_video).is_file()
    assert not infer._temporary_output_path(final_video).exists()
    assert not infer._temporary_metadata_path(final_video).exists()
    saved_metadata = json.loads(infer._metadata_path_for_video(final_video).read_text())
    assert saved_metadata["generated"] == str(final_video)


def test_interrupted_atomic_commit_is_not_complete(monkeypatch, tmp_path) -> None:
    final_video = infer._expected_generated_path(tmp_path, 3, "full_siglip")
    metadata = {
        "sample_index": 3,
        "condition_mode": "full_siglip",
        "generated": str(final_video),
    }
    original_replace = Path.replace

    def save_video(*, output_path, **kwargs) -> None:
        del kwargs
        output_path.write_bytes(b"video")

    def fail_metadata_replace(path: Path, target: Path):
        if path.name == ".metadata.tmp.json":
            raise OSError("metadata replace interrupted")
        return original_replace(path, target)

    monkeypatch.setattr(infer, "save_video", save_video)
    monkeypatch.setattr(Path, "replace", fail_metadata_replace)
    with pytest.raises(OSError, match="metadata replace interrupted"):
        infer._save_sample_outputs_atomically(
            video_tensor=torch.zeros(1),
            final_video=final_video,
            metadata=metadata,
            fps=24.0,
        )

    assert not infer._sample_is_complete(
        output_dir=tmp_path, sample_index=3, condition_mode="full_siglip"
    )


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


def test_failure_file_is_reset_for_each_worker_run(tmp_path) -> None:
    failures_path = tmp_path / "failures_shard_0.jsonl"
    failures_path.write_text('{"error":"old failure"}\n', encoding="utf-8")
    infer._run_selected_samples(
        selected_indices=[0],
        output_dir=tmp_path,
        condition_mode="full_siglip",
        checkpoint_path=tmp_path / "checkpoint.safetensors",
        shard_index=0,
        num_shards=1,
        skip_existing=False,
        continue_on_error=True,
        gc_interval=20,
        device=torch.device("cpu"),
        run_sample=lambda index: tmp_path / f"{index}.mp4",
        write_summary=True,
    )
    assert not failures_path.exists()

    failures_path.write_text('{"error":"old failure"}\n', encoding="utf-8")

    def fail_current_sample(index: int) -> Path:
        raise RuntimeError(f"current failure {index}")

    infer._run_selected_samples(
        selected_indices=[5],
        output_dir=tmp_path,
        condition_mode="full_siglip",
        checkpoint_path=tmp_path / "checkpoint.safetensors",
        shard_index=0,
        num_shards=1,
        skip_existing=False,
        continue_on_error=True,
        gc_interval=20,
        device=torch.device("cpu"),
        run_sample=fail_current_sample,
        write_summary=True,
    )
    current_failures = failures_path.read_text(encoding="utf-8")
    assert "old failure" not in current_failures
    assert "current failure 5" in current_failures


def test_negative_prompt_is_encoded_before_vae_load(monkeypatch, tmp_path) -> None:
    strategy = MultiReferenceVideoStrategy(MultiReferenceVideoConfig(visual_branch_enabled=False))
    cfg = _fake_config(strategy.config)
    events: list[str] = []
    monkeypatch.setattr(infer, "_load_config", Mock(return_value=cfg))
    monkeypatch.setattr(
        infer,
        "_load_transformer_processor_and_strategy",
        Mock(return_value=(nn.Identity(), _Processor(8), strategy, _checkpoint_flags())),
    )

    def encode_negative(**kwargs):
        del kwargs
        events.append("encode_negative")
        return {
            "video_prompt_embeds": torch.zeros(1, 2, 8),
            "audio_prompt_embeds": None,
            "prompt_attention_mask": None,
        }

    def load_vae(*args, **kwargs):
        del args, kwargs
        events.append("load_vae")
        return nn.Identity()

    monkeypatch.setattr(infer, "_encode_negative_prompt_condition", encode_negative)
    monkeypatch.setattr(infer, "load_video_vae_decoder", load_vae)
    infer._load_inference_runtime(
        config_path=tmp_path / "config.yaml",
        checkpoint_path=tmp_path / "checkpoint.safetensors",
        device=torch.device("cpu"),
        dtype=torch.float32,
        guidance_scale=2.0,
        negative_prompt="negative",
    )

    assert events == ["encode_negative", "load_vae"]


def _batch_cli_args(tmp_path: Path) -> list[str]:
    return [
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
        "--all-samples",
        "--no-skip-existing",
        "--continue-on-error",
    ]


def test_batch_cli_returns_two_after_processing_all_samples_with_failures(monkeypatch, tmp_path) -> None:
    rows = [{"video": "a.mp4"}, {"video": "b.mp4"}, {"video": "c.mp4"}]
    processed: list[int] = []
    monkeypatch.setattr(infer, "_read_manifest", Mock(return_value=rows))
    monkeypatch.setattr(infer, "_load_inference_runtime", Mock(return_value=SimpleNamespace()))

    def run_one_sample(**kwargs) -> Path:
        index = kwargs["sample_index"]
        processed.append(index)
        if index == 1:
            raise RuntimeError("broken sample")
        return infer._expected_generated_path(kwargs["output_dir"], index, kwargs["condition_mode"])

    monkeypatch.setattr(infer, "_run_one_sample", run_one_sample)
    result = CliRunner().invoke(infer.app, _batch_cli_args(tmp_path))
    summary = json.loads((tmp_path / "batch_summary_shard_0.json").read_text(encoding="utf-8"))

    assert processed == [0, 1, 2]
    assert summary["num_failed"] == 1
    assert summary["num_success"] == 2
    assert result.exit_code == 2


def test_batch_cli_returns_zero_when_all_samples_succeed(monkeypatch, tmp_path) -> None:
    rows = [{"video": "a.mp4"}, {"video": "b.mp4"}]
    processed: list[int] = []
    monkeypatch.setattr(infer, "_read_manifest", Mock(return_value=rows))
    monkeypatch.setattr(infer, "_load_inference_runtime", Mock(return_value=SimpleNamespace()))

    def run_one_sample(**kwargs) -> Path:
        index = kwargs["sample_index"]
        processed.append(index)
        return infer._expected_generated_path(kwargs["output_dir"], index, kwargs["condition_mode"])

    monkeypatch.setattr(infer, "_run_one_sample", run_one_sample)
    result = CliRunner().invoke(infer.app, _batch_cli_args(tmp_path))

    assert processed == [0, 1]
    assert result.exit_code == 0, result.output


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
