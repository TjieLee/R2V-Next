from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from PIL import Image
from transformers import Gemma3ImageProcessor

from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_trainer.online_data import multitask_dataset
from ltx_trainer.online_data.constants import IMAGE_TASK
from ltx_trainer.online_data.multitask_dataset import (
    OnlineMultiTaskDataset,
    SampleLoadError,
    validate_vlm_reference_tensor,
)
from ltx_trainer.online_data.online_batch_encoder import (
    OnlineBatchEncoder,
    OnlineSampleEncodeError,
)
from ltx_trainer.online_data.reference_audit import audit_online_manifest_references
from ltx_trainer.trainer import LtxvTrainer


def _record(reference_paths: list[str]) -> dict[str, Any]:
    return {
        "sample_key": "sample-key",
        "sample_plan_sha256": "plan",
        "task": "i2i",
        "target_modality": "image",
        "caption": "edit",
        "target_fps": 1.0,
        "target_num_frames": 1,
        "target_source_frame_indices": [0],
        "vlm_target_frame_indices": [0],
        "vlm_source_frame_indices": [0],
        "reference_paths": reference_paths,
    }


def _dataset(monkeypatch: pytest.MonkeyPatch, references: dict[str, torch.Tensor]):
    dataset = object.__new__(OnlineMultiTaskDataset)
    dataset.max_ref_images = 4
    dataset.width = 832
    dataset.height = 480
    dataset.cpu_transform_chunk_frames = 4
    dataset.vlm_reference_preprocess = "original"
    dataset.video_decoder = "pyav"
    dataset.return_load_errors = True
    dataset._read_record = lambda _index: _record(list(references))  # type: ignore[method-assign]
    dataset._load_target = lambda _record: torch.zeros(  # type: ignore[method-assign]
        1, 480, 832, 3, dtype=torch.uint8
    )
    monkeypatch.setattr(
        multitask_dataset,
        "decode_image_rgb",
        lambda path: references[str(path)],
    )
    monkeypatch.setattr(
        multitask_dataset,
        "deterministic_resize_center_crop",
        lambda frames, **_kwargs: torch.zeros(1, 480, 832, 3, dtype=torch.uint8),
    )
    return dataset


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 20, 3), (20, 1, 3)])
def test_dataset_rejects_degenerate_original_vlm_reference(
    shape: tuple[int, int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "/readonly/degenerate.png"
    dataset = _dataset(monkeypatch, {path: torch.zeros(shape, dtype=torch.uint8)})
    sample = dataset[7]
    assert isinstance(sample, SampleLoadError)
    assert sample.reason == "degenerate_vlm_reference_geometry"
    assert sample.manifest_index == 7
    assert sample.sample_key == "sample-key"
    assert sample.task == "i2i"
    assert sample.reference_index == 0
    assert sample.reference_path == path
    assert path in sample.message
    assert f"shape={shape}" in sample.message
    assert "dtype=torch.uint8" in sample.message


def test_two_by_two_reference_passes_and_order_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ["/readonly/first.png", "/readonly/second.png"]
    references = {
        paths[0]: torch.full((2, 2, 3), 1, dtype=torch.uint8),
        paths[1]: torch.full((20, 3, 3), 2, dtype=torch.uint8),
    }
    sample = _dataset(monkeypatch, references)[0]
    assert isinstance(sample, dict)
    assert sample["reference_paths"] == paths
    assert [int(image[0, 0, 0]) for image in sample["reference_images_vlm"]] == [1, 2]


def test_vlm_reference_tensor_contract_rejects_non_rgb_uint8() -> None:
    with pytest.raises(ValueError, match="invalid_vlm_reference_tensor.*dtype=torch.float32"):
        validate_vlm_reference_tensor(
            torch.zeros(2, 2, 3),
            path="/readonly/float.png",
            reference_index=0,
            manifest_index=4,
            sample_key="sample",
            task="r2v",
        )


class _Tokenizer:
    all_special_ids: list[int] = []

    @staticmethod
    def apply_chat_template(*_args: Any, **_kwargs: Any) -> str:
        return "serialized chat"


class _ReferenceProcessor:
    image_seq_length = 256

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        references = kwargs.get("images") or []
        ids: list[int] = [17]
        for _ in references:
            ids.extend(
                [
                    GEMMA3_CONFIG_FOR_LTX.boi_token_index,
                    *([GEMMA3_CONFIG_FOR_LTX.image_token_index] * self.image_seq_length),
                    GEMMA3_CONFIG_FOR_LTX.eoi_token_index,
                ]
            )
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def _encoder(processor: _ReferenceProcessor) -> OnlineBatchEncoder:
    encoder = object.__new__(OnlineBatchEncoder)
    encoder.processor = processor
    encoder.tokenizer = _Tokenizer()
    encoder.system_prompts = {"i2i": "system", "r2v": "system"}
    return encoder


def test_thin_reference_images_are_explicitly_channels_last() -> None:
    processor = _ReferenceProcessor()
    encoder = _encoder(processor)
    images = [Image.new("RGB", (3, 20)), Image.new("RGB", (20, 3))]
    _processed, mask = encoder._process_multimodal_source(
        caption="edit",
        reference_images=images,
        task=IMAGE_TASK,
        source_max_length=2048,
    )
    assert processor.calls[0]["input_data_format"] == "channels_last"
    assert int(mask.sum()) == 2 * (processor.image_seq_length + 2)


@pytest.mark.parametrize("size", [(1, 1), (3, 20), (20, 3), (32, 32)])
def test_transformers_gemma3_image_processor_accepts_explicit_channels_last(
    size: tuple[int, int],
) -> None:
    processor = Gemma3ImageProcessor()
    result = processor(
        images=[Image.new("RGB", size)],
        input_data_format="channels_last",
        return_tensors="pt",
    )
    assert result["pixel_values"].shape[0] >= 1


def test_reference_processor_value_error_is_retryable_and_redacted() -> None:
    processor = _ReferenceProcessor(ValueError("mean mismatch"))
    encoder = _encoder(processor)
    with pytest.raises(OnlineSampleEncodeError) as exc_info:
        encoder._process_multimodal_source(
            caption="caption must not appear in the error",
            reference_images=[Image.new("RGB", (3, 20))],
            task=IMAGE_TASK,
            source_max_length=2048,
        )
    error = exc_info.value
    assert error.reason == "gemma_reference_processor_failure"
    assert "image_count=1" in str(error)
    assert "sizes=[(3, 20)]" in str(error)
    assert "modes=['RGB']" in str(error)
    assert "caption must not appear" not in str(error)


def test_text_only_processor_value_error_and_reference_runtime_error_fail_fast() -> None:
    value_encoder = _encoder(_ReferenceProcessor(ValueError("text bug")))
    with pytest.raises(ValueError, match="text bug"):
        value_encoder._process_multimodal_source(
            caption="text",
            reference_images=[],
            task=IMAGE_TASK,
            source_max_length=2048,
        )
    runtime_encoder = _encoder(_ReferenceProcessor(RuntimeError("cuda failure")))
    with pytest.raises(RuntimeError, match="cuda failure"):
        runtime_encoder._process_multimodal_source(
            caption="image",
            reference_images=[Image.new("RGB", (20, 3))],
            task=IMAGE_TASK,
            source_max_length=2048,
        )


def test_retryable_processor_error_attaches_sample_context() -> None:
    error = OnlineSampleEncodeError(
        "processor failed",
        reason="gemma_reference_processor_failure",
    )
    error.attach_sample_context(
        {
            "manifest_index": torch.tensor([12]),
            "sample_key": ["sample-12"],
            "task": ["r2v"],
            "reference_paths": [["/readonly/ref.png"]],
        }
    )
    payload = error.to_dict()
    assert payload["manifest_index"] == 12
    assert payload["sample_key"] == "sample-12"
    assert payload["task"] == "r2v"
    assert payload["reference_path"] == "/readonly/ref.png"


def test_reject_log_contains_rank_reason_and_sample_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ltx_trainer.online_data.path_safety.assert_write_path_allowed",
        lambda path: Path(path),
    )
    trainer = object.__new__(LtxvTrainer)
    trainer._config = SimpleNamespace(
        data=SimpleNamespace(
            online_encoding=SimpleNamespace(runtime_reject_log_dir=str(tmp_path))
        ),
        output_dir=str(tmp_path / "output"),
    )
    trainer._accelerator = SimpleNamespace(process_index=3)
    trainer._global_step = 15000
    error = SampleLoadError(
        sample_key="bad-sample",
        task="r2v",
        error_type="VLMReferenceValidationError",
        message="degenerate_vlm_reference_geometry: reference_path=/readonly/one.png",
        manifest_index=99,
        reason="degenerate_vlm_reference_geometry",
        reference_index=2,
        reference_path="/readonly/one.png",
    )
    trainer._log_online_reject(error, attempt=1, phase="decode")
    row = json.loads((tmp_path / "runtime_rejected_rank_3.jsonl").read_text())
    assert row["global_step"] == 15000
    assert row["attempt"] == 1
    assert row["phase"] == "decode"
    assert row["rank"] == 3
    assert row["manifest_index"] == 99
    assert row["reference_path"] == "/readonly/one.png"
    assert row["reason"] == "degenerate_vlm_reference_geometry"


def test_reference_audit_deduplicates_decode_and_keeps_all_occurrences(tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (1, 1), "red").save(tiny)
    missing = tmp_path / "missing.png"
    records = [
        {
            "sample_key": "first",
            "task": "i2i",
            "reference_paths": [str(tiny), str(missing)],
        },
        {"sample_key": "second", "task": "r2v", "reference_paths": [str(tiny)]},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    rows, summary = audit_online_manifest_references(manifest, workers=2)
    assert summary["record_count"] == 2
    assert summary["reference_occurrence_count"] == 3
    assert summary["unique_reference_count"] == 2
    assert summary["duplicate_reference_occurrence_count"] == 1
    assert summary["degenerate_unique_reference_count"] == 1
    tiny_rows = [row for row in rows if row["reference_path"] == str(tiny)]
    assert [row["manifest_index"] for row in tiny_rows] == [0, 1]
    assert all(row["shape"] == [1, 1, 3] for row in tiny_rows)
    assert all(row["dtype"] == "torch.uint8" for row in tiny_rows)


def test_resume_refguard_config_is_exact_resume() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "multiref_stage3_resume_step15000_refguard_30k.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["model"]["load_checkpoint"].endswith(
        "stage3_step_15000/lora_weights_step_15000.safetensors"
    )
    assert config["checkpoints"]["no_resume"] is False
    assert config["checkpoints"]["save_training_state"] == "full"
    assert config["checkpoints"]["allow_warm_resume_without_optimizer"] is False
    assert config["optimization"]["steps"] == 30000
    assert config["data"]["online_encoding"]["vlm_reference_preprocess"] == "original"
