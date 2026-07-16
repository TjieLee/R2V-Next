from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn

from ltx_trainer.online_inference.checkpoint_runtime import (
    CheckpointAuditError,
    CheckpointComponentSpec,
    assert_checkpoint_unchanged,
    audit_checkpoint,
    checkpoint_snapshot,
    resolve_checkpoint,
)
from ltx_trainer.online_inference.output_artifacts import (
    atomic_save_png,
    atomic_save_video,
    atomic_write_json,
    output_is_complete,
    sample_output_dir,
    save_contact_sheet,
)
from ltx_trainer.online_inference.vae_decode import decode_video_latents


def _checkpoint(tmp_path: Path, step: int = 25) -> Path:
    path = tmp_path / f"lora_weights_step_{step:05d}.safetensors"
    tensors = {
        "diffusion_model.block.lora_A.weight": torch.ones(1),
        "text_encoder.model.model.language_model.block.lora_A.weight": torch.ones(1),
        "training_strategy.planner_tokens.weight": torch.ones(1),
        "training_strategy.visual_token_projection.weight": torch.ones(1),
        "training_strategy.visual_full_encoder.weight": torch.ones(1),
        "embeddings_processor.video_connector.weight": torch.ones(1),
    }
    save_file(
        tensors,
        path,
        metadata={"global_step": str(step), "training_phase": "stage3"},
    )
    return path


def _component_specs() -> dict[str, CheckpointComponentSpec]:
    return {
        "dit_lora": CheckpointComponentSpec(
            prefix="diffusion_model.",
            expected_state={"block.lora_A.weight": torch.ones(1)},
        ),
        "gemma_lora": CheckpointComponentSpec(
            prefix="text_encoder.model.model.language_model.",
            expected_state={"block.lora_A.weight": torch.ones(1)},
        ),
        "planner": CheckpointComponentSpec(
            prefix="training_strategy.planner_tokens.",
            expected_state={"weight": torch.ones(1)},
        ),
        "visual_projection": CheckpointComponentSpec(
            prefix="training_strategy.visual_token_projection.",
            expected_state={"weight": torch.ones(1)},
        ),
        "visual_full_encoder": CheckpointComponentSpec(
            prefix="training_strategy.visual_full_encoder.",
            expected_state={"weight": torch.ones(1)},
        ),
        "connector": CheckpointComponentSpec(
            prefix="embeddings_processor.video_connector.",
            expected_state={"weight": torch.ones(1)},
        ),
    }


def test_checkpoint_audit_requires_all_six_components(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    audit = audit_checkpoint(checkpoint, component_specs=_component_specs())
    assert audit["checkpoint_step"] == 25
    assert set(audit["component_key_counts"]) == {
        "dit_lora",
        "gemma_lora",
        "planner",
        "visual_projection",
        "visual_full_encoder",
        "connector",
    }
    assert all(count == 1 for count in audit["component_key_counts"].values())
    assert audit["required_missing_keys"] == []
    assert audit["unexpected_checkpoint_keys"] == []
    assert audit["ignored_base_model_missing_keys"]
    assert not any(
        key in audit["required_missing_keys"]
        for key in audit["ignored_base_model_missing_keys"]
    )


@pytest.mark.parametrize(
    ("removed_prefix", "missing_name"),
    [
        ("training_strategy.planner_tokens.", "training_strategy.planner_tokens"),
        ("embeddings_processor.video_connector.", "embeddings_processor.video_connector"),
        ("text_encoder.model.model.language_model.", "text_encoder.model.model.language_model"),
    ],
)
def test_checkpoint_audit_rejects_missing_required_component(
    tmp_path: Path,
    removed_prefix: str,
    missing_name: str,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    tensors = {
        key: value
        for key, value in load_file(checkpoint).items()
        if not key.startswith(removed_prefix)
    }
    checkpoint.unlink()
    save_file(
        tensors,
        checkpoint,
        metadata={"global_step": "25", "training_phase": "stage3"},
    )
    with pytest.raises(CheckpointAuditError, match="required_missing") as exc_info:
        audit_checkpoint(checkpoint, component_specs=_component_specs())
    assert any(missing_name in key for key in exc_info.value.audit["required_missing_keys"])


def test_latest_ready_resolution_consumes_only_complete_marker(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, step=30)
    state = tmp_path / "training_state_step_00030.pt"
    state.write_bytes(b"state")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    marker = tmp_path / "checkpoint_step_00030.ready.json"
    marker.write_text(
        json.dumps(
            {
                "global_step": 30,
                "checkpoint_path": str(checkpoint),
                "training_state_path": str(state),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": digest,
                "metadata_training_phase": "stage3",
                "metadata_global_step": "30",
            }
        ),
        encoding="utf-8",
    )
    resolved, marker_path, payload = resolve_checkpoint(
        checkpoint=None,
        latest_ready_dir=tmp_path,
    )
    assert resolved == checkpoint.resolve()
    assert marker_path == marker
    assert payload["checkpoint_sha256"] == digest


def test_latest_ready_resolution_does_not_fall_back_from_incomplete_latest_marker(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path, step=30)
    state = tmp_path / "training_state_step_00030.pt"
    state.write_bytes(b"state")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (tmp_path / "checkpoint_step_00030.ready.json").write_text(
        json.dumps(
            {
                "global_step": 30,
                "checkpoint_path": str(checkpoint),
                "training_state_path": str(state),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": digest,
                "metadata_training_phase": "stage3",
                "metadata_global_step": "30",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "checkpoint_step_00040.ready.json").write_text(
        json.dumps(
            {
                "global_step": 40,
                "checkpoint_path": str(tmp_path / "missing_step_00040.safetensors"),
                "training_state_path": str(tmp_path / "missing_state_00040.pt"),
                "checkpoint_size_bytes": 1,
                "checkpoint_sha256": "missing",
                "metadata_training_phase": "stage3",
                "metadata_global_step": "40",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="incomplete publication"):
        resolve_checkpoint(checkpoint=None, latest_ready_dir=tmp_path)


def test_atomic_i2i_and_r2v_outputs_publish_only_complete_files(tmp_path: Path) -> None:
    image_path = tmp_path / "generated.png"
    atomic_save_png(Image.new("RGB", (832, 480), "red"), image_path)

    video = torch.zeros(5, 3, 480, 832)
    video_path = tmp_path / "generated.mp4"

    def fake_save_video(*, output_path, **kwargs):
        Path(output_path).write_bytes(b"fake-video")

    atomic_save_video(video, video_path, fps=24.0, save_video=fake_save_video)
    save_contact_sheet(video, tmp_path / "contact_sheet.png")
    atomic_write_json(
        tmp_path / "success.json",
        {
            "status": "success",
            "artifacts": ["generated.png", "generated.mp4", "contact_sheet.png"],
        },
    )
    assert image_path.is_file()
    assert video_path.read_bytes() == b"fake-video"
    assert output_is_complete(tmp_path)


def test_checkpoint_audit_rejects_partial_component(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    tensors = load_file(checkpoint)
    tensors["training_strategy.planner_tokens.bias"] = torch.ones(2)
    save_file(
        tensors,
        checkpoint,
        metadata={"global_step": "25", "training_phase": "stage3"},
    )
    specs = _component_specs()
    specs["planner"] = CheckpointComponentSpec(
        prefix="training_strategy.planner_tokens.",
        expected_state={"weight": torch.ones(1), "bias": torch.ones(2)},
    )
    tensors = load_file(checkpoint)
    tensors.pop("training_strategy.planner_tokens.bias")
    save_file(
        tensors,
        checkpoint,
        metadata={"global_step": "25", "training_phase": "stage3"},
    )
    with pytest.raises(CheckpointAuditError) as exc_info:
        audit_checkpoint(checkpoint, component_specs=specs)
    assert exc_info.value.audit["required_missing_keys"] == [
        "training_strategy.planner_tokens.bias"
    ]


@pytest.mark.parametrize(
    ("component", "missing_key"),
    [
        ("dit_lora", "second.lora_B.weight"),
        ("gemma_lora", "second.lora_B.weight"),
        ("planner", "second.weight"),
        ("visual_projection", "second.weight"),
        ("visual_full_encoder", "second.weight"),
        ("connector", "second.weight"),
    ],
)
def test_checkpoint_audit_rejects_one_missing_key_from_each_component(
    tmp_path: Path,
    component: str,
    missing_key: str,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    specs = _component_specs()
    original = specs[component]
    specs[component] = CheckpointComponentSpec(
        prefix=original.prefix,
        expected_state={**original.expected_state, missing_key: torch.ones(2)},
        key_normalizer=original.key_normalizer,
    )
    with pytest.raises(CheckpointAuditError) as exc_info:
        audit_checkpoint(checkpoint, component_specs=specs)
    assert f"{original.prefix}{missing_key}" in exc_info.value.audit[
        "required_missing_keys"
    ]


def test_checkpoint_audit_rejects_unexpected_checkpoint_owned_key(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    tensors = load_file(checkpoint)
    tensors["training_strategy.visual_full_encoder.unexpected"] = torch.ones(1)
    save_file(
        tensors,
        checkpoint,
        metadata={"global_step": "25", "training_phase": "stage3"},
    )
    with pytest.raises(CheckpointAuditError) as exc_info:
        audit_checkpoint(checkpoint, component_specs=_component_specs())
    assert exc_info.value.audit["unexpected_checkpoint_keys"] == [
        "training_strategy.visual_full_encoder.unexpected"
    ]


def test_checkpoint_modification_during_load_is_detected(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    snapshot = checkpoint_snapshot(checkpoint)
    tensors = load_file(checkpoint)
    tensors["training_strategy.planner_tokens.weight"] = torch.full((1,), 2.0)
    save_file(
        tensors,
        checkpoint,
        metadata={"global_step": "25", "training_phase": "stage3"},
    )
    with pytest.raises(RuntimeError, match="changed while inference runtime was loading"):
        assert_checkpoint_unchanged(checkpoint, snapshot)


def test_checkpoint_filename_and_metadata_step_mismatch_fails(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, step=30)
    tensors = load_file(checkpoint)
    save_file(
        tensors,
        checkpoint,
        metadata={"global_step": "31", "training_phase": "stage3"},
    )
    with pytest.raises(RuntimeError, match="filename=30, metadata=31"):
        audit_checkpoint(checkpoint, component_specs=_component_specs())


@pytest.mark.parametrize(
    ("first_key", "second_key"),
    [
        ("abc/def", "abc_def"),
        ("x" * 140 + "A", "x" * 140 + "B"),
    ],
)
def test_sample_output_directories_do_not_collide(
    tmp_path: Path,
    first_key: str,
    second_key: str,
) -> None:
    first = sample_output_dir(tmp_path, {"task": "i2i", "sample_key": first_key})
    second = sample_output_dir(tmp_path, {"task": "i2i", "sample_key": second_key})
    assert first != second
    assert first == sample_output_dir(
        tmp_path,
        {"task": "i2i", "sample_key": first_key},
    )


class _FakeDecoder(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones((), dtype=dtype))
        self.seen_dtypes: list[torch.dtype] = []

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        self.seen_dtypes.append(latents.dtype)
        return torch.zeros(
            latents.shape[0],
            3,
            latents.shape[2],
            latents.shape[3],
            latents.shape[4],
            device=latents.device,
            dtype=latents.dtype,
        )

    def tiled_decode(self, latents: torch.Tensor, *, tiling_config: object) -> Iterator[torch.Tensor]:
        del tiling_config
        yield self(latents)


class _WrappedDecoder(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module


@pytest.mark.parametrize(
    ("decoder_dtype", "latent_dtype", "decode_tile", "wrapped"),
    [
        (torch.bfloat16, torch.bfloat16, False, False),
        (torch.float16, torch.float32, True, False),
        (torch.float32, torch.bfloat16, False, False),
        (torch.float32, torch.float16, True, True),
    ],
)
def test_vae_decode_aligns_input_to_decoder_compute_dtype(
    decoder_dtype: torch.dtype,
    latent_dtype: torch.dtype,
    decode_tile: bool,
    wrapped: bool,
) -> None:
    decoder = _FakeDecoder(decoder_dtype)
    module: nn.Module = _WrappedDecoder(decoder) if wrapped else decoder
    output, diagnostics = decode_video_latents(
        vae_decoder=module,
        latents=torch.zeros(1, 128, 1, 2, 3, dtype=latent_dtype),
        decode_tile=decode_tile,
    )
    assert decoder.seen_dtypes == [decoder_dtype]
    assert diagnostics.decoder_weight_dtype == str(decoder_dtype)
    assert diagnostics.decoder_input_dtype == str(decoder_dtype)
    assert diagnostics.decoder_output_dtype == str(decoder_dtype)
    assert output.shape == (1, 3, 2, 3)
