from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from ltx_trainer.online_inference.checkpoint_runtime import audit_checkpoint, resolve_checkpoint
from ltx_trainer.online_inference.output_artifacts import (
    atomic_save_png,
    atomic_save_video,
    atomic_write_json,
    output_is_complete,
    save_contact_sheet,
)


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


def test_checkpoint_audit_requires_all_six_components(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    audit = audit_checkpoint(checkpoint)
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


@pytest.mark.parametrize(
    ("removed_prefix", "missing_name"),
    [
        ("training_strategy.planner_tokens.", "planner"),
        ("embeddings_processor.video_connector.", "connector"),
        ("text_encoder.model.model.language_model.", "gemma_lora"),
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
    with pytest.raises(RuntimeError, match=missing_name):
        audit_checkpoint(checkpoint)


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
