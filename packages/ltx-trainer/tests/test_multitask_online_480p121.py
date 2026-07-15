from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ltx_trainer.config import DataConfig, OnlineEncodingConfig
from ltx_trainer.online_data.constants import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VIDEO_NUM_FRAMES,
    VLM_TARGET_INDICES,
    uniform_indices,
)
from ltx_trainer.online_data.manifest import ManifestReject, build_i2i_record, build_r2v_record
from ltx_trainer.online_data.path_safety import assert_write_path_allowed
from ltx_trainer.online_data.visual_token_packing import build_visual_metadata, pack_visual_tokens


def test_fixed_480p121_contract() -> None:
    assert TARGET_WIDTH % 32 == 0
    assert TARGET_HEIGHT % 32 == 0
    assert VIDEO_NUM_FRAMES % 8 == 1
    assert uniform_indices(121, 8) == list(VLM_TARGET_INDICES)
    OnlineEncodingConfig()


def test_image_visual_tokens_forward_one_frame_then_zero_pad() -> None:
    real = torch.ones(1, 256, 3840)
    packed, mask = pack_visual_tokens(real, valid_frames=1)
    assert packed.shape == (1, 2048, 3840)
    assert mask.sum().item() == 256
    torch.testing.assert_close(packed[:, :256], real)
    assert torch.count_nonzero(packed[:, 256:]).item() == 0

    metadata = build_visual_metadata(
        batch_size=1,
        valid_frames=1,
        target_num_frames=1,
        target_fps=1.0,
        sampled_frame_indices=torch.zeros(1, 8, dtype=torch.long),
        device="cpu",
    )
    assert metadata["num_valid_vlm_frames"].item() == 1
    assert metadata["sampled_frame_mask"].tolist() == [[True, False, False, False, False, False, False, False]]


def test_video_visual_tokens_are_all_valid() -> None:
    real = torch.ones(1, 2048, 3840)
    packed, mask = pack_visual_tokens(real, valid_frames=8)
    torch.testing.assert_close(packed, real)
    assert mask.all()


def test_i2i_manifest_does_not_apply_video_frame_filter(tmp_path: Path) -> None:
    record = build_i2i_record(
        {"target": "target.png", "sources": ["source.png"], "instruction": "move it"},
        dataset_name="i2i",
        data_root=tmp_path,
        target_field="target",
        reference_field="sources",
        caption_field="instruction",
    )
    assert record["target_modality"] == "image"
    assert record["target_num_frames"] == 1
    assert record["target_fps"] == 1.0
    assert record["target_source_frame_indices"] == [0]


def test_i2i_schema_mismatch_lists_available_columns() -> None:
    with pytest.raises(ManifestReject, match="Available columns"):
        build_i2i_record(
            {"unexpected": "value"},
            dataset_name="i2i",
            data_root=None,
            target_field="target",
            reference_field="sources",
            caption_field="instruction",
        )


def test_short_r2v_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ltx_trainer.online_data.manifest.probe_video",
        lambda _path: {"fps": 24.0, "frame_count": 100, "width": 832, "height": 480},
    )
    row = {
        "video_path": "/tmp/video.mp4",
        "text": "prompt",
        "crop": [0, 832, 0, 480],
        "face_cut": [0, 100],
        "ref_images": ["/tmp/ref.png"],
    }
    with pytest.raises(ManifestReject, match="121-frame") as exc_info:
        build_r2v_record(row, dataset_name="r2v", data_root=None, manifest_seed=42)
    assert exc_info.value.reason == "insufficient_frames_for_121_at_24fps"


def test_legacy_data_config_defaults_to_precomputed(tmp_path: Path) -> None:
    config = DataConfig(preprocessed_data_root=str(tmp_path), num_dataloader_workers=0)
    assert config.encoding_mode == "precomputed"
    assert config.online_encoding is None


def test_online_data_config_requires_manifest_inputs() -> None:
    with pytest.raises(ValueError, match="online data requires"):
        DataConfig(encoding_mode="online")


def test_online_data_config_accepts_existing_manifest_inputs(tmp_path: Path) -> None:
    source_config = tmp_path / "data.yaml"
    manifest = tmp_path / "train.jsonl"
    source_config.write_text("datasets: []\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    config = DataConfig(
        encoding_mode="online",
        train_data_config=str(source_config),
        manifest_path=str(manifest),
        online_encoding=OnlineEncodingConfig(),
    )
    assert config.preprocessed_data_root is None
    assert config.encoding_mode == "online"


def test_write_path_policy() -> None:
    assert str(assert_write_path_allowed("/mnt/workspace/litengjie/run/output")).startswith(
        "/mnt/workspace/litengjie"
    )
    with pytest.raises(ValueError, match="forbidden"):
        assert_write_path_allowed("/mnt/workspace/liutao/cache/output.pt")
    with pytest.raises(ValueError, match="must stay"):
        assert_write_path_allowed("/tmp/output.pt")
