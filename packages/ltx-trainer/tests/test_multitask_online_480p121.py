from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from ltx_trainer.config import DataConfig, LtxTrainerConfig, OnlineEncodingConfig
from ltx_trainer.online_data.constants import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VIDEO_NUM_FRAMES,
    VLM_TARGET_INDICES,
    uniform_indices,
)
from ltx_trainer.online_data.manifest import ManifestReject, build_i2i_record, build_r2v_record
from ltx_trainer.online_data.manifest_index import build_manifest_offset_index
from ltx_trainer.online_data.media_decoder import decode_image_rgb
from ltx_trainer.online_data.multitask_dataset import OnlineMultiTaskDataset
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


def _write_image(path: Path, *, size: tuple[int, int] = (64, 64), color: tuple[int, int, int] = (1, 2, 3)) -> None:
    Image.new("RGB", size, color=color).save(path)


def test_i2i_manifest_does_not_apply_video_frame_filter(tmp_path: Path) -> None:
    _write_image(tmp_path / "target.png")
    _write_image(tmp_path / "source.png")
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


def test_short_r2v_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    reference_path = tmp_path / "ref.png"
    _write_image(reference_path)
    monkeypatch.setattr(
        "ltx_trainer.online_data.manifest.probe_video",
        lambda _path: {"fps": 24.0, "frame_count": 100, "width": 832, "height": 480},
    )
    row = {
        "video_path": str(video_path),
        "text": "prompt",
        "crop": [0, 832, 0, 480],
        "face_cut": [0, 100],
        "ref_images": [str(reference_path)],
    }
    with pytest.raises(ManifestReject, match="121-frame") as exc_info:
        build_r2v_record(row, dataset_name="r2v", data_root=None, manifest_seed=42)
    assert exc_info.value.reason == "insufficient_frames_for_121_at_24fps"


def test_i2i_static_bad_data_uses_stable_reject_reasons(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    reference = tmp_path / "reference.png"
    _write_image(target, size=(100, 80))
    _write_image(reference)
    common = {
        "dataset_name": "i2i",
        "data_root": None,
        "target_field": "target",
        "reference_field": "sources",
        "caption_field": "instruction",
    }

    cases = [
        (
            {"target": str(tmp_path / "missing.png"), "sources": [str(reference)], "instruction": "edit"},
            {},
            "missing_target",
        ),
        (
            {"target": str(target), "sources": [str(tmp_path / "missing-ref.png")], "instruction": "edit"},
            {},
            "missing_reference",
        ),
        (
            {"target": str(target), "sources": [str(reference)], "instruction": "   "},
            {},
            "empty_caption",
        ),
        (
            {
                "target": str(target),
                "sources": [str(reference)],
                "instruction": "edit",
                "crop": [0, 0, 101, 80],
            },
            {"crop_field": "crop"},
            "invalid_crop",
        ),
    ]
    for row, extra, expected_reason in cases:
        with pytest.raises(ManifestReject) as exc_info:
            build_i2i_record(row, **common, **extra)
        assert exc_info.value.reason == expected_reason


def test_r2v_static_bad_data_uses_stable_reject_reasons(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    reference = tmp_path / "reference.png"
    _write_image(reference)
    base_row = {
        "video_path": str(video),
        "text": "prompt",
        "crop": [0, 832, 0, 480],
        "face_cut": [0, 200],
        "ref_images": [str(reference)],
    }
    header = {"fps": 24.0, "frame_count": 240, "width": 832, "height": 480}

    bad_crop = dict(base_row, crop=[0, 900, 0, 480])
    with pytest.raises(ManifestReject) as crop_error:
        build_r2v_record(
            bad_crop,
            dataset_name="r2v",
            data_root=None,
            manifest_seed=42,
            video_header=header,
        )
    assert crop_error.value.reason == "invalid_crop"

    bad_face_cut = dict(base_row, face_cut=[0, 241])
    with pytest.raises(ManifestReject) as face_cut_error:
        build_r2v_record(
            bad_face_cut,
            dataset_name="r2v",
            data_root=None,
            manifest_seed=42,
            video_header=header,
        )
    assert face_cut_error.value.reason == "invalid_face_cut"

    with pytest.raises(ManifestReject) as header_error:
        build_r2v_record(
            base_row,
            dataset_name="r2v",
            data_root=None,
            manifest_seed=42,
            video_header={"fps": 0, "frame_count": 240, "width": 832, "height": 480},
        )
    assert header_error.value.reason == "invalid_video_header"


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


def test_online_encoding_defaults_to_original_vlm_reference_and_pyav() -> None:
    config = OnlineEncodingConfig()
    assert config.vlm_reference_preprocess == "original"
    assert config.video_decoder == "pyav"


@pytest.mark.parametrize(
    "config_name",
    [
        "multiref_stage1_full_tokens_2048_2000.yaml",
        "multiref_stage2_full_tokens_planner_2048.yaml",
        "multiref_stage3_joint_full_tokens_planner_2048.yaml",
    ],
)
def test_legacy_stage_configs_remain_precomputed(
    config_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / config_name
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_path = tmp_path / "model.safetensors"
    model_path.touch()
    config_data["model"]["model_path"] = str(model_path)
    config_data["data"]["preprocessed_data_root"] = str(tmp_path)
    monkeypatch.setattr(LtxTrainerConfig, "_validate_data_dirs_exist", lambda _self: None)
    config = LtxTrainerConfig.model_validate(config_data)
    assert config.data.encoding_mode == "precomputed"
    assert config.data.online_encoding is None


@pytest.mark.parametrize("policy", ["sequential_cuda", "cpu_offload"])
def test_incomplete_encoder_device_policies_fail_fast(policy: str) -> None:
    with pytest.raises(ValueError, match="resident_cuda"):
        OnlineEncodingConfig(encoder_device_policy=policy)


def test_reference_vae_and_vlm_paths_preserve_shape_and_order(tmp_path: Path) -> None:
    target_path = tmp_path / "target.png"
    first_ref = tmp_path / "portrait_red.png"
    second_ref = tmp_path / "portrait_green.png"
    _write_image(target_path, size=(100, 80), color=(10, 20, 30))
    _write_image(first_ref, size=(120, 300), color=(255, 0, 0))
    _write_image(second_ref, size=(90, 240), color=(0, 255, 0))
    record = build_i2i_record(
        {
            "target": str(target_path),
            "sources": [str(first_ref), str(second_ref)],
            "instruction": "preserve both identities",
        },
        dataset_name="i2i",
        data_root=None,
        target_field="target",
        reference_field="sources",
        caption_field="instruction",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    build_manifest_offset_index(manifest)
    dataset = OnlineMultiTaskDataset(
        manifest,
        vlm_reference_preprocess="original",
        return_load_errors=False,
        require_all_tasks=False,
    )
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert [tuple(image.shape) for image in sample["reference_pixels_vae"]] == [
        (480, 832, 3),
        (480, 832, 3),
    ]
    assert [tuple(image.shape) for image in sample["reference_images_vlm"]] == [
        (300, 120, 3),
        (240, 90, 3),
    ]
    torch.testing.assert_close(sample["reference_images_vlm"][0], decode_image_rgb(first_ref))
    torch.testing.assert_close(sample["reference_images_vlm"][1], decode_image_rgb(second_ref))
    assert sample["reference_images_vlm"][0][0, 0].tolist() == [255, 0, 0]
    assert sample["reference_images_vlm"][1][0, 0].tolist() == [0, 255, 0]


def test_target_crop_vlm_reference_is_explicit_not_default(tmp_path: Path) -> None:
    target_path = tmp_path / "target.png"
    reference_path = tmp_path / "portrait.png"
    _write_image(target_path)
    _write_image(reference_path, size=(100, 300))
    record = build_i2i_record(
        {"target": str(target_path), "sources": [str(reference_path)], "instruction": "edit"},
        dataset_name="i2i",
        data_root=None,
        target_field="target",
        reference_field="sources",
        caption_field="instruction",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = OnlineMultiTaskDataset(
        manifest,
        vlm_reference_preprocess="target_crop",
        return_load_errors=False,
        require_all_tasks=False,
    )
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert tuple(sample["reference_images_vlm"][0].shape) == (480, 832, 3)
    torch.testing.assert_close(sample["reference_images_vlm"][0], sample["reference_pixels_vae"][0])


def test_write_path_policy() -> None:
    assert str(assert_write_path_allowed("/mnt/workspace/litengjie/run/output")).startswith(
        "/mnt/workspace/litengjie"
    )
    with pytest.raises(ValueError, match="forbidden"):
        assert_write_path_allowed("/mnt/workspace/liutao/cache/output.pt")
    with pytest.raises(ValueError, match="must stay"):
        assert_write_path_allowed("/tmp/output.pt")
