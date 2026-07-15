from __future__ import annotations

import json
import resource
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import typer
import yaml
from PIL import Image
from safetensors.torch import save_file

from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_trainer.config import DataConfig, LtxTrainerConfig, OnlineEncodingConfig
from ltx_trainer.online_data import smoke as smoke_helpers
from ltx_trainer.online_data.constants import (
    IMAGE_TASK,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VIDEO_NUM_FRAMES,
    VIDEO_TASK,
    VLM_TARGET_INDICES,
    uniform_indices,
)
from ltx_trainer.online_data.manifest import ManifestReject, build_i2i_record, build_r2v_record
from ltx_trainer.online_data.manifest_index import build_manifest_offset_index
from ltx_trainer.online_data.media_decoder import decode_image_rgb
from ltx_trainer.online_data.multitask_dataset import OnlineMultiTaskDataset
from ltx_trainer.online_data.online_batch_encoder import OnlineBatchEncoder, _build_messages
from ltx_trainer.online_data.path_safety import assert_write_path_allowed
from ltx_trainer.online_data.transforms import deterministic_resize_center_crop
from ltx_trainer.online_data.visual_token_packing import build_visual_metadata, pack_visual_tokens
from scripts import check_multitask_online_training_ready as training_ready
from scripts.check_multitask_online_ddp import _validate_task


def test_fixed_480p121_contract() -> None:
    assert TARGET_WIDTH % 32 == 0
    assert TARGET_HEIGHT % 32 == 0
    assert VIDEO_NUM_FRAMES % 8 == 1
    assert uniform_indices(121, 8) == list(VLM_TARGET_INDICES)
    assert OnlineEncodingConfig().cpu_transform_chunk_frames == 4
    with pytest.raises(ValueError):
        OnlineEncodingConfig(cpu_transform_chunk_frames=0)
    with pytest.raises(ValueError):
        OnlineEncodingConfig(cpu_transform_chunk_frames=17)


def test_seven_gpu_preflight_reports_task_exposures_without_changing_steps() -> None:
    report = training_ready._planned_task_exposure_report(
        optimization_steps=30_000,
        effective_global_batch=28,
        image_ratio=0.3,
        task_counts={IMAGE_TASK: 100_000, VIDEO_TASK: 200_000},
    )
    assert report["planned_task_optimizer_steps"] == {IMAGE_TASK: 9_000, VIDEO_TASK: 21_000}
    assert report["planned_task_exposures"] == {IMAGE_TASK: 252_000, VIDEO_TASK: 588_000}
    assert report["planned_total_exposures"] == 840_000
    assert report["estimated_repeats"] == pytest.approx({IMAGE_TASK: 2.52, VIDEO_TASK: 2.94})


def test_seven_gpu_benchmark_uses_bounded_worker_settings() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "multiref_stage3_multitask_online_480p121_joint_warmstart_benchmark_200_7gpu.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["optimization"]["learning_rate"] == 1.0e-5
    assert payload["optimization"]["steps"] == 200
    assert payload["optimization"]["batch_size"] == 1
    assert payload["optimization"]["gradient_accumulation_steps"] == 4
    assert payload["data"]["num_dataloader_workers"] == 1
    assert payload["data"]["online_encoding"]["prefetch_factor"] == 1
    assert payload["data"]["online_encoding"]["cpu_transform_chunk_frames"] == 4
    assert payload["checkpoints"]["interval"] == 100
    assert payload["checkpoints"]["save_training_state"] == "full"


def test_online_messages_use_task_specific_image_edit_semantics() -> None:
    system_prompt = (
        Path(__file__).resolve().parents[2]
        / "ltx-core"
        / "src"
        / "ltx_core"
        / "text_encoders"
        / "gemma"
        / "encoders"
        / "prompts"
        / "gemma_multiref_image_edit_planner_system_prompt.txt"
    ).read_text(encoding="utf-8")
    messages = _build_messages(system_prompt, "replace the shirt", 2, task=IMAGE_TASK)
    serialized = json.dumps(messages)

    assert "target image" in serialized
    assert "target video" not in serialized
    assert "motion sequence" not in serialized
    assert messages[1]["content"][0]["text"] == "Image editing instruction: replace the shirt"
    assert messages[1]["content"][1]["text"] == "Source/reference images:"


def test_online_video_messages_preserve_existing_serialization() -> None:
    system_prompt = "target video"
    messages = _build_messages(system_prompt, "a person walks", 1, task=VIDEO_TASK)

    assert messages == [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "User Raw Input Prompt: a person walks."},
                {"type": "text", "text": "Reference images:"},
                {"type": "text", "text": "Reference image 1:"},
                {"type": "image"},
            ],
        },
    ]


@pytest.mark.parametrize(("task", "valid_tokens"), [(IMAGE_TASK, 256), (VIDEO_TASK, 2048)])
def test_task_specific_planner_inputs_keep_fixed_capacity_and_ntp_masks(
    task: str,
    valid_tokens: int,
) -> None:
    class _Tokenizer:
        pad_token_id = 0

        def __init__(self) -> None:
            self.messages = None

        def apply_chat_template(self, messages: list[dict[str, Any]], **_kwargs: Any) -> str:
            self.messages = messages
            return "serialized"

    class _Processor:
        image_seq_length = 1

        def __call__(self, **_kwargs: Any) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor(
                    [[
                        10,
                        GEMMA3_CONFIG_FOR_LTX.boi_token_index,
                        GEMMA3_CONFIG_FOR_LTX.image_token_index,
                        GEMMA3_CONFIG_FOR_LTX.eoi_token_index,
                        11,
                    ]],
                    dtype=torch.long,
                ),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
            }

    encoder = object.__new__(OnlineBatchEncoder)
    encoder.config = SimpleNamespace(planner_max_length=2070)
    encoder.tokenizer = _Tokenizer()
    encoder.processor = _Processor()
    encoder.system_prompts = {
        IMAGE_TASK: "target image",
        VIDEO_TASK: "target video",
    }
    planner_output_mask = torch.zeros(1, 2048, dtype=torch.bool)
    planner_output_mask[:, :valid_tokens] = True

    result = encoder._build_planner_vlm_inputs(
        caption="instruction",
        reference_images=[object()],
        planner_output_mask=planner_output_mask,
        task=task,
    )

    assert result["planner_placeholder_mask"].sum().item() == 2048
    assert result["planner_output_mask"].sum().item() == valid_tokens
    assert not bool((result["ntp_label_mask"] & result["planner_region_mask"]).any())
    assert not bool((result["ntp_label_mask"] & result["ref_image_region_mask"]).any())
    serialized_messages = json.dumps(encoder.tokenizer.messages)
    assert ("target image" in serialized_messages) is (task == IMAGE_TASK)
    assert ("target video" in serialized_messages) is (task == VIDEO_TASK)


def _write_complete_stage3_checkpoint(
    path: Path,
    *,
    training_phase: str,
    global_step: int = 2000,
) -> None:
    save_file(
        {
            "diffusion_model.block.lora_A.default.weight": torch.ones(1),
            "diffusion_model.block.lora_B.default.weight": torch.ones(1),
            "training_strategy.planner_tokens.query_tokens": torch.ones(1),
            "training_strategy.visual_token_projection.weight": torch.ones(1),
            "training_strategy.visual_full_encoder.input_norm.weight": torch.ones(1),
            "embeddings_processor.video_connector.weight": torch.ones(1),
            "text_encoder.model.model.language_model.block.lora_A.default.weight": torch.ones(1),
            "text_encoder.model.model.language_model.block.lora_B.default.weight": torch.ones(1),
        },
        path,
        metadata={"training_phase": training_phase, "global_step": str(global_step)},
    )


@pytest.mark.parametrize(
    ("checkpoint_phase", "expected_mode"),
    [
        ("stage2", "stage2_to_stage3_init"),
        ("stage3", "stage3_joint_warmstart"),
    ],
)
def test_stage3_no_resume_preflight_accepts_complete_stage2_or_stage3_checkpoint(
    tmp_path: Path,
    checkpoint_phase: str,
    expected_mode: str,
) -> None:
    checkpoint = tmp_path / "lora_weights_step_02000.safetensors"
    _write_complete_stage3_checkpoint(checkpoint, training_phase=checkpoint_phase)

    report = training_ready._inspect_stage3_initialization(
        checkpoint,
        no_resume=True,
        optimization_steps=30_000,
    )

    assert report["initialization_mode"] == expected_mode
    assert report["initial_checkpoint_training_phase"] == checkpoint_phase
    assert report["strict_component_check_passed"] is True
    assert report["starts_from_global_step"] == 0


def test_exact_stage3_preflight_requires_matching_full_training_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "lora_weights_step_00500.safetensors"
    _write_complete_stage3_checkpoint(checkpoint, training_phase="stage3", global_step=500)
    torch.save(
        {
            "global_step": 500,
            "optimizer_state_dict": {"state": {0: {"step": 500}}},
            "lr_scheduler_state_dict": {"last_epoch": 500, "_last_lr": [1.0e-5]},
            "data_state": {
                "task_schedule_cursor": 500,
                "image_permutation_epoch": 0,
                "image_cursor": 0,
                "video_permutation_epoch": 0,
                "video_cursor": 0,
                "microstep_in_optimizer_step": 0,
                "sampler_seed": 42,
            },
        },
        tmp_path / "training_state_step_00500.pt",
    )

    report = training_ready._inspect_stage3_initialization(
        checkpoint,
        no_resume=False,
        optimization_steps=30_000,
    )
    assert report["initialization_mode"] == "exact_resume"
    assert report["starts_from_global_step"] == 500


def test_ddp_smoke_accepts_exactly_one_task() -> None:
    assert _validate_task("i2i") == "i2i"
    assert _validate_task(" R2V ") == "r2v"
    with pytest.raises(typer.BadParameter, match="i2i or r2v"):
        _validate_task("stage3")


def test_nonmain_smoke_never_constructs_path_from_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broadcast_payload = {
        "success": True,
        "checkpoint_path": "/mnt/workspace/litengjie/run/checkpoint.safetensors",
        "training_state_path": "/mnt/workspace/litengjie/run/training_state.pt",
        "ready_marker_path": "/mnt/workspace/litengjie/run/checkpoint.ready.json",
        "error_message": "",
    }
    path_calls = 0

    def fail_path(*_args: Any, **_kwargs: Any) -> None:
        nonlocal path_calls
        path_calls += 1
        raise AssertionError("non-main rank must not construct a Path from checkpoint=None")

    monkeypatch.setattr(smoke_helpers, "Path", fail_path)
    def fake_broadcast(_payload: Any, *, is_main_process: bool) -> dict[str, Any]:
        del is_main_process
        return broadcast_payload

    monkeypatch.setattr(smoke_helpers, "_broadcast_main_payload", fake_broadcast)

    status = smoke_helpers._checkpoint_artifact_status(
        None,
        global_step=1,
        phase="stage3",
        is_main_process=False,
    )

    assert path_calls == 0
    assert status == broadcast_payload


def _legacy_single_frame_transform(
    frames: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    cropped = frames.permute(0, 3, 1, 2).float()
    scale = max(target_height / cropped.shape[-2], target_width / cropped.shape[-1])
    resized_height = max(target_height, int(round(cropped.shape[-2] * scale)))
    resized_width = max(target_width, int(round(cropped.shape[-1] * scale)))
    resized = F.interpolate(
        cropped,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    top = (resized_height - target_height) // 2
    left = (resized_width - target_width) // 2
    return (
        resized[:, :, top : top + target_height, left : left + target_width]
        .round()
        .clamp_(0, 255)
        .to(dtype=torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
    )


def test_chunked_transform_preserves_single_image_numerics() -> None:
    torch.manual_seed(7)
    image = torch.randint(0, 256, (1, 731, 1193, 3), dtype=torch.uint8)
    expected = _legacy_single_frame_transform(
        image,
        target_height=480,
        target_width=832,
    )
    actual = deterministic_resize_center_crop(
        image,
        target_height=480,
        target_width=832,
        chunk_frames=4,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_chunked_transform_never_interpolates_more_than_configured_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_batch_sizes: list[int] = []
    original_interpolate = F.interpolate

    def recording_interpolate(input_tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        observed_batch_sizes.append(input_tensor.shape[0])
        return original_interpolate(input_tensor, *args, **kwargs)

    monkeypatch.setattr("ltx_trainer.online_data.transforms.F.interpolate", recording_interpolate)
    frames = torch.zeros(9, 96, 160, 3, dtype=torch.uint8)
    output = deterministic_resize_center_crop(
        frames,
        target_height=48,
        target_width=80,
        chunk_frames=4,
    )
    assert output.shape == (9, 48, 80, 3)
    assert observed_batch_sizes == [4, 4, 1]


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def test_1080p_121_frame_transform_has_bounded_rss() -> None:
    # Expanded uint8 input avoids making input allocation part of the transform RSS assertion.
    frame = torch.zeros(1, 1080, 1920, 3, dtype=torch.uint8)
    frames = frame.expand(121, -1, -1, -1)
    before = _peak_rss_bytes()
    output = deterministic_resize_center_crop(
        frames,
        target_height=480,
        target_width=832,
        chunk_frames=4,
    )
    rss_delta = max(0, _peak_rss_bytes() - before)
    assert output.shape == (121, 480, 832, 3)
    assert output.dtype == torch.uint8
    assert rss_delta < 700 * 1024**2


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
