from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from ltx_trainer.online_data.constants import VISUAL_TOKEN_CAPACITY
from ltx_trainer.online_data.online_batch_encoder import OnlineBatchEncoder
from ltx_trainer.online_inference import raw_condition_encoder, runner
from ltx_trainer.online_inference.media_identity import TargetReferenceAliasError


@pytest.mark.parametrize(
    ("task", "num_frames", "fps", "expected_shape"),
    [
        ("i2i", 1, 1.0, (1, 128, 1, 15, 26)),
        ("r2v", 121, 24.0, (1, 128, 16, 15, 26)),
    ],
)
def test_noise_shape_is_geometry_only(
    task: str,
    num_frames: int,
    fps: float,
    expected_shape: tuple[int, ...],
) -> None:
    metadata = runner.build_noise_shape_metadata(
        {
            "task": task,
            "width": 832,
            "height": 480,
            "num_frames": num_frames,
            "fps": fps,
        },
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert tuple(metadata["latents"].shape) == expected_shape
    assert torch.count_nonzero(metadata["latents"]) == 0


@pytest.mark.parametrize(
    ("task", "num_frames", "fps", "valid_tokens"),
    [("i2i", 1, 1.0, 256), ("r2v", 121, 24.0, 2048)],
)
def test_reference_only_encoder_emits_no_target_or_gt_keys(
    task: str,
    num_frames: int,
    fps: float,
    valid_tokens: int,
) -> None:
    encoder = OnlineBatchEncoder.__new__(OnlineBatchEncoder)
    encoder.config = SimpleNamespace(
        width=832,
        height=480,
        image_num_frames=1,
        image_fps=1.0,
        video_num_frames=121,
        video_fps=24.0,
        max_ref_images=4,
        vlm_reference_preprocess="original",
    )
    encoder.device = torch.device("cpu")
    encoder.last_dtype_diagnostics = {}
    encoder._move_frozen_encoders_for_encode = lambda: None
    encoder._offload_frozen_encoders_after_encode = lambda: None
    encoder._reference_images = lambda values: [f"image-{index}" for index, _ in enumerate(values)]
    encoder._encode_reference_latents = lambda values: {
        "latents": torch.zeros(1, len(values[0]), 128, 1, 15, 26),
        "ref_valid_mask": torch.ones(1, len(values[0]), dtype=torch.bool),
        "fps": torch.ones(1),
    }
    encoder._encode_condition = lambda **kwargs: {
        "video_prompt_embeds": torch.zeros(1, 8, 4096),
        "prompt_attention_mask": torch.ones(1, 8, dtype=torch.long),
        "reference_count": len(kwargs["reference_images"]),
    }
    encoder._build_planner_vlm_inputs = lambda **kwargs: {
        "planner_output_mask": kwargs["planner_output_mask"],
        "num_ref_images": torch.tensor([len(kwargs["reference_images"])]),
    }

    references = [torch.zeros(480, 832, 3, dtype=torch.uint8) for _ in range(2)]
    result = encoder.encode_inference_conditions_from_references(
        task=task,
        caption="test instruction",
        reference_pixels_vae=references,
        reference_images_vlm=references,
        width=832,
        height=480,
        num_frames=num_frames,
        fps=fps,
    )

    forbidden = {
        "latents",
        "target_pixels",
        "target_latents",
        "gt_visual_tokens",
        "gt_siglip_tokens",
    }
    assert not forbidden.intersection(result)
    planner_mask = result["planner_vlm_inputs"]["planner_output_mask"]
    assert planner_mask.shape == (1, VISUAL_TOKEN_CAPACITY)
    assert int(planner_mask.sum()) == valid_tokens
    assert result["conditions"]["reference_count"] == 2
    assert result["text_conditions"]["reference_count"] == 0
    assert result["reference_metadata"]["reference_order"] == [0, 1]
    metadata = result["visual_position_metadata"]
    assert int(metadata["num_valid_visual_tokens"].item()) == valid_tokens
    assert int(metadata["target_num_frames"].item()) == num_frames
    assert float(metadata["target_fps"].item()) == fps
    assert int(result["task_system_prompt_id"].item()) == (0 if task == "i2i" else 1)


def test_selected_condition_path_never_passes_or_opens_target(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []

    def fake_decode(path):
        opened.append(str(path))
        if "target" in str(path):
            raise AssertionError("target media was opened")
        return torch.zeros(10, 12, 3, dtype=torch.uint8)

    monkeypatch.setattr(raw_condition_encoder, "decode_image_rgb", fake_decode)
    monkeypatch.setattr(
        raw_condition_encoder,
        "assert_references_do_not_alias_target",
        lambda references, target: None,
    )
    monkeypatch.setattr(
        raw_condition_encoder,
        "deterministic_resize_center_crop",
        lambda frames, **kwargs: torch.zeros(1, 480, 832, 3, dtype=torch.uint8),
    )

    class FakeEncoder:
        config = SimpleNamespace(
            vlm_reference_preprocess="original",
            cpu_transform_chunk_frames=4,
        )

        def encode_inference_conditions_from_references(self, **kwargs):
            assert "target_path" not in kwargs
            assert "target_pixels" not in kwargs
            return {"reference_metadata": {}, "planner_vlm_inputs": {}}

    sample = {
        "task": "i2i",
        "caption": "edit this",
        "reference_paths": ["/fake/reference_a.png", "/fake/reference_b.png"],
        "target_path": "/fake/target.png",
        "width": 832,
        "height": 480,
        "num_frames": 1,
        "fps": 1.0,
    }
    result = raw_condition_encoder.encode_selected_sample_conditions(FakeEncoder(), sample)
    assert opened == sample["reference_paths"]
    assert result["reference_metadata"]["reference_paths"] == sample["reference_paths"]


def test_strict_no_gt_dry_run_succeeds_when_target_open_is_forbidden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 24), "blue").save(reference)
    target = tmp_path / "target.png"
    target.write_bytes(b"unreadable-for-generation")
    original_open = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any):
        if path.resolve() == target.resolve():
            raise AssertionError("strict-no-GT dry-run opened target")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    raw = {
        "multi_ref_latents": {
            "latents": torch.zeros(1, 1, 128, 1, 15, 26),
            "ref_valid_mask": torch.ones(1, 1, dtype=torch.bool),
            "fps": torch.ones(1),
        },
        "multi_reference_latents": {},
        "conditions": {"video_prompt_embeds": torch.zeros(1, 8, 4096)},
        "text_conditions": {"video_prompt_embeds": torch.zeros(1, 8, 4096)},
        "planner_vlm_inputs": {
            "planner_output_mask": torch.ones(1, 2048, dtype=torch.bool),
            "planner_token_count": torch.tensor([2048]),
        },
        "visual_position_metadata": {},
        "task_system_prompt_id": torch.tensor([0]),
        "dtype_diagnostics": {},
        "reference_metadata": {
            "reference_paths_used": [str(reference)],
            "original_reference_paths": [str(reference)],
            "reference_export_modes": [],
        },
        "strict_no_gt_checks": {
            "reference_target_alias_check": "passed",
            "target_path_passed_to_condition_encoder": False,
            "target_path_passed_to_denoiser": False,
            "uses_target_latents": False,
            "uses_gt_siglip_tokens": False,
        },
    }
    raw["multi_reference_latents"] = raw["multi_ref_latents"]
    monkeypatch.setattr(runner, "encode_selected_sample_conditions", lambda encoder, sample: raw)
    guidance = SimpleNamespace(
        positive_conditions={},
        negative_conditions=None,
        no_ref_conditions=None,
        no_visual_conditions=None,
        planner_forward_count=1,
        inference_diagnostics={"final_condition_shape": [1, 2056, 4096]},
    )
    stage2 = SimpleNamespace(
        _autocast_context=lambda device, dtype: nullcontext(),
        _prepare_guidance_condition_bundle=lambda **kwargs: guidance,
    )
    runtime = SimpleNamespace(
        online_encoder=object(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        stage2=stage2,
        strategy=object(),
        negative_conditions=None,
        checkpoint_path=tmp_path / "checkpoint_step_00001.safetensors",
        checkpoint_audit={
            "checkpoint_sha256": "abc",
            "checkpoint_step": 1,
        },
        checkpoint_flags={},
        negative_prompt=None,
        config_path=tmp_path / "config.yaml",
    )
    sample = {
        "sample_key": "strict-no-gt",
        "sample_plan_sha256": "plan",
        "manifest_path": str(tmp_path / "manifest.jsonl"),
        "manifest_index": 0,
        "task": "i2i",
        "caption": "edit reference",
        "reference_paths": [str(reference)],
        "target_path": str(target),
        "width": 832,
        "height": 480,
        "num_frames": 1,
        "fps": 1.0,
    }
    result = runner.run_online_sample(
        runtime=runtime,
        sample=sample,
        output_root=tmp_path / "outputs",
        dry_run=True,
        overwrite=False,
        seed=42,
        num_inference_steps=2,
        guidance_scale=1.0,
        ref_guidance_scale=0.0,
        vision_guidance_scale=0.0,
        ref_guidance_mode="synchronized",
        guidance_rescale=0.0,
        stg_scale=0.0,
        stg_blocks=None,
        decode_tile=True,
        code_commit="test",
    )
    assert result["status"] == "dry_run_success"
    assert result["target_open_count"] == 0
    assert result["target_open_count_instrumented"] is False
    assert result["strict_no_gt_checks"]["reference_target_alias_check"] == "passed"
    sample_dir = Path(result["sample_dir"])
    assert (sample_dir / "reference_00.png").is_file()
    assert (sample_dir / "references.png").is_file()


@pytest.mark.parametrize("alias_kind", ["direct", "symlink", "hardlink"])
def test_runtime_rejects_reference_target_alias_before_decode(
    tmp_path: Path,
    alias_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.png"
    Image.new("RGB", (16, 16), "red").save(target)
    reference = tmp_path / f"reference_{alias_kind}.png"
    if alias_kind == "direct":
        reference = target
    elif alias_kind == "symlink":
        reference.symlink_to(target)
    else:
        os.link(target, reference)
    decode_calls = 0

    def fail_decode(path):
        nonlocal decode_calls
        decode_calls += 1
        raise AssertionError(f"alias reference was decoded: {path}")

    monkeypatch.setattr(raw_condition_encoder, "decode_image_rgb", fail_decode)
    sample = {
        "task": "i2i",
        "reference_paths": [str(reference)],
        "target_path": str(target),
        "width": 832,
        "height": 480,
        "num_frames": 1,
        "fps": 1.0,
    }
    with pytest.raises(TargetReferenceAliasError, match="reference_aliases_target"):
        raw_condition_encoder.load_reference_inputs(
            sample,
            vlm_reference_preprocess="original",
        )
    assert decode_calls == 0
