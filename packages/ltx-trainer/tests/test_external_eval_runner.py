from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from ltx_trainer.online_inference import external_eval_runner, raw_condition_encoder
from ltx_trainer.online_inference.external_eval_runner import (
    filter_external_records,
    run_external_sample,
)
from ltx_trainer.online_inference.raw_condition_encoder import (
    RawReferenceLoadError,
    encode_external_reference_only_conditions,
    encode_selected_sample_conditions,
)
from ltx_trainer.online_inference.read_only_sources import ReadOnlySourcePolicy


def _sample(reference: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_name": "opens2v_open_domain",
        "source_json": str(reference.parent / "source.json"),
        "source_record_id": "singleobj_1",
        "output_id": "singleobj_1",
        "sample_key": "key",
        "task": "r2v",
        "caption": "the object rotates",
        "reference_paths": [str(reference)],
        "original_reference_paths": ["Images/reference.png"],
        "width": 832,
        "height": 480,
        "num_frames": 121,
        "fps": 24.0,
        "dataset_metadata": {"schema_group": "singleobj"},
    }


def test_external_reference_only_encoder_never_accepts_or_opens_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []

    def fake_decode(path: Path) -> torch.Tensor:
        opened.append(str(path))
        return torch.zeros(12, 18, 3, dtype=torch.uint8)

    monkeypatch.setattr(raw_condition_encoder, "decode_image_rgb", fake_decode)
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
            assert not {"target", "target_path", "target_pixels", "target_latents"}.intersection(kwargs)
            return {"reference_metadata": {}}

    sample = {
        "task": "r2v",
        "caption": "targetless",
        "reference_paths": ["/references/one.png", "/references/two.png"],
        "original_reference_paths": ["one.png", "two.png"],
        "width": 832,
        "height": 480,
        "num_frames": 121,
        "fps": 24.0,
    }
    result = encode_external_reference_only_conditions(FakeEncoder(), sample)
    assert opened == sample["reference_paths"]
    assert result["strict_no_gt_checks"] == {
        "strict_no_gt": True,
        "has_target": False,
        "reference_target_alias_check": "not_applicable_no_target",
        "target_open_count": 0,
        "target_path_passed_to_condition_encoder": False,
        "target_path_passed_to_denoiser": False,
        "uses_target_latents": False,
        "uses_gt_siglip_tokens": False,
    }


def test_existing_training_replay_still_requires_target_alias_metadata() -> None:
    encoder = SimpleNamespace(
        config=SimpleNamespace(vlm_reference_preprocess="original", cpu_transform_chunk_frames=4)
    )
    sample = {
        "task": "r2v",
        "caption": "missing target",
        "reference_paths": ["/references/one.png"],
        "width": 832,
        "height": 480,
        "num_frames": 121,
        "fps": 24.0,
    }
    with pytest.raises(RawReferenceLoadError, match="requires target_path"):
        encode_selected_sample_conditions(encoder, sample)


def test_filters_are_stable_and_composable(tmp_path: Path) -> None:
    records = []
    for identifier in ("complex_0", "CASE24_0", "complex_1", "CASE24_1"):
        record = _sample(tmp_path / "reference.png")
        record["source_record_id"] = identifier
        record["output_id"] = identifier
        records.append(record)
    selected = filter_external_records(
        records,
        id_prefixes=("complex_",),
        start_index=1,
        limit=1,
    )
    assert [record["source_record_id"] for record in selected] == ["complex_1"]


def test_external_runner_uses_targetless_encoder_and_exact_flat_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "sources" / "reference.png"
    reference.parent.mkdir()
    Image.new("RGB", (32, 32), "blue").save(reference)
    sample = _sample(reference)
    output_root = tmp_path / "owned" / "external"
    policy = ReadOnlySourcePolicy(
        writable_root=tmp_path / "owned",
        allowed_files=frozenset({reference}),
    )
    observed: dict[str, Any] = {}

    def fake_run_online_sample(**kwargs):
        observed.update(kwargs)
        sample_dir = kwargs["sample_dir_override"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "generated.mp4").write_bytes(b"video")
        return {"status": "success", "sample_dir": str(sample_dir)}

    monkeypatch.setattr(external_eval_runner, "run_online_sample", fake_run_online_sample)
    result = run_external_sample(
        runtime=SimpleNamespace(),
        sample=sample,
        output_root=output_root,
        policy=policy,
        dry_run=False,
        resume=False,
        overwrite_incomplete=False,
        base_seed=42,
        num_inference_steps=50,
        guidance_scale=4.0,
        ref_guidance_scale=2.0,
        vision_guidance_scale=0.0,
        ref_guidance_mode="synchronized",
        guidance_rescale=0.7,
        stg_scale=1.0,
        stg_blocks=[28],
        decode_tile=True,
        code_commit="commit",
    )
    assert observed["condition_encoder"] is encode_external_reference_only_conditions
    assert observed["metadata_overrides"]["has_target"] is False
    assert observed["metadata_overrides"]["target_open_count"] == 0
    flat = output_root / "opens2v_open_domain" / "Generated_Videos" / "singleobj_1.mp4"
    assert flat.read_bytes() == b"video"
    assert result["flat_export_path"] == str(flat.resolve())


def test_resume_repairs_missing_flat_export_without_loading_sample_again(tmp_path: Path) -> None:
    reference = tmp_path / "sources" / "reference.png"
    reference.parent.mkdir()
    Image.new("RGB", (32, 32), "blue").save(reference)
    sample = _sample(reference)
    output_root = tmp_path / "owned" / "external"
    policy = ReadOnlySourcePolicy(
        writable_root=tmp_path / "owned",
        allowed_files=frozenset({reference}),
    )
    sample_dir = (
        output_root
        / "opens2v_open_domain"
        / "samples"
        / "singleobj_1"
    )
    sample_dir.mkdir(parents=True)
    (sample_dir / "generated.mp4").write_bytes(b"video")
    (sample_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (sample_dir / "success.json").write_text(
        '{"artifacts":["generated.mp4","metadata.json"]}',
        encoding="utf-8",
    )
    result = run_external_sample(
        runtime=SimpleNamespace(),
        sample=sample,
        output_root=output_root,
        policy=policy,
        dry_run=False,
        resume=True,
        overwrite_incomplete=False,
        base_seed=42,
        num_inference_steps=50,
        guidance_scale=4.0,
        ref_guidance_scale=2.0,
        vision_guidance_scale=0.0,
        ref_guidance_mode="synchronized",
        guidance_rescale=0.7,
        stg_scale=1.0,
        stg_blocks=[28],
        decode_tile=True,
        code_commit="commit",
    )
    assert result["status"] == "skipped_existing"
    flat = output_root / "opens2v_open_domain" / "Generated_Videos" / "singleobj_1.mp4"
    assert flat.read_bytes() == b"video"
