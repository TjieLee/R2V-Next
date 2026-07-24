from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

import ltx_trainer.online_inference.runner as runner_module
from ltx_trainer.online_inference.external_eval_runner import select_external_records
from ltx_trainer.online_inference.external_eval_schema import (
    manifest_jsonl,
    normalize_external_dataset,
)
from ltx_trainer.online_inference.path_policy import assert_external_eval_output_path
from ltx_trainer.online_inference.semantic_guidance import SemanticGuidanceConfig


def _manifest(tmp_path: Path) -> Path:
    records, errors = normalize_external_dataset(
        dataset_schema="opens2v_open_domain",
        input_json=tmp_path / "source.json",
        payload={
            "alpha_1": {"prompt": "one", "img_paths": ["one.png"]},
            "alpha_2": {"prompt": "two", "img_paths": ["two.png"]},
            "beta_1": {"prompt": "three", "img_paths": ["three.png"]},
        },
    )
    assert not errors
    path = tmp_path / "manifest.jsonl"
    path.write_text(manifest_jsonl(records), encoding="utf-8")
    return path


def test_external_selection_supports_id_prefix_start_and_limit(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    selected = select_external_records(
        manifest,
        dataset_name="opens2v_open_domain",
        id_prefix="alpha_",
        start_index=1,
        limit=1,
    )
    assert [record["source_record_id"] for record in selected] == ["alpha_2"]
    selected = select_external_records(
        manifest,
        dataset_name="opens2v_open_domain",
        ids=["beta_1"],
    )
    assert selected[0]["source_record_id"] == "beta_1"
    assert "target_path" not in selected[0]
    assert "target_latent" not in selected[0]


def test_external_selection_rejects_wrong_dataset_and_unknown_id(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="other datasets"):
        select_external_records(manifest, dataset_name="custom64")
    with pytest.raises(ValueError, match="not found"):
        select_external_records(
            manifest,
            dataset_name="opens2v_open_domain",
            ids=["missing"],
        )


def test_external_output_policy_is_confined_to_user_workspace() -> None:
    assert_external_eval_output_path("/mnt/workspace/litengjie/external_eval/run")
    with pytest.raises(ValueError, match="forbidden|must stay"):
        assert_external_eval_output_path("/mnt/workspace/public/external_eval/run")


def test_normalized_manifest_contains_no_target_contract(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        forbidden = {
            "target_path",
            "target_video_path",
            "target_image_path",
            "target_latent",
            "semantic_teacher_inputs",
            "evidence_tokens",
        }
        assert not forbidden.intersection(record)


def test_external_dry_run_records_all_guidance_metadata_without_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "reference.png"
    Image.new("RGB", (16, 16), "white").save(reference)
    sample = {
        "dataset_name": "opens2v_open_domain",
        "source_json": str(tmp_path / "source.json"),
        "source_record_id": "sample_1",
        "output_id": "sample_1",
        "sample_key": "a" * 64,
        "task": "r2v",
        "caption": "move",
        "reference_paths": [str(reference)],
        "original_reference_paths": ["reference.png"],
        "width": 832,
        "height": 480,
        "num_frames": 121,
        "fps": 24.0,
        "dataset_metadata": {},
    }
    encoded = {
        "reference_metadata": {"reference_count": 1},
        "strict_no_gt_checks": {
            "strict_no_gt": True,
            "target_path_passed_to_condition_encoder": False,
            "uses_target_latents": False,
            "uses_gt_teacher_evidence": False,
        },
    }
    monkeypatch.setattr(
        runner_module,
        "encode_external_reference_only_conditions",
        lambda *_args, **_kwargs: encoded,
    )

    class FakeRuntime:
        online_encoder = object()
        device = torch.device("cpu")
        checkpoint_path = tmp_path / "checkpoint.safetensors"
        checkpoint_audit = {"checkpoint_sha256": "digest"}
        vae_decoder = None
        last_generation_geometry: dict[str, object] = {}

        def prepare_guidance_states(self, *_args, guidance, negative_prompt, **_kwargs):
            self.last_generation_geometry = guidance.metadata(
                negative_prompt=negative_prompt
            )
            self.last_generation_geometry["guidance_branch_generated_noise_identical"] = True
            return object()

    guidance = SemanticGuidanceConfig()
    result = runner_module.run_online_sample(
        runtime=FakeRuntime(),  # type: ignore[arg-type]
        sample=sample,
        output_root=tmp_path / "outputs",
        dry_run=True,
        overwrite=True,
        seed=9,
        num_inference_steps=50,
        decode_tile=True,
        guidance=guidance,
        negative_prompt="negative",
        external_eval=True,
        code_commit="commit",
    )
    assert result["status"] == "dry_run_success"
    assert result["transformer_forwards_per_step"] == 4
    assert result["enabled_guidance_branches"] == ["P", "N", "R", "U"]
    assert result["joint_guided_span"] == "semantic_and_target"
    assert result["has_target"] is False
    sample_dir = tmp_path / "outputs" / "opens2v_open_domain" / "sample_1"
    assert (sample_dir / "dry_run.json").is_file()
    assert (sample_dir / "negative_prompt.txt").read_text(encoding="utf-8") == "negative\n"
