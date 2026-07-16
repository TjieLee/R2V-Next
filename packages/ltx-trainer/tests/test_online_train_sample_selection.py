from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
from PIL import Image

from ltx_trainer.online_data.constants import VLM_TARGET_INDICES
from ltx_trainer.online_data.manifest_index import build_manifest_offset_index
from ltx_trainer.online_inference.raw_condition_encoder import load_reference_inputs
from ltx_trainer.online_inference.runner import read_selected_samples
from ltx_trainer.online_inference.train_sample_selection import (
    _validate_candidate,
    select_online_train_samples,
    write_selection_bundle,
)


def _write_record(
    *,
    root: Path,
    task: str,
    index: int,
    reference_count: int,
) -> dict:
    references = []
    for reference_index in range(reference_count):
        path = root / f"ref_{task}_{index}_{reference_index}.png"
        Image.new("RGB", (16, 12), (reference_index * 32, 64, 128)).save(path)
        references.append(str(path))
    target = root / f"target_{task}_{index}.bin"
    target.write_bytes(b"target-must-not-be-opened")
    is_image = task == "i2i"
    target_indices = [0] if is_image else list(range(121))
    vlm_indices = [0] if is_image else list(VLM_TARGET_INDICES)
    record = {
        "sample_key": f"{task}-{index}",
        "dataset_name": "unit-test",
        "task": task,
        "target_modality": "image" if is_image else "video",
        "target_path": str(target),
        "reference_paths": references,
        "caption": f"caption {task} {index}",
        "crop_xyxy": None,
        "face_cut": None,
        "target_fps": 1.0 if is_image else 24.0,
        "target_num_frames": 1 if is_image else 121,
        "target_width": 832,
        "target_height": 480,
        "target_source_frame_indices": target_indices,
        "vlm_target_frame_indices": vlm_indices,
        "vlm_source_frame_indices": vlm_indices,
    }
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    record["sample_plan_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return record


def _manifest(tmp_path: Path) -> tuple[Path, list[dict]]:
    records = [
        _write_record(root=tmp_path, task=task, index=index, reference_count=reference_count)
        for task in ("i2i", "r2v")
        for index, reference_count in enumerate((1, 2, 3, 4))
    ]
    manifest = tmp_path / "train_unique.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    build_manifest_offset_index(manifest)
    return manifest, records


def test_selection_is_deterministic_unique_and_reference_stratified(tmp_path: Path) -> None:
    manifest, _ = _manifest(tmp_path)
    first = select_online_train_samples(manifest, seed=42)
    second = select_online_train_samples(manifest, seed=42)

    assert [sample["sample_key"] for sample in first.samples] == [
        sample["sample_key"] for sample in second.samples
    ]
    assert len({sample["sample_key"] for sample in first.samples}) == 8
    for task in ("i2i", "r2v"):
        selected = [sample for sample in first.samples if sample["task"] == task]
        assert [sample["reference_count"] for sample in selected] == [1, 2, 3, 4]
    assert first.summary["target_files_decoded"] == 0
    assert first.summary["manifest_index_path"].endswith(".idx")


def test_selection_never_opens_target_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, records = _manifest(tmp_path)
    target_paths = {Path(record["target_path"]).resolve() for record in records}
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() in target_paths:
            raise AssertionError(f"target was opened: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    result = select_online_train_samples(manifest, seed=7)
    assert len(result.samples) == 8


def test_reference_order_and_explicit_key_order_are_preserved(tmp_path: Path) -> None:
    manifest, records = _manifest(tmp_path)
    requested = [records[3]["sample_key"], records[0]["sample_key"]]
    result = select_online_train_samples(
        manifest,
        tasks=("i2i",),
        sample_keys=requested,
    )
    assert [sample["sample_key"] for sample in result.samples] == requested
    assert result.samples[0]["reference_paths"] == [
        str(Path(path).resolve()) for path in records[3]["reference_paths"]
    ]


def test_selection_bundle_is_atomic_and_does_not_export_target(tmp_path: Path) -> None:
    manifest, _ = _manifest(tmp_path)
    result = select_online_train_samples(manifest, samples_per_task=1)
    output = write_selection_bundle(result, tmp_path / "selection")

    assert (output / "selected_samples.jsonl").is_file()
    assert (output / "selection_summary.json").is_file()
    sample_dirs = sorted((output / "samples").iterdir())
    assert len(sample_dirs) == 2
    for sample_dir in sample_dirs:
        assert (sample_dir / "prompt.txt").is_file()
        assert (sample_dir / "sample.json").is_file()
        assert not list(sample_dir.glob("target*"))
    with pytest.raises(FileExistsError):
        write_selection_bundle(result, output)


@pytest.mark.parametrize("alias_kind", ["direct", "symlink", "hardlink"])
def test_selection_rejects_reference_target_alias(
    tmp_path: Path,
    alias_kind: str,
) -> None:
    record = _write_record(root=tmp_path, task="i2i", index=90, reference_count=1)
    target = Path(record["target_path"])
    reference = tmp_path / f"alias_{alias_kind}.bin"
    Path(record["reference_paths"][0]).unlink()
    if alias_kind == "direct":
        reference = target
    elif alias_kind == "symlink":
        reference.symlink_to(target)
    else:
        os.link(target, reference)
    record["reference_paths"] = [str(reference)]
    canonical_record = dict(record)
    canonical_record.pop("sample_plan_sha256")
    canonical = json.dumps(
        canonical_record,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    record["sample_plan_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    candidate, reason = _validate_candidate(
        record,
        manifest_path=tmp_path / "train_unique.jsonl",
        manifest_index=0,
        max_caption_chars=None,
    )
    assert candidate is None
    assert reason == "reference_aliases_target"


def test_selection_accepts_distinct_reference_and_target(tmp_path: Path) -> None:
    record = _write_record(root=tmp_path, task="i2i", index=91, reference_count=1)
    candidate, reason = _validate_candidate(
        record,
        manifest_path=tmp_path / "train_unique.jsonl",
        manifest_index=0,
        max_caption_chars=None,
    )
    assert reason is None
    assert candidate is not None


def test_selection_bundle_copy_survives_move_and_original_reference_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, records = _manifest(tmp_path)
    result = select_online_train_samples(
        manifest,
        tasks=("i2i",),
        samples_per_task=1,
    )

    def fail_symlink(*args, **kwargs):
        raise OSError("copy fallback requested")

    monkeypatch.setattr(Path, "symlink_to", fail_symlink)
    original_bundle = write_selection_bundle(result, tmp_path / "selection")
    moved_bundle = tmp_path / "moved" / "selection"
    moved_bundle.parent.mkdir()
    shutil.move(str(original_bundle), moved_bundle)
    for record in records:
        for reference in record["reference_paths"]:
            Path(reference).unlink(missing_ok=True)

    selected = read_selected_samples(moved_bundle / "selected_samples.jsonl")
    assert len(selected) == 1
    assert selected[0]["reference_export_modes"] == ["copy"]
    assert selected[0]["original_reference_paths"]
    for reference in selected[0]["reference_paths"]:
        path = Path(reference)
        assert path.is_file()
        assert moved_bundle in path.parents
    decoded = load_reference_inputs(
        selected[0],
        vlm_reference_preprocess="original",
    )
    assert len(decoded.reference_pixels_vae) == 1
