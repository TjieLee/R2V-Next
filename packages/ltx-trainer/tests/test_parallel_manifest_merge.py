from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from ltx_trainer.online_data.manifest_index import read_manifest_index, validate_manifest_index
from ltx_trainer.online_data.parallel_manifest import (
    BuildOptions,
    ManifestCollisionError,
    build_task_shards,
    file_sha256,
    merge_manifest_shards,
)


def _built_shards(tmp_path: Path) -> Path:
    target = tmp_path / "target.png"
    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), (10, 20, 30)).save(target)
    Image.new("RGB", (32, 32), (30, 20, 10)).save(reference)
    rows = [
        {"image": str(target), "edit_image": [str(reference)], "prompt": "duplicate"},
        {"image": str(target), "edit_image": [str(reference)], "prompt": "unique one"},
        {"image": str(target), "edit_image": [str(reference)], "prompt": "duplicate"},
        {"image": str(target), "edit_image": [str(reference)], "prompt": "unique two"},
    ]
    annotation = tmp_path / "rows.jsonl"
    annotation.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = tmp_path / "data.yaml"
    config.write_text(
        "datasets:\n"
        "  - name: first_dataset\n"
        "    task: i2i\n"
        f"    ann_path: {annotation}\n",
        encoding="utf-8",
    )
    root = tmp_path / "shards"
    build_task_shards(
        config,
        task="i2i",
        shard_root=root,
        options=BuildOptions(shards_per_task=2, image_workers=2, max_in_flight=4),
    )
    return root


def _merge(root: Path, tmp_path: Path):
    return merge_manifest_shards(
        root,
        tasks=["i2i"],
        output=tmp_path / "train_unique.jsonl",
        reject_output=tmp_path / "rejected.jsonl",
        summary_output=tmp_path / "summary.json",
    )


def test_merge_deduplicates_globally_strips_build_fields_and_builds_ltxidx02(tmp_path: Path) -> None:
    root = _built_shards(tmp_path)
    summary = _merge(root, tmp_path)
    manifest = tmp_path / "train_unique.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [record["caption"] for record in records] == ["duplicate", "unique one", "unique two"]
    assert all(not any(key.startswith("_build_") for key in record) for record in records)
    assert summary["raw_rows"] == 4
    assert summary["accepted_rows"] == 3
    assert summary["duplicate_rows"] == 1
    metadata = validate_manifest_index(manifest)
    assert metadata.manifest_row_count == 3
    offsets, task_indices = read_manifest_index(f"{manifest}.idx", manifest_path=manifest)
    assert len(offsets) == 3
    assert list(task_indices["i2i"]) == [0, 1, 2]


def test_merge_rejects_missing_done_marker(tmp_path: Path) -> None:
    root = _built_shards(tmp_path)
    (root / "i2i" / "shard_00001.done.json").unlink()
    with pytest.raises(ValueError, match="Missing completed shard marker"):
        _merge(root, tmp_path)


def test_merge_rejects_same_key_with_different_plan_and_preserves_existing_output(tmp_path: Path) -> None:
    root = _built_shards(tmp_path)
    second = root / "i2i" / "shard_00001.accepted.jsonl"
    records = [json.loads(line) for line in second.read_text(encoding="utf-8").splitlines()]
    records[0]["sample_plan_sha256"] = "different-plan"
    second.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    marker_path = root / "i2i" / "shard_00001.done.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["files"]["accepted"]["sha256"] = file_sha256(second)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    output = tmp_path / "train_unique.jsonl"
    output.write_text("existing output\n", encoding="utf-8")
    with pytest.raises(ManifestCollisionError, match="different plans"):
        _merge(root, tmp_path)
    assert output.read_text(encoding="utf-8") == "existing output\n"


def test_merge_rejects_corrupt_shard_hash(tmp_path: Path) -> None:
    root = _built_shards(tmp_path)
    accepted = root / "i2i" / "shard_00000.accepted.jsonl"
    accepted.write_text(accepted.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _merge(root, tmp_path)


def test_merge_preserves_dataset_config_order_across_shard_boundaries(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(target)
    Image.new("RGB", (32, 32), (4, 5, 6)).save(reference)
    annotations = []
    for dataset_name, captions in (("first", ["a0", "a1", "a2"]), ("second", ["b0", "b1", "b2"])):
        annotation = tmp_path / f"{dataset_name}.jsonl"
        annotation.write_text(
            "".join(
                json.dumps(
                    {
                        "image": str(target),
                        "edit_image": [str(reference)],
                        "prompt": caption,
                    }
                )
                + "\n"
                for caption in captions
            ),
            encoding="utf-8",
        )
        annotations.append((dataset_name, annotation))
    config = tmp_path / "ordered.yaml"
    config.write_text(
        "datasets:\n"
        + "".join(
            f"  - name: {name}\n    task: i2i\n    ann_path: {annotation}\n"
            for name, annotation in annotations
        ),
        encoding="utf-8",
    )
    root = tmp_path / "ordered_shards"
    build_task_shards(
        config,
        task="i2i",
        shard_root=root,
        options=BuildOptions(shards_per_task=4, image_workers=3, max_in_flight=4),
    )
    summary = merge_manifest_shards(
        root,
        tasks=["i2i"],
        output=tmp_path / "ordered.jsonl",
        reject_output=tmp_path / "ordered_rejected.jsonl",
        summary_output=tmp_path / "ordered_summary.json",
    )
    records = [json.loads(line) for line in (tmp_path / "ordered.jsonl").read_text().splitlines()]
    assert [record["dataset_name"] for record in records] == ["first"] * 3 + ["second"] * 3
    assert [record["caption"] for record in records] == ["a0", "a1", "a2", "b0", "b1", "b2"]
    assert summary["accepted_rows"] == 6
