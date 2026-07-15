from __future__ import annotations

import json
import tracemalloc
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest

from ltx_trainer.online_data.manifest import iter_annotation_rows, read_annotation_rows
from ltx_trainer.online_data.manifest_index import (
    build_manifest_offset_index,
    read_jsonl_record_at,
    read_manifest_index,
)
from scripts import build_multitask_online_manifest as manifest_builder


def test_iter_annotation_rows_streams_100k_jsonl_without_read_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotation = tmp_path / "synthetic.jsonl"
    with annotation.open("w", encoding="utf-8") as handle:
        for index in range(100_000):
            handle.write(json.dumps({"index": index, "text": f"sample-{index}"}) + "\n")

    def _forbid_read_text(*_args, **_kwargs) -> None:
        raise AssertionError("streaming JSONL must not call Path.read_text")

    monkeypatch.setattr(Path, "read_text", _forbid_read_text)
    tracemalloc.start()
    count = sum(1 for _ in iter_annotation_rows(annotation))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert count == 100_000
    assert peak < 16 * 1024 * 1024


def test_streaming_rows_match_eager_wrapper_for_small_input(tmp_path: Path) -> None:
    annotation = tmp_path / "small.jsonl"
    rows = [{"index": index, "value": None if index % 2 else index} for index in range(20)]
    annotation.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert list(iter_annotation_rows(annotation)) == read_annotation_rows(annotation)


def test_binary_manifest_index_preserves_offsets_and_compact_task_indices(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    rows = [
        {"sample_key": f"sample-{index}", "task": "i2i" if index % 2 == 0 else "r2v"}
        for index in range(1000)
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    index_path = build_manifest_offset_index(manifest)
    offsets, task_indices = read_manifest_index(index_path)
    assert len(offsets) == 1000
    assert len(task_indices["i2i"]) == 500
    assert len(task_indices["r2v"]) == 500
    with manifest.open("rb") as handle:
        assert read_jsonl_record_at(handle, offsets[731]) == rows[731]


def test_manifest_index_failure_does_not_replace_existing_index(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.jsonl"
    index_path = tmp_path / "bad.jsonl.idx"
    index_path.write_bytes(b"existing-index")
    manifest.write_text(json.dumps({"task": "unsupported"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported task"):
        build_manifest_offset_index(manifest, index_path)
    assert index_path.read_bytes() == b"existing-index"
    assert not list(tmp_path.glob("bad.jsonl.idx.tmp.*"))


def _patch_synthetic_builder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows_per_task: int,
) -> None:
    monkeypatch.setattr(
        manifest_builder,
        "load_multitask_data_config",
        lambda _path: {
            "datasets": [
                {"name": "images", "task": "i2i", "ann_path": "images.jsonl"},
                {"name": "videos", "task": "r2v", "ann_path": "videos.jsonl"},
            ]
        },
    )

    def _rows(path: str, *, batch_size: int) -> Iterator[dict[str, Any]]:
        del batch_size
        task = "i2i" if "images" in str(path) else "r2v"
        for index in range(rows_per_task):
            yield {"index": index, "task": task}

    def _record(row: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        task = str(row["task"])
        index = int(row["index"])
        return {
            "sample_key": f"{task}-{index:08d}",
            "sample_plan_sha256": f"plan-{task}-{index:08d}",
            "task": task,
        }

    def _probed(
        rows: Iterable[dict[str, Any]],
        **_kwargs: Any,
    ) -> Iterator[tuple[dict[str, Any], dict[str, float], None]]:
        for row in rows:
            yield row, {"fps": 24.0, "frame_count": 240, "width": 832, "height": 480}, None

    monkeypatch.setattr(manifest_builder, "iter_annotation_rows", _rows)
    monkeypatch.setattr(manifest_builder, "build_i2i_record", _record)
    monkeypatch.setattr(manifest_builder, "build_r2v_record", _record)
    monkeypatch.setattr(manifest_builder, "_iter_rows_with_bounded_probes", _probed)
    monkeypatch.setattr(manifest_builder, "assert_write_path_allowed", lambda path: Path(path).resolve())


def _run_synthetic_builder(tmp_path: Path) -> tuple[Path, Path, Path]:
    output = tmp_path / "train_unique.jsonl"
    rejects = tmp_path / "rejected.jsonl"
    summary = tmp_path / "summary.json"
    manifest_builder.main(
        train_data_config="unused.yaml",
        output=str(output),
        reject_output=str(rejects),
        summary_output=str(summary),
        manifest_seed=42,
        annotation_batch_size=4096,
        probe_workers=2,
        probe_batch_size=128,
        i2i_target_field="target",
        i2i_reference_field="sources",
        i2i_caption_field="caption",
        i2i_crop_field=None,
    )
    return output, rejects, summary


def test_manifest_builder_streams_100k_rows_with_bounded_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_builder(monkeypatch, rows_per_task=50_000)
    tracemalloc.start()
    output, rejects, summary_path = _run_synthetic_builder(tmp_path)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    with output.open("r", encoding="utf-8") as handle:
        assert sum(1 for _ in handle) == 100_000
    assert rejects.read_text(encoding="utf-8") == ""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["raw_rows"] == 100_000
    assert summary["accepted_rows"] == 100_000
    assert summary["task_counts"] == {"i2i": 50_000, "r2v": 50_000}
    assert peak < 64 * 1024 * 1024


def test_manifest_builder_failure_preserves_existing_outputs_and_cleans_temps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_builder(monkeypatch, rows_per_task=2)
    output = tmp_path / "train_unique.jsonl"
    output.write_text("existing-manifest\n", encoding="utf-8")

    def _fail_index(*_args, **_kwargs) -> None:
        raise RuntimeError("forced index failure")

    monkeypatch.setattr(manifest_builder, "build_manifest_offset_index", _fail_index)
    with pytest.raises(RuntimeError, match="forced index failure"):
        _run_synthetic_builder(tmp_path)
    assert output.read_text(encoding="utf-8") == "existing-manifest\n"
    assert not list(tmp_path.glob("*.tmp.*"))
    assert not list(tmp_path.glob("*.sqlite"))
