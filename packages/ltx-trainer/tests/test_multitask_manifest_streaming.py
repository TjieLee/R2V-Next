from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tracemalloc
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from ltx_trainer.online_data.manifest import (
    ManifestReject,
    iter_annotation_rows,
    read_annotation_rows,
)
from ltx_trainer.online_data.manifest_index import (
    INDEX_ENTRY,
    INDEX_HEADER,
    INDEX_MAGIC,
    LEGACY_INDEX_MAGICS,
    build_manifest_offset_index,
    read_jsonl_record_at,
    read_manifest_index,
    validate_manifest_index,
)


def _load_online_manifest_builder_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_multitask_online_manifest.py"
    name = "_test_streaming_build_multitask_online_manifest"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


manifest_builder = _load_online_manifest_builder_module()


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
        {"sample_key": f"sample-{index}", "task": "i2i" if index % 2 == 0 else "r2v", "dataset_name": "unit"}
        for index in range(1000)
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    index_path = build_manifest_offset_index(manifest)
    offsets, task_indices, dataset_indices = read_manifest_index(index_path, manifest_path=manifest)
    assert len(offsets) == 1000
    assert len(task_indices["i2i"]) == 500
    assert len(dataset_indices["unit"]) == 1000
    assert len(task_indices["r2v"]) == 500
    with manifest.open("rb") as handle:
        assert read_jsonl_record_at(handle, offsets[731]) == rows[731]
    metadata = validate_manifest_index(manifest, index_path)
    assert metadata.manifest_row_count == 1000
    assert metadata.manifest_size_bytes == manifest.stat().st_size


def test_manifest_index_rejects_same_row_count_content_replacement(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text('{"task":"i2i","dataset_name":"unit","value":"aaaa"}\n', encoding="utf-8")
    index_path = build_manifest_offset_index(manifest)
    manifest.write_text('{"task":"i2i","dataset_name":"unit","value":"bbbb"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256 mismatch.*Rebuild"):
        read_manifest_index(index_path, manifest_path=manifest)


def test_manifest_index_rejects_manifest_size_or_offset_change(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        '{"task":"i2i","dataset_name":"unit"}\n{"task":"r2v","dataset_name":"unit"}\n',
        encoding="utf-8",
    )
    index_path = build_manifest_offset_index(manifest)
    manifest.write_text(
        '{"task":"i2i","dataset_name":"unit","longer":true}\n{"task":"r2v","dataset_name":"unit"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="file-size mismatch.*Rebuild"):
        read_manifest_index(index_path, manifest_path=manifest)


def test_manifest_index_rejects_truncated_index(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text('{"task":"i2i","dataset_name":"unit"}\n', encoding="utf-8")
    index_path = build_manifest_offset_index(manifest)
    index_path.write_bytes(index_path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="index size mismatch.*Rebuild"):
        read_manifest_index(index_path, manifest_path=manifest)


def test_manifest_index_rejects_legacy_v1_with_rebuild_message(tmp_path: Path) -> None:
    index_path = tmp_path / "train.jsonl.idx"
    index_path.write_bytes(next(iter(LEGACY_INDEX_MAGICS)))

    with pytest.raises(ValueError, match="Legacy online manifest index.*Rebuild"):
        read_manifest_index(index_path)


def test_strict_manifest_index_validation_checks_offsets_and_tasks(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        '{"task":"i2i","dataset_name":"unit"}\n{"task":"r2v","dataset_name":"unit"}\n',
        encoding="utf-8",
    )
    index_path = build_manifest_offset_index(manifest)
    payload = bytearray(index_path.read_bytes())
    first_entry = len(INDEX_MAGIC) + INDEX_HEADER.size
    payload[first_entry : first_entry + INDEX_ENTRY.size] = INDEX_ENTRY.pack(1, 0, 0)
    index_path.write_bytes(payload)

    with pytest.raises(ValueError, match="entry mismatch.*Rebuild"):
        validate_manifest_index(manifest, index_path)


def test_manifest_index_failure_does_not_replace_existing_index(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.jsonl"
    index_path = tmp_path / "bad.jsonl.idx"
    index_path.write_bytes(b"existing-index")
    manifest.write_text(json.dumps({"task": "unsupported"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported task"):
        build_manifest_offset_index(manifest, index_path)
    assert index_path.read_bytes() == b"existing-index"
    assert not list(tmp_path.glob("bad.jsonl.idx.tmp.*"))


def test_same_build_image_validation_cache_reuses_results_and_invalidates_on_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = sqlite3.connect(":memory:")
    cache = manifest_builder._MediaValidationCache(connection)
    image_path = tmp_path / "image.bin"
    image_path.write_bytes(b"first")
    calls = 0

    def _verify(_path: str) -> tuple[int, int]:
        nonlocal calls
        calls += 1
        return (64, 48)

    monkeypatch.setattr(manifest_builder, "_verified_image_size", _verify)
    with ThreadPoolExecutor(max_workers=2) as executor:
        cache.prefetch_images([str(image_path), str(image_path)], executor=executor)
        assert cache.require_image(str(image_path)) == (64, 48)
        assert cache.require_image(str(image_path)) == (64, 48)
        assert calls == 1

        image_path.write_bytes(b"second-version-is-longer")
        cache.prefetch_images([str(image_path)], executor=executor)
        assert cache.require_image(str(image_path)) == (64, 48)
    assert calls == 2
    connection.close()


def test_same_build_media_cache_reuses_negative_image_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = sqlite3.connect(":memory:")
    cache = manifest_builder._MediaValidationCache(connection)
    image_path = tmp_path / "broken.bin"
    image_path.write_bytes(b"broken")
    calls = 0

    def _fail(_path: str) -> tuple[int, int]:
        nonlocal calls
        calls += 1
        raise ValueError("bad image")

    monkeypatch.setattr(manifest_builder, "_verified_image_size", _fail)
    with ThreadPoolExecutor(max_workers=2) as executor:
        cache.prefetch_images([str(image_path)], executor=executor)
        for _ in range(2):
            with pytest.raises(RuntimeError, match="Unreadable image"):
                cache.require_image(str(image_path))
    assert calls == 1
    connection.close()


def test_bounded_video_probe_cache_deduplicates_repeated_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = sqlite3.connect(":memory:")
    cache = manifest_builder._MediaValidationCache(connection)
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    calls = 0

    def _probe(_path: str) -> dict[str, float | int]:
        nonlocal calls
        calls += 1
        return {"fps": 24.0, "frame_count": 240, "width": 832, "height": 480}

    monkeypatch.setattr(manifest_builder, "probe_video", _probe)
    with ThreadPoolExecutor(max_workers=2) as executor:
        cache.prefetch_videos([str(video_path)] * 8, executor=executor)
        results = [cache.require_video(str(video_path)).as_probe_result() for _ in range(8)]
    assert calls == 1
    assert all(header is not None and error is None for header, error in results)
    connection.close()


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
                {
                    "name": "videos",
                    "task": "r2v",
                    "dataset_type": "OpenS2VDataset",
                    "ann_path": "videos.jsonl",
                },
            ]
        },
    )

    def _rows(path: str, *, batch_size: int) -> Iterator[tuple[str, dict[str, Any]]]:
        del batch_size
        task = "i2i" if "images" in str(path) else "r2v"
        for index in range(rows_per_task):
            row: dict[str, Any] = {"index": index, "task": task}
            if task == "i2i":
                row.update(
                    target="target.png",
                    sources=["reference.png"],
                    caption=f"caption-{index}",
                )
            else:
                row.update(
                    video_path="target.mp4",
                    text=f"caption-{index}",
                    crop=[0, 64, 0, 48],
                    face_cut=[0, 121],
                    ref_images=["reference.png"],
                )
            yield str(index), row

    def _record(row: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        task = str(row["task"])
        index = int(row["index"])
        return {
            "sample_key": f"{task}-{index:08d}",
            "sample_plan_sha256": f"plan-{task}-{index:08d}",
            "task": task,
            "dataset_name": str(_kwargs["dataset_name"]),
        }

    monkeypatch.setattr(manifest_builder, "iter_annotation_items", _rows)
    monkeypatch.setattr(manifest_builder, "build_i2i_record", _record)
    monkeypatch.setattr(
        manifest_builder,
        "normalize_r2v_source",
        lambda row, **kwargs: SimpleNamespace(
            video_path="synthetic.mp4",
            dataset_name=kwargs["dataset_name"],
            reference_paths=[],
            row=row,
        ),
    )
    monkeypatch.setattr(
        manifest_builder,
        "prepare_canonical_r2v_record",
        lambda source, **kwargs: SimpleNamespace(
            record=_record(source.row, dataset_name=source.dataset_name, **kwargs),
            reference_paths=(),
        ),
    )
    monkeypatch.setattr(
        manifest_builder,
        "finalize_prepared_canonical_r2v_record",
        lambda prepared, **_kwargs: prepared.record,
    )
    monkeypatch.setattr(
        manifest_builder,
        "_probe_resolved_video",
        lambda _path: manifest_builder._ValidationResult({}),
    )
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
        media_workers=2,
        media_batch_size=128,
        progress_interval_seconds=10.0,
        progress_every_rows=10_000,
        count_total_rows=False,
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
    capsys: pytest.CaptureFixture[str],
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
    assert not (tmp_path / "rejected.jsonl").exists()
    assert not Path(f"{output}.idx").exists()
    assert not (tmp_path / "summary.json").exists()
    assert not list(tmp_path.glob("*.tmp.*"))
    assert not list(tmp_path.glob("*.sqlite"))
    events = capsys.readouterr().err
    assert "event=rows_complete" in events
    assert "event=index_start" in events
    assert "event=index_complete" not in events
    assert "event=manifest_complete" not in events


def test_manifest_completion_event_follows_all_published_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synthetic_builder(monkeypatch, rows_per_task=2)
    output = tmp_path / "train_unique.jsonl"
    rejects = tmp_path / "rejected.jsonl"
    summary = tmp_path / "summary.json"
    index = Path(f"{output}.idx")
    events: list[str] = []
    original_maybe_emit = manifest_builder._ManifestProgress.maybe_emit

    def recording_maybe_emit(
        self: Any,
        *,
        force: bool = False,
        event: str = "progress",
    ) -> None:
        original_maybe_emit(self, force=force, event=event)
        if force:
            events.append(event)
        if event == "manifest_complete":
            assert output.is_file()
            assert rejects.is_file()
            assert index.is_file()
            assert summary.is_file()

    monkeypatch.setattr(manifest_builder._ManifestProgress, "maybe_emit", recording_maybe_emit)
    _run_synthetic_builder(tmp_path)

    required_events = [
        "dataset_complete",
        "rows_complete",
        "index_start",
        "index_complete",
        "publish_complete",
        "manifest_complete",
    ]
    positions = [
        max(position for position, recorded_event in enumerate(events) if recorded_event == name)
        for name in required_events
    ]
    assert positions == sorted(positions)


def test_summary_write_failure_never_emits_manifest_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_synthetic_builder(monkeypatch, rows_per_task=2)
    output = tmp_path / "train_unique.jsonl"
    rejects = tmp_path / "rejected.jsonl"
    summary = tmp_path / "summary.json"

    def fail_summary_write(*_args, **_kwargs) -> None:
        raise RuntimeError("forced summary failure")

    monkeypatch.setattr(manifest_builder, "_atomic_write_json", fail_summary_write)
    with pytest.raises(RuntimeError, match="forced summary failure"):
        _run_synthetic_builder(tmp_path)

    assert output.is_file()
    assert rejects.is_file()
    assert Path(f"{output}.idx").is_file()
    assert not summary.exists()
    assert not list(tmp_path.glob("*.tmp.*"))
    events = capsys.readouterr().err
    assert "event=publish_complete" in events
    assert "event=manifest_complete" not in events


def _run_schema_fixture_builder(tmp_path: Path) -> tuple[Path, Path, Path]:
    output = tmp_path / "schema_train.jsonl"
    rejects = tmp_path / "schema_rejected.jsonl"
    summary = tmp_path / "schema_summary.json"
    manifest_builder.main(
        train_data_config="schema_fixture.yaml",
        output=str(output),
        reject_output=str(rejects),
        summary_output=str(summary),
        manifest_seed=42,
        annotation_batch_size=128,
        media_workers=2,
        media_batch_size=128,
        progress_interval_seconds=10.0,
        progress_every_rows=10_000,
        count_total_rows=False,
        i2i_target_field="image",
        i2i_reference_field="edit_image",
        i2i_caption_field="prompt",
        i2i_crop_field=None,
    )
    return output, rejects, summary


def _patch_two_dataset_schema_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    i2i_rows: list[dict[str, Any]],
    i2i_builder: Callable[..., dict[str, Any]],
) -> list[str]:
    started_datasets: list[str] = []
    monkeypatch.setattr(
        manifest_builder,
        "load_multitask_data_config",
        lambda _path: {
            "datasets": [
                {"name": "i2i_first", "task": "i2i", "ann_path": "i2i.jsonl"},
                {
                    "name": "r2v_second",
                    "task": "r2v",
                    "dataset_type": "OpenS2VDataset",
                    "ann_path": "r2v.jsonl",
                },
            ]
        },
    )

    def rows(path: str, *, batch_size: int) -> Iterator[tuple[str, dict[str, Any]]]:
        del batch_size
        dataset = "i2i_first" if path == "i2i.jsonl" else "r2v_second"
        started_datasets.append(dataset)
        if dataset == "i2i_first":
            for index, row in enumerate(i2i_rows):
                yield str(index), row
            return
        yield (
            "r2v-0",
            {
                "video_path": "target.mp4",
                "text": "video instruction",
                "crop": [0, 64, 0, 48],
                "face_cut": [0, 121],
                "ref_images": ["reference.png"],
            },
        )

    def record(task: str, dataset_name: str, source_id: str) -> dict[str, str]:
        return {
            "sample_key": f"{task}-{source_id}",
            "sample_plan_sha256": f"plan-{task}-{source_id}",
            "task": task,
            "dataset_name": dataset_name,
            "source_record_id": source_id,
        }

    monkeypatch.setattr(manifest_builder, "iter_annotation_items", rows)
    monkeypatch.setattr(manifest_builder, "build_i2i_record", i2i_builder)
    monkeypatch.setattr(
        manifest_builder,
        "normalize_r2v_source",
        lambda row, **kwargs: SimpleNamespace(
            video_path=str(row["video_path"]),
            dataset_name=kwargs["dataset_name"],
            reference_paths=[],
            row=row,
        ),
    )
    monkeypatch.setattr(
        manifest_builder,
        "prepare_canonical_r2v_record",
        lambda source, **_kwargs: SimpleNamespace(
            record=record("r2v", source.dataset_name, "r2v-0"),
            reference_paths=(),
        ),
    )
    monkeypatch.setattr(
        manifest_builder,
        "finalize_prepared_canonical_r2v_record",
        lambda prepared, **_kwargs: prepared.record,
    )
    monkeypatch.setattr(
        manifest_builder,
        "_probe_resolved_image",
        lambda _path: manifest_builder._ValidationResult({"width": 64, "height": 48}),
    )
    monkeypatch.setattr(
        manifest_builder,
        "_probe_resolved_video",
        lambda _path: manifest_builder._ValidationResult({"fps": 24.0, "frame_count": 300, "width": 64, "height": 48}),
    )
    monkeypatch.setattr(manifest_builder, "assert_write_path_allowed", lambda path: Path(path).resolve())
    return started_datasets


def test_i2i_schema_preflight_accepts_real_fields_and_preserves_first_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = {
        "image": "target",
        "edit_image": ["ref"],
        "prompt": "edit",
    }
    seen_rows: list[dict[str, Any]] = []

    def build_i2i(row: dict[str, Any], **kwargs: object) -> dict[str, str]:
        seen_rows.append(row)
        return {
            "sample_key": "i2i-first",
            "sample_plan_sha256": "plan-i2i-first",
            "task": "i2i",
            "dataset_name": str(kwargs["dataset_name"]),
            "source_record_id": "i2i-0",
        }

    started = _patch_two_dataset_schema_fixture(
        monkeypatch,
        i2i_rows=[source],
        i2i_builder=build_i2i,
    )
    output, rejects, summary_path = _run_schema_fixture_builder(tmp_path)

    assert seen_rows == [source]
    assert started == ["i2i_first", "r2v_second"]
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["source_record_id"] for record in records] == ["i2i-0", "r2v-0"]
    assert rejects.read_text(encoding="utf-8") == ""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["task_counts"] == {"i2i": 1, "r2v": 1}
    assert "event=schema_preflight_pass dataset_name=i2i_first" in capsys.readouterr().err


def test_i2i_schema_preflight_rejects_wrong_fields_before_any_media_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wrong_rows = [
        {
            "video": f"target-{index}",
            "reference_images": [f"ref-{index}"],
            "caption": f"edit-{index}",
        }
        for index in range(100)
    ]
    started = _patch_two_dataset_schema_fixture(
        monkeypatch,
        i2i_rows=wrong_rows,
        i2i_builder=lambda _row, **_kwargs: pytest.fail("row build must not start"),
    )
    probe_calls = 0

    def forbid_probe(_path: str) -> object:
        nonlocal probe_calls
        probe_calls += 1
        raise AssertionError("schema preflight must fail before media probes")

    monkeypatch.setattr(manifest_builder, "_probe_resolved_image", forbid_probe)
    monkeypatch.setattr(manifest_builder, "_probe_resolved_video", forbid_probe)
    output = tmp_path / "schema_train.jsonl"
    with pytest.raises(RuntimeError, match="Dataset schema preflight failed") as exc_info:
        _run_schema_fixture_builder(tmp_path)

    message = str(exc_info.value)
    for field in (
        "dataset_name",
        "annotation_path",
        "required_fields",
        "available_fields",
        "missing_fields",
    ):
        assert field in message
    assert '"dataset_name": "i2i_first"' in message
    assert '"required_fields": ["image", "edit_image", "prompt"]' in message
    assert '"missing_fields": ["image", "edit_image", "prompt"]' in message
    assert probe_calls == 0
    assert started == ["i2i_first"]
    assert not output.exists()
    assert not Path(f"{output}.idx").exists()
    events = capsys.readouterr().err
    assert "event=dataset_failed dataset_name=i2i_first task=i2i processed=0" in events
    assert "top_reject_reasons=none" in events
    assert "dataset_name=r2v_second" not in events
    assert "event=index_start" not in events
    assert "event=manifest_complete" not in events


def test_dataset_fail_fast_stops_before_next_dataset_and_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    valid_rows = [
        {
            "image": f"target-{index}",
            "edit_image": [f"ref-{index}"],
            "prompt": f"edit-{index}",
        }
        for index in range(100)
    ]

    def reject_i2i(_row: dict[str, Any], **_kwargs: object) -> dict[str, Any]:
        raise ManifestReject("empty_caption", "synthetic rejection")

    started = _patch_two_dataset_schema_fixture(
        monkeypatch,
        i2i_rows=valid_rows,
        i2i_builder=reject_i2i,
    )
    output = tmp_path / "schema_train.jsonl"
    with pytest.raises(RuntimeError, match="Dataset produced no accepted or duplicate records"):
        _run_schema_fixture_builder(tmp_path)

    assert started == ["i2i_first"]
    assert not output.exists()
    assert not Path(f"{output}.idx").exists()
    assert not (tmp_path / "schema_summary.json").exists()
    events = capsys.readouterr().err
    assert "event=dataset_failed dataset_name=i2i_first task=i2i processed=100" in events
    assert "accepted=0" in events
    assert "rejected=100" in events
    assert "duplicates=0" in events
    assert "top_reject_reasons=empty_caption:100" in events
    assert "dataset_name=r2v_second" not in events
    assert "event=index_start" not in events
    assert "event=manifest_complete" not in events
