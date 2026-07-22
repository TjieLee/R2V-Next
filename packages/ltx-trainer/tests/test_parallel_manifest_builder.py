from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from ltx_trainer.online_data import parallel_manifest
from ltx_trainer.online_data.manifest import ManifestReject, finalize_prepared_canonical_r2v_record
from ltx_trainer.online_data.parallel_manifest import (
    AnnotationSource,
    BuildOptions,
    MediaValidationError,
    PersistentMediaCache,
    build_jsonl_offset_index,
    build_task_shards,
    discover_annotation_sources,
    iter_source_range,
    read_jsonl_offset_index,
    validate_done_marker,
)
from ltx_trainer.online_data.path_safety import assert_write_path_allowed
from ltx_trainer.online_data import path_safety


def _load_online_manifest_builder_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_multitask_online_manifest.py"
    name = "_test_build_multitask_online_manifest"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (64, 48), color).save(path)


def _write_i2i_fixture(tmp_path: Path, *, rows: int = 12) -> tuple[Path, list[dict[str, object]]]:
    target = tmp_path / "target.png"
    first_ref = tmp_path / "ref_a.png"
    second_ref = tmp_path / "ref_b.png"
    _write_image(target, (255, 0, 0))
    _write_image(first_ref, (0, 255, 0))
    _write_image(second_ref, (0, 0, 255))
    payload = [
        {
            "image": str(target),
            "edit_image": [str(second_ref), str(first_ref)],
            "prompt": f"edit request {index}",
        }
        for index in range(rows)
    ]
    annotation = tmp_path / "i2i.jsonl"
    annotation.write_text("".join(json.dumps(row) + "\n" for row in payload), encoding="utf-8")
    config = tmp_path / "data.yaml"
    config.write_text(
        "data_root: null\n"
        "datasets:\n"
        "  - name: i2i_fixture\n"
        "    task: i2i\n"
        f"    ann_path: {annotation}\n",
        encoding="utf-8",
    )
    return config, payload


def test_parallel_i2i_builder_preserves_source_and_reference_order(tmp_path: Path) -> None:
    config, source_rows = _write_i2i_fixture(tmp_path)
    root = tmp_path / "shards"
    summary = build_task_shards(
        config,
        task="i2i",
        shard_root=root,
        options=BuildOptions(
            shards_per_task=3,
            image_workers=4,
            max_in_flight=5,
            progress_interval_seconds=0.01,
        ),
    )

    assert summary["accepted_rows"] == len(source_rows)
    records = []
    for shard_id in range(3):
        marker = validate_done_marker(root / "i2i" / f"shard_{shard_id:05d}.done.json")
        assert marker["raw_rows"] == 4
        accepted = Path(marker["files"]["accepted"]["path"])
        records.extend(json.loads(line) for line in accepted.read_text(encoding="utf-8").splitlines())
    assert [record["_build_task_row_index"] for record in records] == list(range(len(source_rows)))
    assert records[0]["reference_paths"] == source_rows[0]["edit_image"]
    progress = json.loads((root / "build_i2i.progress.json").read_text(encoding="utf-8"))
    assert progress["processed"] == len(source_rows)
    assert progress["in_flight"] == 0
    assert summary["media_cache_misses"] == 3


def _write_r2v_fixture(tmp_path: Path, *, rows: int = 12) -> tuple[Path, list[dict[str, object]]]:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic video signature")
    reference = tmp_path / "r2v_reference.png"
    _write_image(reference, (12, 34, 56))
    payload = [
        {
            "video_path": str(video),
            "text": f"video request {index}",
            "crop": [0, 64, 0, 48],
            "face_cut": [0, 300],
            "ref_images": [str(reference)],
        }
        for index in range(rows)
    ]
    annotation = tmp_path / "r2v.jsonl"
    annotation.write_text("".join(json.dumps(row) + "\n" for row in payload), encoding="utf-8")
    config = tmp_path / "r2v_data.yaml"
    config.write_text(
        "datasets:\n"
        "  - name: r2v_fixture\n"
        "    task: r2v\n"
        "    dataset_type: OpenS2VDataset\n"
        f"    ann_path: {annotation}\n",
        encoding="utf-8",
    )
    return config, payload


def test_parallel_r2v_builder_deduplicates_inflight_probe_and_builds_121_frame_plan(
    tmp_path: Path,
) -> None:
    config, source_rows = _write_r2v_fixture(tmp_path)
    calls = 0

    def fake_probe(_path: str, _timeout: float):
        nonlocal calls
        calls += 1
        time.sleep(0.01)
        return {"fps": 24.0, "frame_count": 300, "width": 64, "height": 48}

    root = tmp_path / "r2v_shards"
    summary = build_task_shards(
        config,
        task="r2v",
        shard_root=root,
        options=BuildOptions(shards_per_task=2, video_workers=6, max_in_flight=8),
        probe_runner=fake_probe,
    )
    records = []
    for accepted in sorted((root / "r2v").glob("shard_*.accepted.jsonl")):
        records.extend(json.loads(line) for line in accepted.read_text(encoding="utf-8").splitlines())
    assert summary["accepted_rows"] == len(source_rows)
    assert calls == 1
    assert summary["media_cache_misses"] == 2
    assert all(len(record["target_source_frame_indices"]) == 121 for record in records)
    assert all(
        left < right
        for record in records
        for left, right in zip(
            record["target_source_frame_indices"],
            record["target_source_frame_indices"][1:],
        )
    )


def _canonical_probe_rows(tmp_path: Path) -> list[tuple[str, dict[str, object]]]:
    reference = tmp_path / "reference.png"
    _write_image(reference, (11, 22, 33))
    videos = []
    for index in range(4):
        video = tmp_path / f"video_{index}.mp4"
        video.write_bytes(f"video-{index}".encode())
        videos.append(video)
    paths = [videos[0], videos[0], videos[1], videos[2], videos[3]]
    return [
        (
            f"row-{index}",
            {
                "video_path": str(path),
                "text": f"video request {index}",
                "crop": [0, 64, 0, 48],
                "face_cut": [0, 300],
                "ref_images": [str(reference)],
            },
        )
        for index, path in enumerate(paths)
    ]


def _write_r2v_two_stage_rows(tmp_path: Path) -> list[tuple[str, dict[str, object]]]:
    rows: list[tuple[str, dict[str, object]]] = []
    for index in range(100):
        video = tmp_path / f"two_stage_video_{index:03d}.mp4"
        video.write_bytes(f"video-{index}".encode())
        references = []
        for reference_index in range(2):
            reference = tmp_path / f"two_stage_ref_{index:03d}_{reference_index}.png"
            reference.write_bytes(b"reference")
            references.append(str(reference))
        clip_end = 60 if index < 90 else 160
        rows.append(
            (
                f"row-{index:03d}",
                {
                    "video_path": str(video),
                    "text": f"video request {index}",
                    "crop": [0, 64, 0, 48],
                    "face_cut": [0, clip_end],
                    "ref_images": references,
                },
            )
        )
    return rows


def _build_r2v_two_stage_payload(
    builder,
    rows: list[tuple[str, dict[str, object]]],
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
) -> tuple[bytes, bytes, dict[str, object]]:
    video_calls: list[str] = []
    image_calls: list[str] = []

    def fake_probe_video(path: str) -> dict[str, float | int]:
        video_calls.append(path)
        return {"fps": 24.0, "frame_count": 200, "width": 64, "height": 48}

    def fake_probe_image(path: str):
        image_calls.append(path)
        return builder._ValidationResult({"width": 64, "height": 48})

    monkeypatch.setattr(builder, "probe_video", fake_probe_video)
    monkeypatch.setattr(builder, "_probe_resolved_image", fake_probe_image)
    connection = sqlite3.connect(":memory:")
    cache = builder._MediaValidationCache(connection)
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="manifest-media") as executor:
            for row_index, (row_id, prepared, error) in enumerate(
                builder._iter_canonical_rows_with_bounded_probes(
                    rows,
                    dataset_name="r2v_fixture",
                    dataset_type="OpenS2VDataset",
                    data_root=None,
                    adapter_config=None,
                    manifest_seed=42,
                    anchor_frame_ratio=0.10,
                    executor=executor,
                    batch_size=100,
                    validation_cache=cache,
                )
            ):
                try:
                    if error is not None:
                        raise error
                    assert prepared is not None
                    accepted.append(
                        finalize_prepared_canonical_r2v_record(
                            prepared,
                            image_validator=cache.require_image,
                        )
                    )
                except ManifestReject as exc:
                    rejected.append(
                        {
                            "row_index": row_index,
                            "source_record_id": row_id,
                            "reason": exc.reason,
                        }
                    )
    finally:
        connection.close()
    accepted_payload = b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
        for record in accepted
    )
    rejected_payload = b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
        for record in rejected
    )
    return accepted_payload, rejected_payload, {
        "video_calls": video_calls,
        "image_calls": image_calls,
        "image_probe_submitted": cache.image_probe_submitted,
        "video_probe_submitted": cache.video_probe_submitted,
        "reference_probes_skipped_due_video_reject": cache.reference_probes_skipped_due_video_reject,
        "tmp_path": str(tmp_path),
    }


def test_r2v_reference_probes_defer_until_video_plan_passes_and_remain_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_online_manifest_builder_module()
    rows = _write_r2v_two_stage_rows(tmp_path)

    results = [
        _build_r2v_two_stage_payload(
            builder,
            rows,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            workers=workers,
        )
        for workers in (1, 4, 16)
    ]

    for payload_index in (0, 1):
        assert results[0][payload_index] == results[1][payload_index] == results[2][payload_index]
    stats = results[0][2]
    assert len(stats["video_calls"]) == 100
    assert len(stats["image_calls"]) == 20
    assert stats["video_probe_submitted"] == 100
    assert stats["image_probe_submitted"] == 20
    assert stats["reference_probes_skipped_due_video_reject"] == 180
    for index in range(90):
        assert not any(f"two_stage_ref_{index:03d}_" in path for path in stats["image_calls"])
    accepted = [json.loads(line) for line in results[0][0].splitlines()]
    rejected = [json.loads(line) for line in results[0][1].splitlines()]
    assert [record["source_record_id"] for record in accepted] == [f"row-{index:03d}" for index in range(90, 100)]
    assert [record["source_record_id"] for record in rejected] == [f"row-{index:03d}" for index in range(90)]
    assert {record["reason"] for record in rejected} == {"insufficient_frames_for_121_at_24fps"}


def test_r2v_missing_or_invalid_video_skips_references_but_feasible_invalid_reference_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_online_manifest_builder_module()
    invalid_video = tmp_path / "invalid_video.mp4"
    invalid_video.write_bytes(b"invalid")
    feasible_video = tmp_path / "feasible_video.mp4"
    feasible_video.write_bytes(b"feasible")
    invalid_reference = tmp_path / "invalid_reference.png"
    invalid_reference.write_bytes(b"invalid-ref")
    skipped_reference = tmp_path / "skipped_reference.png"
    skipped_reference.write_bytes(b"skipped-ref")
    rows = [
        (
            "missing-video",
            {
                "video_path": str(tmp_path / "missing_video.mp4"),
                "text": "missing target",
                "crop": [0, 64, 0, 48],
                "face_cut": [0, 160],
                "ref_images": [str(skipped_reference)],
            },
        ),
        (
            "invalid-video",
            {
                "video_path": str(invalid_video),
                "text": "invalid target",
                "crop": [0, 64, 0, 48],
                "face_cut": [0, 160],
                "ref_images": [str(skipped_reference)],
            },
        ),
        (
            "invalid-reference",
            {
                "video_path": str(feasible_video),
                "text": "invalid reference",
                "crop": [0, 64, 0, 48],
                "face_cut": [0, 160],
                "ref_images": [str(invalid_reference)],
            },
        ),
    ]
    image_calls: list[str] = []

    def fake_probe_video(path: str) -> dict[str, float | int]:
        if path == str(invalid_video):
            raise RuntimeError("bad video")
        return {"fps": 24.0, "frame_count": 200, "width": 64, "height": 48}

    def fake_probe_image(path: str):
        image_calls.append(path)
        return builder._ValidationResult(None, "invalid_image", f"bad image: {path}")

    monkeypatch.setattr(builder, "probe_video", fake_probe_video)
    monkeypatch.setattr(builder, "_probe_resolved_image", fake_probe_image)
    connection = sqlite3.connect(":memory:")
    cache = builder._MediaValidationCache(connection)
    rejected: list[tuple[str, str]] = []
    try:
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="manifest-media") as executor:
            for row_id, prepared, error in builder._iter_canonical_rows_with_bounded_probes(
                rows,
                dataset_name="r2v_fixture",
                dataset_type="OpenS2VDataset",
                data_root=None,
                adapter_config=None,
                manifest_seed=42,
                anchor_frame_ratio=0.10,
                executor=executor,
                batch_size=3,
                validation_cache=cache,
            ):
                try:
                    if error is not None:
                        raise error
                    assert prepared is not None
                    finalize_prepared_canonical_r2v_record(
                        prepared,
                        image_validator=cache.require_image,
                    )
                except ManifestReject as exc:
                    rejected.append((row_id, exc.reason))
    finally:
        connection.close()

    assert image_calls == [str(invalid_reference)]
    assert rejected == [
        ("missing-video", "missing_target"),
        ("invalid-video", "invalid_video_header"),
        ("invalid-reference", "missing_reference"),
    ]


def test_canonical_manifest_probe_is_parallel_cached_ordered_and_rejects_probe_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_online_manifest_builder_module()
    rows = _canonical_probe_rows(tmp_path)
    call_paths: list[str] = []
    thread_names: set[str] = set()

    def fake_probe(path: str) -> dict[str, object]:
        call_paths.append(path)
        thread_names.add(threading.current_thread().name)
        time.sleep(0.02)
        if path.endswith("video_2.mp4"):
            raise RuntimeError("synthetic probe failure")
        return {"fps": 24.0, "frame_count": 300, "width": 64, "height": 48}

    monkeypatch.setattr(builder, "probe_video", fake_probe)
    connection = sqlite3.connect(":memory:")
    cache = builder._MediaValidationCache(connection)
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="manifest-media") as executor:
        results = list(
            builder._iter_canonical_rows_with_bounded_probes(
                rows,
                dataset_name="r2v_fixture",
                dataset_type="OpenS2VDataset",
                data_root=None,
                adapter_config=None,
                manifest_seed=42,
                anchor_frame_ratio=0.10,
                executor=executor,
                batch_size=8,
                validation_cache=cache,
            )
        )

    assert [source_record_id for source_record_id, *_ in results] == [row_id for row_id, _ in rows]
    assert len(thread_names) > 1
    assert call_paths.count(str(tmp_path / "video_0.mp4")) == 1
    assert len(call_paths) == 4
    assert results[3][2] is not None
    assert results[3][2].reason == "invalid_video_header"
    assert [result[1].record["source_record_id"] for result in results if result[2] is None] == [
        "row-0",
        "row-1",
        "row-2",
        "row-4",
    ]


def test_canonical_manifest_probe_worker_count_does_not_change_records_or_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_online_manifest_builder_module()
    rows = _canonical_probe_rows(tmp_path)[:3]
    monkeypatch.setattr(
        builder,
        "probe_video",
        lambda _path: {"fps": 24.0, "frame_count": 300, "width": 64, "height": 48},
    )

    def build_payload(workers: int) -> tuple[str, str]:
        connection = sqlite3.connect(":memory:")
        cache = builder._MediaValidationCache(connection)
        records = []
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="manifest-media",
        ) as executor:
            for _row_id, prepared, error in builder._iter_canonical_rows_with_bounded_probes(
                rows,
                dataset_name="r2v_fixture",
                dataset_type="OpenS2VDataset",
                data_root=None,
                adapter_config=None,
                manifest_seed=42,
                anchor_frame_ratio=0.10,
                executor=executor,
                batch_size=2,
                validation_cache=cache,
            ):
                assert error is None
                assert prepared is not None
                records.append(
                    finalize_prepared_canonical_r2v_record(
                        prepared,
                        image_validator=lambda _path: (64, 48),
                    )
                )
        payload = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
        return payload, hashlib.sha256(payload.encode()).hexdigest()

    serial_payload, serial_sha = build_payload(workers=1)
    parallel_payload, parallel_sha = build_payload(workers=4)

    assert serial_payload == parallel_payload
    assert serial_sha == parallel_sha


def _write_v7_manifest_fixture(tmp_path: Path) -> Path:
    target_slow = tmp_path / "target_slow.png"
    target_fast = tmp_path / "target_fast.png"
    reference_slow = tmp_path / "reference_slow.png"
    reference_fast = tmp_path / "reference_fast.png"
    for index, path in enumerate((target_slow, target_fast, reference_slow, reference_fast)):
        _write_image(path, (index * 30, index * 20, index * 10))

    i2i_rows = [
        {
            "source_record_id": "i2i-0",
            "image": str(target_slow),
            "edit_image": [str(reference_slow), str(reference_slow), str(reference_fast)],
            "prompt": "first deterministic edit",
        },
        {
            "source_record_id": "i2i-bad",
            "image": str(target_fast),
            "edit_image": [str(reference_slow)],
        },
        {
            "source_record_id": "i2i-1",
            "image": str(target_fast),
            "edit_image": [str(reference_fast)],
            "prompt": "second deterministic edit",
        },
    ]
    i2i_rows.append(dict(i2i_rows[0]))
    i2i_annotation = tmp_path / "i2i_v7.jsonl"
    i2i_annotation.write_text(
        "".join(json.dumps(row) + "\n" for row in i2i_rows),
        encoding="utf-8",
    )

    video_slow = tmp_path / "video_slow.mp4"
    video_fast = tmp_path / "video_fast.mp4"
    video_slow.write_bytes(b"slow video signature")
    video_fast.write_bytes(b"fast video signature")
    r2v_rows = [
        {
            "source_record_id": "r2v-0",
            "video_path": str(video_slow),
            "text": "first deterministic video",
            "crop": [0, 64, 0, 48],
            "face_cut": [0, 121],
            "ref_images": [str(reference_slow), str(reference_slow)],
        },
        {
            "source_record_id": "r2v-1",
            "video_path": str(video_fast),
            "text": "second deterministic video",
            "crop": [0, 64, 0, 48],
            "face_cut": [0, 121],
            "ref_images": [str(reference_fast)],
        },
        {
            "source_record_id": "r2v-bad",
            "text": "missing target video",
            "crop": [0, 64, 0, 48],
            "face_cut": [0, 121],
            "ref_images": [str(reference_fast)],
        },
    ]
    r2v_rows.append(dict(r2v_rows[0]))
    r2v_annotation = tmp_path / "r2v_v7.jsonl"
    r2v_annotation.write_text(
        "".join(json.dumps(row) + "\n" for row in r2v_rows),
        encoding="utf-8",
    )

    config = tmp_path / "v7_data.yaml"
    config.write_text(
        "datasets:\n"
        "  - name: i2i_v7\n"
        "    task: i2i\n"
        f"    ann_path: {i2i_annotation}\n"
        "  - name: r2v_v7\n"
        "    task: r2v\n"
        "    dataset_type: OpenS2VDataset\n"
        f"    ann_path: {r2v_annotation}\n",
        encoding="utf-8",
    )
    return config


def test_manifest_builder_is_byte_deterministic_for_workers_1_4_16_and_delayed_probes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    builder = _load_online_manifest_builder_module()
    config = _write_v7_manifest_fixture(tmp_path)
    monkeypatch.setattr(builder, "assert_write_path_allowed", lambda path: Path(path).resolve())
    original_image_probe = builder._verified_image_size
    probe_threads: set[str] = set()

    def delayed_image_probe(path: str) -> tuple[int, int]:
        probe_threads.add(threading.current_thread().name)
        time.sleep(0.03 if "slow" in path else 0.001)
        return original_image_probe(path)

    def delayed_video_probe(path: str) -> dict[str, float | int]:
        probe_threads.add(threading.current_thread().name)
        time.sleep(0.03 if "slow" in path else 0.001)
        return {"fps": 24.0, "frame_count": 300, "width": 64, "height": 48}

    monkeypatch.setattr(builder, "_verified_image_size", delayed_image_probe)
    monkeypatch.setattr(builder, "probe_video", delayed_video_probe)
    output = tmp_path / "train.jsonl"
    rejects = tmp_path / "rejected.jsonl"
    summary_path = tmp_path / "summary.json"

    def run(workers: int) -> tuple[bytes, bytes, bytes, dict[str, object]]:
        builder.main(
            train_data_config=str(config),
            output=str(output),
            reject_output=str(rejects),
            summary_output=str(summary_path),
            manifest_seed=42,
            annotation_batch_size=2,
            media_workers=workers,
            media_batch_size=2,
            progress_interval_seconds=10.0,
            progress_every_rows=10_000,
            count_total_rows=True,
            i2i_target_field="image",
            i2i_reference_field="edit_image",
            i2i_caption_field="prompt",
            i2i_crop_field=None,
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return (
            output.read_bytes(),
            rejects.read_bytes(),
            Path(f"{output}.idx").read_bytes(),
            summary,
        )

    results = [run(workers) for workers in (1, 4, 16)]
    for position in range(3):
        assert results[0][position] == results[1][position] == results[2][position]
    ignored_summary_fields = {"elapsed_seconds", "peak_rss_gb", "rows_per_second", "eta_seconds"}
    stable_summaries = [
        {key: value for key, value in result[3].items() if key not in ignored_summary_fields}
        for result in results
    ]
    assert stable_summaries[0] == stable_summaries[1] == stable_summaries[2]

    records = [json.loads(line) for line in results[0][0].splitlines()]
    rejected = [json.loads(line) for line in results[0][1].splitlines()]
    assert [record["source_record_id"] for record in records] == ["i2i-0", "i2i-1", "r2v-0", "r2v-1"]
    assert [record["source_record_id"] for record in rejected] == ["i2i-bad", "r2v-bad"]
    summary = results[0][3]
    assert summary["accepted_rows"] == 4
    assert summary["rejected_rows"] == 2
    assert summary["duplicate_rows"] == 2
    assert summary["task_counts"] == {"i2i": 2, "r2v": 2}
    assert summary["dataset_counts"] == {"i2i_v7": 2, "r2v_v7": 2}
    assert summary["image_probe_submitted"] == 4
    assert summary["video_probe_submitted"] == 2
    assert probe_threads
    assert all(name.startswith("manifest-media") for name in probe_threads)
    assert not list(tmp_path.glob("*.tmp.*"))
    assert not list(tmp_path.glob("*.dedup.*.sqlite"))
    captured = capsys.readouterr()
    assert '"accepted_rows": 4' in captured.out
    assert "event=dataset_start" in captured.err
    assert "event=dataset_complete" in captured.err
    assert "event=manifest_complete" in captured.err


def test_media_prefetch_deduplicates_paths_and_keeps_sqlite_on_main_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_online_manifest_builder_module()
    raw_connection = sqlite3.connect(":memory:")
    sqlite_thread_ids: list[int] = []

    class RecordingConnection:
        def execute(self, *args, **kwargs):
            sqlite_thread_ids.append(threading.get_ident())
            return raw_connection.execute(*args, **kwargs)

    cache = builder._MediaValidationCache(RecordingConnection())
    images = [tmp_path / f"cache_{index}.png" for index in range(3)]
    for index, path in enumerate(images):
        _write_image(path, (index, index, index))
    original_probe = builder._probe_resolved_image
    worker_thread_ids: list[int] = []

    def recording_probe(path: str):
        worker_thread_ids.append(threading.get_ident())
        time.sleep(0.005)
        return original_probe(path)

    monkeypatch.setattr(builder, "_probe_resolved_image", recording_probe)
    main_thread_id = threading.get_ident()
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="manifest-media") as executor:
        with pytest.raises(RuntimeError, match="cache miss after batch prefetch"):
            cache.require_image(str(images[2]))
        cache.prefetch_images(
            [str(images[0]), str(images[0]), str(images[1]), str(images[0])],
            executor=executor,
        )
        assert cache.require_image(str(images[0])) == (64, 48)
        assert cache.require_image(str(images[0])) == (64, 48)
        cache.prefetch_images(
            [str(images[0]), str(images[1]), str(images[2]), str(images[2])],
            executor=executor,
        )
        assert cache.require_image(str(images[2])) == (64, 48)
        with pytest.raises(RuntimeError, match="owner thread"):
            executor.submit(cache.require_image, str(images[0])).result()

    assert cache.image_probe_submitted == 3
    assert len(worker_thread_ids) == 3
    assert all(thread_id != main_thread_id for thread_id in worker_thread_ids)
    assert sqlite_thread_ids
    assert set(sqlite_thread_ids) == {main_thread_id}
    assert cache.image_require_calls == 4
    assert cache.image_cache_hits == 2
    raw_connection.close()


def test_media_require_uses_prefetched_signature_without_restat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_online_manifest_builder_module()
    connection = sqlite3.connect(":memory:")
    cache = builder._MediaValidationCache(connection)
    image_path = str(tmp_path / "repeated.png")
    signature_calls = 0
    probe_calls = 0

    def fake_signature(path: str):
        nonlocal signature_calls
        signature_calls += 1
        return builder._MediaSignature(path, 123, 456)

    def fake_probe(path: str):
        nonlocal probe_calls
        probe_calls += 1
        return builder._ValidationResult({"width": 64, "height": 48})

    monkeypatch.setattr(builder, "_media_signature", fake_signature)
    monkeypatch.setattr(builder, "_probe_resolved_image", fake_probe)
    try:
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="manifest-media") as executor:
            cache.prefetch_images([image_path] * 1000, executor=executor)
            for _ in range(1000):
                assert cache.require_image(image_path) == (64, 48)
    finally:
        connection.close()

    assert signature_calls == 1
    assert probe_calls == 1
    assert cache.signature_submitted == 1
    assert cache.image_probe_submitted == 1
    assert cache.image_unique_paths == 1
    assert cache.image_require_calls == 1000


def test_media_signature_recomputed_per_batch_but_probe_reuses_sqlite_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_online_manifest_builder_module()
    connection = sqlite3.connect(":memory:")
    cache = builder._MediaValidationCache(connection)
    image_path = str(tmp_path / "cached-across-batches.png")
    signature_calls = 0
    probe_calls = 0

    def fake_signature(path: str):
        nonlocal signature_calls
        signature_calls += 1
        return builder._MediaSignature(path, 777, 888)

    def fake_probe(path: str):
        nonlocal probe_calls
        probe_calls += 1
        return builder._ValidationResult({"width": 64, "height": 48})

    monkeypatch.setattr(builder, "_media_signature", fake_signature)
    monkeypatch.setattr(builder, "_probe_resolved_image", fake_probe)
    try:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="manifest-media") as executor:
            cache.prefetch_images([image_path], executor=executor)
            assert cache.require_image(image_path) == (64, 48)
            cache.prefetch_images([image_path], executor=executor)
            assert cache.require_image(image_path) == (64, 48)
    finally:
        connection.close()

    assert signature_calls == 2
    assert probe_calls == 1
    assert cache.image_cache_hits == 1
    assert cache.image_cache_misses == 1
    assert cache.signature_submitted == 2
    assert cache.image_probe_submitted == 1


def test_media_cache_runtime_writes_use_dedicated_writer_thread(tmp_path: Path) -> None:
    image = tmp_path / "cached.png"
    _write_image(image, (9, 8, 7))
    cache = PersistentMediaCache(
        tmp_path / "media_cache.sqlite",
        probe_timeout_seconds=1.0,
    )
    writer_threads: list[str] = []
    original_store = cache._store

    def recording_store(*args, **kwargs):
        writer_threads.append(threading.current_thread().name)
        return original_store(*args, **kwargs)

    cache._store = recording_store
    try:
        assert cache.validate_image(str(image)) == (64, 48)
    finally:
        cache.close()
    assert len(writer_threads) == 1
    assert writer_threads[0].startswith("manifest-cache-writer-")
    assert writer_threads[0] != threading.current_thread().name


def test_i2i_and_r2v_task_builds_share_root_without_overwriting_metadata(tmp_path: Path) -> None:
    i2i_dir = tmp_path / "i2i_source"
    r2v_dir = tmp_path / "r2v_source"
    i2i_dir.mkdir()
    r2v_dir.mkdir()
    i2i_config, _ = _write_i2i_fixture(i2i_dir, rows=4)
    r2v_config, _ = _write_r2v_fixture(r2v_dir, rows=4)
    i2i_annotation = i2i_dir / "i2i.jsonl"
    r2v_annotation = r2v_dir / "r2v.jsonl"
    combined = tmp_path / "combined.yaml"
    combined.write_text(
        "datasets:\n"
        "  - name: i2i_fixture\n"
        "    task: i2i\n"
        f"    ann_path: {i2i_annotation}\n"
        "  - name: r2v_fixture\n"
        "    task: r2v\n"
        "    dataset_type: OpenS2VDataset\n"
        f"    ann_path: {r2v_annotation}\n",
        encoding="utf-8",
    )
    assert i2i_config.is_file() and r2v_config.is_file()
    root = tmp_path / "combined_shards"
    options = BuildOptions(shards_per_task=2, image_workers=2, video_workers=2, max_in_flight=4)

    def build(task: str):
        return build_task_shards(
            combined,
            task=task,
            shard_root=root,
            options=options,
            probe_runner=lambda _path, _timeout: {
                "fps": 24.0,
                "frame_count": 300,
                "width": 64,
                "height": 48,
            },
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        summaries = list(executor.map(build, ["i2i", "r2v"]))
    metadata = json.loads((root / "build_metadata.json").read_text(encoding="utf-8"))
    assert set(metadata["tasks"]) == {"i2i", "r2v"}
    assert all(summary["completed_shards"] == 2 for summary in summaries)
    assert len(list((root / "i2i").glob("shard_*.done.json"))) == 2
    assert len(list((root / "r2v").glob("shard_*.done.json"))) == 2


def test_ordered_parallel_map_bounds_reordering_and_emits_source_order() -> None:
    source = AnnotationSource(
        task="i2i",
        dataset_name="synthetic",
        dataset_order=0,
        annotation_path="/read/only/source.jsonl",
        data_root=None,
        row_count=40,
        task_start_row=0,
        task_end_row=40,
        size=1,
        mtime_ns=1,
        fingerprint="fingerprint",
        suffix=".jsonl",
    )
    items = [(index, source, index, {"index": index}) for index in range(40)]

    def worker(item):
        index = item[0]
        time.sleep((7 - index % 7) * 0.0002)
        return parallel_manifest.RowBuildResult(index, index, 0, "synthetic", record={"index": index})

    emitted = list(
        parallel_manifest._ordered_parallel_map(items, worker, workers=8, max_in_flight=11)
    )
    assert [result.task_row_index for result, _ in emitted] == list(range(40))
    assert max(in_flight for _, in_flight in emitted) <= 11


def test_ordered_parallel_map_emits_heartbeat_while_workers_are_busy() -> None:
    source = AnnotationSource(
        task="i2i",
        dataset_name="synthetic",
        dataset_order=0,
        annotation_path="/read/only/source.jsonl",
        data_root=None,
        row_count=1,
        task_start_row=0,
        task_end_row=1,
        size=1,
        mtime_ns=1,
        fingerprint="fingerprint",
        suffix=".jsonl",
    )

    def worker(item):
        time.sleep(0.02)
        return parallel_manifest.RowBuildResult(item[0], item[2], 0, "synthetic", record={})

    emitted = list(
        parallel_manifest._ordered_parallel_map(
            [(0, source, 0, {})],
            worker,
            workers=1,
            max_in_flight=1,
            heartbeat_seconds=0.001,
        )
    )
    assert emitted[-1][0] is not None
    assert any(result is None and in_flight == 1 for result, in_flight in emitted[:-1])


def test_jsonl_offset_index_streams_100k_and_reads_ranges_directly(tmp_path: Path) -> None:
    annotation = tmp_path / "large.jsonl"
    with annotation.open("w", encoding="utf-8") as handle:
        for index in range(100_000):
            handle.write(json.dumps({"index": index}) + "\n")
    tracemalloc.start()
    index_path, header = build_jsonl_offset_index(annotation, tmp_path / "indexes")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert header["row_count"] == 100_000
    assert peak < 32 * 1024 * 1024
    parsed_header, entries_start = read_jsonl_offset_index(index_path)
    assert parsed_header["offset_entry_size"] == 8
    assert entries_start > 0

    config = tmp_path / "large.yaml"
    config.write_text(
        "datasets:\n"
        "  - name: large\n"
        "    task: i2i\n"
        f"    ann_path: {annotation}\n",
        encoding="utf-8",
    )
    source = discover_annotation_sources(
        config,
        shard_root=tmp_path / "build",
        tasks=["i2i"],
    )["i2i"][0]
    rows = list(iter_source_range(source, 99_990, 100_000, batch_size=4))
    assert [row[1]["index"] for row in rows] == list(range(99_990, 100_000))


def test_parquet_source_reads_only_requested_row_range(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    annotation = tmp_path / "rows.parquet"
    parquet.write_table(
        pyarrow.table({"index": list(range(100)), "text": [f"row-{index}" for index in range(100)]}),
        annotation,
        row_group_size=13,
    )
    config = tmp_path / "parquet.yaml"
    config.write_text(
        "datasets:\n"
        "  - name: parquet_fixture\n"
        "    task: r2v\n"
        "    dataset_type: OpenS2VDataset\n"
        f"    ann_path: {annotation}\n",
        encoding="utf-8",
    )
    source = discover_annotation_sources(
        config,
        shard_root=tmp_path / "parquet_build",
        tasks=["r2v"],
    )["r2v"][0]
    rows = list(iter_source_range(source, 37, 62, batch_size=7))
    assert [row_index for row_index, _ in rows] == list(range(37, 62))
    assert [row["index"] for _, row in rows] == list(range(37, 62))


def test_probe_timeout_terminates_isolated_process(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"terminated": False, "joined": False}

    class Connection:
        def poll(self, _timeout):
            return False

        def close(self):
            return None

    class Process:
        def start(self):
            return None

        def terminate(self):
            state["terminated"] = True

        def join(self, timeout=None):
            state["joined"] = timeout is not None

        def is_alive(self):
            return False

    class Context:
        def Pipe(self, duplex=False):
            assert not duplex
            return Connection(), Connection()

        def Process(self, target, args, daemon):
            assert target is parallel_manifest._probe_child
            assert daemon
            return Process()

    monkeypatch.setattr(parallel_manifest.mp, "get_context", lambda _method: Context())
    with pytest.raises(MediaValidationError, match="timed out") as error:
        parallel_manifest.probe_video_isolated("/read/only/video.mp4", 0.01)
    assert error.value.reason == "video_probe_timeout"
    assert state == {"terminated": True, "joined": True}


def test_write_policy_explicitly_rejects_read_only_source_roots() -> None:
    with pytest.raises(ValueError, match="liutao"):
        assert_write_path_allowed("/mnt/workspace/liutao/output.jsonl")
    with pytest.raises(ValueError, match="public"):
        assert_write_path_allowed("/mnt/workspace/public/output.jsonl")
    with pytest.raises(ValueError, match="jiangyuxiang2"):
        assert_write_path_allowed("/mnt/workspace/jiangyuxiang2/output.jsonl")


def test_write_policy_uses_resolved_paths_and_allows_only_litengjie_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "mnt" / "workspace"
    allowed = workspace / "litengjie"
    liutao = workspace / "liutao"
    public = workspace / "public"
    jiangyuxiang2 = workspace / "jiangyuxiang2"
    outside = tmp_path / "outside"
    for root in (allowed, liutao, public, jiangyuxiang2, outside):
        root.mkdir(parents=True)
    external_target = outside / "target"
    external_target.mkdir()
    symlink = allowed / "link_to_external"
    symlink.symlink_to(external_target, target_is_directory=True)

    monkeypatch.setattr(path_safety, "_ALLOWED_WRITE_ROOT", allowed)
    monkeypatch.setattr(path_safety, "_FORBIDDEN_WRITE_ROOTS", (liutao, public, jiangyuxiang2))
    monkeypatch.chdir(tmp_path)

    permitted = allowed / "subdir" / "file.jsonl"
    assert path_safety.assert_write_path_allowed(permitted) == permitted.resolve()

    rejected_paths = [
        liutao / "file.jsonl",
        public / "file.jsonl",
        jiangyuxiang2 / "file.jsonl",
        tmp_path / "tmp-output.jsonl",
        Path("outside") / "relative.jsonl",
        symlink / "escaped.jsonl",
    ]
    for path in rejected_paths:
        with pytest.raises(ValueError):
            path_safety.assert_write_path_allowed(path)
