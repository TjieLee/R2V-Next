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
from ltx_trainer.online_data.manifest import build_canonical_r2v_record
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

    results = list(
        builder._iter_canonical_rows_with_bounded_probes(
            rows,
            dataset_name="r2v_fixture",
            dataset_type="OpenS2VDataset",
            data_root=None,
            adapter_config=None,
            manifest_seed=42,
            workers=4,
            batch_size=8,
            validation_cache=cache,
        )
    )

    assert [source_record_id for source_record_id, *_ in results] == [row_id for row_id, _ in rows]
    assert len(thread_names) > 1
    assert call_paths.count(str(tmp_path / "video_0.mp4")) == 1
    assert len(call_paths) == 4
    assert results[3][3] is not None
    assert results[3][3].reason == "invalid_video_header"
    assert [result[1].source_record_id for result in results if result[3] is None] == [
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
        for _row_id, canonical, header, error in builder._iter_canonical_rows_with_bounded_probes(
            rows,
            dataset_name="r2v_fixture",
            dataset_type="OpenS2VDataset",
            data_root=None,
            adapter_config=None,
            manifest_seed=42,
            workers=workers,
            batch_size=2,
            validation_cache=cache,
        ):
            assert error is None
            assert canonical is not None and header is not None
            records.append(
                build_canonical_r2v_record(
                    canonical,
                    manifest_seed=42,
                    video_header=header,
                    image_validator=lambda _path: (64, 48),
                    target_path_validated=True,
                )
            )
        payload = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
        return payload, hashlib.sha256(payload.encode()).hexdigest()

    serial_payload, serial_sha = build_payload(workers=1)
    parallel_payload, parallel_sha = build_payload(workers=4)

    assert serial_payload == parallel_payload
    assert serial_sha == parallel_sha


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


def test_write_policy_explicitly_rejects_both_read_only_source_roots() -> None:
    with pytest.raises(ValueError, match="liutao"):
        assert_write_path_allowed("/mnt/workspace/liutao/output.jsonl")
    with pytest.raises(ValueError, match="jiangyuxiang2"):
        assert_write_path_allowed("/mnt/workspace/jiangyuxiang2/output.jsonl")
