from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

import pytest
from PIL import Image

from ltx_trainer.online_data import parallel_manifest
from ltx_trainer.online_data.manifest import ManifestReject, build_r2v_record
from ltx_trainer.online_data.parallel_manifest import (
    AnnotationSource,
    BuildOptions,
    R2V_FILTER_SEMANTIC_VERSION,
    _semantic_build_payload,
    build_task_shards,
    task_build_fingerprint,
)


def _row(*, face_cut: list[int]) -> dict[str, object]:
    return {
        "video_path": "/read/only/video.mp4",
        "text": "a valid prompt",
        "crop": [0, 640, 0, 384],
        "face_cut": face_cut,
        "ref_images": ["/read/only/reference.png"],
    }


def _header(*, fps: float, frame_count: int = 1000) -> dict[str, float | int]:
    return {"fps": fps, "frame_count": frame_count, "width": 640, "height": 384}


def _build(
    row: Mapping[str, object],
    header: Mapping[str, float | int],
    validator: Callable[[str], tuple[int, int]],
) -> dict[str, Any]:
    return build_r2v_record(
        row,
        dataset_name="r2v",
        data_root=None,
        manifest_seed=42,
        video_header=header,
        image_validator=validator,
        target_path_validated=True,
    )


def test_source_fps_below_24_rejects_without_reference_validation() -> None:
    reference_calls = 0

    def validator(_path: str) -> tuple[int, int]:
        nonlocal reference_calls
        reference_calls += 1
        return (640, 384)

    with pytest.raises(ManifestReject) as error:
        _build(_row(face_cut=[0, 500]), _header(fps=23.976), validator)
    assert error.value.reason == "source_fps_below_24"
    assert reference_calls == 0


def test_fps25_uses_exact_minimum_span_before_reference_validation() -> None:
    calls = 0

    def validator(_path: str) -> tuple[int, int]:
        nonlocal calls
        calls += 1
        return (640, 384)

    with pytest.raises(ManifestReject) as error:
        _build(_row(face_cut=[0, 125]), _header(fps=25.0), validator)
    assert error.value.reason == "insufficient_frames_for_121_at_24fps"
    assert calls == 0
    record = _build(_row(face_cut=[0, 126]), _header(fps=25.0), validator)
    assert calls == 1
    assert len(record["target_source_frame_indices"]) == 121


def test_span121_at_24fps_builds_exact_121_frame_plan() -> None:
    record = _build(
        _row(face_cut=[0, 121]),
        _header(fps=24.0, frame_count=121),
        lambda _path: (640, 384),
    )
    assert record["target_source_frame_indices"] == list(range(121))


def test_exact_indices_are_unique_increasing_and_inside_face_cut() -> None:
    record = _build(
        _row(face_cut=[17, 400]),
        _header(fps=59.94),
        lambda _path: (640, 384),
    )
    indices = record["target_source_frame_indices"]
    assert len(indices) == 121
    assert len(set(indices)) == 121
    assert all(left < right for left, right in zip(indices, indices[1:]))
    assert indices[0] >= 17
    assert indices[-1] < 400


def test_legacy_variable_length_reader_would_accept_but_exact_mode_rejects() -> None:
    original_fps = 24.0
    start, end = 0, 100
    frame_step = original_fps / min(original_fps, 24.0)
    legacy_actual_frames = min(121, int((end - 1 - start) // frame_step) + 1)
    assert 40 < legacy_actual_frames < 121
    with pytest.raises(ManifestReject) as error:
        _build(_row(face_cut=[start, end]), _header(fps=original_fps), lambda _path: (640, 384))
    assert error.value.reason == "insufficient_face_cut_span_for_121"


def test_probe_performance_options_do_not_change_semantic_fingerprint() -> None:
    source = AnnotationSource(
        task="r2v",
        dataset_name="r2v",
        dataset_order=0,
        annotation_path="/read/only/rows.parquet",
        data_root=None,
        row_count=100,
        task_start_row=0,
        task_end_row=100,
        size=1,
        mtime_ns=1,
        fingerprint="source",
        suffix=".parquet",
    )
    base = BuildOptions(video_workers=16, max_in_flight=128)
    tuned = replace(
        base,
        video_workers=48,
        max_in_flight=512,
        video_probe_mode="isolated",
        video_probe_max_tasks_per_worker=500,
    )
    assert task_build_fingerprint("r2v", base, [source]) == task_build_fingerprint("r2v", tuned, [source])
    assert _semantic_build_payload("r2v", base, [source])["r2v_filter_semantic_version"] == (
        R2V_FILTER_SEMANTIC_VERSION
    )
    assert "r2v_filter_semantic_version" not in _semantic_build_payload("i2i", base, [source])


def _write_builder_fixture(tmp_path: Path, *, rows: int = 12) -> Path:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic video")
    reference = tmp_path / "reference.png"
    Image.new("RGB", (64, 48), (10, 20, 30)).save(reference)
    annotation = tmp_path / "r2v.jsonl"
    annotation.write_text(
        "".join(
            json.dumps(
                {
                    "video_path": str(video),
                    "text": f"prompt {index}",
                    "crop": [0, 64, 0, 48],
                    "face_cut": [0, 300],
                    "ref_images": [str(reference)],
                }
            )
            + "\n"
            for index in range(rows)
        ),
        encoding="utf-8",
    )
    config = tmp_path / "data.yaml"
    config.write_text(
        "datasets:\n"
        "  - name: r2v_fixture\n"
        "    task: r2v\n"
        f"    ann_path: {annotation}\n",
        encoding="utf-8",
    )
    return config


def test_persistent_builder_submits_duplicate_video_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[Any] = []

    class FakePersistentPool:
        def __init__(self, **_kwargs):
            self.submitted = []
            self.worker_start_count = 4
            self.worker_restart_count = 0
            instances.append(self)

        @property
        def probe_submit_count(self) -> int:
            return len(self.submitted)

        def submit(self, path: str) -> Future[dict[str, float | int]]:
            self.submitted.append(path)
            future = Future()
            future.set_result({"fps": 24.0, "frame_count": 300, "width": 64, "height": 48})
            return future

        def close(self, *, wait: bool = True) -> None:
            del wait

    monkeypatch.setattr(parallel_manifest, "PersistentVideoProbePool", FakePersistentPool)
    summary = build_task_shards(
        _write_builder_fixture(tmp_path),
        task="r2v",
        shard_root=tmp_path / "persistent",
        options=BuildOptions(video_workers=4, max_in_flight=12),
    )
    assert summary["accepted_rows"] == 12
    assert len(instances) == 1
    assert len(instances[0].submitted) == 1
    assert summary["video_probe_submissions"] == 1
    assert summary["video_probe_worker_starts"] == 4


def test_isolated_builder_keeps_legacy_probe_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_isolated_probe(_path: str, _timeout: float) -> dict[str, float | int]:
        nonlocal calls
        calls += 1
        return {"fps": 24.0, "frame_count": 300, "width": 64, "height": 48}

    monkeypatch.setattr(parallel_manifest, "probe_video_isolated", fake_isolated_probe)
    summary = build_task_shards(
        _write_builder_fixture(tmp_path),
        task="r2v",
        shard_root=tmp_path / "isolated",
        options=BuildOptions(video_probe_mode="isolated", video_workers=4, max_in_flight=12),
    )
    assert summary["accepted_rows"] == 12
    assert summary["video_probe_mode"] == "isolated"
    assert calls == 1


def test_sqlite_cache_reuses_probe_when_unfinished_shard_is_rebuilt(tmp_path: Path) -> None:
    calls = 0

    def first_probe(_path: str, _timeout: float) -> dict[str, float | int]:
        nonlocal calls
        calls += 1
        return {"fps": 24.0, "frame_count": 300, "width": 64, "height": 48}

    config = _write_builder_fixture(tmp_path, rows=4)
    shard_root = tmp_path / "resume"
    options = BuildOptions(shards_per_task=1, video_workers=2, max_in_flight=4)
    first = build_task_shards(
        config,
        task="r2v",
        shard_root=shard_root,
        options=options,
        probe_runner=first_probe,
    )
    assert first["accepted_rows"] == 4
    assert calls == 1

    (shard_root / "r2v" / "shard_00000.done.json").unlink()

    def forbidden_probe(_path: str, _timeout: float) -> NoReturn:
        raise AssertionError("cached video header must survive an unfinished-shard rebuild")

    resumed = build_task_shards(
        config,
        task="r2v",
        shard_root=shard_root,
        options=options,
        probe_runner=forbidden_probe,
    )
    assert resumed["accepted_rows"] == 4
    assert resumed["video_cache_hits"] >= 1
