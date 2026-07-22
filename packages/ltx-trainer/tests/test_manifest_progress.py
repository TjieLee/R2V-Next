from __future__ import annotations

import importlib.util
import io
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import typer

from ltx_trainer.online_data.manifest import annotation_row_count


def _load_online_manifest_builder_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_multitask_online_manifest.py"
    name = "_test_progress_build_multitask_online_manifest"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


manifest_builder = _load_online_manifest_builder_module()


class _FakeClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _progress_reporter(
    *,
    clock: _FakeClock,
    stream: io.StringIO,
) -> tuple[manifest_builder._ManifestProgress, sqlite3.Connection]:
    connection = sqlite3.connect(":memory:")
    cache = manifest_builder._MediaValidationCache(connection)
    reporter = manifest_builder._ManifestProgress(
        started_at=clock.value,
        last_emitted_at=clock.value,
        last_emitted_rows=0,
        total_rows=10,
        progress_interval_seconds=10.0,
        progress_every_rows=3,
        validation_cache=cache,
        clock=clock,
        stream=stream,
    )
    return reporter, connection


def test_progress_reporter_obeys_time_and_row_thresholds_and_forces_boundaries() -> None:
    clock = _FakeClock(100.0)
    stream = io.StringIO()
    progress, connection = _progress_reporter(clock=clock, stream=stream)
    try:
        progress.start_dataset(dataset_name="r2v_fixture", task="r2v", total_rows=5)
        assert "event=dataset_start" in stream.getvalue()
        stream.seek(0)
        stream.truncate(0)

        progress.record_accepted()
        progress.maybe_emit()
        assert stream.getvalue() == ""

        clock.value = 109.9
        progress.maybe_emit()
        assert stream.getvalue() == ""

        clock.value = 110.0
        progress.maybe_emit()
        first_progress = stream.getvalue()
        assert first_progress.count("event=progress") == 1
        assert "dataset_processed=1" in first_progress
        assert "accepted=1" in first_progress
        assert "rows_per_second=" in first_progress
        stream.seek(0)
        stream.truncate(0)

        progress.record_rejected("invalid_video_header")
        progress.maybe_emit()
        progress.record_duplicate()
        progress.maybe_emit()
        assert stream.getvalue() == ""
        progress.record_rejected("invalid_video_header")
        progress.maybe_emit()
        row_progress = stream.getvalue()
        assert row_progress.count("event=progress") == 1
        assert "dataset_processed=4" in row_progress
        assert "rejected=2" in row_progress
        assert "duplicates=1" in row_progress
        assert "accept_rate=25.00%" in row_progress
        assert "top_reject_reasons=invalid_video_header:2" in row_progress

        progress.maybe_emit(force=True, event="dataset_complete")
        progress.maybe_emit(force=True, event="manifest_complete")
        boundaries = stream.getvalue()
        assert "event=dataset_complete" in boundaries
        assert "dataset_accepted=1" in boundaries
        assert "dataset_rejected=2" in boundaries
        assert "event=manifest_complete" in boundaries
    finally:
        connection.close()


def test_progress_defaults_to_stderr_and_never_writes_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock = _FakeClock(50.0)
    connection = sqlite3.connect(":memory:")
    try:
        progress = manifest_builder._ManifestProgress(
            started_at=50.0,
            last_emitted_at=50.0,
            last_emitted_rows=0,
            total_rows=None,
            progress_interval_seconds=10.0,
            progress_every_rows=10_000,
            validation_cache=manifest_builder._MediaValidationCache(connection),
            clock=clock,
        )
        progress.start_dataset(dataset_name="i2i_fixture", task="i2i", total_rows=None)
    finally:
        connection.close()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "event=dataset_start" in captured.err
    assert "dataset_total=unknown" in captured.err
    assert "global_total=unknown" in captured.err
    assert "eta_seconds=unknown" in captured.err


def test_probe_heartbeat_emits_before_slow_batch_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = sqlite3.connect(":memory:")
    cache = manifest_builder._MediaValidationCache(connection)
    stream = io.StringIO()
    started = time.perf_counter()
    progress = manifest_builder._ManifestProgress(
        started_at=started,
        last_emitted_at=started,
        last_emitted_rows=0,
        total_rows=20,
        progress_interval_seconds=0.05,
        progress_every_rows=10_000,
        validation_cache=cache,
        stream=stream,
    )
    progress.start_dataset(dataset_name="slow_images", task="i2i", total_rows=20)
    stream.seek(0)
    stream.truncate(0)
    paths = [str(tmp_path / f"slow_{index}.png") for index in range(20)]

    def fake_signature(path: str):
        return manifest_builder._MediaSignature(path, len(path), len(path) * 10)

    def slow_probe(_path: str):
        time.sleep(0.2)
        return manifest_builder._ValidationResult({"width": 64, "height": 48})

    monkeypatch.setattr(manifest_builder, "_media_signature", fake_signature)
    monkeypatch.setattr(manifest_builder, "_probe_resolved_image", slow_probe)
    try:
        with ThreadPoolExecutor(max_workers=10, thread_name_prefix="manifest-media") as executor:
            cache.prefetch_images(
                paths,
                executor=executor,
                progress=progress,
                phase="i2i_image_probe",
                batch_rows=20,
            )
    finally:
        connection.close()

    probe_lines = [
        line for line in stream.getvalue().splitlines()
        if "event=probe_progress" in line
    ]
    assert len(probe_lines) >= 2
    assert any("phase=i2i_image_probe" in line for line in probe_lines)
    assert any("pending=0" not in line for line in probe_lines)


def test_annotation_row_count_handles_streaming_and_in_memory_formats(tmp_path: Path) -> None:
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_bytes(b'{"row": 0}\n{"row": 1}')
    json_list = tmp_path / "rows_list.json"
    json_list.write_text(json.dumps([{"row": 0}, {"row": 1}, {"row": 2}]), encoding="utf-8")
    json_dict = tmp_path / "rows_dict.json"
    json_dict.write_text(json.dumps({"a": {"row": 0}, "b": {"row": 1}}), encoding="utf-8")
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("row\n0\n1\n", encoding="utf-8")

    assert annotation_row_count(jsonl) == 2
    assert annotation_row_count(json_list) == 3
    assert annotation_row_count(json_dict) == 2
    assert annotation_row_count(csv_path) is None


def test_media_cli_names_keep_probe_aliases() -> None:
    command = typer.main.get_command(manifest_builder.app)
    parameters = {parameter.name: parameter for parameter in command.params}

    assert set(parameters["media_workers"].opts) == {"--media-workers", "--probe-workers"}
    assert set(parameters["media_batch_size"].opts) == {"--media-batch-size", "--probe-batch-size"}
