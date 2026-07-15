from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import pytest

from ltx_trainer.online_data.manifest import ManifestReject
from ltx_trainer.online_data.parallel_manifest import (
    BuildOptions,
    build_task_shards,
    prefilter_r2v_annotation,
    run_r2v_prefilter,
)


def _row(*, face_cut: list[int], ref_images: list[str] | None = None) -> dict[str, object]:
    return {
        "video_path": "/read/only/video.mp4",
        "text": "a valid prompt",
        "crop": [0, 64, 0, 48],
        "face_cut": face_cut,
        "ref_images": ["/read/only/reference.png"] if ref_images is None else ref_images,
    }


@pytest.mark.parametrize("face_cut", ([68, 165], [0, 120]))
def test_short_face_cut_is_rejected_by_annotation_only_prefilter(face_cut: list[int]) -> None:
    with pytest.raises(ManifestReject) as error:
        prefilter_r2v_annotation(_row(face_cut=face_cut), data_root=None)
    assert error.value.reason == "insufficient_face_cut_span_for_121"


def test_prefilter_validates_crop_face_cut_and_reference_schema_without_media() -> None:
    with pytest.raises(ManifestReject) as crop_error:
        prefilter_r2v_annotation({**_row(face_cut=[0, 200]), "crop": [0, 0, 0, 48]}, data_root=None)
    assert crop_error.value.reason == "invalid_crop"
    with pytest.raises(ManifestReject) as face_error:
        prefilter_r2v_annotation(_row(face_cut=[-1, 200]), data_root=None)
    assert face_error.value.reason == "invalid_face_cut"
    with pytest.raises(ManifestReject) as reference_error:
        prefilter_r2v_annotation(_row(face_cut=[0, 200], ref_images=[]), data_root=None)
    assert reference_error.value.reason == "missing_reference"


def test_full_builder_does_not_probe_or_validate_reference_for_short_span(tmp_path: Path) -> None:
    annotation = tmp_path / "rows.jsonl"
    annotation.write_text(json.dumps(_row(face_cut=[68, 165])) + "\n", encoding="utf-8")
    config = tmp_path / "data.yaml"
    config.write_text(
        "datasets:\n"
        "  - name: short\n"
        "    task: r2v\n"
        f"    ann_path: {annotation}\n",
        encoding="utf-8",
    )
    probe_calls = 0

    def forbidden_probe(_path: str, _timeout: float) -> NoReturn:
        nonlocal probe_calls
        probe_calls += 1
        raise AssertionError("short span must not reach video probe")

    root = tmp_path / "shards"
    summary = build_task_shards(
        config,
        task="r2v",
        shard_root=root,
        options=BuildOptions(shards_per_task=1, video_workers=2, max_in_flight=2),
        probe_runner=forbidden_probe,
    )
    assert probe_calls == 0
    assert summary["accepted_rows"] == 0
    assert summary["reject_reason_counts"] == {"insufficient_face_cut_span_for_121": 1}
    assert summary["runtime_stats"]["annotation_prefilter_rejected"] == 1
    assert summary["image_cache_misses"] == 0
    assert summary["video_cache_misses"] == 0


def test_prefilter_only_writes_statistics_without_media_access(tmp_path: Path) -> None:
    rows = [_row(face_cut=[0, 120]), _row(face_cut=[0, 121]), _row(face_cut=[10, 250])]
    annotation = tmp_path / "rows.jsonl"
    annotation.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = tmp_path / "data.yaml"
    config.write_text(
        "datasets:\n"
        "  - name: prefilter\n"
        "    task: r2v\n"
        f"    ann_path: {annotation}\n",
        encoding="utf-8",
    )
    output = tmp_path / "prefilter"
    summary = run_r2v_prefilter(config, shard_root=output, options=BuildOptions())
    assert summary["raw_rows"] == 3
    assert summary["prefilter_passed"] == 2
    assert summary["prefilter_rejected"] == 1
    assert summary["media_open_count"] == 0
    assert summary["face_cut_span_histogram"] == {"120": 1, "121": 1, "240": 1}
    assert (output / "r2v_prefilter.accepted.jsonl").is_file()
    assert (output / "r2v_prefilter.rejected.jsonl").is_file()
    assert (output / "r2v_prefilter_summary.json").is_file()
