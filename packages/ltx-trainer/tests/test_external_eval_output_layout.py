from __future__ import annotations

import json
import time
from pathlib import Path

from ltx_trainer.online_inference.external_eval_runner import (
    classify_sample_output,
    external_sample_dir,
    incremental_summary,
    write_static_gallery,
)
from ltx_trainer.online_inference.read_only_sources import ReadOnlySourcePolicy


def _record() -> dict[str, object]:
    return {
        "dataset_name": "opens2v_open_domain",
        "source_record_id": "multiface_3",
        "output_id": "multiface_3",
        "caption": "two people turn toward the camera",
        "reference_paths": ["/read-only/a.png", "/read-only/b.png"],
    }


def test_structured_output_complete_incomplete_and_resume_states(tmp_path: Path) -> None:
    policy = ReadOnlySourcePolicy(writable_root=tmp_path / "owned")
    dataset_root = policy.ensure_directory(tmp_path / "owned" / "opens2v_open_domain")
    sample_dir = external_sample_dir(dataset_root, "multiface_3", policy)
    assert classify_sample_output(sample_dir, dry_run=False) == "missing"
    policy.ensure_directory(sample_dir)
    policy.atomic_write_text(sample_dir / "generated.mp4", "video")
    assert classify_sample_output(sample_dir, dry_run=False) == "incomplete"
    policy.atomic_write_json(
        sample_dir / "success.json",
        {"status": "success", "artifacts": ["generated.mp4"]},
    )
    assert classify_sample_output(sample_dir, dry_run=False) == "complete"


def test_summary_is_atomically_replaceable_and_gallery_is_static(tmp_path: Path) -> None:
    policy = ReadOnlySourcePolicy(writable_root=tmp_path / "owned")
    dataset_root = policy.ensure_directory(tmp_path / "owned" / "opens2v_open_domain")
    record = _record()
    sample_dir = policy.ensure_directory(dataset_root / "samples" / "multiface_3")
    policy.atomic_write_json(
        sample_dir / "metadata.json",
        {
            "seed": 123,
            "elapsed_seconds": 2.5,
            "crop_risks": [{"vertical_crop_risk": True}],
        },
    )
    gallery = write_static_gallery(records=[record], dataset_root=dataset_root, policy=policy)
    source = gallery.read_text(encoding="utf-8")
    assert "multiface_3" in source
    assert "samples/multiface_3/generated.mp4" in source
    assert "reference_00.png" in source and "reference_01.png" in source
    assert "vertical_crop_risk" in source

    started = time.perf_counter()
    summary = incremental_summary(
        dataset_root=dataset_root,
        policy=policy,
        run_metadata={"dataset_name": "opens2v_open_domain"},
        results=[{"status": "success", "output_id": "multiface_3"}],
        failures=[],
        started=started,
    )
    first = json.loads(summary.read_text(encoding="utf-8"))
    assert first["success_count"] == 1
    incremental_summary(
        dataset_root=dataset_root,
        policy=policy,
        run_metadata={"dataset_name": "opens2v_open_domain"},
        results=[
            {"status": "success", "output_id": "multiface_3"},
            {"status": "skipped_existing", "output_id": "singleobj_1"},
        ],
        failures=[],
        started=started,
    )
    second = json.loads(summary.read_text(encoding="utf-8"))
    assert second["result_count"] == 2
    assert second["skipped_existing_count"] == 1
    assert not list(dataset_root.glob(".run_summary.json.tmp.*"))
