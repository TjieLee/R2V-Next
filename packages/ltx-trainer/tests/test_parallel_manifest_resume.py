from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import pytest
from PIL import Image

from ltx_trainer.online_data.parallel_manifest import BuildOptions, ShardLock, build_task_shards


def _fixture(tmp_path: Path, rows: int = 16) -> tuple[Path, Path]:
    target = tmp_path / "target.png"
    reference = tmp_path / "reference.png"
    Image.new("RGB", (48, 48), (1, 2, 3)).save(target)
    Image.new("RGB", (48, 48), (4, 5, 6)).save(reference)
    annotation = tmp_path / "rows.jsonl"
    annotation.write_text(
        "".join(
            json.dumps(
                {
                    "image": str(target),
                    "edit_image": [str(reference)],
                    "prompt": f"prompt {index}",
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
        "  - name: fixture\n"
        "    task: i2i\n"
        f"    ann_path: {annotation}\n",
        encoding="utf-8",
    )
    return config, annotation


def _options(**overrides) -> BuildOptions:
    values = {
        "shards_per_task": 4,
        "image_workers": 4,
        "max_in_flight": 8,
        "progress_interval_seconds": 0.01,
    }
    values.update(overrides)
    return BuildOptions(**values)


def test_resume_reuses_valid_shards_and_only_rebuilds_missing_marker(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    root = tmp_path / "shards"
    build_task_shards(config, task="i2i", shard_root=root, options=_options())
    markers = sorted((root / "i2i").glob("shard_*.done.json"))
    original_mtimes = {path.name: path.stat().st_mtime_ns for path in markers}

    build_task_shards(config, task="i2i", shard_root=root, options=_options())
    assert {path.name: path.stat().st_mtime_ns for path in markers} == original_mtimes

    time.sleep(0.01)
    markers[2].unlink()
    build_task_shards(config, task="i2i", shard_root=root, options=_options())
    updated = {path.name: path.stat().st_mtime_ns for path in markers}
    assert updated[markers[2].name] > original_mtimes[markers[2].name]
    for marker in (markers[0], markers[1], markers[3]):
        assert updated[marker.name] == original_mtimes[marker.name]


def test_source_fingerprint_change_invalidates_completed_shards(tmp_path: Path) -> None:
    config, annotation = _fixture(tmp_path)
    root = tmp_path / "shards"
    build_task_shards(config, task="i2i", shard_root=root, options=_options(manifest_seed=42))
    marker = root / "i2i" / "shard_00000.done.json"
    old_marker = json.loads(marker.read_text(encoding="utf-8"))
    time.sleep(0.01)
    os.utime(annotation, None)
    build_task_shards(config, task="i2i", shard_root=root, options=_options(manifest_seed=42))
    new_marker = json.loads(marker.read_text(encoding="utf-8"))
    assert new_marker["source_fingerprints"] != old_marker["source_fingerprints"]


def test_semantic_build_config_change_invalidates_completed_shards(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    root = tmp_path / "shards"
    build_task_shards(config, task="i2i", shard_root=root, options=_options(manifest_seed=42))
    marker = root / "i2i" / "shard_00000.done.json"
    old_marker = json.loads(marker.read_text(encoding="utf-8"))
    build_task_shards(config, task="i2i", shard_root=root, options=_options(manifest_seed=43))
    new_marker = json.loads(marker.read_text(encoding="utf-8"))
    assert new_marker["build_config_fingerprint"] != old_marker["build_config_fingerprint"]


def test_active_lock_fails_fast_and_stale_lock_requires_explicit_recovery(tmp_path: Path) -> None:
    path = tmp_path / "shard_00000.lock"
    active = ShardLock(path, config_fingerprint="a")
    active.acquire()
    try:
        with pytest.raises(RuntimeError, match="Active shard lock"):
            ShardLock(path, config_fingerprint="a").acquire()
    finally:
        active.release()

    path.write_text(
        json.dumps(
            {
                "pid": 999_999_999,
                "hostname": socket.gethostname(),
                "started_epoch": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="recover-stale-locks"):
        ShardLock(path, config_fingerprint="a").acquire()
    recovered = ShardLock(path, config_fingerprint="a", recover_stale=True)
    recovered.acquire()
    recovered.release()


def test_corrupt_media_cache_is_rebuilt_without_invalidating_done_shards(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    root = tmp_path / "shards"
    build_task_shards(config, task="i2i", shard_root=root, options=_options())
    cache = root / "i2i" / "media_cache.sqlite"
    cache.write_bytes(b"not a sqlite database")
    summary = build_task_shards(config, task="i2i", shard_root=root, options=_options())
    assert summary["completed_shards"] == 4
    assert cache.stat().st_size > len(b"not a sqlite database")
