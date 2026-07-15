from __future__ import annotations

import os
import time
from collections.abc import Callable

import pytest

from ltx_trainer.online_data.video_probe_pool import PersistentVideoProbePool, VideoProbeError


def _synthetic_probe(path: str) -> dict[str, float | int]:
    if path == "timeout":
        time.sleep(0.3)
    if path == "crash":
        os._exit(17)
    if path == "invalid":
        raise ValueError("invalid synthetic header")
    return {"fps": 24.0, "frame_count": 300, "width": 64, "height": 48}


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true before timeout")
        time.sleep(0.005)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_100_probes_reuse_long_lived_workers() -> None:
    pool = PersistentVideoProbePool(
        workers=4,
        timeout_seconds=1.0,
        max_tasks_per_worker=1000,
        start_method="fork",
        probe_fn=_synthetic_probe,
    )
    try:
        results = [future.result(timeout=2.0) for future in [pool.submit(f"video-{i}") for i in range(100)]]
        assert all(result["frame_count"] == 300 for result in results)
        assert pool.probe_submit_count == 100
        assert pool.worker_start_count == 4
        assert pool.worker_restart_count == 0
    finally:
        pool.close()


def test_single_timeout_restarts_only_its_worker_and_pool_continues() -> None:
    pool = PersistentVideoProbePool(
        workers=2,
        timeout_seconds=0.05,
        max_tasks_per_worker=1000,
        start_method="fork",
        probe_fn=_synthetic_probe,
    )
    try:
        timed_out = pool.submit("timeout")
        healthy = pool.submit("healthy")
        assert healthy.result(timeout=1.0)["fps"] == 24.0
        with pytest.raises(VideoProbeError, match="timed out") as error:
            timed_out.result(timeout=1.0)
        assert error.value.reason == "video_probe_timeout"
        assert pool.submit("after-timeout").result(timeout=1.0)["frame_count"] == 300
        _wait_until(lambda: pool.worker_restart_count == 1)
        assert pool.worker_start_count == 3
        assert pool.probe_timeout_count == 1
    finally:
        pool.close()


def test_worker_crash_restarts_slot_and_later_probe_succeeds() -> None:
    pool = PersistentVideoProbePool(
        workers=1,
        timeout_seconds=1.0,
        max_tasks_per_worker=1000,
        start_method="fork",
        probe_fn=_synthetic_probe,
    )
    try:
        with pytest.raises(VideoProbeError, match="crashed") as error:
            pool.submit("crash").result(timeout=2.0)
        assert error.value.reason == "video_probe_worker_crash"
        assert pool.submit("after-crash").result(timeout=1.0)["width"] == 64
        assert pool.probe_crash_count == 1
        assert pool.worker_restart_count == 1
    finally:
        pool.close()


def test_worker_recycles_after_configured_task_count() -> None:
    pool = PersistentVideoProbePool(
        workers=1,
        timeout_seconds=1.0,
        max_tasks_per_worker=2,
        start_method="fork",
        probe_fn=_synthetic_probe,
    )
    try:
        for index in range(5):
            assert pool.submit(f"video-{index}").result(timeout=1.0)["height"] == 48
        _wait_until(lambda: pool.worker_restart_count == 2)
        assert pool.worker_start_count == 3
    finally:
        pool.close()


def test_close_terminates_all_worker_processes() -> None:
    pool = PersistentVideoProbePool(
        workers=3,
        timeout_seconds=1.0,
        start_method="fork",
        probe_fn=_synthetic_probe,
    )
    pids = pool.active_worker_pids
    assert len(pids) == 3
    pool.close()
    assert pool.active_worker_pids == []
    assert all(not _pid_is_alive(pid) for pid in pids)
