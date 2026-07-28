"""Long-lived, individually recoverable worker pool for video header probes."""

from __future__ import annotations

import atexit
import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from ltx_trainer.online_data.manifest import probe_video


class VideoProbeError(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class _ProbeRequest:
    task_id: int
    path: str
    future: Future[dict[str, Any]]


@dataclass
class _WorkerSlot:
    worker_id: int
    generation: int = 0
    process: Any = None
    task_queue: Any = None
    current: _ProbeRequest | None = None
    deadline: float | None = None


def _probe_worker_main(
    worker_id: int,
    generation: int,
    task_queue: Any,
    result_queue: Any,
    max_tasks: int,
    probe_fn: Callable[[str], Mapping[str, Any]],
) -> None:
    completed = 0
    while True:
        request = task_queue.get()
        if request is None:
            return
        task_id, path = request
        recycle = completed + 1 >= max_tasks
        try:
            payload = dict(probe_fn(path))
        except BaseException as exc:
            result_queue.put(
                (
                    worker_id,
                    generation,
                    task_id,
                    False,
                    type(exc).__name__,
                    str(exc),
                    recycle,
                )
            )
        else:
            result_queue.put((worker_id, generation, task_id, True, payload, "", recycle))
        completed += 1
        if recycle:
            return


class PersistentVideoProbePool:
    """Persistent process workers with per-task timeout and per-slot restart."""

    def __init__(
        self,
        workers: int,
        timeout_seconds: float,
        max_tasks_per_worker: int = 1000,
        start_method: str = "spawn",
        *,
        probe_fn: Callable[[str], Mapping[str, Any]] = probe_video,
    ) -> None:
        if workers <= 0:
            raise ValueError("workers must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_tasks_per_worker <= 0:
            raise ValueError("max_tasks_per_worker must be positive")
        if start_method not in mp.get_all_start_methods():
            raise ValueError(f"Unsupported multiprocessing start method: {start_method}")
        self.workers = workers
        self.timeout_seconds = timeout_seconds
        self.max_tasks_per_worker = max_tasks_per_worker
        self.start_method = start_method
        self._probe_fn = probe_fn
        self._context = mp.get_context(start_method)
        self._result_queue = self._context.Queue()
        self._pending: queue.Queue[_ProbeRequest] = queue.Queue()
        self._slots = [_WorkerSlot(worker_id=index) for index in range(workers)]
        self._state_lock = threading.Lock()
        self._wake = threading.Event()
        self._closing = False
        self._force_stop = False
        self._closed = False
        self._next_task_id = 0
        self._worker_start_count = 0
        self._worker_restart_count = 0
        self._probe_submit_count = 0
        self._probe_timeout_count = 0
        self._probe_crash_count = 0
        for slot in self._slots:
            self._start_slot(slot)
        self._monitor = threading.Thread(
            target=self._monitor_loop,
            name="persistent-video-probe-monitor",
            daemon=True,
        )
        self._monitor.start()
        atexit.register(self.close, wait=False)

    @property
    def worker_start_count(self) -> int:
        with self._state_lock:
            return self._worker_start_count

    @property
    def worker_restart_count(self) -> int:
        with self._state_lock:
            return self._worker_restart_count

    @property
    def probe_submit_count(self) -> int:
        with self._state_lock:
            return self._probe_submit_count

    @property
    def probe_timeout_count(self) -> int:
        with self._state_lock:
            return self._probe_timeout_count

    @property
    def probe_crash_count(self) -> int:
        with self._state_lock:
            return self._probe_crash_count

    @property
    def active_worker_pids(self) -> list[int]:
        return [
            int(slot.process.pid)
            for slot in self._slots
            if slot.process is not None and slot.process.is_alive() and slot.process.pid is not None
        ]

    def _start_slot(self, slot: _WorkerSlot) -> None:
        slot.generation += 1
        slot.task_queue = self._context.Queue(maxsize=1)
        slot.process = self._context.Process(
            target=_probe_worker_main,
            args=(
                slot.worker_id,
                slot.generation,
                slot.task_queue,
                self._result_queue,
                self.max_tasks_per_worker,
                self._probe_fn,
            ),
            daemon=True,
            name=f"video-probe-{slot.worker_id}-g{slot.generation}",
        )
        slot.process.start()
        with self._state_lock:
            self._worker_start_count += 1

    @staticmethod
    def _terminate_process(process: Any) -> None:
        if process is None:
            return
        process.join(timeout=0.1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=2.0)

    def _stop_slot_process(self, slot: _WorkerSlot) -> None:
        process = slot.process
        task_queue = slot.task_queue
        self._terminate_process(process)
        if task_queue is not None:
            task_queue.close()
            task_queue.join_thread()
        slot.process = None
        slot.task_queue = None

    def _restart_slot(self, slot: _WorkerSlot) -> None:
        self._stop_slot_process(slot)
        slot.current = None
        slot.deadline = None
        with self._state_lock:
            self._worker_restart_count += 1
        if not self._force_stop:
            self._start_slot(slot)

    def submit(self, path: str) -> Future[dict[str, Any]]:
        with self._state_lock:
            if self._closing or self._closed:
                raise RuntimeError("PersistentVideoProbePool is closed")
            task_id = self._next_task_id
            self._next_task_id += 1
            self._probe_submit_count += 1
        future: Future[dict[str, Any]] = Future()
        self._pending.put(_ProbeRequest(task_id=task_id, path=str(path), future=future))
        self._wake.set()
        return future

    def _complete_result(self, result: tuple[Any, ...]) -> None:
        worker_id, generation, task_id, ok, payload, message, recycle = result
        slot = self._slots[int(worker_id)]
        request = slot.current
        if slot.generation != generation or request is None or request.task_id != task_id:
            return
        if ok:
            if not request.future.done():
                request.future.set_result(dict(payload))
        elif not request.future.done():
            request.future.set_exception(
                VideoProbeError(
                    "invalid_video_header",
                    f"Video probe failed for {request.path}: {payload}: {message}",
                )
            )
        slot.current = None
        slot.deadline = None
        if recycle:
            self._restart_slot(slot)

    def _drain_results(self) -> None:
        while True:
            try:
                result = self._result_queue.get_nowait()
            except queue.Empty:
                return
            self._complete_result(result)

    def _check_worker_health(self) -> None:
        now = time.monotonic()
        for slot in self._slots:
            request = slot.current
            if request is not None and slot.deadline is not None and now >= slot.deadline:
                if not request.future.done():
                    request.future.set_exception(
                        VideoProbeError(
                            "video_probe_timeout",
                            f"Video probe timed out after {self.timeout_seconds}s: {request.path}",
                        )
                    )
                with self._state_lock:
                    self._probe_timeout_count += 1
                self._restart_slot(slot)
                continue
            if slot.process is not None and not slot.process.is_alive():
                if request is not None and not request.future.done():
                    request.future.set_exception(
                        VideoProbeError(
                            "video_probe_worker_crash",
                            f"Video probe worker crashed while probing: {request.path}",
                        )
                    )
                    with self._state_lock:
                        self._probe_crash_count += 1
                self._restart_slot(slot)

    def _assign_pending(self) -> None:
        for slot in self._slots:
            if slot.current is not None or slot.process is None or not slot.process.is_alive():
                continue
            while True:
                try:
                    request = self._pending.get_nowait()
                except queue.Empty:
                    return
                if request.future.cancelled():
                    continue
                slot.current = request
                slot.deadline = time.monotonic() + self.timeout_seconds
                slot.task_queue.put((request.task_id, request.path))
                break

    def _has_work(self) -> bool:
        return not self._pending.empty() or any(slot.current is not None for slot in self._slots)

    def _cancel_remaining(self) -> None:
        while True:
            try:
                request = self._pending.get_nowait()
            except queue.Empty:
                break
            if not request.future.done():
                request.future.set_exception(VideoProbeError("probe_pool_closed", "Video probe pool was closed"))
        for slot in self._slots:
            if slot.current is not None and not slot.current.future.done():
                slot.current.future.set_exception(
                    VideoProbeError("probe_pool_closed", "Video probe pool was closed")
                )
            slot.current = None
            slot.deadline = None

    def _shutdown_workers(self) -> None:
        for slot in self._slots:
            if slot.process is not None and slot.process.is_alive() and slot.task_queue is not None:
                try:
                    slot.task_queue.put_nowait(None)
                except queue.Full:
                    pass
        for slot in self._slots:
            self._stop_slot_process(slot)
        self._result_queue.close()
        self._result_queue.join_thread()

    def _monitor_loop(self) -> None:
        try:
            while True:
                self._drain_results()
                self._check_worker_health()
                if self._force_stop:
                    self._cancel_remaining()
                    return
                self._assign_pending()
                if self._closing and not self._has_work():
                    return
                self._wake.wait(timeout=0.01)
                self._wake.clear()
        finally:
            self._shutdown_workers()
            with self._state_lock:
                self._closed = True

    def close(self, *, wait: bool = True) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closing = True
            if not wait:
                self._force_stop = True
        self._wake.set()
        if self._monitor is not threading.current_thread():
            self._monitor.join()

    def __enter__(self) -> PersistentVideoProbePool:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

