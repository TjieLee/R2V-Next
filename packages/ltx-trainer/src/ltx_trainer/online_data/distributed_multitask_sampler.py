"""Optimizer-step-synchronized distributed sampling for I2I/R2V training."""

from __future__ import annotations

import hashlib
import logging
from array import array
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_data.data_state import OnlineDataState

_TASK_TO_ID = {IMAGE_TASK: 0, VIDEO_TASK: 1}
_ID_TO_TASK = {value: key for key, value in _TASK_TO_ID.items()}
logger = logging.getLogger(__name__)


def _seed_for(*parts: int) -> int:
    payload = ":".join(str(int(part)) for part in parts).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def _indices_tensor(values: Sequence[int]) -> torch.Tensor:
    if isinstance(values, array):
        if values.typecode != "q":
            raise ValueError(f"Compact task index arrays must use signed int64 typecode 'q', got {values.typecode!r}")
        return torch.frombuffer(values, dtype=torch.int64).clone()
    return torch.as_tensor(values, dtype=torch.long).clone()


class DistributedMultiTaskMicrobatchSampler(Sampler[int]):
    """Yield one rank-local sample per accumulation microstep.

    A shuffled optimizer-step task table is shared across ranks. Each step uses
    only one modality, while rank/microstep slots select a unique global block
    whenever the corresponding task dataset has enough samples.
    """

    def __init__(
        self,
        task_indices: Mapping[str, Sequence[int]],
        *,
        total_optimizer_steps: int,
        gradient_accumulation_steps: int,
        rank: int,
        world_size: int,
        seed: int = 42,
        image_ratio: float = 0.3,
        video_ratio: float = 0.7,
    ) -> None:
        if total_optimizer_steps <= 0:
            raise ValueError("total_optimizer_steps must be positive")
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(f"Invalid distributed rank/world_size: {rank}/{world_size}")
        if image_ratio < 0 or video_ratio < 0 or abs(image_ratio + video_ratio - 1.0) > 1.0e-8:
            raise ValueError("image_ratio and video_ratio must be non-negative and sum to 1")

        self.total_optimizer_steps = int(total_optimizer_steps)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.global_samples_per_step = self.world_size * self.gradient_accumulation_steps
        self._indices = {
            task: _indices_tensor(task_indices.get(task, ()))
            for task in (IMAGE_TASK, VIDEO_TASK)
        }
        for task, indices in self._indices.items():
            if indices.numel() == 0:
                raise ValueError(f"No manifest indices are available for task {task!r}")

        image_steps = int(round(self.total_optimizer_steps * image_ratio))
        video_steps = self.total_optimizer_steps - image_steps
        self._task_schedule = self._build_and_share_task_schedule(image_steps=image_steps, video_steps=video_steps)
        self._task_occurrence = self._build_task_occurrence(self._task_schedule)
        self._task_streams = {
            IMAGE_TASK: self._build_task_stream(
                IMAGE_TASK,
                step_count=image_steps,
                task_seed=_seed_for(self.seed, _TASK_TO_ID[IMAGE_TASK], 101),
            ),
            VIDEO_TASK: self._build_task_stream(
                VIDEO_TASK,
                step_count=video_steps,
                task_seed=_seed_for(self.seed, _TASK_TO_ID[VIDEO_TASK], 101),
            ),
        }
        self._state = OnlineDataState(sampler_seed=self.seed)
        self._consumed_indices_in_step: set[int] = set()
        self._warned_retry_fallback_steps: set[int] = set()

    @property
    def current_task(self) -> str:
        if self._state.task_schedule_cursor >= self.total_optimizer_steps:
            raise StopIteration("Online sampler schedule is exhausted")
        return _ID_TO_TASK[int(self._task_schedule[self._state.task_schedule_cursor].item())]

    @property
    def task_schedule(self) -> tuple[str, ...]:
        return tuple(_ID_TO_TASK[int(value)] for value in self._task_schedule.tolist())

    def __iter__(self) -> Iterator[int]:
        start_step = self._state.task_schedule_cursor
        start_microstep = self._state.microstep_in_optimizer_step
        for step in range(start_step, self.total_optimizer_steps):
            task_id = int(self._task_schedule[step].item())
            task = _ID_TO_TASK[task_id]
            occurrence = int(self._task_occurrence[step].item())
            first_microstep = start_microstep if step == start_step else 0
            block = self._task_streams[task][occurrence]
            for microstep in range(first_microstep, self.gradient_accumulation_steps):
                global_slot = microstep * self.world_size + self.rank
                yield int(block[global_slot].item())

    def __len__(self) -> int:
        completed = (
            self._state.task_schedule_cursor * self.gradient_accumulation_steps
            + self._state.microstep_in_optimizer_step
        )
        return self.total_optimizer_steps * self.gradient_accumulation_steps - completed

    def current_normal_block(self) -> torch.Tensor:
        """Return the full normal sample block reserved for the current optimizer step."""
        task = self.current_task
        step = self._state.task_schedule_cursor
        occurrence = int(self._task_occurrence[step].item())
        return self._task_streams[task][occurrence].clone()

    def was_consumed_in_current_step(self, index: int) -> bool:
        """Whether ``index`` was already used by a successful microbatch in this step."""
        return int(index) in self._consumed_indices_in_step

    def mark_microbatch_consumed(self, global_indices: Sequence[int] | None = None) -> None:
        if self._state.task_schedule_cursor >= self.total_optimizer_steps:
            raise RuntimeError("Cannot advance an exhausted online sampler")
        if global_indices is not None:
            consumed = [int(index) for index in global_indices]
            if len(consumed) != self.world_size:
                raise ValueError(
                    f"global_indices must contain one sample per rank ({self.world_size}), got {len(consumed)}"
                )
            task_size = int(self._indices[self.current_task].numel())
            if task_size >= self.global_samples_per_step:
                overlap = self._consumed_indices_in_step.intersection(consumed)
                if overlap or len(set(consumed)) != len(consumed):
                    raise RuntimeError(
                        "Online sampler produced duplicate successful samples within one optimizer step: "
                        f"{sorted(overlap or set(index for index in consumed if consumed.count(index) > 1))}"
                    )
            self._consumed_indices_in_step.update(consumed)
        self._state.microstep_in_optimizer_step += 1
        if self._state.microstep_in_optimizer_step == self.gradient_accumulation_steps:
            self._state.task_schedule_cursor += 1
            self._state.microstep_in_optimizer_step = 0
            self._consumed_indices_in_step.clear()
        self._refresh_derived_state()

    def seek_optimizer_step(self, optimizer_step: int) -> None:
        if not 0 <= optimizer_step <= self.total_optimizer_steps:
            raise ValueError(f"optimizer_step must be in [0,{self.total_optimizer_steps}], got {optimizer_step}")
        self._state.task_schedule_cursor = int(optimizer_step)
        self._state.microstep_in_optimizer_step = 0
        self._consumed_indices_in_step.clear()
        self._refresh_derived_state()

    def retry_index(self, attempt: int) -> int:
        """Return a deterministic same-task backup for the current rank/microstep."""
        if attempt <= 0:
            raise ValueError("Retry attempt must be positive")
        task = self.current_task
        indices = self._indices[task]
        step = self._state.task_schedule_cursor
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _seed_for(
                self.seed,
                step,
                self._state.microstep_in_optimizer_step,
                attempt,
                _TASK_TO_ID[task],
            )
        )
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        normal_block = self.current_normal_block()
        consumed = torch.tensor(sorted(self._consumed_indices_in_step), dtype=torch.long)
        strict_excluded = torch.cat([normal_block, consumed]) if consumed.numel() else normal_block
        strict_candidates = shuffled[~torch.isin(shuffled, strict_excluded)]

        if strict_candidates.numel() >= self.world_size:
            candidate_block = strict_candidates[: self.world_size]
        else:
            # Small datasets cannot always reserve a disjoint retry pool of one
            # complete global block. Prefer successful-step uniqueness, and let
            # the trainer replace any prefetched normal slot that was consumed
            # by this fallback before it enters the model.
            if step not in self._warned_retry_fallback_steps:
                logger.warning(
                    "Retry pool for task %s has %d samples and cannot avoid the full %d-sample normal block; "
                    "falling back to unused-in-step samples. Prefetched collisions will be retried.",
                    task,
                    indices.numel(),
                    self.global_samples_per_step,
                )
                self._warned_retry_fallback_steps.add(step)
            fallback_candidates = shuffled
            if consumed.numel():
                fallback_candidates = fallback_candidates[~torch.isin(fallback_candidates, consumed)]
            if fallback_candidates.numel() < self.world_size:
                raise RuntimeError(
                    "Cannot construct a rank-unique retry microbatch: "
                    f"task={task}, dataset_size={indices.numel()}, already_consumed={consumed.numel()}, "
                    f"world_size={self.world_size}"
                )
            candidate_block = fallback_candidates[: self.world_size]

        if candidate_block.unique().numel() != self.world_size:
            raise RuntimeError("Deterministic retry candidate block contains duplicate rank slots")
        return int(candidate_block[self.rank].item())

    def state_dict(self) -> dict[str, int]:
        self._refresh_derived_state()
        return self._state.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        loaded = OnlineDataState.from_state_dict(dict(state))
        if loaded.sampler_seed != self.seed:
            raise ValueError(f"Sampler seed mismatch: state={loaded.sampler_seed}, config={self.seed}")
        if not 0 <= loaded.task_schedule_cursor <= self.total_optimizer_steps:
            raise ValueError(f"Invalid task_schedule_cursor={loaded.task_schedule_cursor}")
        if not 0 <= loaded.microstep_in_optimizer_step < self.gradient_accumulation_steps:
            raise ValueError(f"Invalid microstep_in_optimizer_step={loaded.microstep_in_optimizer_step}")
        expected = self._derived_state(
            loaded.task_schedule_cursor,
            loaded.microstep_in_optimizer_step,
        )
        for field in ("image_permutation_epoch", "image_cursor", "video_permutation_epoch", "video_cursor"):
            if getattr(loaded, field) != getattr(expected, field):
                raise ValueError(
                    f"Online sampler state mismatch for {field}: "
                    f"saved={getattr(loaded, field)}, expected={getattr(expected, field)}"
                )
        self._state = loaded
        self._consumed_indices_in_step.clear()

    def _build_and_share_task_schedule(self, *, image_steps: int, video_steps: int) -> torch.Tensor:
        schedule: list[int] | None = None
        is_distributed = dist.is_available() and dist.is_initialized()
        if not is_distributed or dist.get_rank() == 0:
            labels = torch.tensor(
                [_TASK_TO_ID[IMAGE_TASK]] * image_steps + [_TASK_TO_ID[VIDEO_TASK]] * video_steps,
                dtype=torch.int8,
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_seed_for(self.seed, 17))
            schedule = labels[torch.randperm(labels.numel(), generator=generator)].tolist()
        if is_distributed:
            payload: list[Any] = [schedule]
            dist.broadcast_object_list(payload, src=0)
            schedule = payload[0]
        if schedule is None or len(schedule) != self.total_optimizer_steps:
            raise RuntimeError("Failed to create the distributed task schedule")
        tensor = torch.tensor(schedule, dtype=torch.int8)
        if int((tensor == _TASK_TO_ID[IMAGE_TASK]).sum().item()) != image_steps:
            raise RuntimeError("Shared task schedule has an incorrect image-step count")
        return tensor

    @staticmethod
    def _build_task_occurrence(schedule: torch.Tensor) -> torch.Tensor:
        occurrence = torch.empty_like(schedule, dtype=torch.long)
        counters = {task_id: 0 for task_id in _ID_TO_TASK}
        for step, raw_task_id in enumerate(schedule.tolist()):
            task_id = int(raw_task_id)
            occurrence[step] = counters[task_id]
            counters[task_id] += 1
        return occurrence

    def _build_task_stream(self, task: str, *, step_count: int, task_seed: int) -> torch.Tensor:
        dataset_indices = self._indices[task]
        dataset_size = int(dataset_indices.numel())
        block_size = self.global_samples_per_step
        stream = torch.empty(step_count, block_size, dtype=torch.long)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(task_seed)
        permutation = dataset_indices[torch.randperm(dataset_size, generator=generator)]
        cursor = 0

        for step in range(step_count):
            block: list[int] = []
            while len(block) < block_size:
                if cursor >= dataset_size:
                    permutation = dataset_indices[torch.randperm(dataset_size, generator=generator)]
                    cursor = 0
                    if dataset_size >= block_size and block:
                        seen = set(block)
                        keep_first = torch.tensor(
                            [int(value) not in seen for value in permutation.tolist()], dtype=torch.bool
                        )
                        permutation = torch.cat([permutation[keep_first], permutation[~keep_first]])
                candidate = int(permutation[cursor].item())
                cursor += 1
                if dataset_size >= block_size and candidate in block:
                    # The stable partition above makes this reachable only for
                    # a malformed stream; keep the guard explicit.
                    continue
                block.append(candidate)
            stream[step] = torch.tensor(block, dtype=torch.long)
        return stream

    def _derived_state(self, step: int, microstep: int) -> OnlineDataState:
        completed_schedule = self._task_schedule[:step]
        image_steps = int((completed_schedule == _TASK_TO_ID[IMAGE_TASK]).sum().item())
        video_steps = step - image_steps
        image_cursor = image_steps * self.global_samples_per_step
        video_cursor = video_steps * self.global_samples_per_step
        if step < self.total_optimizer_steps:
            current_task = _ID_TO_TASK[int(self._task_schedule[step].item())]
            if current_task == IMAGE_TASK:
                image_cursor += microstep * self.world_size
            else:
                video_cursor += microstep * self.world_size
        image_size = int(self._indices[IMAGE_TASK].numel())
        video_size = int(self._indices[VIDEO_TASK].numel())
        return OnlineDataState(
            task_schedule_cursor=step,
            image_permutation_epoch=image_cursor // image_size,
            image_cursor=image_cursor,
            video_permutation_epoch=video_cursor // video_size,
            video_cursor=video_cursor,
            microstep_in_optimizer_step=microstep,
            sampler_seed=self.seed,
        )

    def _refresh_derived_state(self) -> None:
        self._state = self._derived_state(
            self._state.task_schedule_cursor,
            self._state.microstep_in_optimizer_step,
        )
