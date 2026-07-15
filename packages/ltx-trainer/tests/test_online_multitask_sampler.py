from __future__ import annotations

from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_data.distributed_multitask_sampler import DistributedMultiTaskMicrobatchSampler


def _sampler(*, rank: int = 0, steps: int = 100) -> DistributedMultiTaskMicrobatchSampler:
    return DistributedMultiTaskMicrobatchSampler(
        {IMAGE_TASK: list(range(100)), VIDEO_TASK: list(range(100, 240))},
        total_optimizer_steps=steps,
        gradient_accumulation_steps=4,
        rank=rank,
        world_size=8,
        seed=123,
        image_ratio=0.3,
        video_ratio=0.7,
    )


def test_30k_schedule_has_exact_30_70_optimizer_steps() -> None:
    sampler = _sampler(steps=30_000)
    assert sampler.task_schedule.count(IMAGE_TASK) == 9_000
    assert sampler.task_schedule.count(VIDEO_TASK) == 21_000


def test_same_optimizer_step_is_one_task_and_has_32_unique_samples() -> None:
    samplers = [_sampler(rank=rank, steps=10) for rank in range(8)]
    iterators = [iter(sampler) for sampler in samplers]
    for _step in range(10):
        assert len({sampler.current_task for sampler in samplers}) == 1
        global_indices = []
        for _microstep in range(4):
            global_indices.extend(next(iterator) for iterator in iterators)
            for sampler in samplers:
                sampler.mark_microbatch_consumed()
        assert len(global_indices) == 32
        assert len(set(global_indices)) == 32


def test_resume_reproduces_next_task_and_sample_sequence() -> None:
    control = _sampler(steps=40)
    control_iterator = iter(control)
    for _ in range(13):
        next(control_iterator)
        control.mark_microbatch_consumed()
    state = control.state_dict()
    expected = [next(control_iterator) for _ in range(20)]

    resumed = _sampler(steps=40)
    resumed.load_state_dict(state)
    actual = [next(iter(resumed))]
    resumed.mark_microbatch_consumed()
    resumed_iterator = iter(resumed)
    actual.extend(next(resumed_iterator) for _ in range(19))
    assert actual == expected


def test_retry_stays_in_current_task_and_is_deterministic() -> None:
    first = _sampler(rank=3, steps=20)
    second = _sampler(rank=3, steps=20)
    assert first.current_task == second.current_task
    assert first.retry_index(1) == second.retry_index(1)
    index = first.retry_index(1)
    if first.current_task == IMAGE_TASK:
        assert index < 100
    else:
        assert index >= 100


def test_retry_excludes_successful_retry_samples_from_later_microsteps() -> None:
    samplers = [_sampler(rank=rank, steps=2) for rank in range(8)]
    retry_microstep = [sampler.retry_index(1) for sampler in samplers]
    assert len(set(retry_microstep)) == 8

    for sampler in samplers:
        sampler.mark_microbatch_consumed(retry_microstep)

    later_retry = [sampler.retry_index(1) for sampler in samplers]
    assert len(set(later_retry)) == 8
    assert set(later_retry).isdisjoint(retry_microstep)
