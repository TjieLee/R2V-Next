from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_data.distributed_multitask_sampler import (
    DistributedMultiTaskMicrobatchSampler,
)


def _sampler(*, rank: int = 0) -> DistributedMultiTaskMicrobatchSampler:
    image = list(range(80))
    opens2v = list(range(100, 260))
    phantom = list(range(300, 460))
    return DistributedMultiTaskMicrobatchSampler(
        {IMAGE_TASK: image, VIDEO_TASK: opens2v + phantom},
        dataset_indices={"i2i_train": image, "r2v_opens2v": opens2v, "r2v_phantom": phantom},
        video_source_ratios={"r2v_opens2v": 0.5, "r2v_phantom": 0.5},
        total_optimizer_steps=100,
        gradient_accumulation_steps=2,
        rank=rank,
        world_size=4,
        seed=123,
        image_ratio=0.3,
        video_ratio=0.7,
    )


def test_video_source_schedule_has_exact_ratio_and_is_rank_consistent() -> None:
    schedules = [_sampler(rank=rank).video_source_schedule for rank in range(4)]
    assert all(schedule == schedules[0] for schedule in schedules)
    assert schedules[0].count("r2v_opens2v") == 35
    assert schedules[0].count("r2v_phantom") == 35


def test_each_video_step_and_retry_stay_on_one_source() -> None:
    samplers = [_sampler(rank=rank) for rank in range(4)]
    while samplers[0].current_task != VIDEO_TASK:
        for sampler in samplers:
            sampler.mark_microbatch_consumed()
    source = samplers[0].current_source
    assert source is not None
    normal = [next(iter(sampler)) for sampler in samplers]
    retry = [sampler.retry_index(1) for sampler in samplers]
    expected = set(range(100, 260)) if source == "r2v_opens2v" else set(range(300, 460))
    assert set(normal).issubset(expected)
    assert set(retry).issubset(expected)
    assert len(set(normal)) == 4
    assert len(set(retry)) == 4


def test_source_schedule_and_next_index_resume_exactly() -> None:
    control = _sampler()
    for _ in range(11):
        control.mark_microbatch_consumed()
    state = control.state_dict()
    expected_source = control.current_source
    expected_index = next(iter(control))

    resumed = _sampler()
    resumed.load_state_dict(state)
    assert resumed.current_source == expected_source
    assert next(iter(resumed)) == expected_index
