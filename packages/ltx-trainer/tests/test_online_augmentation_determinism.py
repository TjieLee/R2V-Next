import torch

from ltx_trainer.online_data.transforms import (
    OnlineAugmentationConfig,
    augment_reference_image,
    augment_target_frames,
    augmentation_seed,
    deterministic_resize_center_crop,
)


def _pattern(height: int = 32, width: int = 48) -> torch.Tensor:
    values = torch.arange(height * width * 3, dtype=torch.int64).remainder(256)
    return values.to(torch.uint8).reshape(height, width, 3)


def test_augmentation_seed_covers_step_microstep_sample_and_reference() -> None:
    base = augmentation_seed(global_seed=7, optimizer_step=3, microstep=1, sample_key="sample")
    assert base == augmentation_seed(global_seed=7, optimizer_step=3, microstep=1, sample_key="sample")
    assert base != augmentation_seed(global_seed=7, optimizer_step=4, microstep=1, sample_key="sample")
    assert base != augmentation_seed(
        global_seed=7,
        optimizer_step=3,
        microstep=1,
        sample_key="sample",
        reference_index=0,
    )


def test_target_uses_one_transform_for_all_frames_and_is_resume_stable() -> None:
    frame = _pattern()
    frames = frame.unsqueeze(0).repeat(5, 1, 1, 1)
    config = OnlineAugmentationConfig()
    first = augment_target_frames(
        frames,
        seed=1234,
        config=config,
        target_height=24,
        target_width=40,
        chunk_frames=2,
    )
    second = augment_target_frames(
        frames,
        seed=1234,
        config=config,
        target_height=24,
        target_width=40,
        chunk_frames=3,
    )
    assert torch.equal(first, second)
    assert all(torch.equal(first[0], frame_value) for frame_value in first[1:])


def test_reference_augmentation_is_deterministic_and_shared_ready() -> None:
    config = OnlineAugmentationConfig(
        reference_blur_probability=1.0,
        reference_jpeg_probability=1.0,
        reference_erasing_probability=1.0,
    )
    first = augment_reference_image(
        _pattern(),
        seed=901,
        config=config,
        target_height=24,
        target_width=40,
    )
    second = augment_reference_image(
        _pattern(),
        seed=901,
        config=config,
        target_height=24,
        target_width=40,
    )
    assert torch.equal(first, second)
    assert first.shape == (24, 40, 3)
    assert first.dtype == torch.uint8


def test_disabled_augmentation_is_step_independent_and_never_flips_reference() -> None:
    config = OnlineAugmentationConfig(enabled=False)
    source = _pattern()
    frames = source.unsqueeze(0).repeat(3, 1, 1, 1)
    expected_reference = deterministic_resize_center_crop(
        source.unsqueeze(0),
        target_height=24,
        target_width=40,
        chunk_frames=1,
    )[0]
    first = augment_target_frames(
        frames,
        seed=1,
        config=config,
        target_height=24,
        target_width=40,
        chunk_frames=2,
    )
    second = augment_target_frames(
        frames,
        seed=9999,
        config=config,
        target_height=24,
        target_width=40,
        chunk_frames=2,
    )
    reference = augment_reference_image(
        source,
        seed=9999,
        config=config,
        target_height=24,
        target_width=40,
    )
    reference_second = augment_reference_image(
        source,
        seed=1,
        config=config,
        target_height=24,
        target_width=40,
    )

    assert torch.equal(first, frames)
    assert torch.equal(second, frames)
    assert torch.equal(first, second)
    assert torch.equal(reference, expected_reference)
    assert torch.equal(reference_second, expected_reference)
    assert reference.shape == (24, 40, 3)
