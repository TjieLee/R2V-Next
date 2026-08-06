from types import SimpleNamespace

import pytest

from ltx_trainer.progress import TrainingProgress
from ltx_trainer.trainer import format_strategy_loss_components


class _FakeProgress:
    def __init__(self, task_id: int) -> None:
        self.tasks = {task_id: SimpleNamespace(completed=0)}
        self.updates: list[dict[str, object]] = []

    def update(self, task_id: int, **kwargs: object) -> None:
        self.updates.append(dict(kwargs))
        self.tasks[task_id].completed += int(kwargs.get("advance", 0))


def _training_progress() -> tuple[TrainingProgress, _FakeProgress]:
    task_id = 7
    progress = TrainingProgress(enabled=False, total_steps=10)
    fake = _FakeProgress(task_id)
    progress._progress = fake  # type: ignore[assignment]
    progress._train_task = task_id
    return progress, fake


def test_format_strategy_loss_components_for_semantic_repae() -> None:
    metrics = {
        "train/loss_video_flow": 0.2777,
        "train/loss_video_flow_weighted": 0.2777,
        "train/loss_semantic_flow": 1.9987,
        "train/loss_semantic_flow_weighted": 1.9987,
        "train/loss_semantic_projection_repa": 0.9992,
        "train/loss_semantic_projection_repa_weighted": 0.9992,
        "train/loss_semantic_dit_repa": 0.9994,
        "train/loss_semantic_dit_repa_weighted": 0.4997,
    }

    assert format_strategy_loss_components(metrics, compact=False) == (
        "Video flow: 0.2777 (weighted: 0.2777), "
        "Semantic flow: 1.9987 (weighted: 1.9987), "
        "Projection REPA: 0.9992 (weighted: 0.9992), "
        "DiT REPA: 0.9994 (weighted: 0.4997)"
    )
    assert format_strategy_loss_components(metrics, compact=True) == (
        "V 0.2777 | S 1.9987 | P-REPA 0.9992 | D-REPA 0.9994"
    )


def test_format_strategy_loss_components_for_semantic_flow_and_empty_metrics() -> None:
    metrics = {
        "train/loss_video_flow": 0.5,
        "train/loss_semantic_flow": 0.75,
        "train/loss_semantic_reconstruction": 0.25,
        "train/loss_semantic_alignment": 0.125,
    }

    formatted = format_strategy_loss_components(metrics, compact=False)
    assert formatted == (
        "Video flow: 0.5000, Semantic flow: 0.7500, "
        "Reconstruction: 0.2500, Alignment: 0.1250"
    )
    assert "REPA" not in formatted
    assert "n/a" not in formatted
    assert format_strategy_loss_components({}, compact=False) == ""
    assert format_strategy_loss_components({}, compact=True) == ""


@pytest.mark.parametrize("loss_components", [None, ""])
def test_training_progress_without_components_preserves_existing_info(
    loss_components: str | None,
) -> None:
    progress, fake = _training_progress()
    progress.update_training(
        loss=3.7753,
        lr=5.0e-7,
        step_time=10.74,
        loss_components=loss_components,
    )

    assert fake.updates[0]["info"] == "Loss: 3.7753 | LR: 5.00e-07 | 10.74s/step"


def test_training_progress_includes_compact_components_once() -> None:
    progress, fake = _training_progress()
    components = "V 0.2777 | S 1.9987 | P-REPA 0.9992 | D-REPA 0.9994"
    progress.update_training(
        loss=3.7753,
        lr=5.0e-7,
        step_time=10.74,
        loss_components=components,
    )

    info = fake.updates[0]["info"]
    assert info == (
        "Loss: 3.7753 | V 0.2777 | S 1.9987 | P-REPA 0.9992 | "
        "D-REPA 0.9994 | LR: 5.00e-07 | 10.74s/step"
    )
    assert str(info).count(components) == 1
    assert "|  |" not in str(info)
