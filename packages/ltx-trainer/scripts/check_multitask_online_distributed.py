"""Run one real semantic-flow task in a multi-rank FSDP launch."""

from __future__ import annotations

import os

import typer

from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_data.smoke import run_one_step_training_smoke

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


def _validate_task(task: str) -> str:
    normalized = task.strip().lower()
    if normalized not in {IMAGE_TASK, VIDEO_TASK}:
        raise typer.BadParameter("--task must be i2i or r2v")
    return normalized


@app.command()
def main(
    config: str = typer.Option(..., "--config"),
    task: str = typer.Option(..., "--task"),
    init_checkpoint: str | None = typer.Option(None, "--init-checkpoint"),
    output_dir: str = typer.Option(..., "--output-dir"),
) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise RuntimeError(f"Real 22B semantic-flow smoke requires a multi-process launch, got {world_size}")
    run_one_step_training_smoke(
        config,
        task=_validate_task(task),
        output_dir=output_dir,
        init_checkpoint=init_checkpoint,
    )


if __name__ == "__main__":
    app()
