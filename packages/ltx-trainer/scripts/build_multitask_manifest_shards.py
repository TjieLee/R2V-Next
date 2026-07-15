"""Build deterministic I2I/R2V manifest shards with bounded parallelism."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from ltx_trainer.online_data.parallel_manifest import (
    BuildOptions,
    build_task_shards,
    parse_tasks,
    run_r2v_prefilter,
)
from ltx_trainer.online_data.path_safety import assert_write_path_allowed

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


@app.command()
def main(  # noqa: PLR0913
    train_data_config: str = typer.Option(..., "--train-data-config"),
    tasks: str = typer.Option(..., "--tasks", help="i2i, r2v, or i2i,r2v"),
    shard_root: str = typer.Option(..., "--shard-root"),
    shards_per_task: int = typer.Option(8, "--shards-per-task"),
    image_workers: int = typer.Option(16, "--image-workers"),
    video_workers: int = typer.Option(8, "--video-workers"),
    max_in_flight: int = typer.Option(256, "--max-in-flight"),
    annotation_batch_size: int = typer.Option(4096, "--annotation-batch-size"),
    probe_timeout_seconds: float = typer.Option(60.0, "--probe-timeout-seconds"),
    video_probe_mode: str = typer.Option("persistent", "--video-probe-mode"),
    video_probe_max_tasks_per_worker: int = typer.Option(1000, "--video-probe-max-tasks-per-worker"),
    prefilter_only: bool = typer.Option(False, "--prefilter-only"),
    resume_build: bool = typer.Option(True, "--resume-build/--no-resume-build"),
    progress_interval_seconds: float = typer.Option(10.0, "--progress-interval-seconds"),
    manifest_seed: int = typer.Option(42, "--manifest-seed"),
    i2i_target_field: str = typer.Option("image", "--i2i-target-field"),
    i2i_reference_field: str = typer.Option("edit_image", "--i2i-reference-field"),
    i2i_caption_field: str = typer.Option("prompt", "--i2i-caption-field"),
    i2i_crop_field: str | None = typer.Option(None, "--i2i-crop-field"),
    max_samples_per_task: int | None = typer.Option(None, "--max-samples-per-task"),
    recover_stale_locks: bool = typer.Option(False, "--recover-stale-locks"),
    stale_lock_seconds: float = typer.Option(21_600.0, "--stale-lock-seconds"),
) -> None:
    """Build each selected task into independently resumable shard artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = assert_write_path_allowed(shard_root)
    selected_tasks = parse_tasks(tasks)
    options = BuildOptions(
        shards_per_task=shards_per_task,
        image_workers=image_workers,
        video_workers=video_workers,
        max_in_flight=max_in_flight,
        annotation_batch_size=annotation_batch_size,
        probe_timeout_seconds=probe_timeout_seconds,
        video_probe_mode=video_probe_mode,
        video_probe_max_tasks_per_worker=video_probe_max_tasks_per_worker,
        resume_build=resume_build,
        progress_interval_seconds=progress_interval_seconds,
        manifest_seed=manifest_seed,
        i2i_target_field=i2i_target_field,
        i2i_reference_field=i2i_reference_field,
        i2i_caption_field=i2i_caption_field,
        i2i_crop_field=i2i_crop_field,
        max_samples_per_task=max_samples_per_task,
        recover_stale_locks=recover_stale_locks,
        stale_lock_seconds=stale_lock_seconds,
    )
    options.validate()
    root.mkdir(parents=True, exist_ok=True)
    if prefilter_only:
        if selected_tasks != ["r2v"]:
            raise typer.BadParameter("--prefilter-only requires --tasks r2v")
        summary = run_r2v_prefilter(
            Path(train_data_config),
            shard_root=root,
            options=options,
        )
        typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return
    summaries = []
    for task in selected_tasks:
        summary = build_task_shards(
            Path(train_data_config),
            task=task,
            shard_root=root,
            options=options,
        )
        summaries.append(summary)
        typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    typer.echo(
        json.dumps(
            {"completed_tasks": selected_tasks, "task_summaries": summaries},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app()
