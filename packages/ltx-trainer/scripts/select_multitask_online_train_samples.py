#!/usr/bin/env python3
"""Select deterministic I2I/R2V rows from a finalized online manifest."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ltx_trainer.online_inference.path_policy import assert_online_inference_output_path
from ltx_trainer.online_inference.train_sample_selection import (
    SUPPORTED_TASKS,
    select_online_train_samples,
    write_selection_bundle,
)

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)
console = Console()


def _csv_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def _csv_ints(value: str | None) -> list[int] | None:
    values = _csv_strings(value)
    return [int(item) for item in values] if values is not None else None


@app.command()
def main(
    manifest: str = typer.Option(..., "--manifest", help="Final online train JSONL manifest."),
    output_dir: str = typer.Option(..., "--output-dir", help="New selection bundle directory."),
    tasks: str = typer.Option("i2i,r2v", "--tasks", help="Comma-separated task names."),
    samples_per_task: int = typer.Option(4, "--samples-per-task"),
    seed: int = typer.Option(42, "--seed"),
    manifest_indices: str | None = typer.Option(None, "--manifest-indices"),
    sample_keys: str | None = typer.Option(None, "--sample-keys"),
    stratify_by_reference_count: bool = typer.Option(
        True,
        "--stratify-by-reference-count/--no-stratify-by-reference-count",
    ),
    max_caption_chars: int | None = typer.Option(None, "--max-caption-chars"),
    overwrite: bool = typer.Option(False, "--overwrite/--no-overwrite"),
) -> None:
    selected_tasks = _csv_strings(tasks)
    if selected_tasks is None or any(task not in SUPPORTED_TASKS for task in selected_tasks):
        raise typer.BadParameter(f"--tasks must be a subset of {SUPPORTED_TASKS}")
    if samples_per_task < 1:
        raise typer.BadParameter("--samples-per-task must be >= 1")
    if max_caption_chars is not None and max_caption_chars < 1:
        raise typer.BadParameter("--max-caption-chars must be >= 1")
    result = select_online_train_samples(
        Path(manifest),
        tasks=selected_tasks,
        samples_per_task=samples_per_task,
        seed=seed,
        manifest_indices=_csv_ints(manifest_indices),
        sample_keys=_csv_strings(sample_keys),
        stratify_by_reference_count=stratify_by_reference_count,
        max_caption_chars=max_caption_chars,
    )
    destination = write_selection_bundle(
        result,
        assert_online_inference_output_path(Path(output_dir)),
        overwrite=overwrite,
    )
    console.print(f"[green]Selected {len(result.samples)} samples:[/green] {destination}")
    console.print_json(data=result.summary)


if __name__ == "__main__":
    app()
