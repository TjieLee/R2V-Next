#!/usr/bin/env python3
"""Watch sharded precompute progress by counting completed .pt outputs."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Watch preprocessing progress for sharded .pt outputs.",
)


def _read_rows(dataset_file: Path) -> list[dict[str, Any]]:
    suffix = dataset_file.suffix.lower()
    if suffix == ".json":
        data = json.loads(dataset_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return list(data.values())
        if isinstance(data, list):
            return data
        raise ValueError("JSON manifest must contain a list or dict of objects")
    if suffix == ".jsonl":
        return [json.loads(line) for line in dataset_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".csv":
        with dataset_file.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    raise ValueError(f"Unsupported manifest suffix: {dataset_file.suffix}")


def _count_expected(dataset_file: Path) -> int:
    if dataset_file.is_dir():
        total = 0
        for path in sorted(dataset_file.glob("*.json")):
            total += len(_read_rows(path))
        for path in sorted(dataset_file.glob("*.jsonl")):
            total += len(_read_rows(path))
        for path in sorted(dataset_file.glob("*.csv")):
            total += len(_read_rows(path))
        if total == 0:
            raise ValueError(f"No manifest files found in {dataset_file}")
        return total
    return len(_read_rows(dataset_file))


def _count_completed(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    count = 0
    for path in output_dir.rglob("*.pt"):
        name = path.name
        if ".tmp." not in name:
            count += 1
    return count


@app.command()
def main(
    manifest: str = typer.Argument(
        ...,
        help="Full manifest file or a directory containing shard manifests.",
    ),
    output_dir: str = typer.Argument(..., help="Output directory to watch, e.g. .precomputed/gt_siglip_tokens."),
    label: str = typer.Option("precompute", help="Progress label shown in the terminal."),
    interval: float = typer.Option(20.0, help="Refresh interval in seconds."),
) -> None:
    if interval <= 0:
        raise typer.BadParameter("--interval must be greater than 0.")

    manifest_path = Path(manifest)
    output_path = Path(output_dir)
    total = _count_expected(manifest_path)
    console = Console()
    start = time.time()
    last_completed = 0
    last_time = start

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[cyan]{task.fields[rate]} samples/s"),
        console=console,
    ) as progress:
        task = progress.add_task(label, total=total, rate="0.00")
        try:
            while True:
                now = time.time()
                completed = min(_count_completed(output_path), total)
                elapsed_delta = max(now - last_time, 1e-6)
                rate = max((completed - last_completed) / elapsed_delta, 0.0)
                progress.update(task, completed=completed, rate=f"{rate:.2f}")
                if completed >= total:
                    break
                last_completed = completed
                last_time = now
                time.sleep(interval)
        except KeyboardInterrupt:
            completed = min(_count_completed(output_path), total)
            progress.update(task, completed=completed)
            console.print(f"\nStopped watching at {completed:,}/{total:,}.")


if __name__ == "__main__":
    app()
