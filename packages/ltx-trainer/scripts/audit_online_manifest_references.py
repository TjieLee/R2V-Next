#!/usr/bin/env python3
"""Audit online-manifest references without modifying source data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from ltx_trainer.online_data.path_safety import assert_write_path_allowed
from ltx_trainer.online_data.reference_audit import audit_online_manifest_references

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)
console = Console()


def _atomic_write_text(path: Path, value: str) -> None:
    destination = assert_write_path_allowed(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = assert_write_path_allowed(
        destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    )
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )


@app.command()
def main(
    manifest: str = typer.Option(..., "--manifest"),
    output_dir: str = typer.Option(..., "--output-dir"),
    limit: int | None = typer.Option(None, "--limit"),
    workers: int = typer.Option(8, "--workers"),
) -> None:
    if limit is not None and limit < 1:
        raise typer.BadParameter("--limit must be >= 1")
    if workers < 1:
        raise typer.BadParameter("--workers must be >= 1")
    output = assert_write_path_allowed(output_dir)
    rows, summary = audit_online_manifest_references(
        manifest,
        limit=limit,
        workers=workers,
    )
    output.mkdir(parents=True, exist_ok=True)
    details_path = output / "reference_audit.jsonl"
    summary_path = output / "summary.json"
    _atomic_write_text(details_path, _jsonl(rows))
    _atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    console.print_json(
        data={
            **summary,
            "details_path": str(details_path),
            "summary_path": str(summary_path),
        }
    )
    if summary["degenerate_unique_reference_count"] == 0:
        console.print("[yellow]No width<=1 or height<=1 reference was found.[/yellow]")
    else:
        console.print(
            "[green]Located "
            f"{summary['degenerate_unique_reference_count']} degenerate reference path(s).[/green]"
        )


if __name__ == "__main__":
    app()
