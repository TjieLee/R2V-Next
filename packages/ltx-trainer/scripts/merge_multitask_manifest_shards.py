"""Validate and deterministically merge completed multi-task manifest shards."""

from __future__ import annotations

import json

import typer

from ltx_trainer.online_data.parallel_manifest import merge_manifest_shards, parse_tasks
from ltx_trainer.online_data.path_safety import assert_write_path_allowed

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


@app.command()
def main(  # noqa: PLR0913
    shard_root: str = typer.Option(..., "--shard-root"),
    tasks: str = typer.Option("i2i,r2v", "--tasks"),
    output: str = typer.Option(..., "--output"),
    reject_output: str = typer.Option(..., "--reject-output"),
    summary_output: str = typer.Option(..., "--summary-output"),
    dedup_db: str | None = typer.Option(None, "--dedup-db"),
    require_all_shards: bool = typer.Option(True, "--require-all-shards/--allow-missing-shards"),
) -> None:
    """Publish one atomic LTXIDX02 manifest from validated shard markers."""
    root = assert_write_path_allowed(shard_root)
    output_path = assert_write_path_allowed(output)
    reject_path = assert_write_path_allowed(reject_output)
    summary_path = assert_write_path_allowed(summary_output)
    dedup_path = assert_write_path_allowed(dedup_db) if dedup_db is not None else None
    summary = merge_manifest_shards(
        root,
        tasks=parse_tasks(tasks),
        output=output_path,
        reject_output=reject_path,
        summary_output=summary_path,
        dedup_db=dedup_path,
        require_all_shards=require_all_shards,
    )
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
