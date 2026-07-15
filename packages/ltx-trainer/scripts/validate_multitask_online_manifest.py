"""Validate finalized online manifests and optionally re-check video headers."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import typer

from ltx_trainer.online_data.constants import VIDEO_NUM_FRAMES, VIDEO_TASK
from ltx_trainer.online_data.media_decoder import probe_video
from ltx_trainer.online_data.manifest_index import default_manifest_index_path, iter_index_entries
from ltx_trainer.online_data.multitask_dataset import validate_manifest_record
from ltx_trainer.online_data.path_safety import assert_write_path_allowed

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


@app.command()
def main(
    manifest: str = typer.Argument(...),
    check_media_headers: bool = typer.Option(True, "--check-media-headers/--no-check-media-headers"),
) -> None:
    manifest_path = Path(manifest).expanduser().resolve()
    task_counts: dict[str, int] = {}
    row_count = 0
    dedup_path = assert_write_path_allowed(
        manifest_path.with_suffix(manifest_path.suffix + f".validate.{os.getpid()}.sqlite")
    )
    connection = sqlite3.connect(dedup_path)
    try:
        connection.execute("CREATE TABLE keys (sample_key TEXT PRIMARY KEY) WITHOUT ROWID")
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                validate_manifest_record(row, row_count)
                sample_key = str(row["sample_key"])
                try:
                    connection.execute("INSERT INTO keys VALUES (?)", (sample_key,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"Duplicate sample_key at row {row_count}: {sample_key}") from exc
                task_counts[row["task"]] = task_counts.get(row["task"], 0) + 1
                if check_media_headers and row["task"] == VIDEO_TASK:
                    header = probe_video(row["target_path"])
                    source_indices = row["target_source_frame_indices"]
                    if len(source_indices) != VIDEO_NUM_FRAMES:
                        raise ValueError(
                            f"Row {row_count} does not contain exactly {VIDEO_NUM_FRAMES} source indices"
                        )
                    if int(header["frame_count"]) > 0 and source_indices[-1] >= int(header["frame_count"]):
                        raise ValueError(
                            f"Row {row_count} requests frame {source_indices[-1]} beyond the media header"
                        )
                row_count += 1
        connection.commit()
    finally:
        connection.close()
        dedup_path.unlink(missing_ok=True)
    index_path = default_manifest_index_path(manifest_path)
    if index_path.is_file():
        indexed_rows = sum(1 for _ in iter_index_entries(index_path))
        if indexed_rows != row_count:
            raise ValueError(f"Manifest/index row mismatch: manifest={row_count}, index={indexed_rows}")
    typer.echo(f"Validated {row_count} online manifest rows: {task_counts}")


if __name__ == "__main__":
    app()
