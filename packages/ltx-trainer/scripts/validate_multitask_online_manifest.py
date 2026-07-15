"""Validate finalized online manifests and optionally re-check video headers."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ltx_trainer.online_data.constants import VIDEO_NUM_FRAMES, VIDEO_TASK
from ltx_trainer.online_data.media_decoder import probe_video
from ltx_trainer.online_data.multitask_dataset import validate_manifest_record

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


@app.command()
def main(
    manifest: str = typer.Argument(...),
    check_media_headers: bool = typer.Option(True, "--check-media-headers/--no-check-media-headers"),
) -> None:
    manifest_path = Path(manifest).expanduser().resolve()
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen_keys: set[str] = set()
    task_counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        validate_manifest_record(row, index)
        sample_key = str(row["sample_key"])
        if sample_key in seen_keys:
            raise ValueError(f"Duplicate sample_key at row {index}: {sample_key}")
        seen_keys.add(sample_key)
        task_counts[row["task"]] = task_counts.get(row["task"], 0) + 1
        if check_media_headers and row["task"] == VIDEO_TASK:
            header = probe_video(row["target_path"])
            source_indices = row["target_source_frame_indices"]
            if len(source_indices) != VIDEO_NUM_FRAMES:
                raise ValueError(f"Row {index} does not contain exactly {VIDEO_NUM_FRAMES} source indices")
            if int(header["frame_count"]) > 0 and source_indices[-1] >= int(header["frame_count"]):
                raise ValueError(f"Row {index} requests frame {source_indices[-1]} beyond the media header")
    typer.echo(f"Validated {len(rows)} online manifest rows: {task_counts}")


if __name__ == "__main__":
    app()
