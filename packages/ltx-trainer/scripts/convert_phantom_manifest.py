#!/usr/bin/env python3

"""Convert Phantom raw JSON into the flat LTX-2 multi-reference manifest.

The Phantom data used by ``stage1_dataset.py`` is a dict-of-records where each
item stores nested metadata:

    video_path
    metadata.video_caption
    cropped_ref_paths

LTX-2's stock preprocessors expect flat rows with ``video`` and ``caption``.
The multi-reference preprocessing script also expects ``reference_images``.
This converter bridges those schemas without modifying the raw data.
"""

import json
from pathlib import Path
from typing import Any

import typer
from rich.progress import track

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Convert Phantom raw JSON to a flat LTX-2 manifest.",
)


def _resolve_path(path_value: str, root_dir: Path, absolute: bool) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = root_dir / path
    return str(path.resolve()) if absolute else str(path)


def _load_phantom_items(path: Path) -> list[tuple[str, dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [(str(key), value) for key, value in data.items()]
    if isinstance(data, list):
        return [(str(idx), value) for idx, value in enumerate(data)]
    raise ValueError(f"Expected Phantom JSON to be a dict or list, got {type(data).__name__}")


def _is_valid_item(item: dict[str, Any], require_cross_pair: bool) -> bool:
    if require_cross_pair and item.get("cross_pair") is None:
        return False
    if not item.get("video_path"):
        return False
    if not item.get("cropped_ref_paths"):
        return False
    metadata = item.get("metadata") or {}
    if not metadata.get("video_caption"):
        return False
    return True


@app.command()
def main(  # noqa: PLR0913
    input_json: str = typer.Argument(..., help="Path to Phantom raw JSON, e.g. train_data_0202.json"),
    output_json: str = typer.Option(..., help="Output flat manifest JSON path"),
    root_dir: str = typer.Option(
        ...,
        help="Root used to resolve Phantom relative paths, e.g. /mnt/workspace/liutao/phantom_data",
    ),
    part_start: int | None = typer.Option(None, help="Optional inclusive start index after filtering"),
    part_end: int | None = typer.Option(None, help="Optional exclusive end index after filtering"),
    require_cross_pair: bool = typer.Option(True, help="Keep only samples where cross_pair is present"),
    absolute_paths: bool = typer.Option(True, help="Write absolute media/reference paths"),
) -> None:
    input_path = Path(input_json)
    output_path = Path(output_json)
    root_path = Path(root_dir)

    rows = []
    items = _load_phantom_items(input_path)
    filtered = [(key, item) for key, item in items if _is_valid_item(item, require_cross_pair)]
    if part_start is not None or part_end is not None:
        filtered = filtered[part_start:part_end]

    for key, item in track(filtered, description="Converting Phantom manifest"):
        metadata = item["metadata"]
        ref_paths = [
            _resolve_path(ref_path, root_path, absolute_paths)
            for ref_path in item.get("cropped_ref_paths", [])
        ]
        if not ref_paths:
            continue

        rows.append(
            {
                "key": key,
                "video": _resolve_path(item["video_path"], root_path, absolute_paths),
                "caption": metadata["video_caption"],
                "reference_images": ref_paths,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    app()
