#!/usr/bin/env python3
"""Convert Phantom dict/list raw annotations to streaming-friendly JSONL.

This utility preserves the Phantom source schema.  The online manifest builder
still selects ``PhantomDataset`` and performs all normalization, media
validation, clip planning, semantic-anchor planning, and SHA generation.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import typer

from ltx_trainer.online_data.path_safety import assert_write_path_allowed

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


def _iter_items(payload: Any) -> list[tuple[str, Mapping[str, Any]]]:
    if isinstance(payload, Mapping):
        raw_items = [(str(key), value) for key, value in payload.items()]
    elif isinstance(payload, list):
        raw_items = [(str(index), value) for index, value in enumerate(payload)]
    else:
        raise ValueError(f"Phantom JSON must be a dict or list, got {type(payload).__name__}")
    invalid = [row_id for row_id, row in raw_items if not isinstance(row, Mapping)]
    if invalid:
        raise ValueError(f"Phantom JSON contains non-object rows: {invalid[:5]}")
    return raw_items  # type: ignore[return-value]


@app.command()
def main(
    input_json: str = typer.Argument(..., help="Phantom dict/list raw annotation JSON"),
    output_jsonl: str = typer.Option(..., "--output-jsonl", help="Destination raw annotation JSONL"),
) -> None:
    source = Path(input_json).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Phantom annotation does not exist: {source}")
    destination = assert_write_path_allowed(output_jsonl)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.tmp.{os.getpid()}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    items = _iter_items(payload)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row_id, raw_row in items:
                row = dict(raw_row)
                row.setdefault("source_record_id", row_id)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    typer.echo(f"Wrote {len(items)} raw Phantom rows to {destination}")


if __name__ == "__main__":
    app()
