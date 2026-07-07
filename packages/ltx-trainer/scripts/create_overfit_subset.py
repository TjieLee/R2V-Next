#!/usr/bin/env python3
"""Create a deterministic small manifest for multi-reference overfit tests.

The script preserves every original row field and path. It can also create a
precomputed subset root made of symlinks, which is useful because the current
trainer indexes .pt files under data.preprocessed_data_root rather than reading a
manifest at train time.
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Create a small manifest, and optionally a symlinked precomputed root, for overfit tests.",
)
console = Console()

SUPPORTED_SUFFIXES = (".json", ".jsonl", ".csv")
DEFAULT_PRECOMPUTED_SOURCES = (
    "latents,conditions,vlm_conditions,multi_reference_latents,gt_siglip_tokens,planner_vlm_inputs"
)


def _read_manifest_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return list(data.values())
        if isinstance(data, list):
            return data
        raise ValueError(f"JSON manifest must contain a list or dict of objects: {path}")
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    raise ValueError(f"Unsupported manifest suffix: {path.suffix}")


def _read_manifest(input_path: Path) -> list[dict[str, Any]]:
    if input_path.is_dir():
        files: list[Path] = []
        for suffix in SUPPORTED_SUFFIXES:
            files.extend(sorted(input_path.glob(f"*{suffix}")))
        if not files:
            raise FileNotFoundError(f"No JSON/JSONL/CSV manifest shards found in {input_path}")
        rows: list[dict[str, Any]] = []
        for file in files:
            rows.extend(_read_manifest_file(file))
        return rows
    if input_path.is_file():
        return _read_manifest_file(input_path)
    raise FileNotFoundError(f"Input manifest does not exist: {input_path}")


def _write_json(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_path(value: str, root_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root_dir / path


def _output_relative(path: Path, data_root: Path) -> Path:
    try:
        return path.relative_to(data_root)
    except ValueError:
        return Path(*path.parts[1:]) if path.is_absolute() else path


def _parse_sources(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _symlink_file(src: Path, dst: Path) -> str:
    if dst.exists() or dst.is_symlink():
        return "exists"
    if not src.is_file():
        return "missing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)
    return "linked"


def _link_precomputed_subset(
    *,
    rows: list[dict[str, Any]],
    manifest_root: Path,
    video_column: str,
    precomputed_root: Path,
    subset_precomputed_root: Path,
    sources: list[str],
    allow_missing_sources: bool,
) -> None:
    precomputed_root = precomputed_root.resolve()
    if (precomputed_root / ".precomputed").is_dir():
        precomputed_root = precomputed_root / ".precomputed"
    subset_precomputed_root.mkdir(parents=True, exist_ok=True)

    total_linked = 0
    total_existing = 0
    total_missing = 0
    for source in sources:
        source_root = precomputed_root / source
        if not source_root.is_dir():
            message = f"Skipping missing source directory: {source_root}"
            if allow_missing_sources:
                console.print(f"[yellow]{message}[/yellow]")
                continue
            raise FileNotFoundError(message)

        linked = 0
        existing = 0
        missing = 0
        for row in rows:
            if video_column not in row:
                raise ValueError(f"Missing video column '{video_column}' in row: {row}")
            media_path = _resolve_path(str(row[video_column]), manifest_root)
            rel_path = _output_relative(media_path, manifest_root).with_suffix(".pt")
            status = _symlink_file(source_root / rel_path, subset_precomputed_root / source / rel_path)
            if status == "linked":
                linked += 1
            elif status == "exists":
                existing += 1
            else:
                missing += 1

        total_linked += linked
        total_existing += existing
        total_missing += missing
        console.print(
            f"{source}: linked {linked}, already existed {existing}, missing {missing} "
            f"-> {subset_precomputed_root / source}"
        )

    console.print(
        f"Precomputed symlink summary: linked {total_linked}, already existed {total_existing}, missing {total_missing}"
    )


@app.command()
def main(  # noqa: PLR0913
    input_manifest: str = typer.Option(..., help="Original train.json/train.jsonl file or a shard directory."),
    output_manifest: str = typer.Option(..., help="Output JSON manifest for the selected overfit samples."),
    num_samples: int = typer.Option(100, help="Number of samples to select."),
    seed: int = typer.Option(42, help="Random seed for deterministic sampling."),
    copy_files: bool = typer.Option(
        False,
        help="Accepted for compatibility only. Raw media/precomputed files are not copied by this workflow.",
    ),
    video_column: str = typer.Option("video", help="Target video/media column used for precomputed path mirroring."),
    root_dir: str | None = typer.Option(
        None,
        help="Root for resolving relative media paths. Defaults to the input manifest parent.",
    ),
    link_precomputed: bool = typer.Option(
        False,
        help="Create a subset precomputed root using symlinks for the selected samples.",
    ),
    precomputed_root: str | None = typer.Option(
        None,
        help="Original .precomputed root. Required with --link-precomputed.",
    ),
    subset_precomputed_root: str | None = typer.Option(
        None,
        help="Destination .precomputed root made of symlinks. Required with --link-precomputed.",
    ),
    precomputed_sources: str = typer.Option(
        DEFAULT_PRECOMPUTED_SOURCES,
        help="Comma-separated precomputed subdirectories to symlink.",
    ),
    allow_missing_sources: bool = typer.Option(
        True,
        help="Skip precomputed source directories that are not generated yet.",
    ),
) -> None:
    if num_samples < 1:
        raise typer.BadParameter("--num-samples must be >= 1")
    if copy_files:
        raise typer.BadParameter(
            "--copy-files is intentionally unsupported: this workflow does not copy raw media or precomputed files. "
            "Use --link-precomputed to create a lightweight subset root."
        )

    input_path = Path(input_manifest)
    output_path = Path(output_manifest)
    manifest_root = Path(root_dir) if root_dir is not None else (input_path if input_path.is_dir() else input_path.parent)

    rows = _read_manifest(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    rng = random.Random(seed)
    sample_count = min(num_samples, len(rows))
    selected = rng.sample(rows, sample_count)
    _write_json(selected, output_path)

    console.print(f"Selected {sample_count} samples from {len(rows)} total samples")
    console.print(f"Output: {output_path}")

    if link_precomputed:
        if precomputed_root is None or subset_precomputed_root is None:
            raise typer.BadParameter("--precomputed-root and --subset-precomputed-root are required with --link-precomputed")
        _link_precomputed_subset(
            rows=selected,
            manifest_root=manifest_root,
            video_column=video_column,
            precomputed_root=Path(precomputed_root),
            subset_precomputed_root=Path(subset_precomputed_root),
            sources=_parse_sources(precomputed_sources),
            allow_missing_sources=allow_missing_sources,
        )


if __name__ == "__main__":
    app()
