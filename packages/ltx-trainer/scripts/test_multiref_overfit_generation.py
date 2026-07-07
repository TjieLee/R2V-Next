#!/usr/bin/env python3
"""Package and optionally run one overfit generation sample.

This script intentionally avoids reimplementing the multi-reference inference
pipeline. If a standalone inference command exists in your branch, pass it via
--generation-command. Otherwise, use trainer validation to produce a video and
pass it via --generated-video; the script will package GT, references and
metadata into a consistent review directory.
"""

from __future__ import annotations

import csv
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Prepare one overfit sample and optionally run/copy a generated video for qualitative checking.",
)
console = Console()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return list(data.values())
        if isinstance(data, list):
            return data
        raise ValueError("JSON manifest must contain a list or dict of objects")
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    raise ValueError(f"Unsupported manifest suffix: {path.suffix}")


def _resolve_path(value: str, root_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root_dir / path


def _parse_ref_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("reference_images JSON must decode to a list")
            return [str(item) for item in parsed]
        for sep in ("|", ";", ","):
            if sep in text:
                return [part.strip() for part in text.split(sep) if part.strip()]
        return [text]
    raise ValueError(f"Unsupported reference list type: {type(value).__name__}")


def _copy_ref_image(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        with Image.open(src) as image:
            image.convert("RGB").save(dst, quality=95)
        return dst
    except Exception:
        fallback = dst.with_suffix(src.suffix or ".jpg")
        shutil.copy2(src, fallback)
        return fallback


def _format_generation_command(template: str, values: dict[str, str]) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        raise typer.BadParameter(f"Unknown placeholder in --generation-command: {exc}") from exc


@app.command()
def main(  # noqa: PLR0913
    checkpoint: str = typer.Option(..., help="Checkpoint file or directory to test."),
    config: str = typer.Option(..., help="Training/inference config used for the checkpoint."),
    manifest: str = typer.Option(..., help="Overfit subset manifest."),
    sample_index: int = typer.Option(0, help="Sample index inside the overfit manifest."),
    output_dir: str = typer.Option(..., help="Directory that will contain sample_x/ artifacts."),
    video_column: str = typer.Option("video", help="Target video column."),
    caption_column: str = typer.Option("caption", help="Prompt/caption column."),
    reference_column: str = typer.Option("reference_images", help="Reference image column."),
    root_dir: str | None = typer.Option(None, help="Root for relative media paths. Defaults to manifest parent."),
    generated_video: str | None = typer.Option(None, help="Existing generated video to copy into generated.mp4."),
    generation_command: str | None = typer.Option(
        None,
        help=(
            "Optional shell command template. Available placeholders: {checkpoint}, {config}, {manifest}, "
            "{sample_index}, {sample_dir}, {generated}, {prompt}, {seed}. {prompt} is shell-quoted."
        ),
    ),
    seed: int = typer.Option(42, help="Seed recorded in metadata and exposed to generation command templates."),
) -> None:
    manifest_path = Path(manifest)
    rows = _read_manifest(manifest_path)
    if sample_index < 0 or sample_index >= len(rows):
        raise typer.BadParameter(f"--sample-index must be in [0, {len(rows) - 1}]")

    root = Path(root_dir) if root_dir is not None else manifest_path.parent
    sample = rows[sample_index]
    if video_column not in sample:
        raise ValueError(f"Missing video column '{video_column}' in selected sample")

    sample_dir = Path(output_dir) / f"sample_{sample_index}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    generated_path = sample_dir / "generated.mp4"

    gt_src = _resolve_path(str(sample[video_column]), root)
    gt_dst = sample_dir / "gt.mp4"
    shutil.copy2(gt_src, gt_dst)

    ref_paths = [_resolve_path(value, root) for value in _parse_ref_list(sample.get(reference_column))]
    copied_refs: list[str] = []
    for idx, ref_path in enumerate(ref_paths):
        copied = _copy_ref_image(ref_path, sample_dir / f"ref_{idx}.jpg")
        copied_refs.append(str(copied))

    prompt = str(sample.get(caption_column, ""))
    metadata = {
        "sample_index": sample_index,
        "prompt": prompt,
        "reference_paths": [str(path) for path in ref_paths],
        "copied_references": copied_refs,
        "gt_path": str(gt_src),
        "checkpoint": str(Path(checkpoint)),
        "config": str(Path(config)),
        "seed": seed,
        "generated_path": str(generated_path) if generated_path.exists() else None,
    }

    if generated_video is not None:
        shutil.copy2(generated_video, generated_path)
        metadata["generated_path"] = str(generated_path)
        metadata["generation_mode"] = "copied_existing_video"
    elif generation_command is not None:
        command = _format_generation_command(
            generation_command,
            {
                "checkpoint": str(Path(checkpoint)),
                "config": str(Path(config)),
                "manifest": str(manifest_path),
                "sample_index": str(sample_index),
                "sample_dir": str(sample_dir),
                "generated": str(generated_path),
                "prompt": shlex.quote(prompt),
                "seed": str(seed),
            },
        )
        metadata["generation_mode"] = "generation_command"
        metadata["generation_command"] = command
        subprocess.run(command, shell=True, check=True)
        if generated_path.is_file():
            metadata["generated_path"] = str(generated_path)
        else:
            console.print(
                f"[yellow]Generation command finished, but {generated_path} was not created. "
                "Put/copy the generated video there before running compare_overfit_generation.py.[/yellow]"
            )
    else:
        metadata["generation_mode"] = "package_only"
        (sample_dir / "generation_command_template.txt").write_text(
            "No standalone multi-reference inference command was provided.\n"
            "Use trainer validation output with --generated-video, or rerun this script with --generation-command.\n",
            encoding="utf-8",
        )
        console.print(
            "[yellow]Packaged GT/reference artifacts only. Provide --generated-video or --generation-command to add generated.mp4.[/yellow]"
        )

    (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    console.print(f"Overfit sample package written to: {sample_dir}")


if __name__ == "__main__":
    app()
