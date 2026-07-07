#!/usr/bin/env python3
"""Validate a multi-reference precomputed dataset subset before overfit training."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
import typer
from rich.console import Console

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Check that selected multi-reference samples have all required precomputed tensors.",
)
console = Console()

REQUIRED_STAGE2_KEYS = (
    "input_ids",
    "attention_mask",
    "planner_placeholder_mask",
    "planner_region_mask",
    "text_token_mask",
    "ref_image_region_mask",
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


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for pattern in ("*.json", "*.jsonl", "*.csv"):
            for file in sorted(path.glob(pattern)):
                rows.extend(_read_manifest_file(file))
        if not rows:
            raise FileNotFoundError(f"No manifest shards found in {path}")
        return rows
    return _read_manifest_file(path)


def _resolve_path(value: str, root_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root_dir / path


def _output_relative(path: Path, data_root: Path) -> Path:
    try:
        return path.relative_to(data_root)
    except ValueError:
        return Path(*path.parts[1:]) if path.is_absolute() else path


def _normalize_precomputed_root(path: Path) -> Path:
    if (path / ".precomputed").is_dir():
        return path / ".precomputed"
    return path


def _load_pt(path: Path) -> dict[str, Any]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


def _check_gt_visual_tokens(path: Path, *, planner_token_count: int, visual_token_key: str) -> list[str]:
    errors: list[str] = []
    data = _load_pt(path)
    tokens = data.get(visual_token_key)
    if not isinstance(tokens, torch.Tensor):
        return [f"{path}: missing tensor '{visual_token_key}'"]
    if tokens.ndim != 2:
        errors.append(f"{path}: {visual_token_key} must have shape [K, D], got {tuple(tokens.shape)}")
    elif int(tokens.shape[0]) != planner_token_count:
        errors.append(
            f"{path}: {visual_token_key} has K={int(tokens.shape[0])}, expected planner_token_count={planner_token_count}"
        )
    return errors


def _check_multi_reference_latents(path: Path) -> list[str]:
    data = _load_pt(path)
    latents = data.get("latents")
    if not isinstance(latents, torch.Tensor):
        return [f"{path}: missing tensor 'latents'"]
    if latents.ndim >= 5:
        ref_count = int(latents.shape[0])
    elif latents.ndim == 4:
        ref_count = 1
    else:
        return [f"{path}: reference latents must be [R,C,F,H,W] or [C,F,H,W], got {tuple(latents.shape)}"]
    if ref_count <= 0:
        return [f"{path}: reference count must be > 0"]
    num_refs = data.get("num_refs")
    if isinstance(num_refs, torch.Tensor):
        num_refs = int(num_refs.item())
    if num_refs is not None and int(num_refs) <= 0:
        return [f"{path}: num_refs must be > 0, got {num_refs}"]
    return []


def _check_planner_vlm_inputs(path: Path, planner_token_count: int) -> list[str]:
    errors: list[str] = []
    data = _load_pt(path)
    for key in REQUIRED_STAGE2_KEYS:
        if key not in data:
            errors.append(f"{path}: missing key '{key}'")
    mask = data.get("planner_placeholder_mask")
    if isinstance(mask, torch.Tensor):
        count = int(mask.bool().sum().item())
        if count != planner_token_count:
            errors.append(f"{path}: planner_placeholder_mask has {count} true values, expected {planner_token_count}")
    return errors


def _relative_pt_for_row(row: dict[str, Any], *, video_column: str, manifest_root: Path) -> Path:
    if video_column not in row:
        raise ValueError(f"Missing video column '{video_column}' in row: {row}")
    media_path = _resolve_path(str(row[video_column]), manifest_root)
    return _output_relative(media_path, manifest_root).with_suffix(".pt")


@app.command()
def main(  # noqa: PLR0913
    manifest: str = typer.Option(..., help="Selected manifest JSON/JSONL/CSV or shard directory."),
    precomputed_root: str = typer.Option(..., help="Root containing latents/, conditions/, etc."),
    planner_token_count: int = typer.Option(2048, help="Expected GT/planner visual token count K."),
    video_column: str = typer.Option("video", help="Target video/media column used for precomputed path mirroring."),
    root_dir: str | None = typer.Option(None, help="Root for resolving relative media paths. Defaults to manifest parent."),
    stage2: bool = typer.Option(True, "--stage2/--stage1-only", help="Require planner_vlm_inputs for Stage 2."),
    latents_dir: str = typer.Option("latents", help="Target video latent directory name."),
    conditions_dir: str = typer.Option("conditions", help="Text-only condition directory name."),
    vlm_conditions_dir: str = typer.Option("vlm_conditions", help="VLM reference-image condition directory name."),
    reference_latents_dir: str = typer.Option("multi_reference_latents", help="Reference latent directory name."),
    gt_visual_tokens_dir: str = typer.Option("gt_siglip_tokens", help="GT target-video visual token directory name."),
    planner_vlm_inputs_dir: str = typer.Option("planner_vlm_inputs", help="Planner VLM input directory name."),
    visual_token_key: str = typer.Option("visual_tokens", help="Key used inside gt_siglip_tokens/*.pt."),
) -> None:
    if planner_token_count < 1:
        raise typer.BadParameter("--planner-token-count must be >= 1")

    manifest_path = Path(manifest)
    manifest_root = Path(root_dir) if root_dir is not None else (manifest_path if manifest_path.is_dir() else manifest_path.parent)
    root = _normalize_precomputed_root(Path(precomputed_root))
    rows = _read_manifest(manifest_path)

    required_dirs = [latents_dir, reference_latents_dir, conditions_dir, vlm_conditions_dir, gt_visual_tokens_dir]
    if stage2:
        required_dirs.append(planner_vlm_inputs_dir)

    errors: list[str] = []
    valid = 0
    for index, row in enumerate(rows):
        sample_errors: list[str] = []
        try:
            rel_path = _relative_pt_for_row(row, video_column=video_column, manifest_root=manifest_root)
            files = {directory: root / directory / rel_path for directory in required_dirs}
            for directory, file_path in files.items():
                if not file_path.is_file():
                    sample_errors.append(f"sample {index}: missing {directory}: {file_path}")

            if gt_visual_tokens_dir in files and files[gt_visual_tokens_dir].is_file():
                sample_errors.extend(
                    f"sample {index}: {message}"
                    for message in _check_gt_visual_tokens(
                        files[gt_visual_tokens_dir],
                        planner_token_count=planner_token_count,
                        visual_token_key=visual_token_key,
                    )
                )
            if reference_latents_dir in files and files[reference_latents_dir].is_file():
                sample_errors.extend(
                    f"sample {index}: {message}"
                    for message in _check_multi_reference_latents(files[reference_latents_dir])
                )
            if stage2 and planner_vlm_inputs_dir in files and files[planner_vlm_inputs_dir].is_file():
                sample_errors.extend(
                    f"sample {index}: {message}"
                    for message in _check_planner_vlm_inputs(files[planner_vlm_inputs_dir], planner_token_count)
                )
        except Exception as exc:
            sample_errors.append(f"sample {index}: {exc}")

        if sample_errors:
            errors.extend(sample_errors)
        else:
            valid += 1

    if errors:
        console.print(f"[red]Dataset check failed: {valid}/{len(rows)} samples valid[/red]")
        for message in errors[:20]:
            console.print(f"[red]- {message}[/red]")
        if len(errors) > 20:
            console.print(f"[red]... and {len(errors) - 20} more errors[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Dataset check passed: {valid}/{len(rows)} samples valid[/green]")


if __name__ == "__main__":
    app()
