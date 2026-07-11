#!/usr/bin/env python3
"""Validate Stage 2 precomputed files before loading any large model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import typer


app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

REQUIRED_DIRS = (
    "latents",
    "vlm_conditions",
    "conditions",
    "multi_reference_latents",
    "gt_siglip_tokens",
    "planner_vlm_inputs",
)


def _files_by_relative_path(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {str(path.relative_to(directory)): path for path in directory.rglob("*.pt")}


def _load_tensor_file(path: Path) -> dict[str, Any]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")  # noqa: S614
    if not isinstance(data, dict):
        raise ValueError(f"expected dict, got {type(data).__name__}")
    return data


def _scalar_int(value: Any, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar, got shape {tuple(value.shape)}")
        return int(value.item())
    return int(value)


def _scalar_float(value: Any, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar, got shape {tuple(value.shape)}")
        return float(value.item())
    return float(value)


def _require_tensor(data: dict[str, Any], name: str) -> torch.Tensor:
    value = data.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"missing tensor '{name}'")
    return value


def _validate_gt(
    data: dict[str, Any],
    *,
    planner_token_count: int,
    expected_tokens_per_frame: int,
) -> int:
    tokens = _require_tensor(data, "visual_tokens")
    expected_shape = (planner_token_count, 3840)
    if tuple(tokens.shape) != expected_shape:
        raise ValueError(f"visual_tokens shape {tuple(tokens.shape)} != {expected_shape}")
    mask = _require_tensor(data, "visual_token_mask").to(dtype=torch.bool).flatten()
    if mask.numel() != planner_token_count or int(mask.sum().item()) != planner_token_count:
        raise ValueError(
            f"visual_token_mask valid count {int(mask.sum().item())}/{mask.numel()} != {planner_token_count}"
        )
    num_visual_tokens = _scalar_int(data.get("num_visual_tokens"), "num_visual_tokens")
    if num_visual_tokens != planner_token_count:
        raise ValueError(f"num_visual_tokens={num_visual_tokens} != {planner_token_count}")
    tokens_per_frame = _scalar_int(data.get("tokens_per_frame"), "tokens_per_frame")
    if tokens_per_frame != expected_tokens_per_frame:
        raise ValueError(f"tokens_per_frame={tokens_per_frame} != {expected_tokens_per_frame}")
    sampled = _require_tensor(data, "sampled_frame_indices").flatten()
    if planner_token_count % expected_tokens_per_frame != 0:
        raise ValueError(
            f"planner_token_count={planner_token_count} is not divisible by "
            f"expected_tokens_per_frame={expected_tokens_per_frame}"
        )
    expected_frames = planner_token_count // expected_tokens_per_frame
    if sampled.numel() != expected_frames:
        raise ValueError(f"sampled_frame_indices count={sampled.numel()} != {expected_frames}")
    source_fps = _scalar_float(data.get("source_fps"), "source_fps")
    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps}")
    return num_visual_tokens


def _validate_planner(data: dict[str, Any], *, planner_token_count: int) -> int:
    input_ids = _require_tensor(data, "input_ids").flatten()
    attention_mask = _require_tensor(data, "attention_mask").flatten()
    if input_ids.shape != attention_mask.shape:
        raise ValueError(f"input_ids shape {tuple(input_ids.shape)} != attention_mask {tuple(attention_mask.shape)}")

    required_masks = (
        "planner_placeholder_mask",
        "planner_region_mask",
        "ref_image_region_mask",
        "text_token_mask",
        "ntp_label_mask",
    )
    masks = {name: _require_tensor(data, name).to(dtype=torch.bool).flatten() for name in required_masks}
    for name, mask in masks.items():
        if mask.shape != input_ids.shape:
            raise ValueError(f"{name} shape {tuple(mask.shape)} != input_ids {tuple(input_ids.shape)}")

    placeholder_count = int(masks["planner_placeholder_mask"].sum().item())
    if placeholder_count != planner_token_count:
        raise ValueError(f"planner placeholder count={placeholder_count} != {planner_token_count}")

    ntp_labels = _require_tensor(data, "ntp_labels").flatten()
    if ntp_labels.shape != input_ids.shape:
        raise ValueError(f"ntp_labels shape {tuple(ntp_labels.shape)} != input_ids {tuple(input_ids.shape)}")
    label_mask = masks["ntp_label_mask"]
    if not torch.equal(ntp_labels != -100, label_mask):
        raise ValueError("(ntp_labels != -100) does not exactly match ntp_label_mask")
    if not bool(label_mask.any()):
        raise ValueError("sample contains no valid NTP labels")
    if bool((label_mask & ~masks["text_token_mask"]).any()):
        raise ValueError("non-text positions participate in NTP")
    if bool((label_mask & masks["planner_region_mask"]).any()):
        raise ValueError("planner region participates in NTP")
    if bool((label_mask & masks["planner_placeholder_mask"]).any()):
        raise ValueError("planner placeholder positions participate in NTP")
    if bool((label_mask & masks["ref_image_region_mask"]).any()):
        raise ValueError("reference image region participates in NTP")
    padding_mask = ~attention_mask.to(dtype=torch.bool)
    if bool((label_mask & padding_mask).any()):
        raise ValueError("padding positions participate in NTP")
    return placeholder_count


@app.command()
def main(
    precomputed_root: str = typer.Option(..., help="Root containing the six Stage 2 data directories."),
    planner_token_count: int = typer.Option(2048),
    expected_tokens_per_frame: int = typer.Option(256),
    max_samples: int | None = typer.Option(None),
    strict: bool = typer.Option(True, "--strict/--no-strict"),
) -> None:
    if planner_token_count < 1:
        raise typer.BadParameter("--planner-token-count must be >= 1")
    if expected_tokens_per_frame < 1:
        raise typer.BadParameter("--expected-tokens-per-frame must be >= 1")
    if max_samples is not None and max_samples < 1:
        raise typer.BadParameter("--max-samples must be >= 1")
    root = Path(precomputed_root)
    files = {name: _files_by_relative_path(root / name) for name in REQUIRED_DIRS}
    path_sets = {name: set(paths) for name, paths in files.items()}
    all_paths = set().union(*path_sets.values())
    common_paths = set.intersection(*path_sets.values()) if path_sets else set()
    missing = {name: sorted(all_paths - paths) for name, paths in path_sets.items()}

    failures: list[dict[str, Any]] = []
    for name in REQUIRED_DIRS:
        if not (root / name).is_dir():
            failures.append({"relative_path": None, "errors": [f"missing directory: {name}"]})
        if missing[name]:
            failures.append(
                {
                    "relative_path": None,
                    "errors": [f"{name} is missing {len(missing[name])} relative paths"],
                }
            )

    selected_paths = sorted(common_paths)
    if max_samples is not None:
        selected_paths = selected_paths[:max_samples]
    for relative_path in selected_paths:
        errors = []
        try:
            gt_data = _load_tensor_file(files["gt_siglip_tokens"][relative_path])
            gt_count = _validate_gt(
                gt_data,
                planner_token_count=planner_token_count,
                expected_tokens_per_frame=expected_tokens_per_frame,
            )
        except Exception as exc:
            errors.append(f"gt_siglip_tokens: {type(exc).__name__}: {exc}")
            gt_count = None
        try:
            planner_data = _load_tensor_file(files["planner_vlm_inputs"][relative_path])
            planner_count = _validate_planner(planner_data, planner_token_count=planner_token_count)
        except Exception as exc:
            errors.append(f"planner_vlm_inputs: {type(exc).__name__}: {exc}")
            planner_count = None
        if gt_count is not None and planner_count is not None and gt_count != planner_count:
            errors.append(f"GT token count {gt_count} != planner placeholder count {planner_count}")
        if errors:
            failures.append({"relative_path": relative_path, "errors": errors})

    summary = {
        "precomputed_root": str(root),
        "directory_file_counts": {name: len(paths) for name, paths in files.items()},
        "common_sample_count": len(common_paths),
        "checked_sample_count": len(selected_paths),
        "missing_relative_path_counts": {name: len(paths) for name, paths in missing.items()},
        "missing_relative_paths": missing,
        "failure_count": len(failures),
        "failed_samples": failures,
    }
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))
    if strict and failures:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
