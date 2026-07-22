#!/usr/bin/env python3
"""Generate I2I/R2V samples with strict-no-GT semantic/video joint flow."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Literal

import torch
import typer
from rich.console import Console

from ltx_trainer.online_data.online_batch_encoder import OnlineSampleEncodeError
from ltx_trainer.online_inference.checkpoint_runtime import (
    load_online_inference_runtime,
    resolve_checkpoint,
)
from ltx_trainer.online_inference.output_artifacts import atomic_write_json, sample_output_dir
from ltx_trainer.online_inference.path_policy import assert_online_inference_output_path
from ltx_trainer.online_inference.raw_condition_encoder import RawReferenceLoadError
from ltx_trainer.online_inference.runner import read_selected_samples, run_online_sample
from ltx_trainer.online_inference.vae_decode import validate_vae_decode_request

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)
console = Console()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dtype(value: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if value not in mapping:
        raise typer.BadParameter(f"--dtype must be one of {sorted(mapping)}")
    return mapping[value]


@app.command()
def main(
    config: str = typer.Option(..., "--config", help="Semantic-flow online YAML."),
    selected_samples: str = typer.Option(..., "--samples", help="Selection JSONL."),
    output_root: str = typer.Option(..., "--output-root"),
    checkpoint: str | None = typer.Option(None, "--checkpoint"),
    latest_ready_dir: str | None = typer.Option(None, "--latest-ready-dir"),
    task: Literal["both", "i2i", "r2v"] = typer.Option("both", "--task"),
    limit: int | None = typer.Option(None, "--limit"),
    device: str = typer.Option("cuda", "--device"),
    dtype: str = typer.Option("bfloat16", "--dtype"),
    seed: int = typer.Option(42, "--seed"),
    num_inference_steps: int = typer.Option(50, "--num-inference-steps"),
    decode_tile: bool = typer.Option(True, "--decode-tile/--no-decode-tile"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run"),
    overwrite: bool = typer.Option(False, "--overwrite/--no-overwrite"),
    allow_legacy_reference_rope: bool = typer.Option(False, "--allow-legacy-reference-rope"),
) -> None:
    if limit is not None and limit < 1:
        raise typer.BadParameter("--limit must be >= 1")
    if num_inference_steps < 1:
        raise typer.BadParameter("--num-inference-steps must be >= 1")
    checkpoint_path, marker_path, marker = resolve_checkpoint(
        checkpoint=checkpoint,
        latest_ready_dir=latest_ready_dir,
    )
    torch_device = torch.device(device)
    torch_dtype = _dtype(dtype)
    if not dry_run:
        try:
            validate_vae_decode_request(torch_device, torch_dtype)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    output = assert_online_inference_output_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    selected_tasks = None if task == "both" else {task}
    samples = read_selected_samples(selected_samples, tasks=selected_tasks, limit=limit)
    if not samples:
        raise ValueError("No selected samples matched the requested task/limit")
    runtime = load_online_inference_runtime(
        config_path=config,
        checkpoint_path=checkpoint_path,
        ready_marker_path=marker_path,
        ready_marker=marker,
        device=torch_device,
        dtype=torch_dtype,
        load_vae_decoder=not dry_run,
        allow_legacy_reference_rope=allow_legacy_reference_rope,
    )

    started = time.perf_counter()
    commit = _git_commit()
    results: list[dict] = []
    failures: list[dict] = []
    for index, sample in enumerate(samples):
        console.print(
            f"[{index + 1}/{len(samples)}] {sample['task']} {sample['sample_key']} "
            f"refs={len(sample['reference_paths'])}"
        )
        try:
            results.append(
                run_online_sample(
                    runtime=runtime,
                    sample=sample,
                    output_root=output,
                    dry_run=dry_run,
                    overwrite=overwrite,
                    seed=seed + index,
                    num_inference_steps=num_inference_steps,
                    decode_tile=decode_tile,
                    code_commit=commit,
                )
            )
        except (RawReferenceLoadError, OnlineSampleEncodeError) as exc:
            failure = {
                "sample_key": sample.get("sample_key"),
                "task": sample.get("task"),
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            failures.append(failure)
            atomic_write_json(sample_output_dir(output, sample) / "failure.json", failure)
    summary = {
        "architecture": "semantic_flow_v1",
        "code_commit": commit,
        "config": str(Path(config).expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_audit": runtime.checkpoint_audit,
        "strict_no_gt": True,
        "requested_count": len(samples),
        "success_count": sum(result.get("status") in {"success", "dry_run_success"} for result in results),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
    }
    path = output / "run_summary.json"
    if path.exists() and not overwrite:
        path = output / f"run_summary_{int(time.time())}.json"
    atomic_write_json(path, summary)
    console.print_json(data=summary)
    if failures:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
