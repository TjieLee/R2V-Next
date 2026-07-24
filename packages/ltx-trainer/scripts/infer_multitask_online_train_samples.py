#!/usr/bin/env python3
"""Generate I2I/R2V samples with strict-no-GT semantic/video joint flow."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Literal

import torch
import typer
import yaml
from rich.console import Console

from ltx_trainer.config import ValidationConfig
from ltx_trainer.online_data.online_batch_encoder import OnlineSampleEncodeError
from ltx_trainer.online_inference.checkpoint_runtime import (
    load_online_inference_runtime,
    resolve_checkpoint,
)
from ltx_trainer.online_inference.output_artifacts import atomic_write_json, sample_output_dir
from ltx_trainer.online_inference.path_policy import assert_online_inference_output_path
from ltx_trainer.online_inference.raw_condition_encoder import RawReferenceLoadError
from ltx_trainer.online_inference.runner import read_selected_samples, run_online_sample
from ltx_trainer.online_inference.semantic_guidance import (
    SemanticGuidanceConfig,
    parse_stg_blocks,
)
from ltx_trainer.online_inference.vae_decode import validate_vae_decode_request

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)
console = Console()


def _git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to resolve R2V-Next git commit from {repo_root}") from exc


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


def _resolve_negative_prompt(
    cli_value: str | None,
    *,
    config_path: str,
    required: bool,
) -> str | None:
    if cli_value is not None and cli_value.strip():
        return cli_value.strip()
    payload = yaml.safe_load(Path(config_path).expanduser().resolve().read_text(encoding="utf-8"))
    configured = None
    if isinstance(payload, dict):
        validation = payload.get("validation")
        if isinstance(validation, dict):
            value = validation.get("negative_prompt")
            if isinstance(value, str) and value.strip():
                configured = value.strip()
    if configured is None:
        default_value = ValidationConfig().negative_prompt
        configured = default_value.strip() or None
    if required and configured is None:
        raise typer.BadParameter(
            "CFG requires --negative-prompt or validation.negative_prompt in the trainer config"
        )
    return configured


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
    negative_prompt: str | None = typer.Option(None, "--negative-prompt"),
    guidance_scale: float = typer.Option(4.0, "--guidance-scale"),
    ref_guidance_scale: float = typer.Option(1.0, "--ref-guidance-scale"),
    guidance_rescale: float = typer.Option(0.7, "--guidance-rescale"),
    stg_scale: float = typer.Option(0.0, "--stg-scale"),
    stg_blocks: str = typer.Option("28", "--stg-blocks"),
    decode_tile: bool = typer.Option(True, "--decode-tile/--no-decode-tile"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run"),
    overwrite: bool = typer.Option(False, "--overwrite/--no-overwrite"),
    allow_legacy_reference_rope: bool = typer.Option(
        False,
        "--allow-legacy-reference-rope",
        help=(
            "Allow a metadata-free legacy semantic-flow checkpoint only when "
            "reference_rope_mode=native_overlap. It does not convert legacy weights "
            "to the appended Reference RoPE layout."
        ),
    ),
) -> None:
    if limit is not None and limit < 1:
        raise typer.BadParameter("--limit must be >= 1")
    if num_inference_steps < 1:
        raise typer.BadParameter("--num-inference-steps must be >= 1")
    try:
        guidance = SemanticGuidanceConfig(
            guidance_scale=guidance_scale,
            ref_guidance_scale=ref_guidance_scale,
            guidance_rescale=guidance_rescale,
            stg_scale=stg_scale,
            stg_blocks=parse_stg_blocks(stg_blocks),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    resolved_negative_prompt = _resolve_negative_prompt(
        negative_prompt,
        config_path=config,
        required=guidance.need_negative,
    )
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
                    guidance=guidance,
                    negative_prompt=resolved_negative_prompt,
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
        "architecture": "semantic_flow_v2",
        "code_commit": commit,
        "config": str(Path(config).expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_audit": runtime.checkpoint_audit,
        "strict_no_gt": True,
        **guidance.metadata(negative_prompt=resolved_negative_prompt),
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
