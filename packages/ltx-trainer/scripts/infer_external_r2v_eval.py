#!/usr/bin/env python3
"""Run strict-no-GT semantic-flow inference on one normalized external R2V dataset."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import torch
import typer
import yaml
from rich.console import Console

from ltx_trainer.config import ValidationConfig
from ltx_trainer.online_inference.checkpoint_runtime import (
    load_online_inference_runtime,
    resolve_checkpoint,
)
from ltx_trainer.online_inference.external_eval_runner import (
    run_external_records,
    select_external_records,
)
from ltx_trainer.online_inference.external_eval_schema import DATASET_SCHEMAS
from ltx_trainer.online_inference.output_artifacts import atomic_write_json
from ltx_trainer.online_inference.path_policy import assert_external_eval_output_path
from ltx_trainer.online_inference.semantic_guidance import (
    SemanticGuidanceConfig,
    parse_stg_blocks,
)
from ltx_trainer.online_inference.vae_decode import validate_vae_decode_request

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)
console = Console()


def _git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _dtype(value: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise typer.BadParameter(f"--dtype must be one of {sorted(mapping)}") from exc


def _negative_prompt(config_path: str, cli_value: str | None, *, required: bool) -> str | None:
    if cli_value is not None and cli_value.strip():
        return cli_value.strip()
    payload = yaml.safe_load(Path(config_path).expanduser().resolve().read_text(encoding="utf-8"))
    configured = None
    if isinstance(payload, dict) and isinstance(payload.get("validation"), dict):
        value = payload["validation"].get("negative_prompt")
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
    config: str = typer.Option(..., "--config"),
    manifest: str = typer.Option(..., "--manifest"),
    output_root: str = typer.Option(..., "--output-root"),
    dataset_name: str = typer.Option(..., "--dataset-name"),
    checkpoint: str | None = typer.Option(None, "--checkpoint"),
    latest_ready_dir: str | None = typer.Option(None, "--latest-ready-dir"),
    limit: int | None = typer.Option(None, "--limit"),
    ids: list[str] | None = typer.Option(None, "--id"),
    id_prefix: str | None = typer.Option(None, "--id-prefix"),
    start_index: int = typer.Option(0, "--start-index"),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
    overwrite_incomplete: bool = typer.Option(False, "--overwrite-incomplete"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run"),
    device: str = typer.Option("cuda", "--device"),
    dtype: str = typer.Option("bfloat16", "--dtype"),
    base_seed: int = typer.Option(42, "--base-seed"),
    num_inference_steps: int = typer.Option(50, "--num-inference-steps"),
    negative_prompt: str | None = typer.Option(None, "--negative-prompt"),
    guidance_mode: str = typer.Option(
        "positive_ref",
        "--guidance-mode",
        help="Guidance mode: positive_ref, debiased_ref, or latent_ref.",
    ),
    guidance_scale: float = typer.Option(4.0, "--guidance-scale"),
    ref_guidance_scale: float = typer.Option(1.0, "--ref-guidance-scale"),
    guidance_rescale: float = typer.Option(0.7, "--guidance-rescale"),
    stg_scale: float = typer.Option(0.0, "--stg-scale"),
    stg_blocks: str = typer.Option("28", "--stg-blocks"),
    decode_tile: bool = typer.Option(True, "--decode-tile/--no-decode-tile"),
    allow_legacy_reference_rope: bool = typer.Option(
        False,
        "--allow-legacy-reference-rope",
    ),
) -> None:
    if dataset_name not in DATASET_SCHEMAS:
        raise typer.BadParameter(
            f"--dataset-name must be one of {sorted(DATASET_SCHEMAS)}"
        )
    if num_inference_steps < 1:
        raise typer.BadParameter("--num-inference-steps must be >= 1")
    try:
        guidance = SemanticGuidanceConfig(
            guidance_mode=guidance_mode,
            guidance_scale=guidance_scale,
            ref_guidance_scale=ref_guidance_scale,
            guidance_rescale=guidance_rescale,
            stg_scale=stg_scale,
            stg_blocks=parse_stg_blocks(stg_blocks),
        )
        records = select_external_records(
            manifest,
            dataset_name=dataset_name,
            ids=ids or (),
            id_prefix=id_prefix,
            start_index=start_index,
            limit=limit,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    resolved_negative = _negative_prompt(
        config,
        negative_prompt,
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
    output = assert_external_eval_output_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
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
    results, failures = run_external_records(
        runtime=runtime,
        records=records,
        output_root=output,
        base_seed=base_seed,
        num_inference_steps=num_inference_steps,
        guidance=guidance,
        negative_prompt=resolved_negative,
        dry_run=dry_run,
        resume=resume,
        overwrite_incomplete=overwrite_incomplete,
        decode_tile=decode_tile,
        code_commit=commit,
    )
    summary = {
        "architecture": "semantic_flow_v2",
        "code_commit": commit,
        "dataset_name": dataset_name,
        "manifest": str(Path(manifest).expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "strict_no_gt": True,
        "requested_count": len(records),
        "success_count": sum(
            result.get("status") in {"success", "dry_run_success", "skipped_existing"}
            for result in results
        ),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
        **guidance.metadata(negative_prompt=resolved_negative),
    }
    summary_path = output / dataset_name / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary_path, summary)
    console.print_json(data=summary)
    if failures:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
