#!/usr/bin/env python3
"""Run persistent targetless Stage 3 inference over an external R2V manifest."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import typer
from rich.console import Console

from ltx_trainer.online_data.online_batch_encoder import OnlineSampleEncodeError
from ltx_trainer.online_inference.checkpoint_runtime import (
    load_online_inference_runtime,
    resolve_checkpoint,
)
from ltx_trainer.online_inference.external_eval_runner import (
    classify_sample_output,
    dataset_output_root,
    external_sample_dir,
    filter_external_records,
    git_commit,
    incremental_summary,
    run_external_sample,
    write_static_gallery,
)
from ltx_trainer.online_inference.external_eval_schema import (
    PLANNER_TOKEN_COUNT,
    load_normalized_manifest,
)
from ltx_trainer.online_inference.raw_condition_encoder import RawReferenceLoadError
from ltx_trainer.online_inference.read_only_sources import ReadOnlySourcePolicy
from ltx_trainer.online_inference.vae_decode import validate_vae_decode_request

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)
console = Console()
DEFAULT_WRITABLE_ROOT = Path("/mnt/workspace/litengjie")


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


def _source_policy(
    records: list[dict[str, Any]],
    *,
    writable_root: str | Path,
) -> ReadOnlySourcePolicy:
    allowed_files = {
        Path(record["source_json"]) for record in records
    } | {
        Path(reference)
        for record in records
        for reference in record["reference_paths"]
    }
    allowed_roots = {
        Path(record["source_json"]).parent
        for record in records
        if record["dataset_name"] == "opens2v_open_domain"
    }
    return ReadOnlySourcePolicy(
        writable_root=Path(writable_root),
        allowed_roots=tuple(sorted(allowed_roots)),
        allowed_files=frozenset(allowed_files),
    )


def _source_snapshots(
    records: list[dict[str, Any]],
    policy: ReadOnlySourcePolicy,
) -> dict[str, dict[str, Any]]:
    values = {
        record["source_json"] for record in records
    } | {
        reference for record in records for reference in record["reference_paths"]
    }
    return {
        str(path): policy.snapshot(path).__dict__
        for path in sorted(values)
    }


@app.command()
def main(  # noqa: PLR0913, PLR0915
    config: str = typer.Option(..., "--config"),
    checkpoint: str = typer.Option(..., "--checkpoint"),
    manifest: str = typer.Option(..., "--manifest"),
    output_root: str = typer.Option(..., "--output-root"),
    dataset_name: str | None = typer.Option(None, "--dataset-name"),
    limit: int | None = typer.Option(None, "--limit"),
    record_ids: list[str] | None = typer.Option(None, "--id"),
    id_prefixes: list[str] | None = typer.Option(None, "--id-prefix"),
    start_index: int = typer.Option(0, "--start-index"),
    resume: bool = typer.Option(False, "--resume"),
    overwrite_incomplete: bool = typer.Option(False, "--overwrite-incomplete"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    device: str = typer.Option("cuda", "--device"),
    dtype: str = typer.Option("bfloat16", "--dtype"),
    base_seed: int = typer.Option(42, "--base-seed"),
    seed_mode: str = typer.Option("stable-id", "--seed-mode"),
    num_inference_steps: int = typer.Option(50, "--num-inference-steps"),
    negative_prompt: str | None = typer.Option(None, "--negative-prompt"),
    guidance_scale: float = typer.Option(4.0, "--guidance-scale"),
    ref_guidance_scale: float = typer.Option(2.0, "--ref-guidance-scale"),
    vision_guidance_scale: float = typer.Option(0.0, "--vision-guidance-scale"),
    ref_guidance_mode: str = typer.Option("synchronized", "--ref-guidance-mode"),
    guidance_rescale: float = typer.Option(0.7, "--guidance-rescale"),
    stg_scale: float = typer.Option(1.0, "--stg-scale"),
    stg_blocks: str = typer.Option("28", "--stg-blocks"),
    decode_tile: bool = typer.Option(True, "--decode-tile/--no-decode-tile"),
    strict_no_gt: bool = typer.Option(True, "--strict-no-gt"),
) -> None:
    if not strict_no_gt:
        raise typer.BadParameter("External evaluation is always strict-no-GT")
    if seed_mode != "stable-id":
        raise typer.BadParameter("External evaluation only supports --seed-mode stable-id")
    if start_index < 0 or (limit is not None and limit < 1):
        raise typer.BadParameter("--start-index must be >= 0 and --limit must be >= 1")
    if num_inference_steps < 1:
        raise typer.BadParameter("--num-inference-steps must be >= 1")
    if guidance_scale < 1.0 or ref_guidance_scale < 0.0 or vision_guidance_scale < 0.0:
        raise typer.BadParameter("Guidance scales must satisfy cfg>=1, ref>=0, vision>=0")
    if not 0.0 <= guidance_rescale <= 1.0:
        raise typer.BadParameter("--guidance-rescale must be in [0,1]")
    if ref_guidance_mode != "synchronized":
        raise typer.BadParameter("This fixed external benchmark requires synchronized guidance")
    if vision_guidance_scale != 0.0:
        raise typer.BadParameter("synchronized guidance requires --vision-guidance-scale 0")

    all_records = load_normalized_manifest(manifest)
    selected = filter_external_records(
        all_records,
        dataset_name=dataset_name,
        record_ids=set(record_ids) if record_ids else None,
        id_prefixes=tuple(id_prefixes or ()),
        start_index=start_index,
        limit=limit,
    )
    if not selected:
        raise ValueError("No normalized records matched the requested filters")
    datasets = {record["dataset_name"] for record in selected}
    if len(datasets) != 1:
        raise ValueError(f"One invocation must contain exactly one dataset, got {sorted(datasets)}")
    selected_dataset = next(iter(datasets))
    policy = _source_policy(selected, writable_root=DEFAULT_WRITABLE_ROOT)
    policy.assert_write_path(manifest)
    output = policy.assert_write_path(output_root)
    dataset_root = dataset_output_root(output, selected_dataset, policy)
    policy.ensure_directory(dataset_root)
    output_states = {
        sample["output_id"]: classify_sample_output(
            external_sample_dir(dataset_root, sample["output_id"], policy),
            dry_run=dry_run,
        )
        for sample in selected
    }
    incomplete_ids = [
        output_id for output_id, state in output_states.items() if state == "incomplete"
    ]
    complete_ids = [
        output_id for output_id, state in output_states.items() if state == "complete"
    ]
    if incomplete_ids and not overwrite_incomplete:
        raise RuntimeError(
            "Incomplete outputs exist; pass --overwrite-incomplete to rerun only these IDs: "
            f"{incomplete_ids}"
        )
    if complete_ids and not resume:
        raise RuntimeError(
            "Complete outputs exist; pass --resume to skip these IDs: "
            f"{complete_ids}"
        )
    runtime_cache = policy.ensure_directory(output / ".runtime_cache")
    runtime_temp = policy.ensure_directory(output / ".tmp")
    os.environ["TMPDIR"] = str(runtime_temp)
    os.environ["HF_HOME"] = str(runtime_cache / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(runtime_cache / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(runtime_cache / "transformers")
    os.environ["XDG_CACHE_HOME"] = str(runtime_cache / "xdg")
    tempfile.tempdir = str(runtime_temp)
    source_before = _source_snapshots(selected, policy)

    torch_device = torch.device(device)
    torch_dtype = _dtype(dtype)
    if not dry_run:
        try:
            validate_vae_decode_request(torch_device, torch_dtype)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    checkpoint_path, marker_path, marker = resolve_checkpoint(
        checkpoint=checkpoint,
        latest_ready_dir=None,
    )
    runtime = load_online_inference_runtime(
        config_path=config,
        checkpoint_path=checkpoint_path,
        ready_marker_path=marker_path,
        ready_marker=marker,
        device=torch_device,
        dtype=torch_dtype,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        load_vae_decoder=not dry_run,
    )
    if int(runtime.checkpoint_audit["checkpoint_step"]) != 15000:
        raise RuntimeError(
            "External step15000_4201 evaluation requires checkpoint step 15000, got "
            f"{runtime.checkpoint_audit['checkpoint_step']}"
        )
    resolved_stg_blocks = runtime.stage1._parse_stg_blocks(stg_blocks)
    if resolved_stg_blocks != [28]:
        raise RuntimeError(f"External 4201 settings require STG block [28], got {resolved_stg_blocks}")

    started = time.perf_counter()
    code_commit = git_commit()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    run_metadata = {
        "code_commit": code_commit,
        "config": str(Path(config).expanduser().resolve()),
        "manifest": str(Path(manifest).expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": runtime.checkpoint_audit["checkpoint_sha256"],
        "checkpoint_step": runtime.checkpoint_audit["checkpoint_step"],
        "dataset_name": selected_dataset,
        "requested_count": len(selected),
        "strict_no_gt": True,
        "has_target": False,
        "seed_mode": seed_mode,
        "base_seed": base_seed,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "ref_guidance_scale": ref_guidance_scale,
        "vision_guidance_scale": vision_guidance_scale,
        "ref_guidance_mode": ref_guidance_mode,
        "guidance_rescale": guidance_rescale,
        "stg_scale": stg_scale,
        "stg_blocks": resolved_stg_blocks,
        "decode_tile": decode_tile,
        "source_identity_before": source_before,
        "writable_root": str(policy.writable_root),
        "runtime_cache_root": str(runtime_cache),
        "runtime_temp_root": str(runtime_temp),
    }
    incremental_summary(
        dataset_root=dataset_root,
        policy=policy,
        run_metadata=run_metadata,
        results=results,
        failures=failures,
        started=started,
    )

    for index, sample in enumerate(selected, start=1):
        console.print(
            f"[{index}/{len(selected)}] {sample['source_record_id']} "
            f"refs={len(sample['reference_paths'])}"
        )
        try:
            result = run_external_sample(
                runtime=runtime,
                sample=sample,
                output_root=output,
                policy=policy,
                dry_run=dry_run,
                resume=resume,
                overwrite_incomplete=overwrite_incomplete,
                base_seed=base_seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                ref_guidance_scale=ref_guidance_scale,
                vision_guidance_scale=vision_guidance_scale,
                ref_guidance_mode=ref_guidance_mode,
                guidance_rescale=guidance_rescale,
                stg_scale=stg_scale,
                stg_blocks=resolved_stg_blocks,
                decode_tile=decode_tile,
                code_commit=code_commit,
            )
            if result["status"] == "dry_run_success":
                if result["planner_token_count"] != PLANNER_TOKEN_COUNT:
                    raise RuntimeError("Planner token count is not 2048")
                if result["planner_output_mask_true_count"] != PLANNER_TOKEN_COUNT:
                    raise RuntimeError("R2V planner output mask does not contain 2048 true values")
                alias_check = result["strict_no_gt_checks"]["reference_target_alias_check"]
                if alias_check != "not_applicable_no_target":
                    raise RuntimeError("External dry-run did not use targetless strict-no-GT encoding")
            results.append(result)
        except (RawReferenceLoadError, OnlineSampleEncodeError) as exc:
            failure = {
                "source_record_id": sample["source_record_id"],
                "output_id": sample["output_id"],
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            failures.append(failure)
            sample_dir = external_sample_dir(dataset_root, sample["output_id"], policy)
            policy.atomic_write_json(sample_dir / "failure.json", failure)
            console.print(f"[yellow]Sample-local data failure:[/yellow] {exc}")
        write_static_gallery(records=selected, dataset_root=dataset_root, policy=policy)
        incremental_summary(
            dataset_root=dataset_root,
            policy=policy,
            run_metadata=run_metadata,
            results=results,
            failures=failures,
            started=started,
        )

    source_after = _source_snapshots(selected, policy)
    unchanged = source_before == source_after
    final_metadata = {
        **run_metadata,
        "source_identity_after": source_after,
        "source_identity_unchanged": unchanged,
    }
    incremental_summary(
        dataset_root=dataset_root,
        policy=policy,
        run_metadata=final_metadata,
        results=results,
        failures=failures,
        started=started,
    )
    if not unchanged:
        raise RuntimeError("One or more read-only source files changed during inference")
    console.print(
        f"[green]Completed dataset={selected_dataset} results={len(results)} "
        f"failures={len(failures)} source_identity_unchanged={unchanged}[/green]"
    )


if __name__ == "__main__":
    app()
