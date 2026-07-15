#!/usr/bin/env python3
"""Generate I2I/R2V replays from raw online training references and captions."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import torch
import typer
from rich.console import Console

from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_data.media_decoder import decode_image_rgb, decode_video_indices
from ltx_trainer.online_data.online_batch_encoder import OnlineSampleEncodeError
from ltx_trainer.online_data.transforms import deterministic_resize_center_crop
from ltx_trainer.online_inference.checkpoint_runtime import (
    OnlineInferenceRuntime,
    load_online_inference_runtime,
    resolve_checkpoint,
)
from ltx_trainer.online_inference.output_artifacts import (
    atomic_save_png,
    atomic_save_video,
    atomic_write_json,
    sample_output_dir,
    tensor_frame_to_pil,
)
from ltx_trainer.online_inference.path_policy import assert_online_inference_output_path
from ltx_trainer.online_inference.raw_condition_encoder import RawReferenceLoadError
from ltx_trainer.online_inference.runner import read_selected_samples, run_online_sample

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Replay selected raw online Stage 3 training samples without target leakage.",
)
console = Console()
RefGuidanceMode = Literal[
    "synchronized",
    "shared_planner_latent_only",
    "shared_planner_full_vlm_latent_only",
]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _export_ground_truth(
    runtime: OnlineInferenceRuntime,
    sample: dict[str, Any],
    sample_dir: Path,
) -> str:
    """Explicit post-generation comparison export; never used as a model condition."""
    target_path = Path(str(sample["target_path"])).expanduser().resolve()
    if sample["task"] == IMAGE_TASK:
        target = decode_image_rgb(target_path).unsqueeze(0)
        transformed = deterministic_resize_center_crop(
            target,
            target_height=int(sample["height"]),
            target_width=int(sample["width"]),
            crop_xyxy=sample.get("crop_xyxy"),
            chunk_frames=1,
        )[0]
        image = tensor_frame_to_pil(transformed.permute(2, 0, 1).float().div(255.0))
        path = sample_dir / "target_gt.png"
        atomic_save_png(image, path)
    else:
        frames = decode_video_indices(
            target_path,
            sample["target_source_frame_indices"],
            decoder=runtime.cfg.data.online_encoding.video_decoder,
            timeout_seconds=runtime.cfg.data.online_encoding.decode_timeout_seconds,
        )
        transformed = deterministic_resize_center_crop(
            frames,
            target_height=int(sample["height"]),
            target_width=int(sample["width"]),
            crop_xyxy=sample.get("crop_xyxy"),
            chunk_frames=runtime.cfg.data.online_encoding.cpu_transform_chunk_frames,
        )
        video = transformed.permute(0, 3, 1, 2).float().div(255.0)
        path = sample_dir / "target_gt.mp4"
        atomic_save_video(
            video,
            path,
            fps=float(sample["fps"]),
            save_video=runtime.stage1.save_video,
        )
    atomic_write_json(
        sample_dir / "ground_truth_export.json",
        {
            "post_generation_only": True,
            "used_for_conditioning": False,
            "target_path": str(target_path),
            "exported_path": str(path),
        },
    )
    return str(path)


@app.command()
def main(  # noqa: PLR0913, PLR0915
    config: str = typer.Option(..., "--config", help="Stage 3 online YAML."),
    selected_samples: str = typer.Option(
        ...,
        "--samples",
        "--selected-samples",
        help="Selection JSONL.",
    ),
    output_root: str = typer.Option(..., "--output-root"),
    checkpoint: str | None = typer.Option(None, "--checkpoint"),
    latest_ready_dir: str | None = typer.Option(
        None,
        "--latest-ready-checkpoint",
        "--latest-ready-dir",
    ),
    task: Literal["both", "i2i", "r2v"] = typer.Option("both", "--task"),
    limit: int | None = typer.Option(None, "--limit"),
    device: str = typer.Option("cuda", "--device"),
    dtype: str = typer.Option("bfloat16", "--dtype"),
    seed: int = typer.Option(42, "--seed"),
    num_inference_steps: int = typer.Option(50, "--num-inference-steps"),
    negative_prompt: str | None = typer.Option(None, "--negative-prompt"),
    guidance_scale: float = typer.Option(2.0, "--guidance-scale"),
    ref_guidance_scale: float = typer.Option(2.0, "--ref-guidance-scale"),
    vision_guidance_scale: float = typer.Option(0.0, "--vision-guidance-scale"),
    ref_guidance_mode: RefGuidanceMode = typer.Option("synchronized", "--ref-guidance-mode"),
    guidance_rescale: float = typer.Option(0.0, "--guidance-rescale"),
    stg_scale: float = typer.Option(1.0, "--stg-scale"),
    stg_blocks: str | None = typer.Option(None, "--stg-blocks"),
    decode_tile: bool = typer.Option(True, "--decode-tile/--no-decode-tile"),
    strict_no_gt: bool = typer.Option(True, "--strict-no-gt/--allow-gt"),
    export_ground_truth: bool = typer.Option(
        False,
        "--export-ground-truth/--no-export-ground-truth",
    ),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run"),
    overwrite: bool = typer.Option(False, "--overwrite/--no-overwrite"),
) -> None:
    if not strict_no_gt:
        raise typer.BadParameter(
            "Target conditioning is intentionally unsupported; use --export-ground-truth only "
            "for post-generation comparison"
        )
    if limit is not None and limit < 1:
        raise typer.BadParameter("--limit must be >= 1")
    if num_inference_steps < 1:
        raise typer.BadParameter("--num-inference-steps must be >= 1")
    if guidance_scale < 1.0 or ref_guidance_scale < 0.0 or vision_guidance_scale < 0.0:
        raise typer.BadParameter("Guidance scales must satisfy cfg>=1, ref>=0, vision>=0")
    if not 0.0 <= guidance_rescale <= 1.0:
        raise typer.BadParameter("--guidance-rescale must be in [0,1]")
    if vision_guidance_scale != 0.0 and ref_guidance_mode == "synchronized":
        raise typer.BadParameter("Vision guidance requires a shared-planner ref-guidance mode")

    checkpoint_path, marker_path, marker = resolve_checkpoint(
        checkpoint=checkpoint,
        latest_ready_dir=latest_ready_dir,
    )
    torch_device = torch.device(device)
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise typer.BadParameter(f"--dtype must be one of {sorted(dtype_map)}")
    torch_dtype = dtype_map[dtype]
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
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        load_vae_decoder=not dry_run,
    )
    console.print_json(
        data={
            "checkpoint_sha256": runtime.checkpoint_audit["checkpoint_sha256"],
            "checkpoint_step": runtime.checkpoint_audit["checkpoint_step"],
            "base_model_path": runtime.cfg.model.model_path,
            "text_encoder_path": runtime.cfg.model.text_encoder_path,
            "loaded_component_counts": runtime.checkpoint_audit["component_key_counts"],
            "missing_keys": runtime.checkpoint_audit["missing_keys"],
            "unexpected_keys": runtime.checkpoint_audit["unexpected_keys"],
            "task": task,
            "strict_no_gt": strict_no_gt,
        }
    )
    if stg_blocks is None:
        resolved_stg_blocks = runtime.cfg.validation.stg_blocks or [29]
    else:
        resolved_stg_blocks = runtime.stage1._parse_stg_blocks(stg_blocks)

    started = time.perf_counter()
    code_commit = _git_commit()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        console.print(
            f"[{index + 1}/{len(samples)}] {sample['task']} {sample['sample_key']} "
            f"refs={len(sample['reference_paths'])}"
        )
        try:
            result = run_online_sample(
                runtime=runtime,
                sample=sample,
                output_root=output,
                dry_run=dry_run,
                overwrite=overwrite,
                seed=seed + index,
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
                console.print(
                    f"dry-run {sample['task']} refs={result['reference_count']} "
                    f"planner_tokens={result['planner_token_count']} "
                    f"condition={result['planner_diagnostics'].get('final_condition_shape')} "
                    f"target_open_count={result['target_open_count']} "
                    f"peak_vram_gb={result['peak_vram_gb']}"
                )
            if export_ground_truth and not dry_run and result["status"] in {
                "success",
                "skipped_existing",
            }:
                result_sample_dir = sample_output_dir(output, sample)
                gt_name = "target_gt.png" if sample["task"] == IMAGE_TASK else "target_gt.mp4"
                existing_gt = result_sample_dir / gt_name
                if existing_gt.exists() and not overwrite:
                    result["ground_truth_export"] = str(existing_gt)
                else:
                    result["ground_truth_export"] = _export_ground_truth(
                        runtime,
                        sample,
                        result_sample_dir,
                    )
            results.append(result)
        except (RawReferenceLoadError, OnlineSampleEncodeError) as exc:
            failure = {
                "sample_key": sample.get("sample_key"),
                "task": sample.get("task"),
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            failures.append(failure)
            atomic_write_json(sample_output_dir(output, sample) / "failure.json", failure)
            console.print(f"[yellow]Skipped sample-local data failure:[/yellow] {exc}")

    elapsed_values = [
        float(result["elapsed_seconds"])
        for result in results
        if isinstance(result.get("elapsed_seconds"), int | float)
    ]
    peak_values = [
        float(result["peak_vram_gb"])
        for result in results
        if isinstance(result.get("peak_vram_gb"), int | float)
    ]
    summary = {
        "code_commit": code_commit,
        "config": str(Path(config).expanduser().resolve()),
        "selected_samples": str(Path(selected_samples).expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_audit": runtime.checkpoint_audit,
        "strict_no_gt": True,
        "export_ground_truth": export_ground_truth,
        "dry_run": dry_run,
        "requested_count": len(samples),
        "result_count": len(results),
        "success_count": sum(
            result.get("status") in {"success", "dry_run_success"} for result in results
        ),
        "skipped_existing_count": sum(
            result.get("status") == "skipped_existing" for result in results
        ),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
        "average_sample_seconds": (
            sum(elapsed_values) / len(elapsed_values) if elapsed_values else None
        ),
        "peak_vram_gb": max(peak_values) if peak_values else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = output / "run_summary.json"
    if summary_path.exists() and not overwrite:
        summary_path = output / f"run_summary_{int(time.time())}.json"
    atomic_write_json(summary_path, summary)
    console.print(
        f"[green]Completed {len(results)}/{len(samples)} samples; "
        f"sample-local failures={len(failures)}[/green]"
    )


if __name__ == "__main__":
    app()
