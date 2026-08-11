"""Strict-no-GT semantic-flow inference orchestration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_inference.checkpoint_runtime import OnlineInferenceRuntime
from ltx_trainer.online_inference.output_artifacts import (
    atomic_save_png,
    atomic_save_video,
    atomic_write_json,
    atomic_write_text,
    output_is_complete,
    sample_output_dir,
    save_contact_sheet,
    save_reference_images,
    save_reference_montage,
    tensor_frame_to_pil,
)
from ltx_trainer.online_inference.raw_condition_encoder import (
    encode_external_reference_only_conditions,
    encode_selected_sample_conditions,
)
from ltx_trainer.online_inference.semantic_guidance import SemanticGuidanceConfig
from ltx_trainer.online_inference.vae_decode import decode_video_latents


def _bundle_reference_path(bundle_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (bundle_root / path).absolute() if not path.is_absolute() else path.absolute()


def read_selected_samples(
    path: str | Path,
    *,
    tasks: set[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read a selection bundle without opening any target media."""
    selection_path = Path(path).expanduser().resolve()
    bundle_root = selection_path.parent
    selected: list[dict[str, Any]] = []
    with selection_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            task = str(sample.get("task"))
            if task not in {IMAGE_TASK, VIDEO_TASK}:
                raise ValueError(f"Selected sample line {line_number} has invalid task {task!r}")
            if tasks is not None and task not in tasks:
                continue
            reference_values = sample.get("reference_exports", sample.get("reference_paths", []))
            sample["original_reference_paths"] = list(sample.get("reference_paths", reference_values))
            sample["reference_paths"] = [
                str(_bundle_reference_path(bundle_root, str(value))) for value in reference_values
            ]
            selected.append(sample)
            if limit is not None and len(selected) >= limit:
                break
    return selected


def _peak_memory_gib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def run_online_sample(
    *,
    runtime: OnlineInferenceRuntime,
    sample: dict[str, Any],
    output_root: Path,
    dry_run: bool,
    overwrite: bool,
    seed: int,
    num_inference_steps: int,
    decode_tile: bool,
    guidance: SemanticGuidanceConfig | None = None,
    negative_prompt: str | None = None,
    external_eval: bool = False,
    code_commit: str | None = None,
) -> dict[str, Any]:
    """Encode references/text and jointly generate semantics/video; target pixels are never read."""
    guidance = guidance or SemanticGuidanceConfig(
        guidance_scale=1.0,
        ref_guidance_scale=0.0,
        guidance_rescale=0.0,
    )
    started = time.perf_counter()
    sample_dir = (
        output_root / str(sample["dataset_name"]) / str(sample["output_id"])
        if external_eval
        else sample_output_dir(output_root, sample)
    )
    if output_is_complete(sample_dir) and not overwrite:
        return {"status": "skipped_existing", "sample_dir": str(sample_dir)}
    sample_dir.mkdir(parents=True, exist_ok=True)

    condition_encoder = (
        encode_external_reference_only_conditions
        if external_eval
        else encode_selected_sample_conditions
    )
    encoded = condition_encoder(
        runtime.online_encoder,
        sample,
        guidance=guidance,
        negative_prompt=negative_prompt,
    )
    strict_checks = encoded["strict_no_gt_checks"]
    if strict_checks.get("target_path_passed_to_condition_encoder") is not False:
        raise RuntimeError("Strict-no-GT guard failed before denoising")
    metadata: dict[str, Any] = {
        "architecture": runtime.checkpoint_audit["metadata"].get("architecture"),
        "sample_key": sample["sample_key"],
        "task": sample["task"],
        "caption": sample["caption"],
        "reference_count": encoded["reference_metadata"]["reference_count"],
        "reference_metadata": encoded["reference_metadata"],
        "strict_no_gt_checks": strict_checks,
        "semantic_initialization": "independent_noise",
        "video_initialization": "independent_noise",
        "reference_sigma": 0.0,
        "reference_velocity": 0.0,
        "shared_semantic_video_sigma": True,
        "seed": seed,
        "fps": float(sample["fps"]),
        "num_inference_steps": num_inference_steps,
        "checkpoint": str(runtime.checkpoint_path),
        "checkpoint_sha256": runtime.checkpoint_audit["checkpoint_sha256"],
        "code_commit": code_commit,
        "dry_run": dry_run,
        **guidance.metadata(negative_prompt=negative_prompt),
    }
    if external_eval:
        metadata.update(
            {
                "dataset_name": sample["dataset_name"],
                "source_json": sample["source_json"],
                "source_record_id": sample["source_record_id"],
                "output_id": sample["output_id"],
                "dataset_metadata": sample["dataset_metadata"],
                "external_eval": True,
                "has_target": False,
            }
        )
    atomic_write_text(sample_dir / "prompt.txt", f"{sample['caption']}\n")
    if guidance.need_negative:
        atomic_write_text(sample_dir / "negative_prompt.txt", f"{negative_prompt}\n")
    save_reference_montage(list(sample["reference_paths"]), sample_dir / "references.png")
    reference_outputs = save_reference_images(list(sample["reference_paths"]), sample_dir)
    states = runtime.prepare_guidance_states(
        encoded,
        width=int(sample["width"]),
        height=int(sample["height"]),
        num_frames=int(sample["num_frames"]),
        fps=float(sample["fps"]),
        seed=seed,
        guidance=guidance,
        negative_prompt=negative_prompt,
    )
    metadata.update(runtime.last_generation_geometry)
    if dry_run:
        metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata["peak_vram_gib"] = _peak_memory_gib(runtime.device)
        atomic_write_json(sample_dir / "dry_run.json", metadata)
        return {"status": "dry_run_success", "sample_dir": str(sample_dir), **metadata}

    if runtime.vae_decoder is None:
        raise RuntimeError("Generation requires a loaded VAE decoder")
    semantic, generated_latents = runtime.strategy.denoise_joint_guided(
        transformer=runtime.transformer,
        states=states,
        guidance=guidance,
        num_inference_steps=num_inference_steps,
    )
    if not torch.isfinite(semantic).all().item():
        raise RuntimeError("Semantic ODE produced non-finite latents")
    if not torch.isfinite(generated_latents).all().item():
        raise RuntimeError("Video ODE produced non-finite latents")
    decoded, decode_diagnostics = decode_video_latents(
        vae_decoder=runtime.vae_decoder,
        latents=generated_latents,
        decode_tile=decode_tile,
    )
    if not torch.isfinite(decoded).all().item():
        raise RuntimeError("VAE decode produced non-finite pixels")
    expected_shape = (
        int(sample["num_frames"]),
        3,
        int(sample["height"]),
        int(sample["width"]),
    )
    if tuple(decoded.shape) != expected_shape:
        raise RuntimeError(f"Decoded output shape {tuple(decoded.shape)} != expected {expected_shape}")

    artifacts = ["prompt.txt", "references.png", *reference_outputs, "metadata.json"]
    if guidance.need_negative:
        artifacts.append("negative_prompt.txt")
    if sample["task"] == IMAGE_TASK:
        atomic_save_png(tensor_frame_to_pil(decoded[0]), sample_dir / "generated.png")
        artifacts.append("generated.png")
    else:
        from ltx_trainer.video_utils import save_video

        atomic_save_video(decoded, sample_dir / "generated.mp4", fps=float(sample["fps"]), save_video=save_video)
        atomic_save_png(tensor_frame_to_pil(decoded[0]), sample_dir / "generated_first.png")
        atomic_save_png(tensor_frame_to_pil(decoded[-1]), sample_dir / "generated_last.png")
        save_contact_sheet(decoded, sample_dir / "generated_contact_sheet.png")
        artifacts.extend(
            ["generated.mp4", "generated_first.png", "generated_last.png", "generated_contact_sheet.png"]
        )
    metadata.update(
        {
            **runtime.last_generation_geometry,
            "semantic_token_count": int(semantic.shape[1]),
            "semantic_shape": list(semantic.shape),
            "latent_shape": list(generated_latents.shape),
            "output_shape": list(decoded.shape),
            "semantic_latent_finite": True,
            "video_latent_finite": True,
            "decoded_image_finite": True,
            "vae_decode": decode_diagnostics.__dict__,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_vram_gib": _peak_memory_gib(runtime.device),
        }
    )
    atomic_write_json(sample_dir / "metadata.json", metadata)
    atomic_write_json(
        sample_dir / "success.json",
        {"status": "success", "artifacts": artifacts, "sample_key": sample["sample_key"]},
    )
    return {"status": "success", "sample_dir": str(sample_dir), **metadata}


__all__ = ["read_selected_samples", "run_online_sample"]
