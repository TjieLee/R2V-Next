"""One-sample raw online Stage 3 inference orchestration."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ltx_core.types import VideoLatentShape, VideoPixelShape
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
    encode_selected_sample_conditions,
)


def read_selected_samples(
    path: str | Path,
    *,
    tasks: set[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = __import__("json").loads(line)
            task = str(sample.get("task"))
            if task not in {IMAGE_TASK, VIDEO_TASK}:
                raise ValueError(f"Selected sample line {line_number} has invalid task {task!r}")
            if tasks is not None and task not in tasks:
                continue
            selected.append(sample)
            if limit is not None and len(selected) >= limit:
                break
    return selected


def build_noise_shape_metadata(
    sample: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    pixel_shape = VideoPixelShape(
        batch=1,
        frames=int(sample["num_frames"]),
        height=int(sample["height"]),
        width=int(sample["width"]),
        fps=float(sample["fps"]),
    )
    latent_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
    return {
        "latents": torch.zeros(latent_shape.to_torch_shape(), device=device, dtype=dtype),
        "num_frames": torch.tensor([latent_shape.frames], device=device, dtype=torch.long),
        "height": torch.tensor([latent_shape.height], device=device, dtype=torch.long),
        "width": torch.tensor([latent_shape.width], device=device, dtype=torch.long),
        "fps": torch.tensor([pixel_shape.fps], device=device, dtype=torch.float32),
    }


def _clean_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in diagnostics.items():
        if isinstance(value, Tensor):
            if key in {"predicted_visual_tokens", "predicted_visual_token_mask", "token_positions"}:
                continue
            cleaned[key] = list(value.shape)
        else:
            cleaned[key] = value
    return cleaned


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
    guidance_scale: float,
    ref_guidance_scale: float,
    vision_guidance_scale: float,
    ref_guidance_mode: str,
    guidance_rescale: float,
    stg_scale: float,
    stg_blocks: list[int] | None,
    decode_tile: bool,
    code_commit: str | None,
) -> dict[str, Any]:  # noqa: PLR0913, PLR0915
    started = time.perf_counter()
    sample_dir = sample_output_dir(output_root, sample)
    if sample_dir.exists():
        if output_is_complete(sample_dir) and not overwrite:
            return {"status": "skipped_existing", "sample_dir": str(sample_dir)}
        if not overwrite:
            raise FileExistsError(
                f"Incomplete output already exists; pass --overwrite to retry: {sample_dir}"
            )
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)

    raw = encode_selected_sample_conditions(runtime.online_encoder, sample)
    latents_metadata = build_noise_shape_metadata(
        sample,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    batch = {
        "latents": latents_metadata,
        "multi_ref_latents": raw["multi_ref_latents"],
        "multi_reference_latents": raw["multi_reference_latents"],
    }
    with torch.inference_mode(), runtime.stage2._autocast_context(runtime.device, runtime.dtype):
        guidance = runtime.stage2._prepare_guidance_condition_bundle(
            strategy=runtime.strategy,
            conditions=raw["conditions"],
            text_conditions=raw["text_conditions"],
            planner_vlm_inputs=raw["planner_vlm_inputs"],
            latents_metadata=latents_metadata,
            visual_position_metadata=raw["visual_position_metadata"],
            negative_text_conditions=runtime.negative_conditions,
            guidance_scale=guidance_scale,
            ref_guidance_scale=ref_guidance_scale,
            vision_guidance_scale=vision_guidance_scale,
            ref_guidance_mode=ref_guidance_mode,
        )

    diagnostics = _clean_diagnostics(guidance.inference_diagnostics)
    planner_output_mask = raw["planner_vlm_inputs"]["planner_output_mask"]
    output_name = "generated.png" if sample["task"] == IMAGE_TASK else "generated.mp4"
    metadata: dict[str, Any] = {
        "code_commit": code_commit,
        "sample_key": sample["sample_key"],
        "manifest_path": sample.get("manifest_path"),
        "manifest_index": sample.get("manifest_index"),
        "sample_plan_sha256": sample.get("sample_plan_sha256"),
        "task": sample["task"],
        "caption": sample["caption"],
        "reference_paths": list(sample["reference_paths"]),
        "reference_count": len(sample["reference_paths"]),
        "reference_order": list(range(len(sample["reference_paths"]))),
        "strict_no_gt": True,
        "opened_target_during_generation": False,
        "target_open_count": 0,
        "target_opened_during_generation": False,
        "uses_target_latents": False,
        "uses_gt_siglip_tokens": False,
        "uses_precomputed_conditions": False,
        "geometry": {
            "width": int(sample["width"]),
            "height": int(sample["height"]),
            "num_frames": int(sample["num_frames"]),
            "fps": float(sample["fps"]),
        },
        "width": int(sample["width"]),
        "height": int(sample["height"]),
        "frames": int(sample["num_frames"]),
        "fps": float(sample["fps"]),
        "latent_shape": list(latents_metadata["latents"].shape),
        "checkpoint": str(runtime.checkpoint_path),
        "checkpoint_path": str(runtime.checkpoint_path),
        "checkpoint_sha256": runtime.checkpoint_audit["checkpoint_sha256"],
        "checkpoint_step": runtime.checkpoint_audit["checkpoint_step"],
        "config_path": str(runtime.config_path),
        "checkpoint_audit": runtime.checkpoint_audit,
        "checkpoint_flags": runtime.checkpoint_flags,
        "negative_prompt": runtime.negative_prompt if guidance_scale > 1.0 else None,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "ref_guidance_scale": ref_guidance_scale,
        "vision_guidance_scale": vision_guidance_scale,
        "ref_guidance_mode": ref_guidance_mode,
        "guidance_rescale": guidance_rescale,
        "stg_scale": stg_scale,
        "stg_blocks": stg_blocks,
        "planner_forward_count": guidance.planner_forward_count,
        "system_prompt_id": int(raw["task_system_prompt_id"].item()),
        "planner_token_count": int(
            raw["planner_vlm_inputs"]["planner_token_count"].flatten()[0].item()
        ),
        "planner_output_mask_true_count": int(planner_output_mask.sum().item()),
        "planner_diagnostics": diagnostics,
        "dtype_diagnostics": raw["dtype_diagnostics"],
        "dry_run": dry_run,
        "output_path": str(sample_dir / output_name),
    }
    atomic_write_text(sample_dir / "prompt.txt", f"{sample['caption']}\n")
    save_reference_montage(list(sample["reference_paths"]), sample_dir / "references.png")
    reference_outputs = save_reference_images(list(sample["reference_paths"]), sample_dir)

    if dry_run:
        metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata["peak_vram_gb"] = _peak_memory_gib(runtime.device)
        metadata["peak_vram_gib"] = metadata["peak_vram_gb"]
        atomic_write_json(sample_dir / "dry_run.json", metadata)
        atomic_write_json(
            sample_dir / "dry_run.success.json",
            {"status": "dry_run_success", "metadata": "dry_run.json"},
        )
        return {"status": "dry_run_success", "sample_dir": str(sample_dir), **metadata}

    generated_latents = runtime.stage1._denoise_stage1(
        transformer=runtime.transformer,
        strategy=runtime.strategy,
        batch=batch,
        positive_conditions=guidance.positive_conditions,
        negative_conditions=guidance.negative_conditions,
        no_ref_conditions=guidance.no_ref_conditions,
        no_visual_conditions=guidance.no_visual_conditions,
        guidance_scale=guidance_scale,
        cfg_drop_ref_latents_in_negative=False,
        ref_guidance_scale=ref_guidance_scale,
        vision_guidance_scale=vision_guidance_scale,
        siglip_guidance_scale=0.0,
        guidance_rescale=guidance_rescale,
        stg_scale=stg_scale,
        stg_blocks=stg_blocks,
        num_inference_steps=num_inference_steps,
        seed=seed,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    if runtime.vae_decoder is None:
        raise RuntimeError("Generation requires a loaded VAE decoder; dry-run runtime cannot denoise")
    decoded = runtime.stage1._decode_video_latents(
        vae_decoder=runtime.vae_decoder,
        latents=generated_latents,
        device=runtime.device,
        decode_tile=decode_tile,
    )
    expected_frames = int(sample["num_frames"])
    expected_shape = (expected_frames, 3, int(sample["height"]), int(sample["width"]))
    if tuple(decoded.shape) != expected_shape:
        raise RuntimeError(f"Decoded output shape {tuple(decoded.shape)} != expected {expected_shape}")

    artifacts = ["prompt.txt", "references.png", *reference_outputs, "metadata.json"]
    if sample["task"] == IMAGE_TASK:
        atomic_save_png(tensor_frame_to_pil(decoded[0]), sample_dir / "generated.png")
        artifacts.append("generated.png")
    else:
        atomic_save_video(
            decoded,
            sample_dir / "generated.mp4",
            fps=float(sample["fps"]),
            save_video=runtime.stage1.save_video,
        )
        atomic_save_png(tensor_frame_to_pil(decoded[0]), sample_dir / "generated_first.png")
        atomic_save_png(tensor_frame_to_pil(decoded[-1]), sample_dir / "generated_last.png")
        save_contact_sheet(decoded, sample_dir / "generated_contact_sheet.png", frame_count=8)
        artifacts.extend(
            [
                "generated.mp4",
                "generated_first.png",
                "generated_last.png",
                "generated_contact_sheet.png",
            ]
        )
    metadata["elapsed_seconds"] = time.perf_counter() - started
    metadata["peak_vram_gb"] = _peak_memory_gib(runtime.device)
    metadata["peak_vram_gib"] = metadata["peak_vram_gb"]
    metadata["output_shape"] = list(decoded.shape)
    atomic_write_json(sample_dir / "metadata.json", metadata)
    atomic_write_json(
        sample_dir / "success.json",
        {"status": "success", "artifacts": artifacts, "sample_key": sample["sample_key"]},
    )
    return {"status": "success", "sample_dir": str(sample_dir), **metadata}
