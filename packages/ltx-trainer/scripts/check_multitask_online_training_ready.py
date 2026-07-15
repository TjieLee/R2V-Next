"""Preflight an online multi-task Stage 1/2/3 training configuration."""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

import torch
import typer
import yaml
from safetensors import safe_open

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK, VLM_TARGET_INDICES
from ltx_trainer.online_data.multitask_dataset import OnlineMultiTaskDataset, SampleLoadError
from ltx_trainer.online_data.path_safety import assert_write_path_allowed

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

_STAGE3_REQUIRED_PREFIXES = (
    "diffusion_model.",
    "training_strategy.planner_tokens.",
    "training_strategy.visual_token_projection.",
    "training_strategy.visual_full_encoder.",
    "embeddings_processor.video_connector.",
    "text_encoder.model.model.language_model.",
)
_ONLINE_STATE_FIELDS = {
    "task_schedule_cursor",
    "image_permutation_epoch",
    "image_cursor",
    "video_permutation_epoch",
    "video_cursor",
    "microstep_in_optimizer_step",
    "sampler_seed",
}


def _phase(config: LtxTrainerConfig) -> str:
    phase = getattr(config.training_strategy, "training_phase", None)
    return str(phase or "stage1")


def _sample_indices(indices: Any, count: int, seed: int) -> list[int]:
    size = len(indices)
    if size == 0:
        return []
    randomizer = random.Random(seed)
    positions = randomizer.sample(range(size), min(count, size))
    return [int(indices[position]) for position in positions]


def _checkpoint_metadata(path: Path) -> dict[str, str]:
    if not path.is_file() or path.suffix != ".safetensors":
        return {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _strict_stage3_component_audit(path: Path) -> dict[str, Any]:
    """Validate the portable checkpoint invariants before constructing the 22B runtime."""
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        missing = [prefix for prefix in _STAGE3_REQUIRED_PREFIXES if not any(key.startswith(prefix) for key in keys)]
        if missing:
            raise ValueError(f"Stage 3 initialization checkpoint is missing component prefixes: {missing}")

        dit_keys = [key for key in keys if key.startswith("diffusion_model.")]
        non_lora_dit = [key for key in dit_keys if "lora_" not in key]
        if non_lora_dit:
            raise ValueError(f"Stage 3 checkpoint contains non-LoRA DiT weights: {non_lora_dit[:20]}")

        gemma_prefix = "text_encoder.model.model.language_model."
        gemma_keys = [key for key in keys if key.startswith(gemma_prefix)]
        non_lora_gemma = [key for key in gemma_keys if "lora_" not in key]
        if non_lora_gemma:
            raise ValueError(f"Stage 3 checkpoint contains non-LoRA Gemma LM weights: {non_lora_gemma[:20]}")

        frozen_visual = [
            key
            for key in keys
            if "vision_tower" in key or "multi_modal_projector" in key
        ]
        if frozen_visual:
            raise ValueError(
                "Stage 3 checkpoint unexpectedly stores frozen vision/projector weights: "
                f"{frozen_visual[:20]}"
            )

        non_finite: list[str] = []
        for key in keys:
            tensor = handle.get_tensor(key)
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                non_finite.append(key)
                if len(non_finite) == 20:
                    break
        if non_finite:
            raise ValueError(f"Stage 3 checkpoint contains non-finite tensors: {non_finite}")

    return {
        "checkpoint_tensor_count": len(keys),
        "required_component_prefixes": list(_STAGE3_REQUIRED_PREFIXES),
    }


def _validate_exact_stage3_resume(
    checkpoint_path: Path,
    *,
    metadata: dict[str, str],
    optimization_steps: int,
) -> int:
    step_match = re.search(r"step_(\d+)", checkpoint_path.name)
    if step_match is None:
        raise ValueError(f"Cannot parse checkpoint step from {checkpoint_path.name}")
    filename_step = int(step_match.group(1))
    metadata_step_raw = metadata.get("global_step")
    if metadata_step_raw is None:
        raise ValueError("Exact Stage 3 resume checkpoint metadata is missing global_step")
    try:
        metadata_step = int(metadata_step_raw)
    except ValueError as exc:
        raise ValueError(f"Invalid checkpoint metadata global_step={metadata_step_raw!r}") from exc

    state_path = checkpoint_path.parent / f"training_state_step_{step_match.group(1)}.pt"
    if not state_path.is_file():
        raise ValueError(f"Exact Stage 3 resume requires matching training state: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    state_step = int(state.get("global_step", -1))
    if filename_step != metadata_step or filename_step != state_step:
        raise ValueError(
            "Stage 3 resume step mismatch: "
            f"filename={filename_step}, metadata={metadata_step}, training_state={state_step}"
        )
    if state_step < 0 or state_step >= optimization_steps:
        raise ValueError(
            f"Stage 3 resume global_step={state_step} must be in [0, {optimization_steps})"
        )
    if not state.get("optimizer_state_dict"):
        raise ValueError("Exact Stage 3 resume requires optimizer state")
    scheduler_state = state.get("lr_scheduler_state_dict")
    if not scheduler_state:
        raise ValueError("Exact Stage 3 resume requires scheduler state")
    if int(scheduler_state.get("last_epoch", -1)) != state_step:
        raise ValueError(
            "Stage 3 scheduler/global_step mismatch: "
            f"last_epoch={scheduler_state.get('last_epoch')}, global_step={state_step}"
        )
    saved_lrs = scheduler_state.get("_last_lr")
    if not isinstance(saved_lrs, list) or not saved_lrs:
        raise ValueError("Exact Stage 3 resume scheduler state is missing _last_lr")

    data_state = state.get("data_state")
    if not isinstance(data_state, dict):
        raise ValueError("Exact online Stage 3 resume requires online sampler state")
    missing_data_fields = sorted(_ONLINE_STATE_FIELDS - data_state.keys())
    if missing_data_fields:
        raise ValueError(f"Online sampler state is missing fields: {missing_data_fields}")
    if int(data_state["task_schedule_cursor"]) != state_step:
        raise ValueError(
            "Online sampler/global_step mismatch: "
            f"task_schedule_cursor={data_state['task_schedule_cursor']}, global_step={state_step}"
        )
    if int(data_state["microstep_in_optimizer_step"]) != 0:
        raise ValueError("Exact Stage 3 checkpoint must be saved on an optimizer-step boundary")
    return state_step


def _inspect_stage3_initialization(
    checkpoint_path: Path,
    *,
    no_resume: bool,
    optimization_steps: int,
) -> dict[str, Any]:
    metadata = _checkpoint_metadata(checkpoint_path)
    checkpoint_phase = metadata.get("training_phase")
    component_report = _strict_stage3_component_audit(checkpoint_path)
    if no_resume:
        modes = {
            "stage2": "stage2_to_stage3_init",
            "stage3": "stage3_joint_warmstart",
        }
        if checkpoint_phase not in modes:
            raise ValueError(
                "Stage 3 no-resume initialization requires checkpoint metadata.training_phase "
                f"to be stage2 or stage3, got {checkpoint_phase!r}"
            )
        return {
            "initialization_mode": modes[checkpoint_phase],
            "initial_checkpoint_training_phase": checkpoint_phase,
            "strict_component_check_passed": True,
            "starts_from_global_step": 0,
            **component_report,
        }

    if checkpoint_phase != "stage3":
        raise ValueError(
            "Exact Stage 3 resume requires checkpoint metadata.training_phase='stage3', "
            f"got {checkpoint_phase!r}"
        )
    resume_step = _validate_exact_stage3_resume(
        checkpoint_path,
        metadata=metadata,
        optimization_steps=optimization_steps,
    )
    return {
        "initialization_mode": "exact_resume",
        "initial_checkpoint_training_phase": checkpoint_phase,
        "strict_component_check_passed": True,
        "starts_from_global_step": resume_step,
        **component_report,
    }


@app.command()
def main(
    config_path: str = typer.Argument(...),
    samples_per_task: int = typer.Option(1, "--samples-per-task", min=1),
    world_size: int = typer.Option(8, "--world-size", min=1),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    errors: list[str] = []
    warnings: list[str] = []
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = LtxTrainerConfig(**yaml.safe_load(handle))
    phase = _phase(config)
    initialization_report: dict[str, Any] = {
        "initialization_mode": "base_model",
        "initial_checkpoint_training_phase": None,
        "strict_component_check_passed": False,
        "starts_from_global_step": 0,
    }

    if config.data.encoding_mode != "online":
        errors.append("data.encoding_mode must be online")
    if config.data.train_data_config is None or not Path(config.data.train_data_config).expanduser().is_file():
        errors.append(f"data.train_data_config does not exist: {config.data.train_data_config}")
    online = config.data.online_encoding
    if online is None:
        errors.append("data.online_encoding is missing")
    if config.optimization.batch_size != 1:
        errors.append("optimization.batch_size must be 1")
    if config.optimization.gradient_accumulation_steps != 4:
        errors.append("optimization.gradient_accumulation_steps must be 4")
    global_batch = (
        config.optimization.batch_size
        * config.optimization.gradient_accumulation_steps
        * world_size
    )
    if world_size == 8 and global_batch != 32:
        errors.append(f"Eight-rank effective global batch must be 32, got {global_batch}")

    try:
        assert_write_path_allowed(config.output_dir)
        if online is not None and online.runtime_reject_log_dir is not None:
            assert_write_path_allowed(online.runtime_reject_log_dir)
    except ValueError as exc:
        errors.append(str(exc))
    source_root = Path("/mnt/workspace/liutao")
    if source_root.exists() and os.access(source_root, os.W_OK):
        warnings.append(
            "/mnt/workspace/liutao is writable at the OS level; JD-LTX path guards still forbid all online writes"
        )

    checkpoint = config.model.load_checkpoint
    if phase in {"stage2", "stage3"}:
        if not checkpoint:
            errors.append(f"{phase} requires model.load_checkpoint")
        elif not Path(checkpoint).expanduser().exists():
            errors.append(f"{phase} checkpoint does not exist: {checkpoint}")
        else:
            checkpoint_path = Path(checkpoint).expanduser().resolve()
            metadata = _checkpoint_metadata(checkpoint_path)
            if phase == "stage2" and metadata.get("conditioning") != "multi_reference_video":
                errors.append(
                    "Stage 2 must initialize from a Stage 1 multi_reference_video checkpoint; "
                    f"metadata={metadata}"
                )
            if phase == "stage3":
                try:
                    initialization_report = _inspect_stage3_initialization(
                        checkpoint_path,
                        no_resume=config.checkpoints.no_resume,
                        optimization_steps=config.optimization.steps,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    errors.append(f"Stage 3 checkpoint preflight failed: {exc}")

    dataset: OnlineMultiTaskDataset | None = None
    decoded: dict[str, list[dict[str, Any]]] = {IMAGE_TASK: [], VIDEO_TASK: []}
    if config.data.manifest_path is None or online is None:
        errors.append("Online manifest/config is missing")
    else:
        try:
            dataset = OnlineMultiTaskDataset(
                config.data.manifest_path,
                width=online.width,
                height=online.height,
                max_ref_images=online.max_ref_images,
                vlm_reference_preprocess=online.vlm_reference_preprocess,
                video_decoder=online.video_decoder,
                decode_timeout_seconds=online.decode_timeout_seconds,
            )
            for task in (IMAGE_TASK, VIDEO_TASK):
                task_count = len(dataset.task_indices[task])
                if task_count < global_batch:
                    errors.append(
                        f"Manifest task {task} has {task_count} rows, fewer than global block {global_batch}"
                    )
                for index in _sample_indices(dataset.task_indices[task], samples_per_task, seed):
                    sample = dataset[index]
                    if isinstance(sample, SampleLoadError):
                        errors.append(f"CPU decode failed for {task}/{sample.sample_key}: {sample.message}")
                        continue
                    decoded[task].append(
                        {
                            "sample_key": sample["sample_key"],
                            "target_shape": list(sample["target_pixels"].shape),
                            "reference_vae_shapes": [
                                list(reference.shape) for reference in sample["reference_pixels_vae"]
                            ],
                            "reference_vlm_shapes": [
                                list(reference.shape) for reference in sample["reference_images_vlm"]
                            ],
                        }
                    )
                    expected_frames = 1 if task == IMAGE_TASK else 121
                    expected_fps = 1.0 if task == IMAGE_TASK else 24.0
                    if sample["target_pixels"].shape[0] != expected_frames:
                        errors.append(f"{task} decoded frames != {expected_frames}")
                    if float(sample["target_fps"]) != expected_fps:
                        errors.append(f"{task} target_fps != {expected_fps}")
                    if any(
                        tuple(reference.shape) != (online.height, online.width, 3)
                        for reference in sample["reference_pixels_vae"]
                    ):
                        errors.append(f"{task} reference VAE input is not {online.width}x{online.height} RGB")
                    if task == VIDEO_TASK and tuple(sample["vlm_target_frame_indices"].tolist()) != VLM_TARGET_INDICES:
                        errors.append("R2V 121-to-8 indices do not match the fixed contract")
        except Exception as exc:
            errors.append(f"Manifest preflight failed: {exc}")

    readiness = {name: False for name in ("stage1", "stage2", "stage3")}
    if not errors and phase in readiness:
        readiness[phase] = True
    report = {
        "config": str(path),
        "phase": phase,
        "world_size": world_size,
        "effective_global_batch": global_batch,
        "task_counts": (
            {task: len(dataset.task_indices[task]) for task in (IMAGE_TASK, VIDEO_TASK)}
            if dataset is not None
            else {}
        ),
        "decoded_samples": decoded,
        **initialization_report,
        "warnings": warnings,
        "errors": errors,
        **{f"READY_FOR_{name.upper()}": ready for name, ready in readiness.items()},
    }
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
