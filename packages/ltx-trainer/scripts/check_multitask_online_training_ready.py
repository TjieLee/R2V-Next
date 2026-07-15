"""Preflight an online multi-task Stage 1/2/3 training configuration."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import typer
import yaml
from safetensors import safe_open

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK, VLM_TARGET_INDICES
from ltx_trainer.online_data.multitask_dataset import OnlineMultiTaskDataset, SampleLoadError
from ltx_trainer.online_data.path_safety import assert_write_path_allowed

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


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
            if phase == "stage3" and metadata.get("training_phase") != "stage2":
                errors.append(
                    "Stage 3 must initialize from a Stage 2 planner checkpoint; "
                    f"metadata={metadata}"
                )

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
        "warnings": warnings,
        "errors": errors,
        **{f"READY_FOR_{name.upper()}": ready for name, ready in readiness.items()},
    }
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
