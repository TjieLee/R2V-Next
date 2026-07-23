"""Preflight a semantic-flow online multi-task training configuration."""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path

import typer
import yaml

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_NUM_FRAMES, VIDEO_TASK
from ltx_trainer.online_data.manifest import load_multitask_data_config
from ltx_trainer.online_data.manifest_index import validate_manifest_index
from ltx_trainer.online_data.multitask_dataset import OnlineMultiTaskDataset, SampleLoadError

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


@app.command()
def main(
    config_path: str = typer.Argument(...),
    samples_per_task: int = typer.Option(1, "--samples-per-task", min=1),
    world_size: int = typer.Option(8, "--world-size", min=1),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    del seed
    errors: list[str] = []
    path = Path(config_path).expanduser().resolve()
    config = LtxTrainerConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))
    if config.training_strategy.name != "semantic_flow":
        errors.append("training_strategy.name must be semantic_flow")
    if config.model.training_mode != "full" or config.lora is not None:
        errors.append("semantic_flow requires full DiT training with no LoRA config")
    if config.data.encoding_mode != "online" or config.data.online_encoding is None:
        errors.append("semantic_flow readiness requires data.encoding_mode=online and online_encoding")
    if config.data.manifest_path is None:
        errors.append("data.manifest_path is missing")
    if config.data.train_data_config is None:
        errors.append("data.train_data_config is missing")

    dataset = None
    if not errors:
        try:
            assert config.data.manifest_path is not None
            validate_manifest_index(config.data.manifest_path)
            online = config.data.online_encoding
            assert online is not None
            dataset = OnlineMultiTaskDataset(
                config.data.manifest_path,
                width=online.width,
                height=online.height,
                max_ref_images=online.max_ref_images,
                video_decoder=online.video_decoder,
                decode_timeout_seconds=online.decode_timeout_seconds,
                cpu_transform_chunk_frames=online.cpu_transform_chunk_frames,
            )
            for task in (IMAGE_TASK, VIDEO_TASK):
                for manifest_index in islice(dataset.task_indices[task], samples_per_task):
                    sample = dataset[int(manifest_index)]
                    if isinstance(sample, SampleLoadError):
                        errors.append(str(sample.to_dict()))
                        continue
                    anchors = sample["semantic_anchor_target_indices"].tolist()
                    if task == IMAGE_TASK and anchors != [0]:
                        errors.append("I2I semantic anchors must be [0]")
                    if task == VIDEO_TASK and (
                        len(anchors) != 12 or anchors[0] != 0 or anchors[-1] != VIDEO_NUM_FRAMES - 1
                    ):
                        errors.append(f"R2V semantic anchors are invalid: {anchors}")
                    if sample["semantic_teacher_pixels"].shape[0] != len(anchors):
                        errors.append(f"{task} teacher frames were not selected from decoded target pixels")
        except Exception as exc:
            errors.append(f"manifest preflight failed: {type(exc).__name__}: {exc}")

    source_report: dict[str, object] = {}
    if config.data.train_data_config is not None:
        source_config = load_multitask_data_config(config.data.train_data_config)
        source_names = [str(row.get("name")) for row in source_config["datasets"] if row.get("task") == VIDEO_TASK]
        ratios = source_config.get("online_sampling", {}).get("video_source_ratios", {})
        source_report = {"configured_video_sources": source_names, "video_source_ratios": ratios}
        if set(ratios) != set(source_names):
            errors.append(
                f"video_source_ratios keys {sorted(ratios)} do not match R2V datasets {sorted(source_names)}"
            )

    report = {
        "ready": not errors,
        "architecture": "semantic_flow_v2",
        "world_size": world_size,
        "errors": errors,
        "task_counts": (
            {task: len(dataset.task_indices[task]) for task in (IMAGE_TASK, VIDEO_TASK)}
            if dataset is not None
            else {}
        ),
        "dataset_counts": (
            {name: len(indices) for name, indices in dataset.dataset_indices.items()}
            if dataset is not None
            else {}
        ),
        **source_report,
    }
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
