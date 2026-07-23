"""Real-model semantic-flow online encoding and one-step training smoke helpers."""

from __future__ import annotations

import json
import os
import time
from itertools import islice
from pathlib import Path
from typing import Any, Literal

import torch
import typer
import yaml

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_data.multitask_dataset import SampleLoadError, collate_online_raw_batch
from ltx_trainer.online_data.path_safety import assert_write_path_allowed
from ltx_trainer.online_inference.output_artifacts import atomic_write_json
from ltx_trainer.online_inference.startup_memory import host_memory_snapshot
from ltx_trainer.trainer import LtxvTrainer


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(f"{path}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def make_one_step_smoke_config(
    config_path: str | Path,
    *,
    task: Literal["i2i", "r2v"],
    output_dir: str | Path,
    init_checkpoint: str | Path | None = None,
) -> tuple[LtxTrainerConfig, Path]:
    output = assert_write_path_allowed(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(config_path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if init_checkpoint is not None:
        payload["model"]["load_checkpoint"] = str(Path(init_checkpoint).expanduser().resolve())

    source_data_path = Path(payload["data"]["train_data_config"]).expanduser().resolve()
    source_data = yaml.safe_load(source_data_path.read_text(encoding="utf-8"))
    source_data.setdefault("online_sampling", {})
    source_data["online_sampling"]["image_ratio"] = 1.0 if task == IMAGE_TASK else 0.0
    source_data["online_sampling"]["video_ratio"] = 1.0 if task == VIDEO_TASK else 0.0
    smoke_data_path = output / "semantic_flow_smoke_data.yaml"
    _atomic_yaml(smoke_data_path, source_data)

    payload["data"]["train_data_config"] = str(smoke_data_path)
    payload["data"]["online_encoding"]["image_ratio"] = 1.0 if task == IMAGE_TASK else 0.0
    payload["data"]["online_encoding"]["video_ratio"] = 1.0 if task == VIDEO_TASK else 0.0
    payload["optimization"]["steps"] = 1
    payload["optimization"]["batch_size"] = 1
    payload["optimization"]["gradient_accumulation_steps"] = 1
    payload["validation"]["interval"] = None
    payload["validation"]["skip_initial_validation"] = True
    payload["checkpoints"]["interval"] = 1
    payload["checkpoints"]["keep_last_n"] = 1
    payload["checkpoints"]["no_resume"] = True
    payload["wandb"]["enabled"] = False
    strategy = payload["training_strategy"]
    strategy["condition_full_p"] = 1.0
    strategy["condition_drop_text_p"] = 0.0
    strategy["condition_drop_reference_all_p"] = 0.0
    strategy["condition_drop_all_p"] = 0.0
    payload["output_dir"] = str(output)
    smoke_config_path = output / "semantic_flow_smoke_config.yaml"
    _atomic_yaml(smoke_config_path, payload)
    return LtxTrainerConfig(**payload), smoke_config_path


def run_one_step_training_smoke(
    config_path: str | Path,
    *,
    task: Literal["i2i", "r2v"],
    output_dir: str | Path,
    init_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    smoke_start_memory = host_memory_snapshot()
    config, generated_config = make_one_step_smoke_config(
        config_path,
        task=task,
        output_dir=output_dir,
        init_checkpoint=init_checkpoint,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer = LtxvTrainer(config)
    try:
        checkpoint, stats = trainer.train(disable_progress_bars=True, finalize_accelerator=False)
        final_memory = host_memory_snapshot()
        memory_snapshots = {
            "smoke_start": smoke_start_memory,
            **trainer._startup_host_memory,
        }

        def gather_int(value: int) -> list[int]:
            tensor = torch.tensor([int(value)], device=trainer._accelerator.device, dtype=torch.int64)
            return [int(item) for item in trainer._accelerator.gather(tensor).detach().cpu().tolist()]

        memory = {
            f"{name}_{field}": gather_int(snapshot[field])
            for name, snapshot in memory_snapshots.items()
            for field in ("available_host_ram_bytes", "process_rss_bytes")
        }
        memory["peak_host_ram_bytes"] = gather_int(final_memory["process_peak_rss_bytes"])
        memory["per_rank_peak_cuda_memory_bytes"] = gather_int(
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        report = {
            "architecture": "semantic_flow_v2",
            "task": task,
            "world_size": trainer._accelerator.num_processes,
            "generated_config": str(generated_config),
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "global_step": trainer._global_step,
            "training_stats": stats.model_dump(),
            "timings_ms": trainer._last_online_metrics,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0,
            "memory": memory,
        }
        if trainer._accelerator.is_main_process:
            atomic_write_json(Path(output_dir).expanduser().resolve() / "result.json", report)
            typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return report
    finally:
        trainer._accelerator.end_training()


def run_real_encode_check(
    config_path: str | Path,
    *,
    num_image_samples: int,
    num_video_samples: int,
) -> dict[str, Any]:
    payload = yaml.safe_load(Path(config_path).expanduser().resolve().read_text(encoding="utf-8"))
    trainer = LtxvTrainer(LtxTrainerConfig(**payload))
    trainer._init_online_dataloader()
    requested = {IMAGE_TASK: num_image_samples, VIDEO_TASK: num_video_samples}
    reports: list[dict[str, Any]] = []
    for task, count in requested.items():
        for manifest_index in islice(trainer._dataset.task_indices[task], count):
            sample = trainer._dataset[int(manifest_index)]
            if isinstance(sample, SampleLoadError):
                raise RuntimeError(f"Real encode sample failed: {sample.to_dict()}")
            raw_batch = collate_online_raw_batch([sample])
            encoded = trainer._online_batch_encoder.encode_for_strategy(
                raw_batch,
                strategy=trainer._training_strategy,
                global_seed=trainer._config.seed,
            )
            evidence = encoded["semantic_teacher_inputs"]["evidence_tokens"]
            expected_anchors = 1 if task == IMAGE_TASK else 12
            if tuple(evidence.shape[1:3]) != (expected_anchors, 256):
                raise RuntimeError(f"Unexpected {task} native evidence shape: {tuple(evidence.shape)}")
            reports.append(
                {
                    "task": task,
                    "sample_key": sample["sample_key"],
                    "latent_shape": list(encoded["latents"]["latents"].shape),
                    "native_evidence_shape": list(evidence.shape),
                    "condition_mode": encoded["condition_mode"],
                }
            )
    report = {"architecture": "semantic_flow_v2", "samples": reports}
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report
