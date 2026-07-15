"""Helpers for real-model online encoding and one-step training smoke tests."""

from __future__ import annotations

import json
import os
import time
from itertools import islice
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
import typer
import yaml
from safetensors import safe_open

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK, VLM_TARGET_INDICES
from ltx_trainer.online_data.multitask_dataset import SampleLoadError, collate_online_raw_batch
from ltx_trainer.online_data.path_safety import assert_write_path_allowed
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
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if init_checkpoint is not None:
        payload["model"]["load_checkpoint"] = str(Path(init_checkpoint).expanduser().resolve())

    source_data_config = Path(payload["data"]["train_data_config"]).expanduser().resolve()
    with source_data_config.open("r", encoding="utf-8") as handle:
        source_data = yaml.safe_load(handle)
    for dataset in source_data["datasets"]:
        dataset["ratio"] = 1.0 if str(dataset.get("task")) == task else 0.0
    smoke_data_config = output / "multitask_smoke_data.yaml"
    _atomic_yaml(smoke_data_config, source_data)

    payload["data"]["train_data_config"] = str(smoke_data_config)
    payload["data"]["online_encoding"]["image_ratio"] = 1.0 if task == IMAGE_TASK else 0.0
    payload["data"]["online_encoding"]["video_ratio"] = 1.0 if task == VIDEO_TASK else 0.0
    payload["optimization"]["steps"] = 1
    payload["optimization"]["batch_size"] = 1
    payload["optimization"]["gradient_accumulation_steps"] = 4
    payload["validation"]["interval"] = None
    payload["validation"]["skip_initial_validation"] = True
    payload["checkpoints"]["interval"] = 1
    payload["checkpoints"]["keep_last_n"] = 1
    payload["checkpoints"]["no_resume"] = True
    payload["checkpoints"]["save_training_state"] = "full"
    payload["wandb"]["enabled"] = False
    strategy = payload["training_strategy"]
    strategy["cfg_dropout_enabled"] = False
    strategy["cfg_full_p"] = 1.0
    strategy["cfg_drop_text_p"] = 0.0
    strategy["cfg_drop_siglip_p"] = 0.0
    strategy["cfg_drop_ref_latents_p"] = 0.0
    strategy["cfg_drop_all_p"] = 0.0
    strategy["cfg_drop_ref_p"] = 0.0
    strategy["cfg_drop_planner_p"] = 0.0
    payload["output_dir"] = str(output)
    smoke_config_path = output / "smoke_config.yaml"
    _atomic_yaml(smoke_config_path, payload)
    return LtxTrainerConfig(**payload), smoke_config_path


def _optimizer_update_audit(trainer: LtxvTrainer) -> dict[str, Any]:
    modules = {
        "transformer": trainer._transformer,
        "text_encoder": trainer._text_encoder,
        "text_connector": trainer._embeddings_processor.video_connector,
        **{
            f"strategy.{name}": module
            for name, module in trainer._training_strategy.get_trainable_modules().items()
        },
    }
    audit: dict[str, Any] = {}
    for module_name, module in modules.items():
        if module is None:
            continue
        trainable = 0
        updated = 0
        moments_finite = True
        parameters_finite = True
        gradient_parameters = 0
        gradients_finite = True
        nonzero_gradients = 0
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            trainable += parameter.numel()
            parameters_finite = parameters_finite and bool(torch.isfinite(parameter).all())
            gradient_state = trainer._last_gradient_audit_by_parameter_id.get(id(parameter))
            if gradient_state is not None:
                gradient_parameters += 1
                gradients_finite = gradients_finite and gradient_state["finite"]
                nonzero_gradients += int(gradient_state["nonzero"])
            state = trainer._optimizer.state.get(parameter, {})
            moment = state.get("exp_avg")
            if isinstance(moment, torch.Tensor):
                moments_finite = moments_finite and bool(torch.isfinite(moment).all())
                updated += int(torch.count_nonzero(moment).item() > 0)
        if trainable:
            audit[module_name] = {
                "trainable_parameters": trainable,
                "parameters_with_nonzero_adam_moment": updated,
                "moments_finite": moments_finite,
                "parameters_finite": parameters_finite,
                "gradient_parameters": gradient_parameters,
                "gradients_finite": gradients_finite,
                "parameters_with_nonzero_gradient": nonzero_gradients,
            }
    return audit


def _assert_optimizer_update_audit(audit: dict[str, Any], *, phase: str) -> None:
    required_by_phase = {
        "stage1": {
            "transformer",
            "strategy.visual_token_projection",
            "strategy.visual_full_encoder",
        },
        "stage2": {
            "text_encoder",
            "text_connector",
            "strategy.planner_tokens",
            "strategy.visual_token_projection",
            "strategy.visual_full_encoder",
        },
        "stage3": {
            "transformer",
            "text_encoder",
            "text_connector",
            "strategy.planner_tokens",
            "strategy.visual_token_projection",
            "strategy.visual_full_encoder",
        },
    }
    required = required_by_phase[phase]
    missing = sorted(required - audit.keys())
    if missing:
        raise RuntimeError(f"{phase} smoke is missing required trainable modules: {missing}")
    failed = sorted(
        name
        for name in required
        if not audit[name]["moments_finite"]
        or not audit[name]["parameters_finite"]
        or not audit[name]["gradients_finite"]
        or audit[name]["gradient_parameters"] <= 0
        or audit[name]["parameters_with_nonzero_adam_moment"] <= 0
    )
    if failed:
        raise RuntimeError(f"{phase} smoke found no finite nonzero Adam update for: {failed}")


def _frozen_parameter_audit(trainer: LtxvTrainer) -> dict[str, Any]:
    text_encoder = getattr(trainer._text_encoder, "module", trainer._text_encoder)
    gemma_model = text_encoder.model.model
    modules = {
        "transformer": trainer._transformer,
        "text_encoder": trainer._text_encoder,
        "vision_tower": gemma_model.vision_tower,
        "multi_modal_projector": gemma_model.multi_modal_projector,
    }
    audit: dict[str, Any] = {}
    for name, module in modules.items():
        trainable = 0
        frozen = 0
        frozen_with_gradient = 0
        for parameter in module.parameters():
            if parameter.requires_grad:
                trainable += parameter.numel()
            else:
                frozen += parameter.numel()
                frozen_with_gradient += int(parameter.grad is not None)
        audit[name] = {
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "frozen_parameters_with_gradient": frozen_with_gradient,
        }
        if frozen_with_gradient:
            raise RuntimeError(f"Frozen {name} parameters received gradients")
    for name in ("vision_tower", "multi_modal_projector"):
        if audit[name]["trainable_parameters"] != 0:
            raise RuntimeError(f"Frozen Gemma {name} unexpectedly contains trainable parameters")
    return audit


def _strategy_loss_audit(trainer: LtxvTrainer, *, phase: str) -> dict[str, float]:
    get_metrics = getattr(trainer._training_strategy, "get_last_training_metrics", None)
    if not callable(get_metrics):
        return {}
    metrics = {
        name: float(value.detach().float().mean().item())
        for name, value in get_metrics().items()
    }
    for name, value in metrics.items():
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError(f"{phase} smoke produced non-finite metric {name}={value}")
    if phase in {"stage2", "stage3"}:
        required = {"train/loss_flow", "train/loss_siglip", "train/loss_ntp"}
        missing = sorted(required - metrics.keys())
        if missing:
            raise RuntimeError(f"{phase} smoke did not report all three losses: {missing}")
    return metrics


def _broadcast_main_payload(payload: dict[str, Any] | None, *, is_main_process: bool) -> dict[str, Any]:
    if not dist.is_available() or not dist.is_initialized():
        if payload is None:
            raise RuntimeError("Main-process smoke payload is unavailable")
        return payload
    objects: list[dict[str, Any] | None] = [payload if is_main_process else None]
    dist.broadcast_object_list(objects, src=0)
    if objects[0] is None:
        raise RuntimeError("Main process broadcast an empty smoke payload")
    return objects[0]


def _checkpoint_artifact_status(
    checkpoint: Path | None,
    *,
    global_step: int,
    phase: str,
    is_main_process: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] | None = None
    if is_main_process:
        try:
            if checkpoint is None:
                raise RuntimeError("Main rank received no checkpoint path")
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_file():
                raise RuntimeError(f"Checkpoint does not exist: {checkpoint_path}")
            training_state_path = (
                checkpoint_path.parent / f"training_state_step_{global_step:05d}.pt"
            )
            if not training_state_path.is_file():
                raise RuntimeError(
                    f"Matching training state does not exist: {training_state_path}"
                )
            with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
                metadata = dict(handle.metadata() or {})
            if metadata.get("global_step") != str(global_step):
                raise RuntimeError(
                    "Checkpoint metadata global_step mismatch: "
                    f"expected={global_step}, actual={metadata.get('global_step')!r}"
                )
            if phase in {"stage2", "stage3"} and metadata.get("training_phase") != phase:
                raise RuntimeError(
                    "Checkpoint metadata training_phase mismatch: "
                    f"expected={phase}, actual={metadata.get('training_phase')!r}"
                )
            if phase == "stage1" and metadata.get("conditioning") != "multi_reference_video":
                raise RuntimeError("Stage 1 checkpoint metadata has unexpected conditioning")
            ready_marker_path = (
                checkpoint_path.parent / f"checkpoint_step_{global_step:05d}.ready.json"
            )
            if not ready_marker_path.is_file():
                raise RuntimeError(f"Checkpoint ready marker does not exist: {ready_marker_path}")
            ready_marker = json.loads(ready_marker_path.read_text(encoding="utf-8"))
            if (
                int(ready_marker.get("global_step", -1)) != global_step
                or Path(ready_marker.get("checkpoint_path", "")) != checkpoint_path.resolve()
                or Path(ready_marker.get("training_state_path", ""))
                != training_state_path.resolve()
            ):
                raise RuntimeError("Checkpoint ready marker does not match the published artifact pair")
            payload = {
                "success": True,
                "checkpoint_path": str(checkpoint_path),
                "training_state_path": str(training_state_path),
                "ready_marker_path": str(ready_marker_path),
                "error_message": "",
            }
        except Exception as exc:
            payload = {
                "success": False,
                "checkpoint_path": "",
                "training_state_path": "",
                "ready_marker_path": "",
                "error_message": str(exc),
            }
    return _broadcast_main_payload(payload, is_main_process=is_main_process)


def _all_ranks_succeeded(success: bool, *, device: torch.device) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return success
    success_tensor = torch.tensor(int(success), device=device, dtype=torch.int32)
    dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN)
    return bool(success_tensor.item())


def _build_one_step_smoke_report(  # noqa: PLR0915
    trainer: LtxvTrainer,
    *,
    config: LtxTrainerConfig,
    generated_config: Path,
    checkpoint: Path | None,
    stats: Any,
    task: Literal["i2i", "r2v"],
    started: float,
) -> dict[str, Any]:
    phase = str(getattr(config.training_strategy, "training_phase", None) or "stage1")
    scheduler_last_epoch = int(trainer._lr_scheduler.state_dict().get("last_epoch", -1))
    optimizer_audit: dict[str, Any] = {}
    frozen_audit: dict[str, Any] = {}
    loss_audit: dict[str, float] = {}
    local_error = ""
    try:
        if trainer._global_step != 1 or scheduler_last_epoch != 1:
            raise RuntimeError(
                "One-step smoke did not preserve optimizer/scheduler semantics: "
                f"global_step={trainer._global_step}, scheduler_last_epoch={scheduler_last_epoch}"
            )
        optimizer_audit = _optimizer_update_audit(trainer)
        _assert_optimizer_update_audit(optimizer_audit, phase=phase)
        frozen_audit = _frozen_parameter_audit(trainer)
        loss_audit = _strategy_loss_audit(trainer, phase=phase)
    except Exception as exc:
        local_error = str(exc)

    artifact_status = _checkpoint_artifact_status(
        checkpoint,
        global_step=trainer._global_step,
        phase=phase,
        is_main_process=trainer._accelerator.is_main_process,
    )
    local_success = not local_error and bool(artifact_status["success"])
    all_success = _all_ranks_succeeded(local_success, device=trainer._accelerator.device)
    if not all_success:
        message = local_error or str(artifact_status["error_message"])
        if not message:
            message = "Another DDP rank failed its local gradient/optimizer/frozen/loss audit"
        raise RuntimeError(f"Distributed one-step smoke failed: {message}")

    return {
        "phase": phase,
        "task": task,
        "generated_config": str(generated_config),
        "checkpoint": artifact_status["checkpoint_path"],
        "training_state": artifact_status["training_state_path"],
        "ready_marker": artifact_status["ready_marker_path"],
        "global_step": trainer._global_step,
        "scheduler_last_epoch": scheduler_last_epoch,
        "elapsed_seconds": time.perf_counter() - started,
        "training_stats": stats.model_dump(),
        "timings_ms": trainer._last_online_metrics,
        "optimizer_update_audit": optimizer_audit,
        "frozen_parameter_audit": frozen_audit,
        "strategy_loss_audit": loss_audit,
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        ),
    }


def run_one_step_training_smoke(
    config_path: str | Path,
    *,
    task: Literal["i2i", "r2v"],
    output_dir: str | Path,
    init_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
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
    trainer._capture_gradient_audit = True
    try:
        checkpoint, stats = trainer.train(
            disable_progress_bars=True,
            finalize_accelerator=False,
        )
    except BaseException:
        trainer._accelerator.end_training()
        raise
    try:
        report = _build_one_step_smoke_report(
            trainer,
            config=config,
            generated_config=generated_config,
            checkpoint=checkpoint,
            stats=stats,
            task=task,
            started=started,
        )
        if trainer._accelerator.is_main_process:
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
    with Path(config_path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        config = LtxTrainerConfig(**yaml.safe_load(handle))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = LtxvTrainer(config)
    trainer._init_online_dataloader()
    requested = {IMAGE_TASK: num_image_samples, VIDEO_TASK: num_video_samples}
    encoded_reports: list[dict[str, Any]] = []
    for task, count in requested.items():
        for manifest_index in islice(trainer._dataset.task_indices[task], count):
            sample = trainer._dataset[int(manifest_index)]
            if isinstance(sample, SampleLoadError):
                raise RuntimeError(f"Real encode sample failed: {sample.to_dict()}")
            raw_batch = collate_online_raw_batch([sample])
            started = time.perf_counter()
            encoded = trainer._online_batch_encoder.encode_for_strategy(
                raw_batch,
                strategy=trainer._training_strategy,
                training_phase=getattr(trainer._training_strategy.config, "training_phase", "stage1"),
            )
            visual = encoded["gt_visual_tokens"]
            expected_valid = 256 if task == IMAGE_TASK else 2048
            expected_prompt_id = 0 if task == IMAGE_TASK else 1
            if int(visual["visual_token_mask"].sum().item()) != expected_valid:
                raise RuntimeError(f"{task} valid visual token count is not {expected_valid}")
            if int(encoded["task_system_prompt_id"].item()) != expected_prompt_id:
                raise RuntimeError(f"{task} used the wrong task system prompt id")
            if task == VIDEO_TASK and tuple(raw_batch["vlm_target_frame_indices"][0].tolist()) != VLM_TARGET_INDICES:
                raise RuntimeError("Real R2V encode used unexpected target frame indices")
            expected_latent_frames = 1 if task == IMAGE_TASK else 16
            latent_frames = int(encoded["latents"]["latents"].shape[2])
            if latent_frames != expected_latent_frames:
                raise RuntimeError(
                    f"{task} latent temporal length {latent_frames} != {expected_latent_frames}"
                )
            encoded_reports.append(
                {
                    "task": task,
                    "sample_key": sample["sample_key"],
                    "target_shape": list(raw_batch["target_pixels"].shape),
                    "latent_shape": list(encoded["latents"]["latents"].shape),
                    "visual_shape": list(visual["visual_tokens"].shape),
                    "valid_visual_tokens": expected_valid,
                    "task_system_prompt_id": expected_prompt_id,
                    "reference_vae_shapes": [
                        list(reference.shape) for reference in sample["reference_pixels_vae"]
                    ],
                    "reference_vlm_shapes": [
                        list(reference.shape) for reference in sample["reference_images_vlm"]
                    ],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    report = {
        "samples": encoded_reports,
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        ),
    }
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report
