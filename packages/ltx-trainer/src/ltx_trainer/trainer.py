import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import wandb
import yaml
from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs, gather_object, set_seed
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from pydantic import BaseModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    StepLR,
)
from torch.utils.data import DataLoader

from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.config_display import print_config
from ltx_trainer.datasets import PrecomputedDataset, collate_precomputed_batch
from ltx_trainer.gpu_utils import free_gpu_memory, get_gpu_memory_gb
from ltx_trainer.hf_hub_utils import push_to_hub
from ltx_trainer.model_loader import (
    load_embeddings_processor,
    load_text_encoder,
    load_transformer,
    load_video_vae_encoder,
)
from ltx_trainer.online_inference.checkpoint_runtime import (
    checkpoint_contains_semantic_flow_modules,
    read_checkpoint_metadata,
    validate_reference_rope_checkpoint_metadata,
    validate_semantic_flow_checkpoint_architecture,
)
from ltx_trainer.online_inference.startup_memory import host_memory_snapshot
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.quantization import quantize_model
from ltx_trainer.sigma_tracker import SigmaBucketTracker
from ltx_trainer.timestep_samplers import SAMPLERS
from ltx_trainer.training_state import ConfigFingerprint, RngStates, TrainingState
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.training_strategies.semantic_flow_bridge import (
    phase2_bridge_audit_rows,
    phase2_bridge_parameters,
    validate_and_load_phase2_bridge_state,
)
from ltx_trainer.validation_runner import ValidationRunner

# Disable irrelevant warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Silence bitsandbytes warnings about casting
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

# Disable progress bars if not main process
IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"
if not IS_MAIN_PROCESS:
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()

StepCallback = Callable[[int, int, list[Path]], None]  # (step, total, list[sampled_video_path]) -> None

MEMORY_CHECK_INTERVAL = 200


def _normalize_fsdp_config_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    name = getattr(value, "name", None)
    if name is not None:
        value = name
    return str(value).split(".")[-1].upper()


def _read_fsdp_plugin(accelerator: Accelerator) -> Any | None:
    plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)
    return plugin


def _read_fsdp_version_name(accelerator: Accelerator) -> str | None:
    plugin = _read_fsdp_plugin(accelerator)
    if plugin is None:
        return None
    for attribute in ("fsdp_version", "version"):
        version = _normalize_fsdp_config_value(getattr(plugin, attribute, None))
        if version is not None:
            return version
    return None


def _read_fsdp_sharding_strategy_name(accelerator: Accelerator) -> str | None:
    plugin = _read_fsdp_plugin(accelerator)
    if plugin is None:
        return None
    for attribute in ("sharding_strategy", "reshard_after_forward"):
        strategy = _normalize_fsdp_config_value(getattr(plugin, attribute, None))
        if strategy is not None:
            return strategy
    return None


def _read_fsdp_state_dict_type_name(accelerator: Accelerator) -> str | None:
    plugin = _read_fsdp_plugin(accelerator)
    if plugin is None:
        return None
    for attribute in ("state_dict_type", "fsdp_state_dict_type"):
        state_dict_type = _normalize_fsdp_config_value(getattr(plugin, attribute, None))
        if state_dict_type is not None:
            return state_dict_type
    return None


def _enforce_semantic_flow_fsdp_runtime_safety(config: LtxTrainerConfig, accelerator: Accelerator) -> None:
    if not (config.training_strategy.name == "semantic_flow" and config.model.training_mode == "full"):
        return
    if accelerator.distributed_type != DistributedType.FSDP:
        raise RuntimeError(
            "Full-DiT semantic-flow training requires Accelerate FSDP FULL_SHARD. "
            "Plain DDP and single-process full training are disabled to prevent OOM."
        )
    fsdp_version = _read_fsdp_version_name(accelerator)
    if fsdp_version not in {"1", "FSDP1"}:
        raise RuntimeError(
            "Full-DiT semantic-flow training currently supports only FSDP1. "
            f"Configured FSDP version is {fsdp_version!r}."
        )
    sharding_strategy = _read_fsdp_sharding_strategy_name(accelerator)
    if sharding_strategy is None:
        raise RuntimeError("Unable to identify FSDP sharding strategy; refusing to load semantic-flow models.")
    if sharding_strategy != "FULL_SHARD":
        raise RuntimeError(
            "Full-DiT semantic-flow training requires Accelerate FSDP FULL_SHARD. "
            f"Configured FSDP sharding strategy is {sharding_strategy!r}."
        )
    state_dict_type = _read_fsdp_state_dict_type_name(accelerator)
    if state_dict_type is None:
        raise RuntimeError("Unable to identify FSDP state-dict type; refusing to load semantic-flow models.")
    if state_dict_type != "FULL_STATE_DICT":
        raise RuntimeError(
            "Full-DiT semantic-flow training requires FSDP FULL_STATE_DICT checkpoint collection. "
            f"Configured FSDP state-dict type is {state_dict_type!r}."
        )


def _find_scalar_trainable_parameters(
    models_to_prepare: list[tuple[str, nn.Module]],
) -> list[str]:
    scalar_parameters: list[str] = []
    for model_name, module in models_to_prepare:
        for parameter_name, parameter in module.named_parameters():
            if parameter.requires_grad and parameter.ndim == 0:
                scalar_parameters.append(
                    f"{model_name}.{parameter_name}: shape={tuple(parameter.shape)}"
                )
    return scalar_parameters


def normalize_peft_adapter_key(key: str) -> str:
    """Normalize PEFT adapter keys across saved and in-memory naming variants."""
    normalized = key
    while normalized.startswith("base_model.model."):
        normalized = normalized.removeprefix("base_model.model.")
    for adapter_key in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
        normalized = normalized.replace(f".{adapter_key}.default.", f".{adapter_key}.")
    return normalized


class TrainingStats(BaseModel):
    """Statistics collected during training"""

    total_time_seconds: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    local_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int
    num_processes: int


@dataclass(frozen=True)
class TrainingStepOutput:
    """Output from a single training step."""

    loss: Tensor  # [B,] per-element loss (unreduced)
    sigma: Tensor  # [B,] sampled sigma, detached from computational graph


class LtxvTrainer:
    def __init__(self, trainer_config: LtxTrainerConfig) -> None:
        self._config = trainer_config
        self._online_batch_encoder = None
        self._online_vae_encoder = None
        self._online_sampler = None
        self._pending_online_data_state: dict[str, Any] | None = None
        self._resume_initial_step = 0
        self._last_online_metrics: dict[str, float] = {}
        self._capture_gradient_audit = False
        self._last_gradient_audit_by_parameter_id: dict[int, dict[str, bool]] = {}
        self._embeddings_processor_trainable_modules: dict[str, nn.Module] = {}
        self._optimizer_group_parameter_counts: dict[str, int] = {}
        self._last_optimizer_group_metrics: dict[str, float] = {}
        self._last_phase2_accelerator_state_path: Path | None = None
        self._last_bridge_checkpoint_key_count = 0
        if IS_MAIN_PROCESS:
            print_config(trainer_config)
        self._training_strategy = get_training_strategy(self._config.training_strategy)
        self._setup_accelerator()
        self._startup_host_memory = {"before_model_load": host_memory_snapshot()}

        # ValidationRunner loads its own models (text encoder, VAE encoder/decoder, etc.),
        # caches prompt embeddings and conditioning media, then unloads encoders.
        self._validation_runner = ValidationRunner(
            config=self._config.validation,
            model_path=self._config.model.model_path,
            text_encoder_path=self._config.model.text_encoder_path,
            load_text_encoder_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
        )

        self._load_models()
        self._startup_host_memory["after_model_load"] = host_memory_snapshot()
        self._setup_trainable_model_wrappers()
        self._loaded_checkpoint_path: Path | None = None
        self._load_checkpoint()
        self._collect_trainable_params()
        self._prepare_models_for_training()
        self._startup_host_memory["after_fsdp_prepare"] = host_memory_snapshot()
        self._dataset = None
        self._global_step = -1
        self._checkpoint_paths: list[Path] = []
        self._training_state_paths: list[Path] = []
        self._last_saved_step: int | None = None
        self._last_saved_weights_path: Path | None = None
        self._training_state_size_warned = False
        self._sigma_tracker = SigmaBucketTracker()
        self._wandb_run = None

    def train(  # noqa: PLR0912, PLR0915
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback | None = None,
        finalize_accelerator: bool = True,
    ) -> tuple[Path | None, TrainingStats]:
        """
        Start the training process.
        Args:
            disable_progress_bars: Disable Rich progress bars (useful for multi-process runs).
            step_callback: Optional callback invoked after each optimization step.
            finalize_accelerator: End trackers/process groups before returning. Smoke callers may defer this
                until their final distributed audits have completed.
        Returns:
            Tuple of (saved_model_path, training_stats)
        """
        device = self._accelerator.device
        cfg = self._config
        start_mem = get_gpu_memory_gb(device)

        train_start_time = time.time()

        initial_step, training_state = self._resume_state
        resuming = training_state is not None

        set_seed(cfg.seed)
        logger.debug(f"Process {self._accelerator.process_index} using seed: {cfg.seed}")

        self._init_optimizer()

        if training_state is not None and self._is_semantic_flow_phase2():
            self._restore_phase2_accelerator_state(training_state)
        elif training_state is not None and not self._restore_training_state(training_state):
            initial_step = 0
            resuming = False

        # Initialize W&B after restore so we only resume the run when state restore succeeds.
        resume_run_id = training_state.wandb_run_id if resuming and training_state is not None else None
        self._init_wandb(resume_run_id=resume_run_id)

        self._resume_initial_step = initial_step
        if training_state is not None:
            self._pending_online_data_state = training_state.data_state
        self._init_dataloader()
        data_iter = iter(self._dataloader)
        self._init_timestep_sampler()

        # Synchronize all processes after initialization
        self._accelerator.wait_for_everyone()

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Save the training configuration as YAML
        self._save_config()

        remaining_steps = cfg.optimization.steps - initial_step
        if remaining_steps <= 0:
            raise ValueError(
                f"No remaining training steps: initial_step={initial_step} >= "
                f"target_steps={cfg.optimization.steps}. Nothing to train."
            )

        if resuming:
            logger.info(f"🚀 Resuming training from step {initial_step} → {cfg.optimization.steps}")
        else:
            logger.info("🚀 Starting training...")

        # Create progress tracking (disabled for non-main processes or when explicitly disabled)
        progress_enabled = IS_MAIN_PROCESS and not disable_progress_bars
        progress = TrainingProgress(
            enabled=progress_enabled,
            total_steps=remaining_steps,
        )

        if IS_MAIN_PROCESS and disable_progress_bars:
            logger.warning("Progress bars disabled. Intermediate status messages will be logged instead.")

        if self._train_transformer:
            self._transformer.train()
        else:
            self._transformer.eval()
        if self._train_embeddings_processor:
            self._embeddings_processor.train()
        else:
            self._embeddings_processor.eval()
        if self._text_encoder is not None:
            self._text_encoder.train(self._train_text_encoder)
        enforce_frozen_eval = getattr(self._training_strategy, "enforce_frozen_module_eval", None)
        if callable(enforce_frozen_eval):
            enforce_frozen_eval()
        self._global_step = initial_step

        peak_mem_during_training = start_mem

        sampled_videos_paths = None

        with progress:
            if cfg.validation.interval and not cfg.validation.skip_initial_validation:
                with self._offloaded_optimizer_state():
                    sampled_videos_paths = self._run_validation(progress)

            self._accelerator.wait_for_everyone()

            micro_step = 0
            while self._global_step < cfg.optimization.steps:
                # Get next batch, reset the dataloader if needed
                try:
                    data_wait_started = time.perf_counter()
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self._dataloader)
                    batch = next(data_iter)
                data_wait_ms = (time.perf_counter() - data_wait_started) * 1000.0
                if cfg.data.encoding_mode == "online":
                    batch = self._prepare_online_batch_with_retry(batch)
                    batch.setdefault("_online_metrics", {})["data_wait_ms"] = data_wait_ms

                step_start_time = time.time()
                with self._accelerator.accumulate(*self._accumulation_models):
                    is_optimization_step = self._accelerator.sync_gradients
                    if is_optimization_step:
                        self._global_step += 1

                    output = self._training_step(batch)
                    backward_started = time.perf_counter()
                    self._accelerator.backward(output.loss.mean())
                    if cfg.data.encoding_mode == "online":
                        self._last_online_metrics["backward_ms"] = (
                            time.perf_counter() - backward_started
                        ) * 1000.0

                    optimizer_started = time.perf_counter()
                    if self._accelerator.sync_gradients:
                        self._last_optimizer_group_metrics = self._optimizer_group_metrics()
                    if self._accelerator.sync_gradients and cfg.optimization.max_grad_norm > 0:
                        self._accelerator.clip_grad_norm_(
                            self._trainable_params,
                            cfg.optimization.max_grad_norm,
                        )

                    if self._accelerator.sync_gradients and self._capture_gradient_audit:
                        self._last_gradient_audit_by_parameter_id = {
                            id(parameter): {
                                "finite": bool(torch.isfinite(parameter.grad).all()),
                                "nonzero": bool(torch.count_nonzero(parameter.grad).item()),
                            }
                            for parameter in self._trainable_params
                            if parameter.grad is not None
                        }

                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    self._step_lr_scheduler(
                        self._lr_scheduler,
                        sync_gradients=self._accelerator.sync_gradients,
                    )
                    if cfg.data.encoding_mode == "online":
                        self._last_online_metrics["optimizer_ms"] = (
                            time.perf_counter() - optimizer_started
                        ) * 1000.0
                    if cfg.data.encoding_mode == "online":
                        manifest_indices = self._accelerator.gather(
                            batch["manifest_index"].to(device=self._accelerator.device, dtype=torch.long)
                        )
                        self._online_sampler.mark_microbatch_consumed(
                            manifest_indices.detach().cpu().flatten().tolist()
                        )

                    # Run validation if needed (handles DDP/FSDP work distribution internally)
                    if (
                        cfg.validation.interval
                        and self._global_step > 0
                        and self._global_step % cfg.validation.interval == 0
                        and is_optimization_step
                    ):
                        with self._offloaded_optimizer_state():
                            sampled_videos_paths = self._run_validation(progress)

                    # Save checkpoint if needed
                    if (
                        cfg.checkpoints.interval
                        and self._global_step > 0
                        and self._global_step % cfg.checkpoints.interval == 0
                        and is_optimization_step
                    ):
                        self._save_checkpoint()

                    self._accelerator.wait_for_everyone()

                    # Call step callback if provided
                    if step_callback and is_optimization_step:
                        step_callback(self._global_step, cfg.optimization.steps, sampled_videos_paths)

                    self._accelerator.wait_for_everyone()

                    # Update progress and log metrics
                    current_lr = self._optimizer.param_groups[0]["lr"]
                    step_time = (time.time() - step_start_time) * cfg.optimization.gradient_accumulation_steps
                    step_loss = output.loss.detach().mean().item()
                    strategy_metrics: dict[str, float] = {}
                    if IS_MAIN_PROCESS and is_optimization_step:
                        get_strategy_metrics = getattr(self._training_strategy, "get_last_training_metrics", None)
                        if callable(get_strategy_metrics):
                            strategy_metrics = {
                                name: float(value.detach().float().mean().item())
                                for name, value in get_strategy_metrics().items()
                            }

                    progress.update_training(
                        loss=step_loss,
                        lr=current_lr,
                        step_time=step_time,
                        advance=is_optimization_step,
                    )

                    # Log metrics to W&B (only on main process and optimization steps)
                    if IS_MAIN_PROCESS and is_optimization_step:
                        # Track per-element loss by sigma bucket
                        self._sigma_tracker.update(output.sigma.cpu().tolist(), output.loss.detach().cpu().tolist())
                        metrics = {
                            "train/loss": step_loss,
                            "train/loss_total": step_loss,
                            "train/learning_rate": current_lr,
                            "train/step_time": step_time,
                            "train/global_step": self._global_step,
                        }
                        metrics.update(strategy_metrics)
                        metrics.update(self._last_optimizer_group_metrics)
                        metrics.update(
                            {f"train/{name}": value for name, value in self._last_online_metrics.items()}
                        )
                        metrics.update(self._sigma_tracker.get_metrics())
                        self._log_metrics(metrics)

                    # Fallback logging when progress bars are disabled
                    if disable_progress_bars and IS_MAIN_PROCESS and is_optimization_step:
                        elapsed = time.time() - train_start_time
                        steps_done = self._global_step - initial_step
                        if steps_done > 0:
                            total_estimated = elapsed / steps_done * remaining_steps
                            total_time = f"{total_estimated // 3600:.0f}h {(total_estimated % 3600) // 60:.0f}m"
                        else:
                            total_time = "calculating..."
                        video_text = self._format_optional_metric(strategy_metrics, "train/loss_video_flow")
                        semantic_text = self._format_optional_metric(strategy_metrics, "train/loss_semantic_flow")
                        reconstruction_text = self._format_optional_metric(
                            strategy_metrics, "train/loss_semantic_reconstruction"
                        )
                        alignment_text = self._format_optional_metric(
                            strategy_metrics, "train/loss_semantic_alignment"
                        )
                        logger.info(
                            f"Step {self._global_step}/{cfg.optimization.steps} - "
                            f"Total: {step_loss:.4f}, Video flow: {video_text}, Semantic flow: {semantic_text}, "
                            f"Reconstruction: {reconstruction_text}, Alignment: {alignment_text}, "
                            f"LR: {current_lr:.2e}, "
                            f"Time/Step: {step_time:.2f}s, Total Time: {total_time}",
                        )

                    # Sample GPU memory periodically
                    if micro_step % MEMORY_CHECK_INTERVAL == 0:
                        current_mem = get_gpu_memory_gb(device)
                        peak_mem_during_training = max(peak_mem_during_training, current_mem)
                micro_step += 1

        # Collect final stats
        train_end_time = time.time()
        end_mem = get_gpu_memory_gb(device)
        peak_mem = max(start_mem, end_mem, peak_mem_during_training)

        # Calculate steps/second over entire training
        total_time_seconds = train_end_time - train_start_time
        steps_per_second = remaining_steps / total_time_seconds

        effective_global_batch_size = self._effective_global_batch_size(
            batch_size=cfg.optimization.batch_size,
            num_processes=self._accelerator.num_processes,
            gradient_accumulation_steps=cfg.optimization.gradient_accumulation_steps,
        )
        samples_per_second = steps_per_second * effective_global_batch_size

        stats = TrainingStats(
            total_time_seconds=total_time_seconds,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            peak_gpu_memory_gb=peak_mem,
            local_batch_size=cfg.optimization.batch_size,
            gradient_accumulation_steps=cfg.optimization.gradient_accumulation_steps,
            num_processes=self._accelerator.num_processes,
            global_batch_size=effective_global_batch_size,
        )

        saved_path = self._save_checkpoint()

        if IS_MAIN_PROCESS:
            if saved_path is None:
                raise RuntimeError("Main process did not receive the final checkpoint path")
            # Log the training statistics
            self._log_training_stats(stats)

            # Upload artifacts to hub if enabled
            if cfg.hub.push_to_hub:
                push_to_hub(saved_path, sampled_videos_paths, self._config)

            # Log final stats to W&B
            if self._wandb_run is not None:
                self._log_metrics(
                    {
                        "stats/total_time_minutes": stats.total_time_seconds / 60,
                        "stats/steps_per_second": stats.steps_per_second,
                        "stats/samples_per_second": stats.samples_per_second,
                        "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
                    }
                )
                self._wandb_run.finish()

        self._accelerator.wait_for_everyone()
        if finalize_accelerator:
            self._accelerator.end_training()

        return saved_path, stats

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> TrainingStepOutput:
        """Perform a single training step using the configured strategy."""
        with self._accelerator.autocast():
            return self._training_step_autocast(batch)

    def _training_step_autocast(self, batch: dict[str, dict[str, Tensor]]) -> TrainingStepOutput:
        """Perform the full training step inside the caller's autocast context."""
        # Apply embedding connectors to transform pre-computed text embeddings
        conditions = batch["conditions"]
        condition_started = time.perf_counter()
        conditions = self._training_strategy.prepare_conditions(batch, conditions)
        if self._config.data.encoding_mode == "online":
            online_metrics = batch.setdefault("_online_metrics", {})
            condition_prepare_ms = (time.perf_counter() - condition_started) * 1000.0
            online_metrics["condition_prepare_ms"] = condition_prepare_ms
        batch["conditions"] = conditions

        if "video_prompt_embeds" in conditions:
            # New format: separate video/audio features from precompute()
            video_features = conditions["video_prompt_embeds"]
            audio_features = conditions.get("audio_prompt_embeds")
        else:
            # Legacy format: single prompt_embeds tensor — duplicate for both modalities
            video_features = conditions["prompt_embeds"]
            audio_features = conditions["prompt_embeds"]

        mask = conditions["prompt_attention_mask"]
        additive_mask = convert_to_additive_mask(mask, video_features.dtype)
        video_embeds, audio_embeds, attention_mask = self._embeddings_processor.create_embeddings(
            video_features, audio_features, additive_mask
        )

        conditions["video_prompt_embeds"] = video_embeds
        conditions["audio_prompt_embeds"] = audio_embeds
        conditions["prompt_attention_mask"] = attention_mask

        conditions = self._training_strategy.postprocess_conditions_after_connector(batch, conditions)
        batch["conditions"] = conditions

        # Use strategy to prepare training inputs (returns ModelInputs with Modality objects)
        model_inputs = self._training_strategy.prepare_training_inputs(batch, self._timestep_sampler)

        # Run transformer forward pass with Modality-based interface
        dit_started = time.perf_counter()
        video_pred, audio_pred = self._transformer(
            video=model_inputs.video,
            audio=model_inputs.audio,
            perturbations=None,
        )
        if self._config.data.encoding_mode == "online":
            batch["_online_metrics"]["dit_ms"] = (time.perf_counter() - dit_started) * 1000.0
            self._last_online_metrics = {
                name: float(value)
                for name, value in batch["_online_metrics"].items()
            }
            task = str(batch["task"][0])
            self._last_online_metrics.update(
                {
                    "task_id": 0.0 if task == "i2i" else 1.0,
                    "is_image": 1.0 if task == "i2i" else 0.0,
                    "target_num_frames": 1.0 if task == "i2i" else 121.0,
                }
            )

        # Use strategy to compute loss (returns per-element [B,] for sigma-bucket tracking)
        loss = self._training_strategy.compute_loss(video_pred, audio_pred, model_inputs)

        # Sigma comes from whichever modality is generated (video preferred, else audio).
        if model_inputs.video is not None and model_inputs.video.enabled:
            sigma = model_inputs.video.sigma.detach()
        else:
            sigma = model_inputs.audio.sigma.detach()

        return TrainingStepOutput(loss=loss, sigma=sigma)

    def _load_models(self) -> None:
        """Load the transformer and embeddings processor for training."""
        logger.debug("Loading transformer...")
        self._transformer = load_transformer(
            checkpoint_path=self._config.model.model_path,
            device="cpu",
            dtype=torch.bfloat16,
        )

        # Accelerator is initialized before model loading, so every rank resolves to its
        # assigned device before Gemma, VAE, and connector weights are materialized.
        init_device = self._accelerator.device if torch.cuda.is_available() else torch.device("cpu")

        logger.debug("Loading embeddings processor...")
        self._embeddings_processor = load_embeddings_processor(
            checkpoint_path=self._config.model.model_path,
            device=init_device,
            dtype=torch.bfloat16,
        )
        if self._config.data.encoding_mode == "precomputed":
            self._embeddings_processor.feature_extractor = None
        self._embeddings_processor.requires_grad_(False)

        self._text_encoder = None
        if self._training_strategy.requires_text_encoder() or self._config.data.encoding_mode == "online":
            logger.debug("Loading Gemma/VLM text encoder for training strategy...")
            self._text_encoder = load_text_encoder(
                gemma_model_path=self._config.model.text_encoder_path,
                device=init_device,
                dtype=torch.bfloat16,
                load_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
            )
            self._text_encoder.requires_grad_(False)
            self._setup_text_encoder_lora()

        transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32
        self._transformer = self._transformer.to(dtype=transformer_dtype)

        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")

            logger.info(f'Quantizing model with "{self._config.acceleration.quantization}". This may take a while...')
            self._transformer = quantize_model(
                self._transformer,
                precision=self._config.acceleration.quantization,
            )

        self._transformer.requires_grad_(False)
        self._training_strategy.attach_models(
            transformer=self._transformer,
            embeddings_processor=self._embeddings_processor,
            text_encoder=self._text_encoder,
        )
        if self._config.data.encoding_mode == "online":
            if self._text_encoder is None or self._config.data.online_encoding is None:
                raise RuntimeError("Online encoding requires a loaded Gemma text encoder and online_encoding config")
            online_config = self._config.data.online_encoding
            encoder_dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[online_config.encoder_dtype]
            vae_device = init_device if online_config.encoder_device_policy == "resident_cuda" else torch.device("cpu")
            logger.debug("Loading frozen video VAE encoder for online target/reference encoding...")
            self._online_vae_encoder = load_video_vae_encoder(
                self._config.model.model_path,
                device=vae_device,
                dtype=encoder_dtype,
            )
            from ltx_trainer.online_data.online_batch_encoder import OnlineBatchEncoder  # noqa: PLC0415

            self._online_batch_encoder = OnlineBatchEncoder(
                config=online_config,
                model_path=self._config.model.model_path,
                text_encoder_path=self._config.model.text_encoder_path,
                vae_encoder=self._online_vae_encoder,
                text_encoder=self._text_encoder,
                embeddings_processor=self._embeddings_processor,
                device=init_device,
            )

    def _setup_trainable_model_wrappers(self) -> None:
        """Create LoRA wrappers before loading an initialization checkpoint."""
        self._train_transformer = self._training_strategy.train_transformer()
        self._train_embeddings_processor = self._training_strategy.train_embeddings_processor()
        self._train_text_encoder = self._training_strategy.train_text_encoder() or (
            self._text_encoder is not None
            and any(parameter.requires_grad for parameter in self._text_encoder.parameters())
        )

        if self._config.model.training_mode == "lora":
            self._setup_lora()
        elif self._config.model.training_mode == "full":
            pass
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

    def _collect_trainable_params(self) -> None:
        """Configure and collect trainable parameters after checkpoint initialization."""
        if self._config.model.training_mode == "lora":
            configure_transformer = getattr(
                self._training_strategy,
                "configure_transformer_trainability",
                None,
            )
            if callable(configure_transformer):
                configure_transformer(self._transformer)
            elif not self._train_transformer:
                self._transformer.requires_grad_(False)
        else:
            self._transformer.requires_grad_(self._train_transformer)

        self._embeddings_processor.requires_grad_(False)
        if self._train_embeddings_processor:
            configure_processor = getattr(
                self._training_strategy,
                "configure_embeddings_processor_trainability",
                None,
            )
            if callable(configure_processor):
                configure_processor(self._embeddings_processor)
            else:
                self._embeddings_processor.video_connector.requires_grad_(True)
            get_processor_modules = getattr(
                self._training_strategy,
                "get_embeddings_processor_trainable_modules",
                None,
            )
            if callable(get_processor_modules):
                self._embeddings_processor_trainable_modules = get_processor_modules(
                    self._embeddings_processor
                )
            else:
                self._embeddings_processor_trainable_modules = {
                    "video_connector": self._embeddings_processor.video_connector
                }
            if not self._embeddings_processor_trainable_modules:
                raise RuntimeError("Embedding-processor training resolved to zero modules")
        if self._text_encoder is not None and not self._train_text_encoder:
            self._text_encoder.requires_grad_(False)

        strategy_modules = self._training_strategy.get_trainable_modules()
        for module in strategy_modules.values():
            module.requires_grad_(True)

        candidate_params = [p for p in self._transformer.parameters() if p.requires_grad]
        if self._train_embeddings_processor:
            for module in self._embeddings_processor_trainable_modules.values():
                candidate_params.extend(p for p in module.parameters() if p.requires_grad)
        if self._train_text_encoder:
            if self._text_encoder is None:
                raise ValueError("Training strategy requested text encoder training, but no text encoder was loaded.")
            candidate_params.extend(p for p in self._text_encoder.parameters() if p.requires_grad)
        for module in strategy_modules.values():
            candidate_params.extend(p for p in module.parameters() if p.requires_grad)

        self._trainable_params = self._deduplicate_parameters(candidate_params)

        if not self._trainable_params:
            raise ValueError("No trainable parameters were found for the selected training strategy.")

        logger.debug(f"Trainable params count: {sum(p.numel() for p in self._trainable_params):,}")
        self._validate_phase2_trainability(strategy_modules)
        self._write_phase2_parameter_audit(strategy_modules)
        self._log_parameter_summary(strategy_modules)

    def _is_semantic_flow_phase2(self) -> bool:
        strategy_config = getattr(self._training_strategy, "config", None)
        return getattr(strategy_config, "training_phase", "phase1") == "phase2"

    def _validate_phase2_trainability(
        self,
        strategy_modules: dict[str, torch.nn.Module],
    ) -> None:
        if not self._is_semantic_flow_phase2():
            return
        frozen_dit = [
            name
            for name, parameter in self._transformer.named_parameters()
            if not parameter.requires_grad
        ]
        if frozen_dit:
            raise RuntimeError(
                f"Phase 2 requires full DiT training; frozen parameters: {frozen_dit[:20]}"
            )
        frozen_semantic = [
            f"{module_name}.{parameter_name}"
            for module_name, module in strategy_modules.items()
            for parameter_name, parameter in module.named_parameters()
            if not parameter.requires_grad
        ]
        if frozen_semantic:
            raise RuntimeError(
                "Phase 2 requires all semantic modules to be trainable; "
                f"frozen parameters: {frozen_semantic[:20]}"
            )
        bridge_parameters = self._deduplicate_parameters(
            [
                parameter
                for module in self._embeddings_processor_trainable_modules.values()
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
        )
        if not bridge_parameters:
            raise RuntimeError("Phase 2 conditioning bridge has zero trainable parameters")
        expected_bridge = phase2_bridge_parameters(self._embeddings_processor)
        frozen_bridge = [
            item.name
            for item in expected_bridge
            if not item.parameter.requires_grad
        ]
        if frozen_bridge:
            raise RuntimeError(
                f"Phase 2 bridge allowlist contains frozen parameters: {frozen_bridge[:20]}"
            )
        expected_bridge_ids = {id(item.parameter) for item in expected_bridge}
        actual_bridge_ids = {id(parameter) for parameter in bridge_parameters}
        if actual_bridge_ids != expected_bridge_ids:
            raise RuntimeError(
                "Phase 2 prepared bridge modules do not match the explicit allowlist"
            )
        dit_semantic_parameters = self._deduplicate_parameters(
            [
                *[parameter for parameter in self._transformer.parameters() if parameter.requires_grad],
                *[
                    parameter
                    for module in strategy_modules.values()
                    for parameter in module.parameters()
                    if parameter.requires_grad
                ],
            ]
        )
        overlap = {id(parameter) for parameter in bridge_parameters} & {
            id(parameter) for parameter in dit_semantic_parameters
        }
        if overlap:
            raise RuntimeError("Phase 2 bridge and DiT/semantic optimizer groups overlap")
        if self._text_encoder is None or any(
            parameter.requires_grad for parameter in self._text_encoder.parameters()
        ):
            raise RuntimeError("Phase 2 Gemma/SigLIP/projector must remain frozen")
        if self._online_vae_encoder is None or any(
            parameter.requires_grad for parameter in self._online_vae_encoder.parameters()
        ):
            raise RuntimeError("Phase 2 VAE encoder must remain frozen")
        audio_connector = getattr(self._embeddings_processor, "audio_connector", None)
        if isinstance(audio_connector, nn.Module) and any(
            parameter.requires_grad for parameter in audio_connector.parameters()
        ):
            raise RuntimeError("Phase 2 audio connector must remain frozen")
        unexpected_processor = [
            name
            for name, parameter in self._embeddings_processor.named_parameters()
            if parameter.requires_grad and id(parameter) not in expected_bridge_ids
        ]
        if unexpected_processor:
            raise RuntimeError(
                f"Unexpected Phase 2 trainable embedding-processor parameters: {unexpected_processor}"
            )
        self._optimizer_group_parameter_counts = {
            "dit_semantic": sum(parameter.numel() for parameter in dit_semantic_parameters),
            "conditioning_bridge": sum(parameter.numel() for parameter in bridge_parameters),
        }
        logger.info(
            "Phase 2 optimizer ownership: dit_semantic=%s conditioning_bridge=%s",
            f"{self._optimizer_group_parameter_counts['dit_semantic']:,}",
            f"{self._optimizer_group_parameter_counts['conditioning_bridge']:,}",
        )

    def _write_phase2_parameter_audit(
        self,
        strategy_modules: dict[str, torch.nn.Module],
    ) -> None:
        if not self._is_semantic_flow_phase2() or not IS_MAIN_PROCESS:
            return
        from ltx_trainer.online_data.path_safety import assert_write_path_allowed  # noqa: PLC0415

        rows: list[dict[str, Any]] = []
        rows.extend(
            {
                "parameter_name": f"transformer.{name}",
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "requires_grad": bool(parameter.requires_grad),
                "owner_group": "dit_semantic",
            }
            for name, parameter in self._transformer.named_parameters()
        )
        for module_name, module in strategy_modules.items():
            rows.extend(
                {
                    "parameter_name": f"training_strategy.{module_name}.{name}",
                    "shape": list(parameter.shape),
                    "numel": parameter.numel(),
                    "requires_grad": bool(parameter.requires_grad),
                    "owner_group": "dit_semantic",
                }
                for name, parameter in module.named_parameters()
            )
        rows.extend(phase2_bridge_audit_rows(self._embeddings_processor))
        if self._text_encoder is not None:
            rows.extend(
                {
                    "parameter_name": f"text_encoder.{name}",
                    "shape": list(parameter.shape),
                    "numel": parameter.numel(),
                    "requires_grad": bool(parameter.requires_grad),
                    "owner_group": "frozen_vlm",
                }
                for name, parameter in self._text_encoder.named_parameters()
            )
        if self._online_vae_encoder is not None:
            rows.extend(
                {
                    "parameter_name": f"vae_encoder.{name}",
                    "shape": list(parameter.shape),
                    "numel": parameter.numel(),
                    "requires_grad": bool(parameter.requires_grad),
                    "owner_group": "frozen_other",
                }
                for name, parameter in self._online_vae_encoder.named_parameters()
            )
        unexpected = [
            row["parameter_name"]
            for row in rows
            if row["requires_grad"]
            and row["owner_group"] in {"frozen_vlm", "frozen_other"}
        ]
        if unexpected:
            raise RuntimeError(
                f"Phase 2 parameter audit found unexpected trainable parameters: {unexpected[:20]}"
            )
        audit_path = assert_write_path_allowed(
            Path(self._config.output_dir) / "phase2_parameter_audit_rank0.jsonl"
        )
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        logger.info("Phase 2 parameter audit written to %s (%d rows)", audit_path, len(rows))

    @staticmethod
    def _deduplicate_parameters(parameters: list[Tensor]) -> list[Tensor]:
        unique: list[Tensor] = []
        seen: set[int] = set()
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            if parameter_id not in seen:
                seen.add(parameter_id)
                unique.append(parameter)
        return unique

    def _log_parameter_summary(self, strategy_modules: dict[str, torch.nn.Module]) -> None:
        """Log the full-rank/adapter and frozen parameter ownership clearly."""
        count = lambda parameters: sum(parameter.numel() for parameter in parameters)  # noqa: E731
        trainable_dit = [parameter for parameter in self._transformer.parameters() if parameter.requires_grad]
        frozen_dit = [parameter for parameter in self._transformer.parameters() if not parameter.requires_grad]
        frozen_gemma = []
        frozen_vision_projector = []
        if self._text_encoder is not None:
            frozen_gemma = [
                parameter
                for name, parameter in self._text_encoder.named_parameters()
                if not parameter.requires_grad
                and "vision_tower" not in name
                and "multi_modal_projector" not in name
            ]
            frozen_vision_projector = [
                parameter
                for name, parameter in self._text_encoder.named_parameters()
                if not parameter.requires_grad
                and ("vision_tower" in name or "multi_modal_projector" in name)
            ]
        connector_trainable = [
            parameter
            for module in self._embeddings_processor_trainable_modules.values()
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        logger.info(f"Trainable DiT params: {count(trainable_dit):,}")
        for module_name, module in sorted(strategy_modules.items()):
            logger.info(f"Trainable strategy module {module_name}: {count(module.parameters()):,}")
        logger.info(f"Trainable text connector params: {count(connector_trainable):,}")
        logger.info(f"Total trainable params: {count(self._trainable_params):,}")
        logger.info(f"Frozen base DiT params: {count(frozen_dit):,}")
        logger.info(f"Frozen Gemma params: {count(frozen_gemma):,}")
        logger.info(f"Frozen vision/projector params: {count(frozen_vision_projector):,}")

    def _init_timestep_sampler(self) -> None:
        """Initialize the timestep sampler based on the config."""
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    def _setup_lora(self) -> None:
        """Configure LoRA adapters for the transformer. Only called in LoRA training mode."""
        logger.debug(f"Adding LoRA adapter with rank {self._config.lora.rank}")
        lora_config = LoraConfig(
            r=self._config.lora.rank,
            lora_alpha=self._config.lora.alpha,
            target_modules=self._config.lora.target_modules,
            lora_dropout=self._config.lora.dropout,
            init_lora_weights=True,
        )
        # Wrap the transformer with PEFT to add LoRA layers
        # noinspection PyTypeChecker
        self._transformer = get_peft_model(self._transformer, lora_config)

    def _setup_text_encoder_lora(self) -> None:
        config = self._config.text_encoder_lora
        if not config.enabled:
            requires_lora = getattr(self._training_strategy, "requires_text_encoder_lora", None)
            if callable(requires_lora) and requires_lora():
                raise ValueError("This training strategy requires text_encoder_lora.enabled=true")
            return
        if self._text_encoder is None:
            raise ValueError("text_encoder_lora.enabled=true requires a loaded text encoder")
        gemma_model = self._text_encoder.model.model
        language_model = getattr(gemma_model, "language_model", None)
        if language_model is None:
            raise ValueError("Gemma model does not expose model.language_model for text_encoder_lora")
        language_model.requires_grad_(False)
        language_model = get_peft_model(
            language_model,
            LoraConfig(
                r=config.rank,
                lora_alpha=config.alpha,
                lora_dropout=config.dropout,
                target_modules=config.target_modules,
                init_lora_weights=True,
            ),
        )
        gemma_model.language_model = language_model
        logger.info(f"Added Gemma language-model LoRA with rank {config.rank}")

    def _load_checkpoint(self) -> None:
        """Load checkpoint if specified in config, then resolve resume state."""
        if not self._config.model.load_checkpoint:
            self._resume_state: tuple[int, TrainingState | None] = (0, None)
            return

        checkpoint_path = self._find_checkpoint(self._config.model.load_checkpoint)
        if not checkpoint_path:
            logger.warning(f"⚠️ Could not find checkpoint at {self._config.model.load_checkpoint}")
            self._resume_state = (0, None)
            return

        self._loaded_checkpoint_path = checkpoint_path
        logger.info(f"📥 Loading checkpoint from {checkpoint_path}")

        if self._config.model.training_mode == "full":
            self._load_full_checkpoint(checkpoint_path)
        else:  # LoRA mode
            self._load_lora_checkpoint(checkpoint_path)

        if self._is_semantic_flow_phase2():
            metadata = read_checkpoint_metadata(checkpoint_path)
            if getattr(self._training_strategy, "phase2_initialization_source", None) == "phase1_parent":
                self._training_strategy.phase2_parent_checkpoint_sha256 = self._sha256_file(
                    checkpoint_path
                )
            elif metadata.get("training_phase") == "phase2":
                parent_sha = metadata.get("parent_checkpoint_sha256")
                if not parent_sha:
                    raise RuntimeError("Phase 2 resume checkpoint is missing parent_checkpoint_sha256")
                self._training_strategy.phase2_parent_checkpoint_sha256 = parent_sha

        self._resume_state = self._resolve_resume_state()

    def _load_full_checkpoint(self, checkpoint_path: Path) -> None:
        """Load full model checkpoint."""
        checkpoint_metadata = self._validate_full_checkpoint_metadata(checkpoint_path)
        state_dict = load_file(checkpoint_path)
        self._load_auxiliary_checkpoint_state(
            state_dict,
            checkpoint_metadata=checkpoint_metadata,
        )

        transformer_state = self._filter_auxiliary_checkpoint_state(state_dict)
        if transformer_state:
            self._transformer.load_state_dict(transformer_state, strict=True)
        else:
            logger.info("No full transformer weights found in checkpoint; loaded auxiliary weights only")

        logger.info("✅ Full model checkpoint loaded successfully")

    def _validate_full_checkpoint_metadata(self, checkpoint_path: Path) -> dict[str, str]:
        if self._config.training_strategy.name != "semantic_flow":
            return read_checkpoint_metadata(checkpoint_path)
        if not checkpoint_contains_semantic_flow_modules(checkpoint_path):
            return read_checkpoint_metadata(checkpoint_path)
        metadata = read_checkpoint_metadata(checkpoint_path)
        validate_reference_rope_checkpoint_metadata(
            metadata,
            expected_mode=self._training_strategy.config.reference_rope_mode,
            allow_legacy=False,
        )
        validate_semantic_flow_checkpoint_architecture(
            metadata,
            allow_v1_warm_start=True,
        )
        return metadata

    @staticmethod
    def _index_peft_adapter_state(
        state_dict: dict[str, Tensor],
        *,
        label: str,
    ) -> dict[str, tuple[str, Tensor]]:
        indexed: dict[str, tuple[str, Tensor]] = {}
        for key, value in state_dict.items():
            normalized = normalize_peft_adapter_key(key)
            if normalized in indexed:
                first_key = indexed[normalized][0]
                raise RuntimeError(
                    f"{label} contains duplicate normalized adapter key {normalized!r}: "
                    f"{first_key!r} and {key!r}"
                )
            indexed[normalized] = (key, value)
        return indexed

    @classmethod
    def _validate_peft_adapter_state(
        cls,
        model: torch.nn.Module,
        checkpoint_state: dict[str, Tensor],
        *,
        label: str,
    ) -> tuple[dict[str, tuple[str, Tensor]], dict[str, tuple[str, Tensor]]]:
        model = getattr(model, "module", model)
        expected_state = get_peft_model_state_dict(model)
        expected = cls._index_peft_adapter_state(expected_state, label=f"expected {label}")
        checkpoint = cls._index_peft_adapter_state(checkpoint_state, label=f"checkpoint {label}")
        expected_keys = set(expected)
        checkpoint_keys = set(checkpoint)
        missing = sorted(expected_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - expected_keys)
        if missing or unexpected:
            raise RuntimeError(
                f"Incomplete {label} checkpoint: missing={missing[:20]}, "
                f"unexpected={unexpected[:20]}"
            )

        shape_mismatches = [
            (
                normalized,
                tuple(checkpoint[normalized][1].shape),
                tuple(expected[normalized][1].shape),
            )
            for normalized in sorted(expected_keys)
            if checkpoint[normalized][1].shape != expected[normalized][1].shape
        ]
        if shape_mismatches:
            raise RuntimeError(f"{label} checkpoint shape mismatch: {shape_mismatches[:20]}")
        return expected, checkpoint

    @classmethod
    def _strict_load_peft_adapter_state(
        cls,
        model: torch.nn.Module,
        checkpoint_state: dict[str, Tensor],
        *,
        label: str,
    ) -> int:
        """Load one PEFT adapter only after exact key and shape validation."""
        model = getattr(model, "module", model)
        expected, checkpoint = cls._validate_peft_adapter_state(
            model,
            checkpoint_state,
            label=label,
        )
        expected_keys = set(expected)

        load_state = {
            expected[normalized][0]: checkpoint[normalized][1]
            for normalized in sorted(expected_keys)
        }
        try:
            load_result = set_peft_model_state_dict(model, load_state)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Failed to load the complete {label} checkpoint") from exc

        adapter_missing = [key for key in load_result.missing_keys if "lora_" in key]
        adapter_unexpected = [key for key in load_result.unexpected_keys if "lora_" in key]
        if adapter_missing or adapter_unexpected:
            raise RuntimeError(
                f"PEFT rejected part of {label}: missing={adapter_missing[:20]}, "
                f"unexpected={adapter_unexpected[:20]}"
            )

        loaded = cls._index_peft_adapter_state(
            get_peft_model_state_dict(model),
            label=f"loaded {label}",
        )
        different = [
            normalized
            for normalized in sorted(expected_keys)
            if not torch.equal(
                loaded[normalized][1].detach().cpu().to(checkpoint[normalized][1].dtype),
                checkpoint[normalized][1].detach().cpu(),
            )
        ]
        if different:
            raise RuntimeError(f"{label} tensors differ after loading: {different[:20]}")
        return len(expected_keys)

    @staticmethod
    def _validate_module_state(
        module: torch.nn.Module,
        checkpoint_state: dict[str, Tensor],
        *,
        label: str,
    ) -> None:
        if not checkpoint_state:
            raise RuntimeError(f"Checkpoint is missing {label}")
        expected_state = module.state_dict()
        missing = sorted(set(expected_state) - set(checkpoint_state))
        unexpected = sorted(set(checkpoint_state) - set(expected_state))
        if missing or unexpected:
            raise RuntimeError(
                f"Incomplete {label} checkpoint: missing={missing[:20]}, "
                f"unexpected={unexpected[:20]}"
            )
        shape_mismatches = [
            (key, tuple(checkpoint_state[key].shape), tuple(expected_state[key].shape))
            for key in sorted(expected_state)
            if checkpoint_state[key].shape != expected_state[key].shape
        ]
        if shape_mismatches:
            raise RuntimeError(f"{label} checkpoint shape mismatch: {shape_mismatches[:20]}")

    @classmethod
    def _strict_load_module_state(
        cls,
        module: torch.nn.Module,
        checkpoint_state: dict[str, Tensor],
        *,
        label: str,
    ) -> int:
        """Strictly load a regular module and verify the loaded tensor values."""
        cls._validate_module_state(module, checkpoint_state, label=label)
        try:
            module.load_state_dict(checkpoint_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to load the complete {label} checkpoint") from exc
        loaded_state = module.state_dict()
        different = [
            key
            for key in sorted(checkpoint_state)
            if not torch.equal(
                loaded_state[key].detach().cpu().to(checkpoint_state[key].dtype),
                checkpoint_state[key].detach().cpu(),
            )
        ]
        if different:
            raise RuntimeError(f"{label} tensors differ after loading: {different[:20]}")
        return len(checkpoint_state)

    def _load_lora_checkpoint(self, checkpoint_path: Path) -> None:
        """Load LoRA checkpoint with DDP/FSDP compatibility."""
        checkpoint_metadata = read_checkpoint_metadata(checkpoint_path)
        if self._config.training_strategy.name == "semantic_flow":
            validate_semantic_flow_checkpoint_architecture(
                checkpoint_metadata,
                allow_v1_warm_start=True,
            )
        state_dict = load_file(checkpoint_path)
        validate_initial = getattr(self._training_strategy, "validate_initial_checkpoint_state_dict", None)
        if callable(validate_initial):
            validate_initial(state_dict)
        self._load_auxiliary_checkpoint_state(
            state_dict,
            checkpoint_metadata=checkpoint_metadata,
        )

        # Adjust layer names to match internal format.
        # (Weights are saved in ComfyUI-compatible format, with "diffusion_model." prefix)
        state_dict = {
            key.replace("diffusion_model.", "", 1): value
            for key, value in state_dict.items()
            if key.startswith("diffusion_model.")
        }

        if not state_dict:
            logger.info("No LoRA weights found in checkpoint; loaded auxiliary weights only")
            return

        base_model = self._transformer.get_base_model()
        try:
            set_peft_model_state_dict(base_model, state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                "LoRA config does not match checkpoint. Use the original training config or matching "
                "rank/target_modules."
            ) from exc

        logger.info("✅ LoRA checkpoint loaded successfully")

    @staticmethod
    def _filter_auxiliary_checkpoint_state(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("training_strategy.")
            and not key.startswith("embeddings_processor.")
            and not key.startswith("text_encoder.")
        }

    def _load_auxiliary_checkpoint_state(
        self,
        state_dict: dict[str, Tensor],
        *,
        checkpoint_metadata: dict[str, str] | None = None,
    ) -> None:
        self._training_strategy.load_extra_checkpoint_state_dict(
            state_dict,
            checkpoint_metadata=checkpoint_metadata,
        )

        processor_state = {
            key.removeprefix("embeddings_processor."): value
            for key, value in state_dict.items()
            if key.startswith("embeddings_processor.")
        }
        if self._is_semantic_flow_phase2() and (checkpoint_metadata or {}).get(
            "training_phase"
        ) == "phase2":
            loaded = validate_and_load_phase2_bridge_state(
                self._embeddings_processor,
                state_dict,
            )
            self._training_strategy.phase2_loaded_bridge_key_count = loaded
            logger.info("✅ Strictly loaded %d Phase 2 conditioning-bridge tensors", loaded)
        elif self._is_semantic_flow_phase2():
            if processor_state:
                logger.info(
                    "Ignoring Phase 1 embeddings_processor weights during Phase 2 warm start; "
                    "using the base LTX-2.3 bridge"
                )
            else:
                logger.info("Phase 2 warm start is using the base LTX-2.3 bridge")
        elif processor_state:
            missing, unexpected = self._embeddings_processor.load_state_dict(processor_state, strict=False)
            if missing:
                logger.debug(f"Missing embeddings processor keys while loading auxiliary checkpoint: {missing}")
            if unexpected:
                logger.debug(f"Unexpected embeddings processor keys while loading auxiliary checkpoint: {unexpected}")
            logger.info("✅ Embeddings processor checkpoint loaded successfully")
        else:
            logger.info("No embeddings_processor.* weights found in checkpoint; using base connector weights.")

        text_encoder_state = {
            key.removeprefix("text_encoder."): value
            for key, value in state_dict.items()
            if key.startswith("text_encoder.")
        }
        if text_encoder_state:
            if self._text_encoder is None:
                logger.warning("Text encoder weights found in checkpoint but no text encoder is loaded")
            else:
                missing, unexpected = self._text_encoder.load_state_dict(text_encoder_state, strict=False)
                if missing:
                    logger.debug(f"Missing text encoder keys while loading auxiliary checkpoint: {missing}")
                if unexpected:
                    logger.debug(f"Unexpected text encoder keys while loading auxiliary checkpoint: {unexpected}")
                logger.info("✅ Text encoder checkpoint loaded successfully")

    def _load_legacy_auxiliary_checkpoint_state(self, state_dict: dict[str, Tensor]) -> None:
        connector_prefix = "embeddings_processor.video_connector."
        connector_state = {
            key.removeprefix(connector_prefix): value
            for key, value in state_dict.items()
            if key.startswith(connector_prefix)
        }
        self._strict_load_module_state(
            self._embeddings_processor.video_connector,
            connector_state,
            label="video connector",
        )
        logger.info("✅ Complete video connector checkpoint loaded successfully")

        gemma_prefix = "text_encoder.model.model.language_model."
        gemma_adapter_state = {
            key.removeprefix(gemma_prefix): value
            for key, value in state_dict.items()
            if key.startswith(gemma_prefix)
        }
        if self._text_encoder is None:
            raise RuntimeError("Legacy checkpoint contains Gemma LoRA weights but no text encoder is loaded")
        get_language_model = getattr(self._training_strategy, "_get_language_model", None)
        if not callable(get_language_model):
            raise RuntimeError("Legacy strategy does not expose its Gemma PEFT language model")
        self._strict_load_peft_adapter_state(
            get_language_model(),
            gemma_adapter_state,
            label="legacy Gemma LoRA",
        )
        logger.info("✅ Complete Gemma LoRA checkpoint loaded successfully")

    def _is_legacy_phase(self) -> bool:
        strategy_config = getattr(self._training_strategy, "config", None)
        return getattr(strategy_config, "legacy_phase", None) is not None

    def _is_strict_legacy_resume(self) -> bool:
        return self._is_legacy_phase() and not self._config.checkpoints.no_resume

    def _is_warm_legacy_resume(self) -> bool:
        checkpoints = self._config.checkpoints
        return (
            self._is_strict_legacy_resume()
            and checkpoints.save_training_state == "minimal"
            and getattr(checkpoints, "allow_legacy_warm_resume", False)
        )

    def _resolve_resume_state(self) -> tuple[int, TrainingState | None]:
        """Determine resume state by looking for a training state file next to the loaded checkpoint.
        Returns (initial_step, TrainingState or None).
        If no_resume config is set, no checkpoint loaded, or no state file found: returns (0, None).
        """
        if self._config.checkpoints.no_resume or self._loaded_checkpoint_path is None:
            return 0, None
        if getattr(self._training_strategy, "checkpoint_loaded_as_warm_start", False):
            logger.warning(
                "Checkpoint weights were warm-migrated to a new architecture; "
                "optimizer, scheduler, RNG, and global-step state will not be resumed."
            )
            return 0, None

        strict_legacy_resume = self._is_strict_legacy_resume()
        checkpoint_metadata: dict[str, str] | None = None
        if strict_legacy_resume:
            checkpoint_metadata = self._read_safetensors_metadata(self._loaded_checkpoint_path)
            if checkpoint_metadata.get("legacy_phase") is None:
                raise RuntimeError(self._legacy_resume_error_message())

        state = self._load_training_state(self._loaded_checkpoint_path)
        if state is None:
            if strict_legacy_resume:
                raise RuntimeError(self._legacy_resume_error_message())
            return 0, None

        if self._is_semantic_flow_phase2():
            metadata = self._read_safetensors_metadata(self._loaded_checkpoint_path)
            if metadata.get("training_phase") != "phase2":
                raise RuntimeError(
                    "Phase 2 exact resume requires a Phase 2 checkpoint; "
                    "use checkpoints.no_resume=true for a Phase 1 warm start"
                )
            accelerator_state_path = self._phase2_accelerator_state_path(
                self._loaded_checkpoint_path
            )
            if not accelerator_state_path.is_dir():
                raise RuntimeError(
                    "Phase 2 exact resume is missing its distributed Accelerate state: "
                    f"{accelerator_state_path}"
                )

        if strict_legacy_resume:
            assert checkpoint_metadata is not None
            self._validate_legacy_resume_state(
                checkpoint_path=self._loaded_checkpoint_path,
                metadata=checkpoint_metadata,
                state=state,
            )

        fp = state.config_fingerprint
        cfg = self._config
        mismatches: list[str] = []
        if fp.optimizer_type != cfg.optimization.optimizer_type:
            mismatches.append(f"optimizer_type: {fp.optimizer_type} → {cfg.optimization.optimizer_type}")
        if fp.scheduler_type != cfg.optimization.scheduler_type:
            mismatches.append(f"scheduler_type: {fp.scheduler_type} → {cfg.optimization.scheduler_type}")
        if fp.training_mode != cfg.model.training_mode:
            mismatches.append(f"training_mode: {fp.training_mode} → {cfg.model.training_mode}")
        if (
            cfg.model.training_mode == "lora"
            and cfg.lora is not None
            and fp.lora_rank is not None
            and fp.lora_rank != cfg.lora.rank
        ):
            mismatches.append(f"lora_rank: {fp.lora_rank} → {cfg.lora.rank}")
        if mismatches:
            if strict_legacy_resume:
                raise RuntimeError(
                    f"Legacy training state config mismatch: {', '.join(mismatches)}"
                )
            logger.warning(
                f"⚠️ Training state config mismatch ({', '.join(mismatches)}). "
                "Starting from step 0. Set checkpoints.no_resume=true to silence this warning."
            )
            return 0, None

        if state.global_step < 0:
            if strict_legacy_resume:
                raise RuntimeError(f"Legacy training state has invalid global_step={state.global_step!r}")
            logger.warning(
                f"⚠️ Training state has invalid global_step={state.global_step!r}. Starting from step 0."
            )
            return 0, None
        if self._is_warm_legacy_resume():
            logger.warning("Warm legacy resume: optimizer moments are reset.")
        logger.info(f"📌 Resuming from step {state.global_step}")
        return state.global_step, state

    @classmethod
    def _phase2_accelerator_state_path(cls, checkpoint_path: Path) -> Path:
        step = cls._checkpoint_step(checkpoint_path)
        if step is None:
            raise RuntimeError(
                f"Cannot resolve Phase 2 Accelerate state for {checkpoint_path.name}"
            )
        return checkpoint_path.parent / f"accelerator_state_step_{step:05d}"

    def _restore_phase2_accelerator_state(self, training_state: TrainingState) -> None:
        if self._loaded_checkpoint_path is None:
            raise RuntimeError("Phase 2 resume has no loaded checkpoint path")
        state_path = self._phase2_accelerator_state_path(self._loaded_checkpoint_path)
        try:
            self._accelerator.load_state(str(state_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to restore exact Phase 2 distributed state from {state_path}"
            ) from exc
        scheduler_epoch = (
            int(getattr(self._lr_scheduler, "last_epoch", -1))
            if self._lr_scheduler is not None
            else training_state.global_step
        )
        if scheduler_epoch != training_state.global_step:
            raise RuntimeError(
                "Phase 2 restored scheduler/global-step mismatch: "
                f"scheduler={scheduler_epoch}, state={training_state.global_step}"
            )
        group_names = [str(group.get("name", "")) for group in self._optimizer.param_groups]
        if group_names != ["dit_semantic", "conditioning_bridge"]:
            raise RuntimeError(
                f"Phase 2 restored optimizer groups are invalid: {group_names}"
            )
        logger.info(
            "Restored exact Phase 2 distributed state from %s at local step %d",
            state_path,
            training_state.global_step,
        )

    def _validate_legacy_resume_state(
        self,
        *,
        checkpoint_path: Path,
        metadata: dict[str, str],
        state: TrainingState,
    ) -> None:
        filename_step = self._checkpoint_step(checkpoint_path)
        if filename_step is None:
            raise RuntimeError(f"Cannot parse legacy resume step from checkpoint filename: {checkpoint_path.name}")
        metadata_step_raw = metadata.get("global_step")
        if metadata_step_raw is None:
            raise RuntimeError(
                "Legacy checkpoint metadata is missing global_step. "
                "This checkpoint predates strict resume metadata; migrate it by re-saving it with the updated trainer."
            )
        try:
            metadata_step = int(metadata_step_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid legacy checkpoint metadata global_step={metadata_step_raw!r}") from exc

        state_step = state.global_step
        if filename_step != metadata_step or filename_step != state_step:
            raise RuntimeError(
                "Legacy resume step mismatch:\n"
                f"filename={filename_step}, metadata={metadata_step}, training_state={state_step}"
            )
        if state_step < 0:
            raise RuntimeError(f"Legacy resume global_step must be non-negative, got {state_step}")
        if state_step >= self._config.optimization.steps:
            raise RuntimeError(
                f"Legacy resume global_step={state_step} must be less than "
                f"optimization.steps={self._config.optimization.steps}"
            )

        scheduler_state = state.lr_scheduler_state_dict
        if self._config.optimization.scheduler_type != "constant":
            if scheduler_state is None:
                raise RuntimeError("Legacy resume is missing LR scheduler state")
            scheduler_last_epoch = scheduler_state.get("last_epoch")
            if scheduler_last_epoch is None:
                raise RuntimeError("Legacy LR scheduler state is missing last_epoch")
            expected_scheduler_last_epoch = state_step
            if int(scheduler_last_epoch) != expected_scheduler_last_epoch:
                raise RuntimeError(
                    "Legacy scheduler/global_step mismatch: "
                    f"last_epoch={scheduler_last_epoch}, global_step={state_step}, "
                    f"expected_last_epoch={expected_scheduler_last_epoch}"
                )

        if not self._is_warm_legacy_resume() and state.optimizer_state_dict is None:
            raise RuntimeError("Exact legacy resume requires optimizer state in the matching training state file")

    @staticmethod
    def _checkpoint_step(path: Path) -> int | None:
        match = re.search(r"step_(\d+)", path.name)
        return int(match.group(1)) if match else None

    @staticmethod
    def _read_safetensors_metadata(checkpoint_path: Path) -> dict[str, str]:
        try:
            with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
                return checkpoint.metadata() or {}
        except Exception as exc:
            raise RuntimeError(f"Could not read checkpoint metadata from {checkpoint_path}") from exc

    @staticmethod
    def _legacy_resume_error_message() -> str:
        return (
            "Legacy resume requires a matching checkpoint and training state. "
            "Use no_resume=true when initializing from weights only."
        )

    @staticmethod
    def _load_training_state(checkpoint_path: Path) -> TrainingState | None:
        """Load training state file that corresponds to a checkpoint weights file."""
        match = re.search(r"step_(\d+)", checkpoint_path.name)
        if not match:
            return None

        step_str = match.group(1)
        state_path = checkpoint_path.parent / f"training_state_step_{step_str}.pt"

        if not state_path.exists():
            return None

        try:
            raw: dict = torch.load(state_path, map_location="cpu", weights_only=False)
            state = TrainingState.from_save_dict(raw)
            logger.info(f"📥 Loaded training state from {state_path}")
            return state
        except Exception as e:
            logger.warning(f"⚠️ Failed to load training state from {state_path}: {e}. Starting from step 0.")
            return None

    def _restore_training_state(self, training_state: TrainingState) -> bool:
        """Restore optimizer, scheduler, and RNG states from a loaded TrainingState.
        Must be called after _init_optimizer() (which calls accelerator.prepare).
        Returns True if restore succeeded, False if it failed (caller should fall back to step 0).
        """
        try:
            if training_state.optimizer_state_dict is not None:
                self._optimizer.load_state_dict(training_state.optimizer_state_dict)
                logger.debug("Restored optimizer state (full mode)")

            if training_state.lr_scheduler_state_dict is not None:
                if self._lr_scheduler is None:
                    raise RuntimeError("Training state contains an LR scheduler but no scheduler is configured")
                self._lr_scheduler.load_state_dict(training_state.lr_scheduler_state_dict)
                self._restore_optimizer_learning_rates(training_state.lr_scheduler_state_dict)
                logger.debug("Restored LR scheduler state")
        except Exception as e:
            logger.warning(f"⚠️ Failed to restore training state: {e}. Starting from step 0.")
            return False

        rng = training_state.rng_states
        if self._accelerator.num_processes > 1:
            logger.debug("Skipping RNG restore in multi-process mode (only main process state was saved)")
        else:
            if rng.torch_state is not None:
                torch.random.set_rng_state(rng.torch_state)
            if rng.cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(rng.cuda_state)
            logger.debug("Restored RNG states")

        return True

    def _restore_optimizer_learning_rates(self, scheduler_state: dict[str, Any]) -> None:
        saved_lrs = scheduler_state.get("_last_lr")
        if saved_lrs is None:
            return
        optimizer_groups = self._optimizer.param_groups
        if len(saved_lrs) != len(optimizer_groups):
            raise RuntimeError(
                "LR scheduler state has a different number of learning rates and optimizer parameter groups: "
                f"saved_lrs={len(saved_lrs)}, param_groups={len(optimizer_groups)}"
            )
        restored_lrs = [float(saved_lr) for saved_lr in saved_lrs]
        for parameter_group, saved_lr in zip(optimizer_groups, restored_lrs, strict=True):
            parameter_group["lr"] = saved_lr

        scheduler_lrs = [float(lr) for lr in self._lr_scheduler.get_last_lr()]
        optimizer_lrs = [float(group["lr"]) for group in optimizer_groups]
        if optimizer_lrs != scheduler_lrs:
            raise RuntimeError(
                "Restored optimizer learning rates do not match scheduler learning rates: "
                f"optimizer={optimizer_lrs}, scheduler={scheduler_lrs}"
            )
        logger.info(f"Resumed learning rates: {optimizer_lrs}")

    def _prepare_models_for_training(self) -> None:
        """Prepare models for training with Accelerate."""

        if self._accelerator.distributed_type == DistributedType.FSDP and not self._train_transformer:
            raise RuntimeError("Frozen-Transformer FSDP is not implemented. Use DDP or single GPU.")

        # For FSDP + LoRA: Cast entire model to FP32.
        # FSDP requires uniform dtype across all parameters in wrapped modules.
        # In LoRA mode, PEFT creates LoRA params in FP32 while base model is BF16.
        # We cast the base model to FP32 to match the LoRA params.
        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.model.training_mode == "lora":
            logger.debug("FSDP: casting transformer to FP32 for uniform dtype")
            self._transformer = self._transformer.to(dtype=torch.float32)

        # Enable gradient checkpointing if requested
        # For PeftModel, we need to access the underlying base model
        transformer = (
            self._transformer.get_base_model() if hasattr(self._transformer, "get_base_model") else self._transformer
        )

        transformer.set_gradient_checkpointing(self._config.optimization.enable_gradient_checkpointing)

        strategy_modules = self._training_strategy.get_trainable_modules()
        models_to_prepare = []
        if self._train_transformer:
            models_to_prepare.append(("transformer", self._transformer))
        else:
            self._transformer = self._transformer.to(self._accelerator.device)
            self._transformer.requires_grad_(False)
            self._transformer.eval()
        if self._train_embeddings_processor:
            models_to_prepare.extend(
                (f"embeddings_processor.{name}", module)
                for name, module in self._embeddings_processor_trainable_modules.items()
            )
        if self._train_text_encoder:
            get_text_module = getattr(self._training_strategy, "get_text_encoder_trainable_module", None)
            if callable(get_text_module):
                models_to_prepare.append(("text_encoder_trainable", get_text_module()))
            else:
                models_to_prepare.append(("text_encoder", self._text_encoder))
        models_to_prepare.extend((f"strategy.{name}", module) for name, module in strategy_modules.items())

        if not models_to_prepare:
            raise RuntimeError("No trainable modules were provided to accelerator.prepare()")

        if self._accelerator.distributed_type == DistributedType.FSDP:
            scalar_parameters = _find_scalar_trainable_parameters(models_to_prepare)
            if scalar_parameters:
                raise RuntimeError(
                    "FSDP does not support scalar trainable parameters: "
                    + ", ".join(scalar_parameters)
                )

        prepared_models = self._accelerator.prepare(*(module for _, module in models_to_prepare))
        if len(models_to_prepare) == 1:
            prepared_models = (prepared_models,)

        prepared_strategy_modules = {}
        prepared_processor_modules = {}
        for (name, _module), prepared_module in zip(models_to_prepare, prepared_models, strict=True):
            if name == "transformer":
                self._transformer = prepared_module
            elif name == "embeddings_processor":
                self._embeddings_processor = prepared_module
            elif name.startswith("embeddings_processor."):
                prepared_processor_modules[name.removeprefix("embeddings_processor.")] = prepared_module
            elif name == "text_encoder":
                self._text_encoder = prepared_module
            elif name == "text_encoder_trainable":
                set_text_module = getattr(self._training_strategy, "set_text_encoder_trainable_module", None)
                if not callable(set_text_module):
                    raise RuntimeError("Training strategy cannot receive its prepared text encoder module")
                set_text_module(prepared_module)
            elif name.startswith("strategy."):
                prepared_strategy_modules[name.removeprefix("strategy.")] = prepared_module

        if prepared_strategy_modules:
            self._training_strategy.set_trainable_modules(prepared_strategy_modules)
        if prepared_processor_modules:
            set_processor_modules = getattr(
                self._training_strategy,
                "set_embeddings_processor_trainable_modules",
                None,
            )
            if callable(set_processor_modules):
                set_processor_modules(
                    self._embeddings_processor,
                    prepared_processor_modules,
                )
            elif set(prepared_processor_modules) == {"video_connector"}:
                self._embeddings_processor.video_connector = prepared_processor_modules["video_connector"]
            else:
                raise RuntimeError(
                    "Training strategy cannot receive prepared embedding-processor modules: "
                    f"{sorted(prepared_processor_modules)}"
                )
            self._embeddings_processor_trainable_modules = prepared_processor_modules
        if self._train_text_encoder and self._text_encoder is not None:
            set_text_encoder = getattr(self._training_strategy, "set_text_encoder", None)
            if callable(set_text_encoder):
                set_text_encoder(self._text_encoder)

        self._accumulation_models = []
        if self._train_transformer:
            self._accumulation_models.append(self._transformer)
        if self._train_embeddings_processor:
            self._accumulation_models.extend(self._embeddings_processor_trainable_modules.values())
        if self._train_text_encoder:
            get_text_module = getattr(self._training_strategy, "get_text_encoder_trainable_module", None)
            self._accumulation_models.append(get_text_module() if callable(get_text_module) else self._text_encoder)
        self._accumulation_models.extend(self._training_strategy.get_trainable_modules().values())
        if not self._accumulation_models:
            raise RuntimeError("At least one trainable module is required for gradient accumulation")

        # Log GPU memory usage after model preparation
        vram_usage_gb = torch.cuda.memory_allocated() / 1024**3
        logger.debug(f"GPU memory usage after models preparation: {vram_usage_gb:.2f} GB")

    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        """Find the checkpoint file to load, handling both file and directory paths."""
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            if not checkpoint_path.suffix == ".safetensors":
                raise ValueError(f"Checkpoint file must have a .safetensors extension: {checkpoint_path}")
            return checkpoint_path

        if checkpoint_path.is_dir():
            # Look for checkpoint files in the directory
            checkpoints = list(checkpoint_path.rglob("*step_*.safetensors"))

            if not checkpoints:
                return None

            # Sort by step number and return the latest
            def _get_step_num(p: Path) -> int:
                try:
                    return int(p.stem.split("step_")[1])
                except (IndexError, ValueError):
                    return -1

            latest = max(checkpoints, key=_get_step_num)
            return latest

        else:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}. Must be a file or directory.")

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        if self._config.data.encoding_mode == "online":
            self._init_online_dataloader()
            return
        if self._dataset is None:
            # Get data sources from the training strategy
            data_sources = self._config.training_strategy.get_data_sources()

            self._dataset = PrecomputedDataset(self._config.data.preprocessed_data_root, data_sources=data_sources)
            logger.debug(f"Loaded dataset with {len(self._dataset):,} samples from sources: {list(data_sources)}")

        num_workers = self._config.data.num_dataloader_workers
        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_precomputed_batch,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
        )

        self._dataloader = self._accelerator.prepare(dataloader)

    def _init_online_dataloader(self) -> None:
        from ltx_trainer.online_data.distributed_multitask_sampler import (  # noqa: PLC0415
            DistributedMultiTaskMicrobatchSampler,
        )
        from ltx_trainer.online_data.manifest import load_multitask_data_config  # noqa: PLC0415
        from ltx_trainer.online_data.multitask_dataset import (  # noqa: PLC0415
            OnlineMultiTaskDataset,
            collate_online_raw_batch,
        )

        data_config = self._config.data
        online_config = data_config.online_encoding
        if online_config is None or data_config.manifest_path is None or data_config.train_data_config is None:
            raise RuntimeError("Online dataloader requires manifest_path, train_data_config, and online_encoding")
        if self._dataset is None:
            self._dataset = OnlineMultiTaskDataset(
                data_config.manifest_path,
                width=online_config.width,
                height=online_config.height,
                max_ref_images=online_config.max_ref_images,
                vlm_reference_preprocess=online_config.vlm_reference_preprocess,
                video_decoder=online_config.video_decoder,
                decode_timeout_seconds=online_config.decode_timeout_seconds,
                cpu_transform_chunk_frames=online_config.cpu_transform_chunk_frames,
            )

        source_config = load_multitask_data_config(data_config.train_data_config)
        sampling_config = source_config.get("online_sampling", {})
        if not isinstance(sampling_config, dict):
            raise ValueError("train_data_config online_sampling must be a mapping")
        image_ratio = float(sampling_config.get("image_ratio", online_config.image_ratio))
        video_ratio = float(sampling_config.get("video_ratio", online_config.video_ratio))
        if (
            abs(image_ratio - online_config.image_ratio) > 1.0e-8
            or abs(video_ratio - online_config.video_ratio) > 1.0e-8
        ):
            raise ValueError(
                "train_data_config online_sampling ratios do not match online_encoding: "
                f"data={image_ratio}/{video_ratio}, model={online_config.image_ratio}/{online_config.video_ratio}"
            )
        raw_source_ratios = sampling_config.get("video_source_ratios", {})
        if not isinstance(raw_source_ratios, dict):
            raise ValueError("online_sampling.video_source_ratios must be a mapping")
        video_source_ratios = {str(name): float(ratio) for name, ratio in raw_source_ratios.items()}

        raw_augmentation = source_config.get("online_augmentation")
        if raw_augmentation is not None:
            if not isinstance(raw_augmentation, dict):
                raise ValueError("train_data_config online_augmentation must be a mapping")
            from ltx_trainer.online_data.transforms import OnlineAugmentationConfig  # noqa: PLC0415

            online_config.augmentation = OnlineAugmentationConfig.from_mapping(raw_augmentation)

        self._online_sampler = DistributedMultiTaskMicrobatchSampler(
            self._dataset.task_indices,
            total_optimizer_steps=self._config.optimization.steps,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
            rank=self._accelerator.process_index,
            world_size=self._accelerator.num_processes,
            seed=self._config.seed,
            image_ratio=online_config.image_ratio,
            video_ratio=online_config.video_ratio,
            dataset_indices=self._dataset.dataset_indices,
            video_source_ratios=video_source_ratios,
        )
        if self._pending_online_data_state is not None:
            self._online_sampler.load_state_dict(self._pending_online_data_state)
            sampler_state = self._online_sampler.state_dict()
            if (
                sampler_state["task_schedule_cursor"] != self._resume_initial_step
                or sampler_state["microstep_in_optimizer_step"] != 0
            ):
                raise RuntimeError(
                    "Online sampler/training-state step mismatch: "
                    f"global_step={self._resume_initial_step}, "
                    f"task_schedule_cursor={sampler_state['task_schedule_cursor']}, "
                    f"microstep={sampler_state['microstep_in_optimizer_step']}"
                )
        elif self._resume_initial_step > 0:
            logger.warning(
                "Online resume state has no sampler fields; reconstructing the deterministic sampler "
                f"at optimizer step {self._resume_initial_step}."
            )
            self._online_sampler.seek_optimizer_step(self._resume_initial_step)

        workers = data_config.num_dataloader_workers
        dataloader_kwargs: dict[str, Any] = {
            "dataset": self._dataset,
            "batch_size": 1,
            "sampler": self._online_sampler,
            "shuffle": False,
            "drop_last": True,
            "collate_fn": collate_online_raw_batch,
            "num_workers": workers,
            "pin_memory": online_config.pin_memory,
            "persistent_workers": workers > 0,
        }
        if workers > 0:
            dataloader_kwargs["prefetch_factor"] = online_config.prefetch_factor
        # The sampler is already rank-aware. Preparing this DataLoader would
        # shard it a second time under Accelerate.
        self._dataloader = DataLoader(**dataloader_kwargs)

        image_steps = int(round(self._config.optimization.steps * online_config.image_ratio))
        video_steps = self._config.optimization.steps - image_steps
        effective_batch = self._effective_global_batch_size(
            batch_size=1,
            num_processes=self._accelerator.num_processes,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
        )
        logger.info(
            "Online multi-task schedule: "
            f"{image_steps} I2I optimizer steps, {video_steps} R2V optimizer steps, "
            f"effective global batch={effective_batch}, "
            f"total exposures={effective_batch * self._config.optimization.steps}."
        )

    def _prepare_online_batch_with_retry(self, initial_raw_batch: dict[str, Any]) -> dict[str, Any]:
        from ltx_trainer.online_data.multitask_dataset import (  # noqa: PLC0415
            SampleLoadError,
            collate_online_raw_batch,
        )
        from ltx_trainer.online_data.online_batch_encoder import OnlineSampleEncodeError  # noqa: PLC0415

        if self._online_batch_encoder is None or self._online_sampler is None:
            raise RuntimeError("Online batch encoder/sampler were not initialized")
        online_config = self._config.data.online_encoding
        if online_config is None:
            raise RuntimeError("online_encoding config is missing")

        raw_batch = initial_raw_batch
        last_error: Exception | SampleLoadError | None = None
        for attempt in range(online_config.runtime_max_retries + 1):
            raw_errors = raw_batch.get("sample_load_errors", [])
            manifest_indices = raw_batch.get("manifest_index")
            local_prefetch_collision = bool(
                isinstance(manifest_indices, Tensor)
                and any(
                    self._online_sampler.was_consumed_in_current_step(int(index))
                    for index in manifest_indices.flatten().tolist()
                )
            )
            local_raw_failed = bool(raw_errors) or local_prefetch_collision
            if self._synchronize_online_failure(local_raw_failed):
                if local_raw_failed:
                    if raw_errors:
                        last_error = raw_errors[0]
                        self._log_online_reject(raw_errors[0], attempt=attempt, phase="decode")
                    else:
                        last_error = RuntimeError(
                            "Prefetched normal sample was already consumed by a retry in this optimizer step"
                        )
                        self._log_online_reject(last_error, attempt=attempt, phase="prefetch_collision")
                raw_batch = self._load_online_retry_batch(attempt + 1, collate_online_raw_batch)
                continue

            encoded_batch = None
            encode_error: Exception | None = None
            try:
                sampler_state = self._online_sampler.state_dict()
                encoded_batch = self._online_batch_encoder.encode_for_strategy(
                    raw_batch,
                    strategy=self._training_strategy,
                    optimizer_step=sampler_state["task_schedule_cursor"],
                    microstep=sampler_state["microstep_in_optimizer_step"],
                    global_seed=self._config.seed,
                )
            except Exception as exc:  # synchronize before any rank enters the trainable graph
                if isinstance(exc, OnlineSampleEncodeError):
                    exc.attach_sample_context(raw_batch)
                encode_error = exc
            any_encode_failure = self._synchronize_online_failure(encode_error is not None)
            if not any_encode_failure:
                if encoded_batch is None:
                    raise RuntimeError("Online encoding returned no batch without reporting an error")
                return encoded_batch

            local_programming_failure = encode_error is not None and not isinstance(
                encode_error,
                OnlineSampleEncodeError,
            )
            any_programming_failure = self._synchronize_online_failure(local_programming_failure)
            if any_programming_failure:
                if local_programming_failure:
                    assert encode_error is not None
                    raise encode_error.with_traceback(encode_error.__traceback__)
                raise RuntimeError(
                    "Online encoding failed with a non-retryable error on a peer rank; "
                    "all ranks are stopping before entering the trainable graph."
                )
            if encode_error is not None:
                self._log_online_reject(encode_error, attempt=attempt, phase="encode")
            last_error = self._synchronize_online_retryable_error(encode_error)
            raw_batch = self._load_online_retry_batch(attempt + 1, collate_online_raw_batch)

        raise RuntimeError(
            f"Online data exceeded runtime_max_retries={online_config.runtime_max_retries}; "
            f"last error: {last_error}"
        )

    def _load_online_retry_batch(self, attempt: int, collate_fn: Callable) -> dict[str, Any]:
        retry_index = self._online_sampler.retry_index(attempt)
        return collate_fn([self._dataset[retry_index]])

    def _synchronize_online_failure(self, local_failed: bool) -> bool:
        flag = torch.tensor(
            [1 if local_failed else 0],
            dtype=torch.int32,
            device=self._accelerator.device,
        )
        reduced = self._accelerator.reduce(flag, reduction="max")
        return bool(reduced.item())

    def _synchronize_online_retryable_error(self, local_error: Exception | None) -> Exception:
        from ltx_trainer.online_data.online_batch_encoder import OnlineSampleEncodeError  # noqa: PLC0415

        if getattr(self._accelerator, "num_processes", 1) <= 1:
            if local_error is None:
                return OnlineSampleEncodeError(
                    "retryable online encoding failure was reported without a local exception",
                    reason="synchronized_peer_data_error",
                )
            return local_error

        payload = []
        if local_error is not None:
            payload.append(
                (
                    int(self._accelerator.process_index),
                    str(getattr(local_error, "reason", "online_sample_encode_error")),
                    str(local_error),
                )
            )
        gathered = sorted(gather_object(payload), key=lambda item: item[0])
        if not gathered:
            return OnlineSampleEncodeError(
                "all ranks observed a retryable failure but no rank supplied error details",
                reason="synchronized_peer_data_error",
            )
        rank, reason, message = gathered[0]
        return OnlineSampleEncodeError(
            f"rank {rank}: {message}",
            reason=reason,
        )

    def _log_online_reject(self, error: Any, *, attempt: int, phase: str) -> None:
        from ltx_trainer.online_data.path_safety import assert_write_path_allowed  # noqa: PLC0415

        online_config = self._config.data.online_encoding
        if online_config is None:
            return
        log_dir = (
            Path(online_config.runtime_reject_log_dir)
            if online_config.runtime_reject_log_dir is not None
            else Path(self._config.output_dir).parent / "logs"
        )
        log_path = assert_write_path_allowed(
            log_dir / f"runtime_rejected_rank_{self._accelerator.process_index}.jsonl"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(error, "to_dict"):
            payload = error.to_dict()
        else:
            payload = {"error_type": type(error).__name__, "message": str(error)}
        payload.update(
            {
                "attempt": attempt,
                "phase": phase,
                "global_step": self._global_step,
                "rank": int(self._accelerator.process_index),
            }
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights for the transformer."""
        logger.debug("Initializing LoRA weights...")
        for _, module in self._transformer.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.reset_lora_parameters(adapter_name="default", init_lora_weights=True)

    def _init_optimizer(self) -> None:
        """Initialize the optimizer and learning rate scheduler."""
        opt_cfg = self._config.optimization

        lr = opt_cfg.learning_rate
        optimizer_parameters: Any = self._trainable_params
        if self._is_semantic_flow_phase2():
            strategy_modules = self._training_strategy.get_trainable_modules()
            dit_semantic = self._deduplicate_parameters(
                [
                    *[parameter for parameter in self._transformer.parameters() if parameter.requires_grad],
                    *[
                        parameter
                        for module in strategy_modules.values()
                        for parameter in module.parameters()
                        if parameter.requires_grad
                    ],
                ]
            )
            conditioning_bridge = self._deduplicate_parameters(
                [
                    parameter
                    for module in self._embeddings_processor_trainable_modules.values()
                    for parameter in module.parameters()
                    if parameter.requires_grad
                ]
            )
            grouped_ids = {id(parameter) for parameter in dit_semantic + conditioning_bridge}
            if grouped_ids != {id(parameter) for parameter in self._trainable_params}:
                raise RuntimeError("Phase 2 optimizer groups do not cover the exact trainable parameter set")
            bridge_lr = opt_cfg.bridge_learning_rate or opt_cfg.learning_rate
            optimizer_parameters = [
                {
                    "name": "dit_semantic",
                    "params": dit_semantic,
                    "lr": opt_cfg.learning_rate,
                },
                {
                    "name": "conditioning_bridge",
                    "params": conditioning_bridge,
                    "lr": bridge_lr,
                },
            ]
        if opt_cfg.optimizer_type == "adamw":
            optimizer = AdamW(optimizer_parameters, lr=lr)
        elif opt_cfg.optimizer_type == "adamw8bit":
            # noinspection PyUnresolvedReferences
            from bitsandbytes.optim import AdamW8bit  # noqa: PLC0415

            optimizer = AdamW8bit(optimizer_parameters, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.optimizer_type}")

        lr_scheduler = self._create_scheduler(optimizer)

        # noinspection PyTypeChecker
        self._optimizer, self._lr_scheduler = self._accelerator.prepare(optimizer, lr_scheduler)
        if self._is_semantic_flow_phase2():
            logger.info(
                "Phase 2 optimizer learning rates: dit_semantic=%g conditioning_bridge=%g",
                opt_cfg.learning_rate,
                opt_cfg.bridge_learning_rate or opt_cfg.learning_rate,
            )

    def _optimizer_group_metrics(self) -> dict[str, float]:
        if not self._is_semantic_flow_phase2():
            return {}
        metrics: dict[str, float] = {}
        for group in self._optimizer.param_groups:
            name = str(group.get("name", ""))
            if name not in {"dit_semantic", "conditioning_bridge"}:
                continue
            parameters = [parameter for parameter in group["params"] if parameter.requires_grad]
            grad_square = sum(
                parameter.grad.detach().float().pow(2).sum()
                for parameter in parameters
                if parameter.grad is not None
            )
            grad_norm = float(torch.sqrt(grad_square).item()) if isinstance(grad_square, Tensor) else 0.0
            metrics[f"train/lr_{name}"] = float(group["lr"])
            metrics[f"train/grad_norm_{name}"] = grad_norm
            if name == "conditioning_bridge":
                parameter_square = sum(
                    parameter.detach().float().pow(2).sum()
                    for parameter in parameters
                )
                parameter_norm = (
                    float(torch.sqrt(parameter_square).item())
                    if isinstance(parameter_square, Tensor)
                    else 0.0
                )
                update_ratio = float(group["lr"]) * grad_norm / max(parameter_norm, 1.0e-12)
                metrics.update(
                    {
                        "train/bridge_grad_norm": grad_norm,
                        "train/bridge_parameter_norm": parameter_norm,
                        "train/bridge_update_ratio": update_ratio,
                        "train/update_ratio_conditioning_bridge": update_ratio,
                        "train/bridge_trainable_parameter_count": float(
                            self._optimizer_group_parameter_counts.get(name, 0)
                        ),
                    }
                )
        return metrics

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        """Create learning rate scheduler based on config."""
        scheduler_type = self._config.optimization.scheduler_type
        steps = self._config.optimization.steps
        params = self._config.optimization.scheduler_params or {}

        if scheduler_type is None:
            return None

        if scheduler_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=params.pop("start_factor", 1.0),
                end_factor=params.pop("end_factor", 0.1),
                total_iters=steps,
                **params,
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=steps,
                eta_min=params.pop("eta_min", 0),
                **params,
            )
        elif scheduler_type == "cosine_with_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.pop("T_0", steps // 4),
                T_mult=params.pop("T_mult", 1),
                eta_min=params.pop("eta_min", 5e-5),
                **params,
            )
        elif scheduler_type == "polynomial":
            scheduler = PolynomialLR(
                optimizer,
                total_iters=steps,
                power=params.pop("power", 1.0),
                **params,
            )
        elif scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=params.pop("step_size", steps // 2),
                gamma=params.pop("gamma", 0.1),
                **params,
            )
        elif scheduler_type == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler

    def _setup_accelerator(self) -> None:
        """Initialize the Accelerator with the appropriate settings."""

        # find_unused_parameters=True keeps DDP happy when LoRA targets a branch the forward
        # pass skips (e.g. audio LoRA with `with_audio: false`, or short module patterns like
        # "to_k" that match the audio branch unintentionally). It's a no-op for FSDP and
        # single-GPU runs. The probing cost is paid only on the first step.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        # All distributed setup (DDP/FSDP, number of processes, etc.) is controlled by
        # the user's Accelerate configuration (accelerate config / accelerate launch).
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
        )
        _enforce_semantic_flow_fsdp_runtime_safety(self._config, self._accelerator)

        logger.info(
            "Training runtime configuration:\n"
            f"Sharing strategy: {torch.multiprocessing.get_sharing_strategy()}\n"
            f"Visible GPU count: {torch.cuda.device_count()}\n"
            f"World size: {self._accelerator.num_processes}\n"
            f"Checkpoint interval: {self._config.checkpoints.interval}\n"
            f"Checkpoint keep_last_n: {self._config.checkpoints.keep_last_n}"
        )

        if self._accelerator.num_processes > 1:
            logger.info(
                f"{self._accelerator.distributed_type.value} distributed training enabled "
                f"with {self._accelerator.num_processes} processes"
            )

        local_batch = self._config.optimization.batch_size
        accumulation_steps = self._config.optimization.gradient_accumulation_steps
        global_batch = self._effective_global_batch_size(
            batch_size=local_batch,
            num_processes=self._accelerator.num_processes,
            gradient_accumulation_steps=accumulation_steps,
        )
        logger.info(
            "Training batch configuration:\n"
            f"Local micro batch size: {local_batch}\n"
            f"Gradient accumulation steps: {accumulation_steps}\n"
            f"Number of processes: {self._accelerator.num_processes}\n"
            f"Effective global batch size: {global_batch}"
        )

        # Log torch.compile status from Accelerate's dynamo plugin
        is_compile_enabled = (
            hasattr(self._accelerator.state, "dynamo_plugin") and self._accelerator.state.dynamo_plugin.backend != "NO"
        )
        if is_compile_enabled:
            plugin = self._accelerator.state.dynamo_plugin
            logger.info(f"🔥 torch.compile enabled via Accelerate: backend={plugin.backend}, mode={plugin.mode}")

            if self._accelerator.distributed_type == DistributedType.FSDP:
                logger.warning(
                    "⚠️ FSDP + torch.compile is experimental and may hang on the first training iteration. "
                    "If this occurs, disable torch.compile by removing dynamo_config from your Accelerate config."
                )

        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.acceleration.quantization:
            logger.warning(
                f"FSDP with quantization ({self._config.acceleration.quantization}) may have compatibility issues."
                "Monitor training stability and consider disabling quantization if issues arise."
            )

    @contextlib.contextmanager
    def _offloaded_optimizer_state(self) -> Iterator[None]:
        """Context manager that offloads optimizer state to CPU during validation.
        Opt-in via `acceleration.offload_optimizer_during_validation`. Frees VRAM for
        validation video generation when optimizer state is large (e.g. full fine-tune
        AdamW, high-rank LoRA). No-op for FSDP (sharded state -- manual `.cpu()` breaks
        metadata).
        """
        enabled = (
            self._config.acceleration.offload_optimizer_during_validation
            and self._accelerator.distributed_type != DistributedType.FSDP
        )

        # Track exactly which tensors we move so we don't promote ones that were
        # intentionally on CPU (e.g. AdamW's `step` scalar on recent PyTorch).
        offloaded: list[tuple[dict, str]] = []
        if enabled:
            offloaded_bytes = 0
            for state in self._optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        offloaded.append((state, k))
                        offloaded_bytes += v.nbytes
            if offloaded:
                logger.info(f"Offloading optimizer state to CPU ({offloaded_bytes / 1e9:.1f} GB)")
                for state, k in offloaded:
                    state[k] = state[k].cpu()

        try:
            yield
        finally:
            device = self._accelerator.device
            for state, k in offloaded:
                state[k] = state[k].to(device)

    def _run_validation(self, progress: TrainingProgress) -> list[Path]:
        """Run distributed validation by delegating to the ValidationRunner.
        Each rank generates its assigned subset of validation samples (round-robin by
        `process_index`/`num_processes`), so all GPUs stay busy and no rank idles long
        enough to trigger NCCL timeouts. Paths are gathered across ranks so rank 0 has
        the full list for W&B logging.
        Under FSDP with multiple processes, ranks pad with extra generate passes
        (same sample, no disk write) so every rank runs the same number of forwards --
        avoids collective mismatch.
        Note: Multi-node training requires a shared filesystem so rank 0 can read
        videos written by other ranks.
        """
        self._optimizer.zero_grad(set_to_none=True)
        free_gpu_memory()

        num_samples = len(self._config.validation.samples)
        if num_samples == 0:
            return []

        rank = self._accelerator.process_index
        world_size = self._accelerator.num_processes

        rank_indices = list(range(rank, num_samples, world_size))
        work_items: list[tuple[int, bool]] = [(i, True) for i in rank_indices]
        if self._accelerator.distributed_type == DistributedType.FSDP and world_size > 1:
            # FSDP forwards run collective ops; pad short ranks with no-save duplicates so
            # every rank executes the same number of forwards. A rank with empty
            # rank_indices (world_size > num_samples) still pads with sample 0 to stay in
            # sync with the others.
            max_per_rank = math.ceil(num_samples / world_size)
            pad_seed = rank_indices[-1] if rank_indices else 0
            work_items += [(pad_seed, False)] * (max_per_rank - len(work_items))

        # W&B logging is handled by the trainer (after gathering across ranks),
        # so we always pass wandb_run=None to the runner.
        sampled = self._validation_runner.run(
            transformer=self._transformer,
            step=self._global_step,
            output_dir=Path(self._config.output_dir),
            device=self._accelerator.device,
            progress=progress,
            wandb_run=None,
            work_items=work_items,
        )

        if world_size > 1:
            sampled = sorted(gather_object(sampled), key=lambda x: x[0])

        paths = [p for _, p in sampled]

        if (
            self._accelerator.is_main_process
            and paths
            and self._config.wandb.log_validation_videos
            and self._wandb_run is not None
        ):
            self._validation_runner.log_to_wandb(self._wandb_run, paths, self._global_step)

        # Non-main ranks must not reach checkpoint collectives while main is still logging to W&B.
        self._accelerator.wait_for_everyone()

        return paths

    @staticmethod
    def _effective_global_batch_size(
        *,
        batch_size: int,
        num_processes: int,
        gradient_accumulation_steps: int,
    ) -> int:
        return batch_size * num_processes * gradient_accumulation_steps

    @staticmethod
    def _step_lr_scheduler(lr_scheduler: Any | None, *, sync_gradients: bool) -> None:
        """Advance the scheduler once at a real optimizer-step boundary."""
        if lr_scheduler is not None and sync_gradients:
            lr_scheduler.step()

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        """Log training statistics."""
        stats_str = (
            "📊 Training Statistics:\n"
            f" - Total time: {stats.total_time_seconds / 60:.1f} minutes\n"
            f" - Training speed: {stats.steps_per_second:.2f} steps/second\n"
            f" - Samples/second: {stats.samples_per_second:.2f}\n"
            f" - Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB\n"
            f" - Local micro batch size: {stats.local_batch_size}\n"
            f" - Gradient accumulation steps: {stats.gradient_accumulation_steps}\n"
            f" - Number of processes: {stats.num_processes}\n"
            f" - Effective global batch size: {stats.global_batch_size}"
        )
        logger.info(stats_str)

    def _save_checkpoint(self) -> Path | None:
        """Save the model weights."""
        is_lora = self._config.model.training_mode == "lora"
        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP

        # Prepare paths
        save_dir = Path(self._config.output_dir) / "checkpoints"
        prefix = "lora" if is_lora else "model"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename

        if (
            self._last_saved_step == self._global_step
            and self._last_saved_weights_path is not None
            and self._last_saved_weights_path.is_file()
        ):
            logger.debug(
                f"Checkpoint for step {self._global_step} already exists; skipping duplicate save"
            )
            return self._last_saved_weights_path if IS_MAIN_PROCESS else None

        # Get state dict (collective operation - all processes must participate)
        self._accelerator.wait_for_everyone()
        full_state_dict = self._accelerator.get_state_dict(self._transformer)
        strategy_checkpoint_states = self._collect_strategy_checkpoint_state_dicts()
        processor_checkpoint_states = self._collect_embeddings_processor_checkpoint_state_dicts()
        self._save_phase2_accelerator_state(save_dir)

        if not IS_MAIN_PROCESS:
            self._last_saved_step = self._global_step
            self._last_saved_weights_path = saved_weights_path
            return None

        save_dir.mkdir(exist_ok=True, parents=True)
        # A same-step republish must not leave a stale marker pointing at new bytes with an old hash.
        (save_dir / f"checkpoint_step_{self._global_step:05d}.ready.json").unlink(
            missing_ok=True
        )

        # Determine save precision
        save_dtype = torch.bfloat16 if self._config.checkpoints.precision == "bfloat16" else torch.float32
        auxiliary_state_dict = self._collect_auxiliary_checkpoint_state(
            save_dtype,
            precollected_strategy_states=strategy_checkpoint_states,
            precollected_processor_states=processor_checkpoint_states,
        )

        # For LoRA: extract only adapter weights; for full: use as-is
        checkpoint_metadata = self._build_checkpoint_metadata()
        if is_lora:
            unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
            # For FSDP, pass full_state_dict since model params aren't directly accessible
            state_dict = get_peft_model_state_dict(unwrapped, state_dict=full_state_dict if is_fsdp else None)

            # Remove "base_model.model." prefix added by PEFT
            state_dict = {k.replace("base_model.model.", "", 1): v for k, v in state_dict.items()}

            # Convert to ComfyUI-compatible format (add "diffusion_model." prefix)
            state_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items()}

            # Cast to configured precision
            state_dict = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in state_dict.items()}
            state_dict.update(auxiliary_state_dict)

            validate_checkpoint = getattr(self._training_strategy, "validate_checkpoint_state_dict", None)
            if callable(validate_checkpoint):
                validate_checkpoint(state_dict)

            # Publish only after the complete safetensors file passes structural validation.
            self._atomic_save_safetensors(
                state_dict,
                saved_weights_path,
                metadata=checkpoint_metadata,
                required_prefixes=(),
            )
        else:
            # Cast to configured precision
            full_state_dict = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in full_state_dict.items()}
            full_state_dict.update(auxiliary_state_dict)

            required_prefixes = tuple(
                f"training_strategy.{name}."
                for name in self._training_strategy.get_trainable_modules()
            )
            if self._is_semantic_flow_phase2():
                required_prefixes += (
                    "embeddings_processor.feature_extractor.",
                    "embeddings_processor.video_connector.",
                )
            self._atomic_save_safetensors(
                full_state_dict,
                saved_weights_path,
                metadata=checkpoint_metadata,
                required_prefixes=required_prefixes,
            )

        rel_path = saved_weights_path.relative_to(self._config.output_dir)
        logger.info(f"💾 {prefix.capitalize()} weights for step {self._global_step} saved in {rel_path}")

        training_state_path = self._save_training_state(save_dir)
        if training_state_path is not None:
            self._publish_checkpoint_ready_marker(
                checkpoint_path=saved_weights_path,
                training_state_path=training_state_path,
                metadata=checkpoint_metadata,
            )

        self._last_saved_step = self._global_step
        self._last_saved_weights_path = saved_weights_path
        if saved_weights_path not in self._checkpoint_paths:
            self._checkpoint_paths.append(saved_weights_path)
        self._cleanup_checkpoints()

        return saved_weights_path

    @staticmethod
    def _fsync_file(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _atomic_save_safetensors(
        self,
        state_dict: dict[str, Tensor],
        final_path: Path,
        *,
        metadata: dict[str, str],
        required_prefixes: tuple[str, ...],
    ) -> None:
        temporary_path = Path(f"{final_path}.tmp.{os.getpid()}")
        try:
            save_file(state_dict, temporary_path, metadata=metadata)
            self._fsync_file(temporary_path)
            if temporary_path.stat().st_size <= 0:
                raise RuntimeError(f"Checkpoint temporary file is empty: {temporary_path}")
            with safe_open(str(temporary_path), framework="pt", device="cpu") as checkpoint:
                saved_metadata = dict(checkpoint.metadata() or {})
                keys = list(checkpoint.keys())
            expected_step = str(self._global_step)
            if saved_metadata.get("global_step") != expected_step:
                raise RuntimeError(
                    "Checkpoint global_step validation failed: "
                    f"expected={expected_step}, actual={saved_metadata.get('global_step')!r}"
                )
            missing_prefixes = [
                prefix
                for prefix in required_prefixes
                if not any(key.startswith(prefix) for key in keys)
            ]
            if missing_prefixes:
                raise RuntimeError(
                    f"Checkpoint is missing required component prefixes: {missing_prefixes}"
                )
            os.replace(temporary_path, final_path)
            self._fsync_directory(final_path.parent)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def _publish_checkpoint_ready_marker(
        self,
        *,
        checkpoint_path: Path,
        training_state_path: Path,
        metadata: dict[str, str],
    ) -> Path:
        if not checkpoint_path.is_file() or not training_state_path.is_file():
            raise RuntimeError(
                "Cannot publish checkpoint ready marker before weights and training state exist"
            )
        marker_path = checkpoint_path.parent / f"checkpoint_step_{self._global_step:05d}.ready.json"
        payload = {
            "global_step": self._global_step,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "training_state_path": str(training_state_path.resolve()),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": self._sha256_file(checkpoint_path),
            "metadata_architecture": metadata.get("architecture"),
            "metadata_global_step": metadata.get("global_step"),
            "config_path": str((Path(self._config.output_dir) / "training_config.yaml").resolve()),
        }
        if self._is_semantic_flow_phase2():
            payload.update(
                {
                    "training_phase": "phase2",
                    "parent_checkpoint_step": int(metadata["parent_checkpoint_step"]),
                    "effective_total_step": int(metadata["effective_total_step"]),
                    "bridge_key_count": int(metadata["bridge_key_count"]),
                    "bridge_parameter_count": int(metadata["bridge_parameter_count"]),
                    "accelerator_state_path": str(
                        self._last_phase2_accelerator_state_path.resolve()
                    ),
                }
            )
        temporary_path = Path(f"{marker_path}.tmp.{os.getpid()}")
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, marker_path)
            self._fsync_directory(marker_path.parent)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return marker_path

    def _collect_strategy_checkpoint_state_dicts(self) -> dict[str, dict[str, Tensor]]:
        """Collect strategy module state on every rank before main-only checkpoint writes."""
        return {
            name: self._accelerator.get_state_dict(module)
            for name, module in self._training_strategy.get_trainable_modules().items()
        }

    def _collect_embeddings_processor_checkpoint_state_dicts(
        self,
    ) -> dict[str, dict[str, Tensor]]:
        if not self._train_embeddings_processor:
            return {}
        return {
            name: self._accelerator.get_state_dict(module)
            for name, module in self._embeddings_processor_trainable_modules.items()
        }

    def _save_phase2_accelerator_state(self, save_dir: Path) -> Path | None:
        if (
            not self._is_semantic_flow_phase2()
            or self._config.checkpoints.save_training_state == "off"
        ):
            return None
        state_path = save_dir / f"accelerator_state_step_{self._global_step:05d}"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        self._accelerator.save_state(
            output_dir=str(state_path),
            safe_serialization=True,
        )
        self._accelerator.wait_for_everyone()
        if IS_MAIN_PROCESS and not state_path.is_dir():
            raise RuntimeError(
                f"Phase 2 distributed Accelerate state was not created: {state_path}"
            )
        self._last_phase2_accelerator_state_path = state_path
        return state_path

    def _collect_auxiliary_checkpoint_state(
        self,
        save_dtype: torch.dtype,
        *,
        precollected_strategy_states: dict[str, dict[str, Tensor]] | None = None,
        precollected_processor_states: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        state_dict = self._training_strategy.get_extra_checkpoint_state_dict(
            self._accelerator,
            precollected_states=precollected_strategy_states,
        )

        if self._train_embeddings_processor:
            processor_state = self._collect_trainable_embeddings_processor_state(
                precollected_states=precollected_processor_states,
            )
            self._last_bridge_checkpoint_key_count = len(processor_state)
            state_dict.update({f"embeddings_processor.{key}": value for key, value in processor_state.items()})
        if self._train_text_encoder and self._text_encoder is not None:
            text_encoder_state = self._collect_trainable_text_encoder_state()
            state_dict.update({f"text_encoder.{key}": value for key, value in text_encoder_state.items()})

        return {key: value.to(save_dtype) if isinstance(value, Tensor) else value for key, value in state_dict.items()}

    def _collect_trainable_embeddings_processor_state(
        self,
        *,
        precollected_states: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        collected: dict[str, Tensor] = {}
        for module_name, module in self._embeddings_processor_trainable_modules.items():
            unwrapped = self._accelerator.unwrap_model(module, keep_torch_compile=False)
            trainable_names = {
                name
                for name, parameter in unwrapped.named_parameters()
                if parameter.requires_grad
            }
            full_state = (
                (precollected_states or {}).get(module_name)
                if precollected_states is not None
                else None
            )
            if full_state is None:
                full_state = self._accelerator.get_state_dict(module)
            selected = {
                key: value
                for key, value in full_state.items()
                if key in trainable_names
            }
            missing = sorted(trainable_names - set(selected))
            if missing:
                raise RuntimeError(
                    f"Trainable embedding-processor state is incomplete for {module_name}: {missing[:20]}"
                )
            collected.update(
                {
                    f"{module_name}.{key}": value
                    for key, value in selected.items()
                }
            )
        return collected

    def _collect_trainable_text_encoder_state(self) -> dict[str, Tensor]:
        get_strategy_state = getattr(self._training_strategy, "get_text_encoder_checkpoint_state_dict", None)
        if callable(get_strategy_state):
            return get_strategy_state(self._accelerator)
        unwrapped = self._accelerator.unwrap_model(self._text_encoder, keep_torch_compile=False)
        trainable_names = {name for name, param in unwrapped.named_parameters() if param.requires_grad}
        full_state = self._accelerator.get_state_dict(self._text_encoder)
        return {key: value for key, value in full_state.items() if key in trainable_names}

    @staticmethod
    def _checkpoint_substate(state_dict: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
        return {
            key.removeprefix(prefix): value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }

    def _validate_legacy_checkpoint_components(self, state_dict: dict[str, Tensor]) -> None:
        """Reject incomplete legacy adapter checkpoints before they are written."""
        transformer = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
        self._validate_peft_adapter_state(
            transformer,
            self._checkpoint_substate(state_dict, "diffusion_model."),
            label="legacy DiT LoRA",
        )

        get_language_model = getattr(self._training_strategy, "_get_language_model", None)
        if not callable(get_language_model):
            raise RuntimeError("Legacy strategy does not expose its Gemma PEFT language model")
        language_model = self._accelerator.unwrap_model(
            get_language_model(),
            keep_torch_compile=False,
        )
        self._validate_peft_adapter_state(
            language_model,
            self._checkpoint_substate(
                state_dict,
                "text_encoder.model.model.language_model.",
            ),
            label="legacy Gemma LoRA",
        )

        strategy_modules = self._training_strategy.get_trainable_modules()
        required_strategy_modules = {
            "semantic_query": "semantic query initializer",
            "semantic_encoder": "semantic encoder",
            "semantic_reconstruction_decoder": "semantic reconstruction decoder",
        }
        for module_name, label in required_strategy_modules.items():
            module = strategy_modules.get(module_name)
            if module is None:
                raise RuntimeError(f"Legacy strategy is missing required module {module_name}")
            module = self._accelerator.unwrap_model(module, keep_torch_compile=False)
            self._validate_module_state(
                module,
                self._checkpoint_substate(
                    state_dict,
                    f"training_strategy.{module_name}.",
                ),
                label=label,
            )

        connector = self._accelerator.unwrap_model(
            self._embeddings_processor.video_connector,
            keep_torch_compile=False,
        )
        self._validate_module_state(
            connector,
            self._checkpoint_substate(
                state_dict,
                "embeddings_processor.video_connector.",
            ),
            label="video connector",
        )

    def _cleanup_checkpoints(self) -> None:
        """Scan disk and clean checkpoint/training-state pairs by step."""
        save_dir = Path(self._config.output_dir) / "checkpoints"
        if not save_dir.is_dir():
            self._checkpoint_paths = []
            self._training_state_paths = []
            return

        loaded_path = getattr(self, "_loaded_checkpoint_path", None)
        loaded_resolved = loaded_path.resolve() if loaded_path is not None else None
        weights_by_step: dict[int, list[Path]] = {}
        for path in save_dir.glob("*_weights_step_*.safetensors"):
            step = self._checkpoint_step(path)
            if step is not None:
                weights_by_step.setdefault(step, []).append(path)
        states_by_step: dict[int, Path] = {}
        for path in save_dir.glob("training_state_step_*.pt"):
            step = self._checkpoint_step(path)
            if step is not None:
                states_by_step[step] = path
        markers_by_step: dict[int, Path] = {}
        for path in save_dir.glob("checkpoint_step_*.ready.json"):
            step = self._checkpoint_step(path)
            if step is not None:
                markers_by_step[step] = path
        accelerator_states_by_step: dict[int, Path] = {}
        for path in save_dir.glob("accelerator_state_step_*"):
            step = self._checkpoint_step(path)
            if step is not None and path.is_dir():
                accelerator_states_by_step[step] = path

        # Keep exactly one weight path per step, preferring the currently loaded path.
        selected_weights: dict[int, Path] = {}
        for step, paths in weights_by_step.items():
            unique = {path.resolve(): path for path in paths}
            selected = next(
                (path for resolved, path in unique.items() if resolved == loaded_resolved),
                sorted(unique.values())[0],
            )
            selected_weights[step] = selected
            for path in unique.values():
                if path != selected:
                    path.unlink(missing_ok=True)
                    logger.info(f"Removed duplicate checkpoint for step {step}: {path}")

        ordered_steps = sorted(selected_weights)
        keep_n = self._config.checkpoints.keep_last_n
        retained_steps = set(ordered_steps if keep_n <= 0 else ordered_steps[-keep_n:])
        loaded_step = next(
            (
                step
                for step, path in selected_weights.items()
                if loaded_resolved is not None and path.resolve() == loaded_resolved
            ),
            None,
        )
        if keep_n > 0 and loaded_step is not None and loaded_step not in retained_steps:
            if retained_steps:
                retained_steps.remove(min(retained_steps))
            retained_steps.add(loaded_step)

        for step in ordered_steps:
            if step in retained_steps:
                continue
            checkpoint_path = selected_weights[step]
            if loaded_resolved is not None and checkpoint_path.resolve() == loaded_resolved:
                continue
            checkpoint_path.unlink(missing_ok=True)
            logger.info(f"Removed old checkpoint: {checkpoint_path}")
            state_path = states_by_step.pop(step, None)
            if state_path is not None:
                state_path.unlink(missing_ok=True)
                logger.debug(f"Removed matching training state: {state_path}")
            marker_path = markers_by_step.pop(step, None)
            if marker_path is not None:
                marker_path.unlink(missing_ok=True)
                logger.debug(f"Removed matching checkpoint ready marker: {marker_path}")
            accelerator_state_path = accelerator_states_by_step.pop(step, None)
            if accelerator_state_path is not None:
                shutil.rmtree(accelerator_state_path)
                logger.debug(
                    f"Removed matching Phase 2 Accelerate state: {accelerator_state_path}"
                )

        remaining_weights = {
            step: path
            for step, path in selected_weights.items()
            if path.is_file()
        }
        for step, state_path in list(states_by_step.items()):
            if step not in remaining_weights:
                state_path.unlink(missing_ok=True)
                states_by_step.pop(step)
                logger.debug(f"Removed orphan training state: {state_path}")
        for step, marker_path in list(markers_by_step.items()):
            if step not in remaining_weights or step not in states_by_step:
                marker_path.unlink(missing_ok=True)
                markers_by_step.pop(step)
                logger.debug(f"Removed orphan checkpoint ready marker: {marker_path}")
        for step, accelerator_state_path in list(accelerator_states_by_step.items()):
            if step not in remaining_weights or step not in states_by_step:
                shutil.rmtree(accelerator_state_path)
                accelerator_states_by_step.pop(step)
                logger.debug(
                    f"Removed orphan Phase 2 Accelerate state: {accelerator_state_path}"
                )

        self._checkpoint_paths = [remaining_weights[step] for step in sorted(remaining_weights)]
        self._training_state_paths = [
            states_by_step[step]
            for step in sorted(states_by_step)
            if states_by_step[step].is_file()
        ]

    def _save_training_state(self, save_dir: Path) -> Path | None:
        """Save training state alongside checkpoint for resume.
        Respects checkpoints.save_training_state config:
        - "full": optimizer + scheduler + RNG + step
        - "minimal": scheduler + RNG + step only
        - "off": skip entirely
        """
        if not IS_MAIN_PROCESS:
            return None

        mode = self._config.checkpoints.save_training_state
        if mode == "off":
            return None

        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP

        optimizer_state = None
        if mode == "full":
            if is_fsdp:
                logger.warning(
                    "⚠️ save_training_state='full' is not supported with FSDP. "
                    "Saving 'minimal' state (scheduler + RNG only)."
                )
            else:
                optimizer_state = self._optimizer.state_dict()

        state = TrainingState(
            global_step=self._global_step,
            config_fingerprint=ConfigFingerprint(
                optimizer_type=self._config.optimization.optimizer_type,
                scheduler_type=self._config.optimization.scheduler_type,
                training_mode=self._config.model.training_mode,
                lora_rank=self._config.lora.rank if self._config.lora is not None else None,
            ),
            rng_states=RngStates(
                torch_state=torch.random.get_rng_state(),
                cuda_state=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            ),
            lr_scheduler_state_dict=self._lr_scheduler.state_dict() if self._lr_scheduler is not None else None,
            optimizer_state_dict=optimizer_state,
            wandb_run_id=self._wandb_run.id if self._wandb_run is not None else None,
            data_state=(
                self._online_sampler.state_dict()
                if self._config.data.encoding_mode == "online" and self._online_sampler is not None
                else None
            ),
        )

        state_path = save_dir / f"training_state_step_{self._global_step:05d}.pt"
        tmp_path = Path(f"{state_path}.tmp.{os.getpid()}")
        try:
            torch.save(state.to_save_dict(), tmp_path)
            self._fsync_file(tmp_path)
            if tmp_path.stat().st_size <= 0:
                raise RuntimeError(f"Training-state temporary file is empty: {tmp_path}")
            os.replace(tmp_path, state_path)
            self._fsync_directory(state_path.parent)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        file_size_gb = state_path.stat().st_size / (1024**3)
        if file_size_gb > 1.0 and not self._training_state_size_warned:
            self._training_state_size_warned = True
            if self._is_legacy_phase() and mode == "full":
                logger.warning(
                    f"⚠️ Training state file is {file_size_gb:.1f} GB. Full optimizer state is required "
                    "for exact legacy resume; use warm resume only if resetting Adam moments "
                    "is intentional."
                )
            else:
                logger.warning(
                    f"⚠️ Training state file is {file_size_gb:.1f} GB (full mode includes optimizer state). "
                    f'Set checkpoints.save_training_state="minimal" to save only scheduler/RNG/step (~few KB), '
                    f'or "off" to disable entirely.'
                )

        if not self._training_state_paths or self._training_state_paths[-1] != state_path:
            self._training_state_paths.append(state_path)

        rel_path = state_path.relative_to(self._config.output_dir)
        logger.debug(f"Training state saved to {rel_path}")
        return state_path

    def _cleanup_training_states(self) -> None:
        """Compatibility wrapper; checkpoint cleanup now manages paired state files."""
        self._cleanup_checkpoints()

    def _build_checkpoint_metadata(self) -> dict[str, str]:
        """Build metadata dictionary for safetensors checkpoint.
        Delegates to the training strategy to get strategy-specific metadata
        that downstream inference pipelines may need.
        Returns:
            Dictionary of string key-value pairs for safetensors metadata.
            Values are converted to strings for safetensors compatibility.
        """
        raw_metadata = self._training_strategy.get_checkpoint_metadata()
        raw_metadata["global_step"] = self._global_step
        if self._is_semantic_flow_phase2():
            parent_sha = self._training_strategy.phase2_parent_checkpoint_sha256
            if not parent_sha:
                raise RuntimeError("Phase 2 checkpoint metadata is missing the parent SHA256")
            if self._last_bridge_checkpoint_key_count <= 0:
                raise RuntimeError("Phase 2 checkpoint collected zero bridge tensors")
            parent_step = int(self._training_strategy.config.parent_checkpoint_step)
            raw_metadata.update(
                {
                    "dit_semantic_initial_lr": self._config.optimization.learning_rate,
                    "conditioning_bridge_initial_lr": (
                        self._config.optimization.bridge_learning_rate
                        or self._config.optimization.learning_rate
                    ),
                    "phase2_local_global_step": self._global_step,
                    "effective_total_step": parent_step + self._global_step,
                    "bridge_key_count": self._last_bridge_checkpoint_key_count,
                    "bridge_parameter_count": self._optimizer_group_parameter_counts[
                        "conditioning_bridge"
                    ],
                }
            )
        if self._config.text_encoder_lora.enabled:
            raw_metadata.update(
                {
                    "gemma_lora_rank": self._config.text_encoder_lora.rank,
                    "gemma_lora_alpha": self._config.text_encoder_lora.alpha,
                }
            )
        # Convert all values to strings for safetensors compatibility
        metadata = {k: str(v) for k, v in raw_metadata.items()}
        if metadata:
            logger.info(f"Saving checkpoint metadata: {metadata}")
        return metadata

    def _save_config(self) -> None:
        """Save the training configuration as a YAML file in the output directory."""
        if not IS_MAIN_PROCESS:
            return

        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)

        logger.info(f"💾 Training configuration saved to: {config_path.relative_to(self._config.output_dir)}")

    def _init_wandb(self, resume_run_id: str | None = None) -> None:
        """Initialize Weights & Biases run, resuming an existing run if its id is provided."""
        if not self._config.wandb.enabled or not IS_MAIN_PROCESS:
            self._wandb_run = None
            return

        wandb_config = self._config.wandb
        init_kwargs: dict[str, Any] = {
            "project": wandb_config.project,
            "entity": wandb_config.entity,
            "name": Path(self._config.output_dir).name,
            "tags": wandb_config.tags,
            "config": self._config.model_dump(),
        }
        if resume_run_id is not None:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "must"
        run = wandb.init(**init_kwargs)
        self._wandb_run = run

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to Weights & Biases."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)

    @staticmethod
    def _format_optional_metric(metrics: dict[str, float], name: str) -> str:
        value = metrics.get(name)
        return "n/a" if value is None else f"{value:.4f}"
