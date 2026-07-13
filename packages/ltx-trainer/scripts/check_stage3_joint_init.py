#!/usr/bin/env python3
"""Validate Stage 3 checkpoint initialization and one real joint-training step."""

from pathlib import Path

import torch
import typer
import yaml
from peft import get_peft_model_state_dict
from safetensors.torch import load_file

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.trainer import LtxvTrainer, normalize_peft_adapter_key
from ltx_trainer.training_strategies.multi_reference_planner_stage2 import (
    MultiReferencePlannerStage2Strategy,
)


app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


def _index_state(
    state: dict[str, torch.Tensor],
    *,
    normalize_peft: bool,
) -> dict[str, torch.Tensor]:
    indexed: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        normalized = normalize_peft_adapter_key(key) if normalize_peft else key
        if normalized in indexed:
            raise RuntimeError(f"Duplicate normalized checkpoint key: {normalized}")
        indexed[normalized] = value
    return indexed


def _assert_exact_tensor_group(
    *,
    label: str,
    checkpoint_state: dict[str, torch.Tensor],
    loaded_state: dict[str, torch.Tensor],
    normalize_peft: bool = False,
) -> int:
    checkpoint = _index_state(checkpoint_state, normalize_peft=normalize_peft)
    loaded = _index_state(loaded_state, normalize_peft=normalize_peft)
    missing = sorted(set(checkpoint) - set(loaded))
    unexpected = sorted(set(loaded) - set(checkpoint))
    if missing or unexpected:
        raise RuntimeError(
            f"{label} key mismatch: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    shape_mismatches = [
        (key, tuple(checkpoint[key].shape), tuple(loaded[key].shape))
        for key in sorted(checkpoint)
        if checkpoint[key].shape != loaded[key].shape
    ]
    if shape_mismatches:
        raise RuntimeError(f"{label} shape mismatch: {shape_mismatches[:20]}")
    different = [
        key
        for key in sorted(checkpoint)
        if not torch.equal(
            loaded[key].detach().cpu().to(checkpoint[key].dtype),
            checkpoint[key].detach().cpu(),
        )
    ]
    if different:
        raise RuntimeError(f"{label} differs from the source checkpoint: {different[:20]}")
    if not checkpoint:
        raise RuntimeError(f"{label} checkpoint group is empty")
    return len(checkpoint)


def _checkpoint_group(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def _has_finite_nonzero_gradient(module: torch.nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return bool(gradients) and all(torch.isfinite(gradient).all() for gradient in gradients) and any(
        torch.any(gradient != 0) for gradient in gradients
    )


def _assert_only_lora_trainable(module: torch.nn.Module, label: str) -> list[str]:
    names = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    if not names:
        raise RuntimeError(f"{label} has no trainable LoRA parameters")
    invalid = [name for name in names if "lora_" not in name]
    if invalid:
        raise RuntimeError(f"{label} contains trainable base parameters: {invalid[:20]}")
    return names


def _assert_frozen_module(module: torch.nn.Module | None, label: str) -> None:
    if module is None:
        raise RuntimeError(f"Missing required frozen module: {label}")
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise RuntimeError(f"{label} must remain frozen")
    if module.training:
        raise RuntimeError(f"{label} must remain in eval mode")


@app.command()
def main(
    config: str = typer.Option(..., help="Stage 3 joint-training YAML."),
    stage2_checkpoint: str | None = typer.Option(
        None,
        "--stage2-checkpoint",
        help="Optional override for model.load_checkpoint.",
    ),
) -> None:
    config_path = Path(config)
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if stage2_checkpoint is not None:
        config_data["model"]["load_checkpoint"] = stage2_checkpoint
    cfg = LtxTrainerConfig.model_validate(config_data)
    if cfg.training_strategy.training_phase != "stage3":
        raise RuntimeError("check_stage3_joint_init.py requires training_phase=stage3")

    trainer = LtxvTrainer(cfg)
    strategy = trainer._training_strategy
    if not isinstance(strategy, MultiReferencePlannerStage2Strategy):
        raise RuntimeError("Config did not create MultiReferencePlannerStage2Strategy")
    if trainer._loaded_checkpoint_path is None:
        raise RuntimeError("Stage 3 did not load the Stage 2 checkpoint")
    if not trainer._train_transformer:
        raise RuntimeError("Stage 3 must train the Stage 1 DiT LoRA")

    text_encoder = strategy._unwrap_text_encoder()
    modules = strategy.get_trainable_modules()
    required_modules = {
        "planner_tokens",
        "visual_token_projection",
        "visual_full_encoder",
    }
    missing_modules = required_modules - modules.keys()
    if missing_modules:
        raise RuntimeError(f"Missing Stage 3 strategy modules: {sorted(missing_modules)}")
    connector = trainer._embeddings_processor.video_connector
    checkpoint_state = load_file(trainer._loaded_checkpoint_path)
    dit = trainer._accelerator.unwrap_model(trainer._transformer, keep_torch_compile=False)
    language_model = trainer._accelerator.unwrap_model(
        strategy._get_language_model(),
        keep_torch_compile=False,
    )
    verified_counts = {
        "DiT LoRA": _assert_exact_tensor_group(
            label="DiT LoRA",
            checkpoint_state=_checkpoint_group(checkpoint_state, "diffusion_model."),
            loaded_state=get_peft_model_state_dict(dit),
            normalize_peft=True,
        ),
        "Gemma LoRA": _assert_exact_tensor_group(
            label="Gemma LoRA",
            checkpoint_state=_checkpoint_group(
                checkpoint_state,
                "text_encoder.model.model.language_model.",
            ),
            loaded_state=get_peft_model_state_dict(language_model),
            normalize_peft=True,
        ),
        "Planner": _assert_exact_tensor_group(
            label="Planner",
            checkpoint_state=_checkpoint_group(
                checkpoint_state,
                "training_strategy.planner_tokens.",
            ),
            loaded_state=trainer._accelerator.unwrap_model(
                modules["planner_tokens"], keep_torch_compile=False
            ).state_dict(),
        ),
        "Visual projection": _assert_exact_tensor_group(
            label="Visual projection",
            checkpoint_state=_checkpoint_group(
                checkpoint_state,
                "training_strategy.visual_token_projection.",
            ),
            loaded_state=trainer._accelerator.unwrap_model(
                modules["visual_token_projection"], keep_torch_compile=False
            ).state_dict(),
        ),
        "Visual3DTokenEncoder": _assert_exact_tensor_group(
            label="Visual3DTokenEncoder",
            checkpoint_state=_checkpoint_group(
                checkpoint_state,
                "training_strategy.visual_full_encoder.",
            ),
            loaded_state=trainer._accelerator.unwrap_model(
                modules["visual_full_encoder"], keep_torch_compile=False
            ).state_dict(),
        ),
        "Text connector": _assert_exact_tensor_group(
            label="Text connector",
            checkpoint_state=_checkpoint_group(
                checkpoint_state,
                "embeddings_processor.video_connector.",
            ),
            loaded_state=trainer._accelerator.unwrap_model(
                connector, keep_torch_compile=False
            ).state_dict(),
        ),
    }
    for label, count in verified_counts.items():
        typer.echo(f"{label} tensors loaded: {count}/{count}")
    del checkpoint_state

    _assert_only_lora_trainable(trainer._transformer, "Stage 1 DiT")
    _assert_only_lora_trainable(text_encoder, "Gemma language model")
    gemma_model = text_encoder.model.model
    _assert_frozen_module(getattr(gemma_model, "vision_tower", None), "Gemma vision tower")
    _assert_frozen_module(
        getattr(gemma_model, "multi_modal_projector", None),
        "Gemma multimodal projector",
    )

    if not all(parameter.requires_grad for parameter in connector.parameters()):
        raise RuntimeError("Stage 3 text connector must be trainable")

    optimizer_ids = [
        id(parameter)
        for parameter_group in torch.optim.AdamW(trainer._trainable_params, lr=1.0e-5).param_groups
        for parameter in parameter_group["params"]
    ]
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise RuntimeError("Stage 3 optimizer contains duplicate parameters")
    if any(not parameter.requires_grad for parameter in trainer._trainable_params):
        raise RuntimeError("Stage 3 optimizer candidates contain frozen parameters")

    trainer._init_optimizer()
    trainer._init_dataloader()
    trainer._init_timestep_sampler()
    batch = next(iter(trainer._dataloader))
    trainer._optimizer.zero_grad(set_to_none=True)
    device = trainer._accelerator.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    cfg_enabled = strategy.config.cfg_dropout_enabled
    strategy.config.cfg_dropout_enabled = False
    try:
        output = trainer._training_step(batch)
    finally:
        strategy.config.cfg_dropout_enabled = cfg_enabled
    if not torch.isfinite(output.loss).all():
        raise RuntimeError("Stage 3 total loss is non-finite")
    metrics = strategy.get_last_training_metrics()
    for metric_name in ("train/loss_flow", "train/loss_siglip", "train/loss_ntp"):
        metric = metrics.get(metric_name)
        if metric is None or not torch.isfinite(metric).all():
            raise RuntimeError(f"Missing or non-finite Stage 3 metric: {metric_name}")

    trainer._accelerator.backward(output.loss.mean())
    gradient_modules = {
        "Stage 1 DiT LoRA": trainer._transformer,
        "Stage 2 Gemma LoRA": strategy.get_text_encoder_trainable_module(),
        "Planner": modules["planner_tokens"],
        "Visual projection": modules["visual_token_projection"],
        "Visual3DTokenEncoder": modules["visual_full_encoder"],
        "Text connector": connector,
    }
    for label, module in gradient_modules.items():
        if not _has_finite_nonzero_gradient(module):
            raise RuntimeError(f"No finite nonzero gradient for {label}")

    frozen_dit_gradients = [
        name
        for name, parameter in trainer._transformer.named_parameters()
        if "lora_" not in name and parameter.grad is not None
    ]
    frozen_gemma_gradients = [
        name
        for name, parameter in text_encoder.named_parameters()
        if "lora_" not in name and parameter.grad is not None
    ]
    if frozen_dit_gradients or frozen_gemma_gradients:
        raise RuntimeError(
            "Frozen base parameters received gradients: "
            f"dit={frozen_dit_gradients[:10]}, gemma={frozen_gemma_gradients[:10]}"
        )
    trainer._accelerator.clip_grad_norm_(trainer._trainable_params, cfg.optimization.max_grad_norm)
    trainer._optimizer.step()

    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    typer.echo("Stage 3 checkpoint loaded: true")
    typer.echo("Stage 1 DiT LoRA loaded: true")
    typer.echo("Stage 2 Gemma LoRA loaded: true")
    typer.echo("Planner loaded: true")
    typer.echo("Visual projection loaded: true")
    typer.echo("Visual3DTokenEncoder loaded: true")
    typer.echo("Text connector loaded: true")
    typer.echo(f"Total loss: {output.loss.detach().float().mean().item():.6f}")
    typer.echo(f"Flow loss: {metrics['train/loss_flow'].item():.6f}")
    typer.echo(f"SigLIP loss: {metrics['train/loss_siglip'].item():.6f}")
    typer.echo(f"NTP loss: {metrics['train/loss_ntp'].item():.6f}")
    typer.echo(f"Peak VRAM: {peak_gib:.3f} GiB")
    typer.echo("Stage 3 optimizer step: success")


if __name__ == "__main__":
    app()
