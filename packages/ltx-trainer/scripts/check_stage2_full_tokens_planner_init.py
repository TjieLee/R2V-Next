#!/usr/bin/env python3
"""Initialize the real Stage 2 stack and smoke-test its full-token bridge."""

import time
from pathlib import Path

import torch
import typer
import yaml

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.trainer import LtxvTrainer
from ltx_trainer.training_strategies.multi_reference_planner_stage2 import (
    MultiReferencePlannerStage2Strategy,
)


app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


def _dtype(value: str) -> torch.dtype:
    mapping = {"bf16": torch.bfloat16, "float32": torch.float32}
    if value not in mapping:
        raise typer.BadParameter("--dtype must be bf16 or float32")
    return mapping[value]


def _mean_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _assert_frozen_gradients(trainer: LtxvTrainer, strategy: MultiReferencePlannerStage2Strategy) -> None:
    if any(parameter.grad is not None for parameter in trainer._transformer.parameters()):
        raise RuntimeError("Frozen LTX base/Stage 1 DiT LoRA received gradients")
    text_encoder = strategy._unwrap_text_encoder()
    for name, parameter in text_encoder.named_parameters():
        if "lora_" not in name and parameter.grad is not None:
            raise RuntimeError(f"Frozen Gemma parameter received a gradient: {name}")


def _run_real_batch_smoke(
    trainer: LtxvTrainer,
    strategy: MultiReferencePlannerStage2Strategy,
) -> None:
    trainer._init_dataloader()
    trainer._init_timestep_sampler()
    batch = next(iter(trainer._dataloader))
    for parameter in trainer._trainable_params:
        parameter.grad = None

    device = trainer._accelerator.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    forward_start = time.perf_counter()
    cfg_dropout_enabled = strategy.config.cfg_dropout_enabled
    strategy.config.cfg_dropout_enabled = False
    try:
        output = trainer._training_step(batch)
    finally:
        strategy.config.cfg_dropout_enabled = cfg_dropout_enabled
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_start

    if not torch.isfinite(output.loss).all():
        raise RuntimeError("Real-batch total loss is not finite")
    strategy_metrics = strategy.get_last_training_metrics()
    for name in ("train/loss_flow", "train/loss_siglip", "train/loss_ntp"):
        value = strategy_metrics.get(name)
        if value is None or not torch.isfinite(value).all():
            raise RuntimeError(f"Real-batch metric is missing or non-finite: {name}")

    backward_start = time.perf_counter()
    trainer._accelerator.backward(output.loss.mean())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_start

    embeddings_processor = trainer._accelerator.unwrap_model(
        trainer._embeddings_processor,
        keep_torch_compile=False,
    )
    required_modules = {
        "Gemma LoRA": strategy.get_text_encoder_trainable_module(),
        "planner_tokens": strategy.get_trainable_modules()["planner_tokens"],
        "visual_token_projection": strategy.get_trainable_modules()["visual_token_projection"],
        "visual_full_encoder": strategy.get_trainable_modules()["visual_full_encoder"],
        "embeddings_processor.video_connector": embeddings_processor.video_connector,
    }
    for name, module in required_modules.items():
        if not _mean_nonzero_grad(module):
            raise RuntimeError(f"No nonzero real-batch gradient for {name}")
    _assert_frozen_gradients(trainer, strategy)

    allocated = torch.cuda.memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    typer.echo(f"total loss: {output.loss.detach().float().mean().item():.6f}")
    typer.echo(f"flow loss: {strategy_metrics['train/loss_flow'].item():.6f}")
    typer.echo(f"siglip loss: {strategy_metrics['train/loss_siglip'].item():.6f}")
    typer.echo(f"ntp loss: {strategy_metrics['train/loss_ntp'].item():.6f}")
    cosine = strategy_metrics.get("train/siglip_cosine")
    typer.echo(f"siglip cosine: {cosine.item():.6f}" if cosine is not None else "siglip cosine: n/a")
    typer.echo(f"allocated VRAM: {allocated:.3f} GiB")
    typer.echo(f"peak VRAM: {peak:.3f} GiB")
    typer.echo(f"forward seconds: {forward_seconds:.3f}")
    typer.echo(f"backward seconds: {backward_seconds:.3f}")


@app.command()
def main(
    config: str = typer.Option(..., help="Stage 2 full-token planner YAML."),
    stage1_checkpoint: str = typer.Option(..., help="Stage 1 rank-128 LoRA/full visual checkpoint."),
    device: str = typer.Option("cuda", help="Smoke-test device."),
    dtype: str = typer.Option("bf16", help="bf16 or float32."),
    real_batch: bool = typer.Option(False, "--real-batch/--no-real-batch"),
) -> None:
    config_path = Path(config)
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_data["model"]["load_checkpoint"] = stage1_checkpoint
    cfg = LtxTrainerConfig.model_validate(config_data)
    requested_dtype = _dtype(dtype)

    trainer = LtxvTrainer(cfg)
    strategy = trainer._training_strategy
    if not isinstance(strategy, MultiReferencePlannerStage2Strategy):
        raise RuntimeError("Config did not create MultiReferencePlannerStage2Strategy")
    if trainer._train_transformer:
        raise RuntimeError("Stage 1 DiT LoRA must be frozen in Stage 2")
    if any(parameter.requires_grad for parameter in trainer._transformer.parameters()):
        raise RuntimeError("Found trainable LTX base/DiT LoRA parameters")
    if (
        strategy._visual_connector is not None
        or strategy._visual_gate is not None
        or strategy._visual_resampler is not None
    ):
        raise RuntimeError("Legacy visual connector/gate/resampler exists in full-token Stage 2")

    text_encoder = trainer._text_encoder
    if text_encoder is None:
        raise RuntimeError("Stage 2 online planner did not load Gemma")
    trainable_text_names = [name for name, parameter in text_encoder.named_parameters() if parameter.requires_grad]
    if not trainable_text_names or any("lora_" not in name for name in trainable_text_names):
        raise RuntimeError("Gemma trainable parameters must contain only language-model LoRA weights")
    gemma_model = strategy._unwrap_text_encoder().model.model
    for name in ("vision_tower", "multi_modal_projector"):
        module = getattr(gemma_model, name, None)
        if module is None or any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError(f"Gemma {name} must exist and remain frozen")
        if module.training:
            raise RuntimeError(f"Gemma {name} must remain in eval mode")

    modules = strategy.get_trainable_modules()
    required = {"planner_tokens", "visual_token_projection", "visual_full_encoder"}
    if not required.issubset(modules):
        raise RuntimeError(f"Missing Stage 2 modules: {sorted(required - modules.keys())}")
    if not trainer._train_embeddings_processor:
        raise RuntimeError("Stage 2 must train embeddings_processor.video_connector")

    if real_batch:
        base_transformer = (
            trainer._transformer.get_base_model()
            if hasattr(trainer._transformer, "get_base_model")
            else trainer._transformer
        )
        if (
            cfg.optimization.enable_gradient_checkpointing
            and not base_transformer._enable_gradient_checkpointing
        ):
            raise RuntimeError("Frozen DiT gradient checkpointing is not enabled")
        if trainer._transformer.training:
            raise RuntimeError("Frozen DiT must remain in eval mode")
        lm_head = strategy._unwrap_text_encoder().model.lm_head
        lm_head_param = next(lm_head.parameters())
        typer.echo("NTP final hidden dtype will be adapted to lm_head dtype")
        typer.echo(f"lm_head dtype: {lm_head_param.dtype}")
        typer.echo(f"lm_head device: {lm_head_param.device}")
        _run_real_batch_smoke(trainer, strategy)
        return

    planner = modules["planner_tokens"]
    projection = modules["visual_token_projection"]
    full_encoder = modules["visual_full_encoder"]
    parameter = next(planner.parameters())
    smoke_device = torch.device(device)
    fake_inputs_embeds = torch.randn(
        1,
        cfg.training_strategy.planner_token_count,
        cfg.training_strategy.planner_source_dim,
        device=smoke_device,
        dtype=requested_dtype,
        requires_grad=True,
    )
    language_model = strategy.get_text_encoder_trainable_module()
    planner_hidden, final_hidden = strategy._forward_language_model_for_planner(
        inputs_embeds=fake_inputs_embeds.to(dtype=parameter.dtype),
        attention_mask=torch.ones(fake_inputs_embeds.shape[:2], dtype=torch.long, device=smoke_device),
    )
    times = torch.arange(8, device=smoke_device, dtype=torch.float32).unsqueeze(0) / 6.0
    positions = strategy._make_visual_positions(
        times,
        height=torch.tensor([384.0], device=smoke_device),
        width=torch.tensor([640.0], device=smoke_device),
        spatial_grid=16,
        dtype=torch.float32,
    )
    valid_mask = torch.ones(planner_hidden.shape[:2], dtype=torch.bool, device=smoke_device)

    predicted_raw = planner(
        planner_hidden=planner_hidden.to(dtype=parameter.dtype),
        planner_mask=valid_mask,
        token_positions=positions,
    )
    projected = projection(predicted_raw)
    encoded, encoded_mask = full_encoder(
        tokens=projected,
        token_positions=positions,
        token_mask=valid_mask,
    )
    siglip_target = torch.randn_like(predicted_raw)
    siglip_loss = (predicted_raw.float() - siglip_target.float()).square().mean().reshape(1)
    flow_loss = encoded.float().square().mean().reshape(1)
    ntp_labels = torch.full(
        planner_hidden.shape[:2],
        -100,
        dtype=torch.long,
        device=smoke_device,
    )
    ntp_labels[:, 1] = 0
    ntp_loss = strategy._compute_lm_loss(final_hidden, ntp_labels)
    total = strategy._combine_losses(flow_loss, siglip_loss, ntp_loss)
    if not torch.isfinite(total).all():
        raise RuntimeError("Stage 2 smoke loss is not finite")
    total.mean().backward()

    for name in required:
        if not _mean_nonzero_grad(modules[name]):
            raise RuntimeError(f"No nonzero first-backward gradient for {name}")
    if not _mean_nonzero_grad(language_model):
        raise RuntimeError("No nonzero first-backward gradient for Gemma LoRA")

    typer.echo(f"planner hidden: {list(planner_hidden.shape)}")
    typer.echo(f"predicted raw: {list(predicted_raw.shape)}")
    typer.echo(f"projected: {list(projected.shape)}")
    typer.echo(f"visual encoder output: {list(encoded.shape)}, valid={int(encoded_mask.sum())}")
    typer.echo(
        f"losses finite: flow={flow_loss.item():.6f}, "
        f"siglip={siglip_loss.item():.6f}, ntp={ntp_loss.item():.6f}"
    )
    typer.echo(
        "TRAINABLE:\n- Gemma LoRA\n- planner\n- visual_token_projection"
        "\n- visual_full_encoder\n- embeddings_processor.video_connector"
    )
    typer.echo("FROZEN:\n- LTX base\n- Stage 1 DiT LoRA\n- Gemma base\n- vision tower\n- multimodal projector")


if __name__ == "__main__":
    app()
