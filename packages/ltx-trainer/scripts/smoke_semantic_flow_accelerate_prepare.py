#!/usr/bin/env python3
"""Validate Accelerate FSDP prepare for the semantic-flow multi-model topology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ltx_trainer.online_data.path_safety import assert_write_path_allowed
from ltx_trainer.online_inference.output_artifacts import atomic_write_json


class BasicAVTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.activation(self.proj(value))


class TinyTransformer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.block = BasicAVTransformerBlock(hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.block(value))


def _local_module_gradient_report(module: nn.Module) -> dict[str, Any]:
    trainable_parameter_count = 0
    gradient_parameter_count = 0
    missing_gradient_parameters: list[str] = []
    nonfinite_gradient_parameters: list[str] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_parameter_count += 1
        gradient = parameter.grad
        if gradient is None:
            missing_gradient_parameters.append(name)
            continue
        gradient_parameter_count += 1
        if not bool(torch.isfinite(gradient).all().item()):
            nonfinite_gradient_parameters.append(name)
    return {
        "passed": (
            trainable_parameter_count > 0
            and not missing_gradient_parameters
            and not nonfinite_gradient_parameters
        ),
        "trainable_parameter_count": trainable_parameter_count,
        "gradient_parameter_count": gradient_parameter_count,
        "missing_gradient_parameters": missing_gradient_parameters,
        "nonfinite_gradient_parameters": nonfinite_gradient_parameters,
    }


def _module_gradient_report(module: nn.Module) -> dict[str, Any]:
    if isinstance(module, FSDP):
        with FSDP.summon_full_params(
            module,
            recurse=True,
            writeback=False,
            with_grads=True,
        ):
            return _local_module_gradient_report(module)
    return _local_module_gradient_report(module)


def _snapshot_local_trainable_parameters(
    module: nn.Module,
) -> list[tuple[str, nn.Parameter, torch.Tensor]]:
    return [
        (name, parameter, parameter.detach().clone())
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.numel() > 0
    ]


def _optimizer_update_checks(
    accelerator: Accelerator,
    snapshots: dict[str, list[tuple[str, nn.Parameter, torch.Tensor]]],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name, module_snapshots in snapshots.items():
        local_updated = any(
            not torch.equal(parameter.detach(), before)
            for _parameter_name, parameter, before in module_snapshots
        )
        gathered = accelerator.gather(
            torch.tensor([int(local_updated)], device=accelerator.device, dtype=torch.int64)
        )
        checks[name] = bool(gathered.max().item())
    return checks


def _finite_nonempty_state(accelerator: Accelerator, module: nn.Module) -> dict[str, torch.Tensor]:
    state = accelerator.get_state_dict(module)
    if accelerator.is_main_process and not state:
        raise RuntimeError(f"Accelerate returned an empty state dict for {type(module).__name__}")
    nonfinite = [key for key, value in state.items() if not torch.isfinite(value).all()]
    if nonfinite:
        raise RuntimeError(f"Non-finite tensors in {type(module).__name__} state dict: {nonfinite}")
    return state


def run(output_dir: Path) -> dict[str, Any] | None:
    accelerator = Accelerator()
    if accelerator.num_processes != 2:
        raise RuntimeError(f"Expected exactly 2 Accelerate processes, got {accelerator.num_processes}")
    if accelerator.distributed_type.value != "FSDP":
        raise RuntimeError(f"Expected Accelerate FSDP, got {accelerator.distributed_type.value}")

    torch.manual_seed(1234)
    hidden_dim = 16
    transformer = TinyTransformer(hidden_dim)
    semantic_query = nn.Linear(hidden_dim, hidden_dim)
    semantic_encoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
    semantic_reconstruction_decoder = nn.Linear(hidden_dim, hidden_dim)
    semantic_alignment_head = nn.Linear(hidden_dim, hidden_dim)
    original_parameters = [
        parameter
        for module in (
            transformer,
            semantic_query,
            semantic_encoder,
            semantic_reconstruction_decoder,
            semantic_alignment_head,
        )
        for parameter in module.parameters()
    ]

    prepared = accelerator.prepare(
        transformer,
        semantic_query,
        semantic_encoder,
        semantic_reconstruction_decoder,
        semantic_alignment_head,
    )
    transformer, semantic_query, semantic_encoder, semantic_reconstruction_decoder, semantic_alignment_head = prepared
    optimizer = torch.optim.AdamW(original_parameters, lr=1.0e-3)
    optimizer = accelerator.prepare(optimizer)

    inputs = torch.randn(2, 4, hidden_dim, device=accelerator.device)
    hidden = transformer(inputs)
    hidden = hidden + semantic_query(hidden)
    semantic = semantic_encoder(hidden)
    reconstruction = semantic_reconstruction_decoder(semantic)
    alignment = semantic_alignment_head(semantic)
    loss = reconstruction.float().square().mean() + alignment.float().square().mean()
    accelerator.backward(loss)

    modules = {
        "transformer": transformer,
        "semantic_query": semantic_query,
        "semantic_encoder": semantic_encoder,
        "semantic_reconstruction_decoder": semantic_reconstruction_decoder,
        "semantic_alignment_head": semantic_alignment_head,
    }
    gradient_reports = {name: _module_gradient_report(module) for name, module in modules.items()}
    gradient_checks = {name: bool(report["passed"]) for name, report in gradient_reports.items()}
    if not all(gradient_checks.values()):
        raise RuntimeError(
            "Missing or non-finite gradients after Accelerate prepare: "
            f"{json.dumps(gradient_reports, sort_keys=True)}"
        )
    parameter_snapshots = {
        name: _snapshot_local_trainable_parameters(module)
        for name, module in modules.items()
    }
    optimizer.step()
    optimizer_update_checks = _optimizer_update_checks(accelerator, parameter_snapshots)
    if not all(optimizer_update_checks.values()):
        raise RuntimeError(
            "Optimizer did not update every trainable module after Accelerate prepare: "
            f"{optimizer_update_checks}"
        )

    state_dicts = {name: _finite_nonempty_state(accelerator, module) for name, module in modules.items()}
    accelerator.wait_for_everyone()
    completed_ranks = accelerator.gather(torch.ones(1, device=accelerator.device, dtype=torch.int64)).sum().item()
    if int(completed_ranks) != accelerator.num_processes:
        raise RuntimeError(f"Only {completed_ranks} ranks completed the Accelerate smoke")

    if not accelerator.is_main_process:
        accelerator.end_training()
        return None
    result = {
        "world_size": accelerator.num_processes,
        "distributed_type": accelerator.distributed_type.value,
        "accelerator_multimodel_prepare_passed": True,
        "all_trainable_modules_have_finite_gradients": True,
        "gradient_checks": gradient_checks,
        "gradient_reports": gradient_reports,
        "optimizer_updates_all_trainable_modules": True,
        "optimizer_update_checks": optimizer_update_checks,
        "state_dict_key_counts": {name: len(state) for name, state in state_dicts.items()},
        "completed_ranks": int(completed_ranks),
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(accelerator.device)),
    }
    accelerator.end_training()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = assert_write_path_allowed(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run(output_dir)
    if result is not None:
        atomic_write_json(output_dir / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
