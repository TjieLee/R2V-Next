#!/usr/bin/env python3
"""Two-rank FSDP semantic-flow checkpoint round-trip smoke test.

Run from the repository root with:

    torchrun --nproc_per_node=2 packages/ltx-trainer/scripts/smoke_semantic_flow_fsdp_checkpoint.py \
        --output-dir /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/semantic_flow_v2/fsdp_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "ltx-trainer" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "ltx-core" / "src"))

from ltx_trainer.online_data.path_safety import assert_write_path_allowed  # noqa: E402
from ltx_trainer.online_inference.checkpoint_runtime import audit_checkpoint  # noqa: E402
from ltx_trainer.online_inference.output_artifacts import atomic_write_json  # noqa: E402
from ltx_trainer.training_strategies.semantic_flow import SemanticFlowConfig, SemanticFlowStrategy  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _init_distributed() -> tuple[int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"Expected exactly 2 ranks, got {world_size}")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device("cuda", local_rank)


def _full_state(module: nn.Module) -> dict[str, torch.Tensor]:
    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, config):
        return {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }


class _FsdpCollectAccelerator:
    def get_state_dict(self, module: nn.Module) -> dict[str, torch.Tensor]:
        return _full_state(module)


class _TinySemanticTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Linear(8, 8)
        self.semantic_token_type_embedding = nn.Embedding(3, 8)
        self.semantic_entity_embedding = nn.Embedding(5, 8)
        self.semantic_position_adapter = nn.Sequential(
            nn.Linear(6, 8),
            nn.SiLU(),
            nn.Linear(8, 8),
        )
        self.semantic_norm_out = nn.RMSNorm(8, elementwise_affine=True)
        self.semantic_proj_out = nn.Linear(8, 6)


def _fill_deterministic(module: nn.Module, *, offset: float) -> None:
    with torch.no_grad():
        for index, (_name, parameter) in enumerate(sorted(module.named_parameters())):
            values = torch.arange(parameter.numel(), dtype=torch.float32, device=parameter.device)
            values = values.reshape_as(parameter).div(1000.0).add(offset + index)
            parameter.copy_(values.to(dtype=parameter.dtype))


def _wrap(module: nn.Module, device: torch.device) -> FSDP:
    return FSDP(
        module.to(device),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=device,
    )


def _build_transformer(device: torch.device) -> FSDP:
    transformer = _TinySemanticTransformer()
    _fill_deterministic(transformer, offset=10.0)
    return _wrap(transformer, device)


def _build_strategy_modules(device: torch.device) -> tuple[SemanticFlowStrategy, dict[str, FSDP]]:
    strategy = SemanticFlowStrategy(SemanticFlowConfig())
    semantic_query = nn.Linear(8, 8, bias=False)
    semantic_encoder = nn.Sequential(nn.Linear(8, 4), nn.SiLU(), nn.Linear(4, 6))
    semantic_reconstruction_decoder = nn.Sequential(nn.Linear(6, 4), nn.SiLU(), nn.Linear(4, 8))
    _fill_deterministic(semantic_query, offset=20.0)
    _fill_deterministic(semantic_encoder, offset=30.0)
    _fill_deterministic(semantic_reconstruction_decoder, offset=40.0)
    modules: dict[str, FSDP] = {
        "semantic_query": _wrap(semantic_query, device),
        "semantic_encoder": _wrap(semantic_encoder, device),
        "semantic_reconstruction_decoder": _wrap(semantic_reconstruction_decoder, device),
    }
    strategy.set_trainable_modules(modules)
    return strategy, modules


def _load_transformer_with_full_state_context(
    transformer: FSDP,
    state: dict[str, torch.Tensor],
) -> None:
    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(transformer, StateDictType.FULL_STATE_DICT, config):
        transformer.load_state_dict(state, strict=True)


def _load_with_full_state_context(
    strategy: SemanticFlowStrategy,
    modules: dict[str, FSDP],
    state: dict[str, torch.Tensor],
) -> None:
    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with ExitStack() as stack:
        for module in modules.values():
            stack.enter_context(FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, config))
        strategy.load_extra_checkpoint_state_dict(state)


def _perturb_modules(modules: dict[str, nn.Module]) -> None:
    with torch.no_grad():
        for module in modules.values():
            for parameter in module.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.25)


def _max_abs_diff(
    lhs: dict[str, dict[str, torch.Tensor]],
    rhs: dict[str, dict[str, torch.Tensor]],
) -> float:
    max_diff = 0.0
    for module_name, module_state in lhs.items():
        for key, value in module_state.items():
            diff = (value.float() - rhs[module_name][key].float()).abs().max().item()
            max_diff = max(max_diff, diff)
    return max_diff


def run(output_dir: Path) -> dict[str, Any] | None:
    rank, world_size, device = _init_distributed()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "semantic_flow_fsdp_roundtrip.safetensors"
    ready_marker_path = output_dir / "checkpoint_step_00001.ready.json"

    transformer = _build_transformer(device)
    strategy, modules = _build_strategy_modules(device)
    accelerator = _FsdpCollectAccelerator()
    transformer_state = accelerator.get_state_dict(transformer)
    precollected = {
        name: accelerator.get_state_dict(module)
        for name, module in modules.items()
    }
    flat_state = strategy.get_extra_checkpoint_state_dict(
        accelerator,
        precollected_states=precollected,
    )
    flat_state.update(transformer_state)
    dist.barrier()
    if rank == 0:
        metadata = {"architecture": "semantic_flow_v1", "global_step": "1"}
        save_file(flat_state, checkpoint_path, metadata=metadata)
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        ready_marker_path.write_text(
            json.dumps(
                {
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                    "global_step": 1,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    dist.barrier()

    del transformer, strategy, modules
    torch.cuda.empty_cache()

    transformer = _build_transformer(device)
    strategy, modules = _build_strategy_modules(device)
    _perturb_modules(modules)
    _perturb_modules({"transformer": transformer})
    loaded_state = load_file(checkpoint_path, device="cpu")
    loaded_strategy_state = {
        key: value
        for key, value in loaded_state.items()
        if key.startswith("training_strategy.")
    }
    loaded_transformer_state = {
        key: value
        for key, value in loaded_state.items()
        if not key.startswith("training_strategy.")
    }
    _load_transformer_with_full_state_context(transformer, loaded_transformer_state)
    _load_with_full_state_context(strategy, modules, loaded_strategy_state)
    post_load = {name: accelerator.get_state_dict(module) for name, module in modules.items()}
    max_diff = max(
        _max_abs_diff({"transformer": transformer_state}, {"transformer": accelerator.get_state_dict(transformer)}),
        _max_abs_diff(precollected, post_load),
    )
    max_diff_tensor = torch.tensor(max_diff, device=device)
    dist.all_reduce(max_diff_tensor, op=dist.ReduceOp.MAX)
    if max_diff_tensor.item() != 0.0:
        raise RuntimeError(
            "Tiny FSDP checkpoint round-trip changed tensors: "
            f"max_abs_diff={max_diff_tensor.item()}"
        )

    if rank != 0:
        return None
    audit = audit_checkpoint(checkpoint_path)
    memory = torch.cuda.max_memory_allocated(device)
    return {
        "world_size": world_size,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "ready_marker_path": str(ready_marker_path.resolve()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_key_count": audit["checkpoint_key_count"],
        "metadata": audit["metadata"],
        "max_abs_tensor_diff_after_reload": max_diff_tensor.item(),
        "peak_cuda_memory_allocated_bytes": memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        output_dir = assert_write_path_allowed(args.output_dir)
        result = run(output_dir)
        if result is not None:
            atomic_write_json(output_dir / "result.json", result)
            print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
