"""Build and enforce exact semantic-flow runtime locks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ltx_trainer.online_inference.output_artifacts import atomic_write_json


class SemanticFlowRuntimeLockError(RuntimeError):
    """The current runtime does not satisfy its semantic-flow lock."""


def _package_version(report: dict[str, Any], name: str) -> str:
    package = (report.get("packages") or {}).get(name) or {}
    if package.get("installed") is not True or not package.get("version"):
        raise SemanticFlowRuntimeLockError(f"Required package is unavailable: {name}")
    return str(package["version"])


def build_semantic_flow_runtime_lock(
    runtime_report: dict[str, Any],
    fsdp_smoke_result: dict[str, Any],
) -> dict[str, Any]:
    if runtime_report.get("ready") is not True:
        raise SemanticFlowRuntimeLockError("Cannot lock a runtime whose capability audit did not pass")
    if fsdp_smoke_result.get("world_size") != 2:
        raise SemanticFlowRuntimeLockError("Tiny FSDP checkpoint smoke must run with exactly two processes")
    if float(fsdp_smoke_result.get("max_abs_tensor_diff_after_reload", float("inf"))) != 0.0:
        raise SemanticFlowRuntimeLockError("Tiny FSDP checkpoint smoke did not reload tensors exactly")

    torch_runtime = runtime_report.get("torch_runtime") or {}
    gemma = runtime_report.get("gemma") or {}
    accelerate_capability = ((runtime_report.get("capabilities") or {}).get("accelerate_fsdp_plugin") or {})
    if gemma.get("sliding_window") != 1024:
        raise SemanticFlowRuntimeLockError(
            f"Production Gemma sliding_window must be 1024, got {gemma.get('sliding_window')}"
        )
    return {
        "runtime_lock_version": 1,
        "python": str((runtime_report.get("python") or {}).get("runtime_version")),
        "torch": _package_version(runtime_report, "torch"),
        "transformers": _package_version(runtime_report, "transformers"),
        "accelerate": _package_version(runtime_report, "accelerate"),
        "safetensors": _package_version(runtime_report, "safetensors"),
        "cuda": torch_runtime.get("cuda_version"),
        "nccl": torch_runtime.get("nccl_version"),
        "gemma_sliding_window": 1024,
        "fsdp_version": accelerate_capability.get("fsdp_version"),
        "sharding_strategy": accelerate_capability.get("sharding_strategy"),
        "state_dict_type": accelerate_capability.get("state_dict_type"),
        "checkpoint_roundtrip_world_size": 2,
        "checkpoint_roundtrip_max_abs_tensor_diff": 0.0,
    }


def write_or_validate_semantic_flow_runtime_lock(
    path: str | Path,
    current: dict[str, Any],
    *,
    refresh: bool,
) -> str:
    resolved = Path(path).expanduser().resolve()
    if refresh:
        atomic_write_json(resolved, current)
        return "refreshed"
    if not resolved.is_file():
        raise SemanticFlowRuntimeLockError(
            f"Runtime lock is missing: {resolved}; run smoke or pass --refresh-runtime-lock explicitly"
        )
    try:
        locked = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticFlowRuntimeLockError(f"Invalid runtime lock {resolved}: {exc}") from exc
    if not isinstance(locked, dict):
        raise SemanticFlowRuntimeLockError(f"Runtime lock must be a JSON object: {resolved}")
    if locked != current:
        differences = {
            key: {"locked": locked.get(key), "current": current.get(key)}
            for key in sorted(set(locked) | set(current))
            if locked.get(key) != current.get(key)
        }
        raise SemanticFlowRuntimeLockError(f"Runtime differs from lock {resolved}: {differences}")
    return "validated"


__all__ = [
    "SemanticFlowRuntimeLockError",
    "build_semantic_flow_runtime_lock",
    "write_or_validate_semantic_flow_runtime_lock",
]
