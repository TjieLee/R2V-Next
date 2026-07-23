#!/usr/bin/env python3
"""Audit required semantic-flow runtime capabilities and enforce an optional version lock."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from ltx_trainer.online_data.path_safety import assert_write_path_allowed
from ltx_trainer.online_inference.runtime_lock import (
    build_semantic_flow_runtime_lock,
    collect_visible_cuda_hardware,
    write_or_validate_semantic_flow_runtime_lock,
)
from ltx_trainer.online_inference.startup_memory import (
    build_startup_host_ram_report,
    enforce_startup_host_ram_report,
    model_parameter_count,
)


def _package_version(name: str) -> dict[str, Any]:
    try:
        return {"installed": True, "version": importlib.metadata.version(name)}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}


def _load_yaml(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        return {}, f"PyYAML is not installed: {exc}"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return {}, f"Expected a YAML mapping in {path}"
    return payload, None


def _torch_runtime() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    cuda_available = bool(torch.cuda.is_available())
    gpu_hardware = collect_visible_cuda_hardware(torch)
    try:
        nccl_version = torch.cuda.nccl.version() if cuda_available else None
        if isinstance(nccl_version, tuple):
            nccl_version = list(nccl_version)
    except Exception as exc:
        nccl_version = f"{type(exc).__name__}: {exc}"
    return {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "nccl_version": nccl_version,
        **gpu_hardware,
    }


def _find_sliding_window(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("sliding_window", "sliding_window_size"):
            if key in value:
                return value[key]
        for nested in value.values():
            found = _find_sliding_window(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_sliding_window(nested)
            if found is not None:
                return found
    return None


def _gemma_sliding_window(path: str | None) -> dict[str, Any]:
    if not path:
        return {"path": None, "sliding_window": None, "error": "model.text_encoder_path is not configured"}
    root = Path(path).expanduser()
    config_path = root / "config.json" if root.is_dir() else root
    if not config_path.is_file():
        return {"path": str(root), "sliding_window": None, "error": f"config.json not found at {config_path}"}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(root), "sliding_window": None, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": str(root),
        "config_path": str(config_path),
        "sliding_window": _find_sliding_window(payload),
    }


def _fsdp_config_report(accelerate_config: dict[str, Any], config_error: str | None) -> dict[str, Any]:
    fsdp_config = accelerate_config.get("fsdp_config") or {}
    return {
        "config_error": config_error,
        "distributed_type": accelerate_config.get("distributed_type"),
        "mixed_precision": accelerate_config.get("mixed_precision"),
        "num_processes": accelerate_config.get("num_processes"),
        "version": fsdp_config.get("fsdp_version"),
        "sharding_strategy": fsdp_config.get("fsdp_sharding_strategy"),
        "auto_wrap_policy": fsdp_config.get("fsdp_auto_wrap_policy"),
        "transformer_layer_cls_to_wrap": fsdp_config.get("fsdp_transformer_layer_cls_to_wrap"),
        "state_dict_type": fsdp_config.get("fsdp_state_dict_type"),
        "sync_module_states": fsdp_config.get("fsdp_sync_module_states"),
        "cpu_ram_efficient_loading": fsdp_config.get("fsdp_cpu_ram_efficient_loading"),
        "use_orig_params": fsdp_config.get("fsdp_use_orig_params"),
    }


def _enum_name(value: Any) -> str | None:
    name = getattr(value, "name", None)
    return str(name) if name is not None else None


def _transformers_mask_mapping_capability() -> dict[str, Any]:
    try:
        import torch
        import transformers

        from ltx_core.multicond.gemma3_attention import build_gemma3_attention_masks
        from ltx_core.multicond.semantic_tokens import build_multimodal_prefix_attention_mask

        config = transformers.Gemma3TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            sliding_window=4,
            layer_types=["full_attention", "sliding_attention"],
            use_cache=False,
        )
        model = transformers.Gemma3TextModel(config).eval()
        inputs = torch.randn(1, 8, config.hidden_size)
        valid = torch.ones(1, 8, dtype=torch.bool)
        image = torch.zeros_like(valid)
        image[:, 2:4] = True
        custom = build_multimodal_prefix_attention_mask(valid, image_token_mask=image)
        masks = build_gemma3_attention_masks(
            valid_token_mask=valid,
            image_token_mask=image,
            custom_visibility=custom,
            sliding_window=config.sliding_window,
            dtype=inputs.dtype,
        )
        with torch.no_grad():
            outputs = model(
                inputs_embeds=inputs,
                attention_mask=masks.as_mapping(),
                position_ids=torch.arange(inputs.shape[1]).unsqueeze(0),
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        output_shape = list(outputs.hidden_states[-1].shape)
        if output_shape != list(inputs.shape):
            raise RuntimeError(f"Unexpected tiny Gemma output shape: {output_shape}")
        return {"passed": True, "output_shape": output_shape}
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def _accelerate_fsdp_plugin_capability(accelerate_config: dict[str, Any]) -> dict[str, Any]:
    fsdp_config = accelerate_config.get("fsdp_config") or {}
    try:
        from accelerate import FullyShardedDataParallelPlugin

        plugin = FullyShardedDataParallelPlugin(
            fsdp_version=fsdp_config.get("fsdp_version"),
            sharding_strategy=fsdp_config.get("fsdp_sharding_strategy"),
            backward_prefetch=fsdp_config.get("fsdp_backward_prefetch"),
            mixed_precision_policy=accelerate_config.get("mixed_precision"),
            auto_wrap_policy=fsdp_config.get("fsdp_auto_wrap_policy"),
            cpu_offload=bool(fsdp_config.get("fsdp_offload_params", False)),
            state_dict_type=fsdp_config.get("fsdp_state_dict_type"),
            use_orig_params=fsdp_config.get("fsdp_use_orig_params"),
            sync_module_states=fsdp_config.get("fsdp_sync_module_states"),
            forward_prefetch=fsdp_config.get("fsdp_forward_prefetch"),
            activation_checkpointing=fsdp_config.get("fsdp_activation_checkpointing"),
            cpu_ram_efficient_loading=fsdp_config.get("fsdp_cpu_ram_efficient_loading"),
            transformer_cls_names_to_wrap=[fsdp_config.get("fsdp_transformer_layer_cls_to_wrap")],
        )
        report = {
            "passed": True,
            "fsdp_version": getattr(plugin, "fsdp_version", None),
            "sharding_strategy": _enum_name(getattr(plugin, "sharding_strategy", None)),
            "state_dict_type": _enum_name(getattr(plugin, "state_dict_type", None)),
        }
        expected = {
            "fsdp_version": 1,
            "sharding_strategy": "FULL_SHARD",
            "state_dict_type": "FULL_STATE_DICT",
        }
        mismatches = {
            key: {"expected": value, "actual": report.get(key)}
            for key, value in expected.items()
            if report.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Accelerate FSDP plugin fields mismatch: {mismatches}")
        return report
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(f"{path}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _real_smoke_memory(path: Path, *, task: str, expected_world_size: int) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("task") != task:
        raise ValueError(f"Expected {task} smoke result JSON: {path}")
    memory = payload.get("memory")
    required = {
        "smoke_start_available_host_ram_bytes",
        "smoke_start_process_rss_bytes",
        "after_model_load_available_host_ram_bytes",
        "after_model_load_process_rss_bytes",
        "after_fsdp_prepare_available_host_ram_bytes",
        "after_fsdp_prepare_process_rss_bytes",
        "peak_host_ram_bytes",
        "per_rank_peak_cuda_memory_bytes",
    }
    if not isinstance(memory, dict) or required - set(memory):
        raise ValueError(f"{task} smoke result is missing memory fields: {sorted(required - set(memory or {}))}")
    world_size = int(payload.get("world_size", 0))
    if world_size != expected_world_size:
        raise ValueError(
            f"{task} real smoke world_size {world_size} does not match accelerate num_processes {expected_world_size}"
        )
    for field in required:
        values = memory[field]
        if not isinstance(values, list) or len(values) != world_size or any(int(value) < 0 for value in values):
            raise ValueError(f"Invalid {task} memory field {field}: {values}")
    available_at_start = min(int(value) for value in memory["smoke_start_available_host_ram_bytes"])
    observed_peak = sum(int(value) for value in memory["peak_host_ram_bytes"])
    if observed_peak > available_at_start * 0.9:
        raise ValueError(
            f"{task} observed peak host RAM {observed_peak} exceeds 90% of starting available RAM {available_at_start}"
        )
    return {
        "result_path": str(path.expanduser().resolve()),
        "world_size": world_size,
        "memory": memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--accelerate-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path)
    parser.add_argument("--fsdp-smoke-result", type=Path)
    parser.add_argument("--accelerate-prepare-smoke-result", type=Path)
    parser.add_argument("--i2i-smoke-result", type=Path)
    parser.add_argument("--r2v-smoke-result", type=Path)
    parser.add_argument("--refresh-runtime-lock", action="store_true")
    args = parser.parse_args()

    trainer_config, trainer_config_error = _load_yaml(args.config.expanduser().resolve())
    accelerate_config, accelerate_config_error = _load_yaml(args.accelerate_config.expanduser().resolve())
    output = assert_write_path_allowed(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch_runtime = _torch_runtime()
    gemma = _gemma_sliding_window((trainer_config.get("model") or {}).get("text_encoder_path"))
    capabilities = {
        "transformers_mask_mapping": _transformers_mask_mapping_capability(),
        "accelerate_fsdp_plugin": _accelerate_fsdp_plugin_capability(accelerate_config),
    }
    errors = []
    for label, error in (
        ("training config", trainer_config_error),
        ("accelerate config", accelerate_config_error),
    ):
        if error:
            errors.append(f"{label}: {error}")
    if torch_runtime.get("available") is not True or torch_runtime.get("cuda_available") is not True:
        errors.append("CUDA-enabled PyTorch is required")
    if gemma.get("sliding_window") != 1024:
        errors.append(f"Production Gemma sliding_window must be 1024, got {gemma.get('sliding_window')}")
    for name, capability in capabilities.items():
        if capability.get("passed") is not True:
            errors.append(f"{name}: {capability.get('error', 'capability check failed')}")

    startup_memory = None
    try:
        startup_memory = build_startup_host_ram_report(
            parameter_count=model_parameter_count((trainer_config.get("model") or {}).get("model_path")),
            num_processes=int(accelerate_config.get("num_processes", 0)),
            training_dtype=str(accelerate_config.get("mixed_precision", "")),
        )
        enforce_startup_host_ram_report(startup_memory)
    except Exception as exc:
        errors.append(f"startup_host_ram: {type(exc).__name__}: {exc}")

    report: dict[str, Any] = {
        "architecture": "semantic_flow_v2",
        "ready": not errors,
        "errors": errors,
        "python": {
            "version": sys.version,
            "runtime_version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "accelerate", "safetensors")
        },
        "torch_runtime": torch_runtime,
        "config": {
            "path": str(args.config.expanduser().resolve()),
            "error": trainer_config_error,
        },
        "accelerate": _fsdp_config_report(accelerate_config, accelerate_config_error),
        "gemma": gemma,
        "capabilities": capabilities,
        "startup_memory": startup_memory,
    }

    real_smoke_memory = {}
    expected_world_size = int((report.get("accelerate") or {}).get("num_processes") or 0)
    for task, path in (("i2i", args.i2i_smoke_result), ("r2v", args.r2v_smoke_result)):
        if path is not None:
            try:
                real_smoke_memory[task] = _real_smoke_memory(
                    path,
                    task=task,
                    expected_world_size=expected_world_size,
                )
            except Exception as exc:
                errors.append(f"{task}_smoke_memory: {type(exc).__name__}: {exc}")
    if real_smoke_memory:
        report["real_smoke_memory"] = real_smoke_memory

    if args.refresh_runtime_lock and args.runtime_lock is None:
        errors.append("--refresh-runtime-lock requires --runtime-lock")
    if args.runtime_lock is not None:
        try:
            if args.fsdp_smoke_result is None:
                raise ValueError("--runtime-lock requires --fsdp-smoke-result")
            if args.accelerate_prepare_smoke_result is None:
                raise ValueError("--runtime-lock requires --accelerate-prepare-smoke-result")
            if args.i2i_smoke_result is None or args.r2v_smoke_result is None:
                raise ValueError("--runtime-lock requires both --i2i-smoke-result and --r2v-smoke-result")
            smoke_path = args.fsdp_smoke_result.expanduser().resolve()
            smoke_result = json.loads(smoke_path.read_text(encoding="utf-8"))
            if not isinstance(smoke_result, dict):
                raise ValueError(f"FSDP smoke result must be a JSON object: {smoke_path}")
            accelerate_smoke_path = args.accelerate_prepare_smoke_result.expanduser().resolve()
            accelerate_smoke_result = json.loads(accelerate_smoke_path.read_text(encoding="utf-8"))
            if not isinstance(accelerate_smoke_result, dict):
                raise ValueError(f"Accelerate smoke result must be a JSON object: {accelerate_smoke_path}")
            report["ready"] = not errors
            runtime_lock = build_semantic_flow_runtime_lock(
                report,
                smoke_result,
                accelerate_smoke_result,
            )
            lock_path = assert_write_path_allowed(args.runtime_lock)
            action = write_or_validate_semantic_flow_runtime_lock(
                lock_path,
                runtime_lock,
                refresh=args.refresh_runtime_lock,
            )
            report["runtime_lock"] = {
                "path": str(lock_path),
                "action": action,
                "values": runtime_lock,
            }
        except Exception as exc:
            errors.append(f"runtime_lock: {type(exc).__name__}: {exc}")
    report["ready"] = not errors
    report["errors"] = errors
    _atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
