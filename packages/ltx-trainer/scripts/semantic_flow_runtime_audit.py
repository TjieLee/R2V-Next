#!/usr/bin/env python3
"""Write a semantic-flow runtime/dependency audit JSON before smoke or training."""

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
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}, None
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def _torch_runtime() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    cuda_available = bool(torch.cuda.is_available())
    gpu_models = []
    if cuda_available:
        gpu_models = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    try:
        nccl_version = torch.cuda.nccl.version() if cuda_available else None
    except Exception as exc:
        nccl_version = f"{type(exc).__name__}: {exc}"
    return {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "nccl_version": nccl_version,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_models": gpu_models,
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


def _fsdp_report(accelerate_config: dict[str, Any], accelerate_error: str | None) -> dict[str, Any]:
    fsdp_config = accelerate_config.get("fsdp_config") or {}
    return {
        "config_error": accelerate_error,
        "distributed_type": accelerate_config.get("distributed_type"),
        "mixed_precision": accelerate_config.get("mixed_precision"),
        "version": fsdp_config.get("fsdp_version"),
        "sharding_strategy": fsdp_config.get("fsdp_sharding_strategy") or fsdp_config.get("fsdp_reshard_after_forward"),
        "auto_wrap_policy": fsdp_config.get("fsdp_auto_wrap_policy"),
        "transformer_layer_cls_to_wrap": fsdp_config.get("fsdp_transformer_layer_cls_to_wrap"),
        "state_dict_type": fsdp_config.get("fsdp_state_dict_type"),
        "sync_module_states": fsdp_config.get("fsdp_sync_module_states"),
        "cpu_ram_efficient_loading": fsdp_config.get("fsdp_cpu_ram_efficient_loading"),
        "use_orig_params": fsdp_config.get("fsdp_use_orig_params"),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--accelerate-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trainer_config, trainer_config_error = _load_yaml(args.config.expanduser().resolve())
    accelerate_config, accelerate_config_error = _load_yaml(args.accelerate_config.expanduser().resolve())
    output = assert_write_path_allowed(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "architecture": "semantic_flow_v1",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "accelerate", "safetensors")
        },
        "torch_runtime": _torch_runtime(),
        "config": {
            "path": str(args.config.expanduser().resolve()),
            "error": trainer_config_error,
        },
        "accelerate": _fsdp_report(accelerate_config, accelerate_config_error),
        "gemma": _gemma_sliding_window((trainer_config.get("model") or {}).get("text_encoder_path")),
    }
    _atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
