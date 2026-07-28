#!/usr/bin/env python3
"""Fail-closed production preflight for Semantic Flow Phase 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.model_loader import load_embeddings_processor
from ltx_trainer.online_data.path_safety import assert_write_path_allowed
from ltx_trainer.online_inference.checkpoint_runtime import read_checkpoint_metadata
from ltx_trainer.training_strategies.semantic_flow_bridge import (
    configure_phase2_bridge_trainability,
    phase2_bridge_audit_rows,
    resolve_phase2_bridge_modules,
)

PHASE2_TRAINING_ARTIFACT_PATTERNS = (
    "checkpoint_step_*.ready.json",
    "model_weights_step_*.safetensors",
    "training_state_step_*.pt",
    "accelerator_state_step_*",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _load_ready_marker(checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    step = checkpoint.stem.rsplit("_step_", 1)[-1]
    marker = checkpoint.parent / f"checkpoint_step_{step}.ready.json"
    if not marker.is_file():
        raise RuntimeError(f"Checkpoint ready marker is missing: {marker}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if Path(str(payload.get("checkpoint_path", ""))).resolve() != checkpoint.resolve():
        raise RuntimeError("Ready marker checkpoint_path does not match configured checkpoint")
    actual_sha = _sha256(checkpoint)
    if payload.get("checkpoint_sha256") != actual_sha:
        raise RuntimeError("Parent checkpoint SHA256 does not match its ready marker")
    return marker, payload


def assert_phase2_start_output_is_empty(output_dir: Path, *, mode: str) -> None:
    if mode != "start":
        return
    checkpoint_dir = output_dir / "checkpoints"
    artifacts = sorted({path for pattern in PHASE2_TRAINING_ARTIFACT_PATTERNS for path in checkpoint_dir.glob(pattern)})
    if artifacts:
        raise RuntimeError(
            "Phase 2 output already contains training artifacts. "
            "Use `run_semantic_flow_phase2_8gpu.sh resume` instead of `start`. "
            f"First artifact: {artifacts[0]}"
        )


def validate_phase2_config_contract(
    raw_config: dict[str, Any],
    accelerate_config: dict[str, Any],
    *,
    mode: str,
    expected_processes: int = 8,
) -> dict[str, Any]:  # noqa: PLR0912
    strategy = raw_config.get("training_strategy") or {}
    optimization = raw_config.get("optimization") or {}
    checkpoints = raw_config.get("checkpoints") or {}
    model = raw_config.get("model") or {}
    probabilities = strategy.get("phase2_condition_probabilities") or {}
    expected_modes = {
        "til_111",
        "til_110",
        "til_101",
        "til_011",
        "til_100",
        "til_010",
        "til_001",
        "til_000",
    }
    errors: list[str] = []
    if strategy.get("name") != "semantic_flow" or strategy.get("training_phase") != "phase2":
        errors.append("training_strategy must explicitly select semantic_flow phase2")
    if raw_config.get("text_encoder_lora", {}).get("enabled") is not False:
        errors.append("Phase 2 Gemma LoRA must be disabled")
    if model.get("training_mode") != "full":
        errors.append("Phase 2 requires full DiT training")
    dit_lr = float(optimization.get("learning_rate", 0.0))
    raw_bridge_lr = optimization.get("bridge_learning_rate")
    bridge_lr = dit_lr if raw_bridge_lr is None else float(raw_bridge_lr)
    if dit_lr != 5.0e-6:
        errors.append("Phase 2 DiT/semantic learning rate must be 5e-6")
    if bridge_lr not in {3.0e-6, 5.0e-6}:
        errors.append("Phase 2 bridge learning rate must be 3e-6 or 5e-6")
    if set(probabilities) != expected_modes:
        errors.append("Phase 2 condition probabilities must define all eight T/I/L modes")
    if abs(sum(float(value) for value in probabilities.values()) - 1.0) > 1.0e-6:
        errors.append("Phase 2 condition probabilities must sum to 1")
    if int(accelerate_config.get("num_processes", 0)) != expected_processes:
        errors.append(f"Phase 2 accelerate config must use {expected_processes} processes")
    if accelerate_config.get("distributed_type") != "FSDP":
        errors.append("Phase 2 production requires FSDP")
    fsdp = accelerate_config.get("fsdp_config") or {}
    if fsdp.get("fsdp_version") != 1:
        errors.append("Phase 2 production requires FSDP1")
    if fsdp.get("fsdp_sharding_strategy") != "FULL_SHARD":
        errors.append("Phase 2 production requires FULL_SHARD")
    if fsdp.get("fsdp_state_dict_type") != "FULL_STATE_DICT":
        errors.append("Phase 2 production requires FULL_STATE_DICT")
    if accelerate_config.get("mixed_precision") != "bf16":
        errors.append("Phase 2 production requires bf16")
    if mode == "start" and checkpoints.get("no_resume") is not True:
        errors.append("Phase 2 start must reset state with checkpoints.no_resume=true")
    if mode == "resume" and checkpoints.get("no_resume") is not False:
        errors.append("Phase 2 resume must set checkpoints.no_resume=false")
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "condition_probability_sum": sum(float(value) for value in probabilities.values()),
        "condition_modes": sorted(probabilities),
        "dit_semantic_learning_rate": dit_lr,
        "conditioning_bridge_learning_rate": bridge_lr,
        "num_processes": int(accelerate_config["num_processes"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--accelerate-config", type=Path, required=True)
    parser.add_argument("--mode", choices=("start", "resume"), required=True)
    parser.add_argument("--expected-processes", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameter-audit", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    accelerate_path = args.accelerate_config.expanduser().resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    accelerate_config = yaml.safe_load(accelerate_path.read_text(encoding="utf-8")) or {}
    contract = validate_phase2_config_contract(
        raw_config,
        accelerate_config,
        mode=args.mode,
        expected_processes=args.expected_processes,
    )
    cfg = LtxTrainerConfig(**raw_config)
    output_dir = assert_write_path_allowed(cfg.output_dir)
    audit_path = assert_write_path_allowed(args.output)
    parameter_audit_path = assert_write_path_allowed(args.parameter_audit)
    phase1_output = Path("/mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/semantic_flow_v2/train").resolve()
    if output_dir.resolve() == phase1_output:
        raise RuntimeError("Phase 2 output directory must differ from Phase 1")
    assert_phase2_start_output_is_empty(output_dir, mode=args.mode)

    checkpoint = Path(str(cfg.model.load_checkpoint)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    marker_path, marker = _load_ready_marker(checkpoint)
    metadata = read_checkpoint_metadata(checkpoint)
    if args.mode == "start":
        if int(metadata.get("global_step", -1)) != cfg.training_strategy.parent_checkpoint_step:
            raise RuntimeError("Phase 2 parent checkpoint is not the configured Phase 1 step")
        if metadata.get("training_phase", "phase1") != "phase1":
            raise RuntimeError("Phase 2 start requires a Phase 1 parent checkpoint")
    elif metadata.get("training_phase") != "phase2":
        raise RuntimeError("Phase 2 resume requires a Phase 2 checkpoint")

    if torch.cuda.device_count() != args.expected_processes:
        raise RuntimeError(
            "Phase 2 preflight visible GPU count mismatch: "
            f"expected={args.expected_processes}, actual={torch.cuda.device_count()}"
        )

    processor = load_embeddings_processor(
        cfg.model.model_path,
        device="cpu",
        dtype=torch.bfloat16,
    )
    configure_phase2_bridge_trainability(processor)
    modules = resolve_phase2_bridge_modules(processor)
    rows = phase2_bridge_audit_rows(processor)
    bridge_rows = [row for row in rows if row["owner_group"] == "bridge"]
    if not bridge_rows:
        raise RuntimeError("Phase 2 bridge parameter audit is empty")
    parameter_audit_path.parent.mkdir(parents=True, exist_ok=True)
    with parameter_audit_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    repo_root = Path(__file__).resolve().parents[3]
    report = {
        "ready": True,
        "mode": args.mode,
        "git_commit": _git_commit(repo_root),
        "config": str(config_path),
        "accelerate_config": str(accelerate_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "ready_marker": str(marker_path),
        "ready_marker_global_step": marker.get("global_step"),
        "output_dir": str(output_dir),
        "visible_gpu_count": torch.cuda.device_count(),
        "contract": contract,
        "bridge_module_tree": {
            name: [
                {
                    "module_name": child_name,
                    "module_type": type(child).__name__,
                }
                for child_name, child in module.named_modules()
            ]
            for name, module in modules.items()
        },
        "bridge_key_count": len(bridge_rows),
        "bridge_parameter_count": sum(int(row["numel"]) for row in bridge_rows),
        "parameter_audit": str(parameter_audit_path),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
