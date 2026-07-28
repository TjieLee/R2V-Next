#!/usr/bin/env python3
"""Fail-closed production preflight for Semantic Flow Phase 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.model_loader import load_embeddings_processor
from ltx_trainer.online_data.path_safety import assert_write_path_allowed
from ltx_trainer.online_inference.checkpoint_runtime import read_checkpoint_metadata
from ltx_trainer.phase2_distributed_state import (
    validate_phase2_distributed_state,
)
from ltx_trainer.training_state import TrainingState
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


def _artifact_step(path: Path) -> int:
    match = re.search(r"_step_(\d+)", path.name)
    if match is None:
        raise RuntimeError(f"Cannot parse checkpoint step from {path.name}")
    return int(match.group(1))


def validate_phase2_resume_bundle(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    step = _artifact_step(checkpoint)
    if not checkpoint.is_file():
        raise RuntimeError(f"Phase 2 checkpoint is missing: {checkpoint}")

    marker_path, marker = _load_ready_marker(checkpoint)
    expected_training_state = (
        checkpoint.parent / f"training_state_step_{step:05d}.pt"
    ).resolve()
    expected_accelerator_state = (
        checkpoint.parent / f"accelerator_state_step_{step:05d}"
    ).resolve()
    try:
        marker_step = int(marker["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Phase 2 ready marker has invalid global_step: {marker_path}"
        ) from exc
    if marker_step != step:
        raise RuntimeError(
            f"Phase 2 ready marker step mismatch: filename={step}, marker={marker_step}"
        )
    marker_training_state = Path(
        str(marker.get("training_state_path", ""))
    ).expanduser().resolve()
    if marker_training_state != expected_training_state:
        raise RuntimeError(
            "Phase 2 ready marker training_state_path does not match its step"
        )
    marker_accelerator_state = Path(
        str(marker.get("accelerator_state_path", ""))
    ).expanduser().resolve()
    if marker_accelerator_state != expected_accelerator_state:
        raise RuntimeError(
            "Phase 2 ready marker accelerator_state_path does not match its step"
        )
    if not expected_training_state.is_file():
        raise RuntimeError(
            f"Phase 2 training state is missing: {expected_training_state}"
        )
    if not expected_accelerator_state.is_dir():
        raise RuntimeError(
            "Phase 2 distributed optimizer state is missing: "
            f"{expected_accelerator_state}"
        )
    distributed_state_manifest = validate_phase2_distributed_state(
        expected_accelerator_state,
        expected_step=step,
        validate_rank_payloads=True,
    )

    metadata = read_checkpoint_metadata(checkpoint)
    if metadata.get("training_phase") != "phase2":
        raise RuntimeError("Phase 2 resume checkpoint metadata is not training_phase=phase2")
    try:
        metadata_step = int(metadata["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Phase 2 checkpoint metadata has invalid global_step") from exc
    if metadata_step != step:
        raise RuntimeError(
            f"Phase 2 checkpoint metadata step mismatch: filename={step}, metadata={metadata_step}"
        )
    if marker.get("training_phase") != "phase2":
        raise RuntimeError("Phase 2 ready marker is missing training_phase=phase2")
    marker_metadata_step = marker.get("metadata_global_step")
    if marker_metadata_step is not None and int(marker_metadata_step) != step:
        raise RuntimeError(
            "Phase 2 ready marker metadata_global_step does not match its step"
        )

    try:
        raw_state = torch.load(
            expected_training_state,
            map_location="cpu",
            weights_only=False,
        )
        training_state = TrainingState.from_save_dict(raw_state)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Phase 2 training state: {expected_training_state}"
        ) from exc
    if training_state.global_step != step:
        raise RuntimeError(
            "Phase 2 training-state step mismatch: "
            f"filename={step}, training_state={training_state.global_step}"
        )
    scheduler_state = training_state.lr_scheduler_state_dict
    if not isinstance(scheduler_state, dict) or int(
        scheduler_state.get("last_epoch", -1)
    ) != step:
        raise RuntimeError(
            "Phase 2 training-state scheduler does not match the checkpoint step"
        )
    data_state = training_state.data_state
    if (
        not isinstance(data_state, dict)
        or data_state.get("task_schedule_cursor") != step
        or data_state.get("microstep_in_optimizer_step") != 0
    ):
        raise RuntimeError(
            "Phase 2 training-state sampler does not match the checkpoint step"
        )
    return {
        "global_step": step,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "ready_marker": str(marker_path),
        "training_state": str(expected_training_state),
        "accelerator_state": str(expected_accelerator_state),
        "distributed_optimizer_state": str(expected_accelerator_state),
        "distributed_optimizer_world_size": int(
            distributed_state_manifest["world_size"]
        ),
    }


def find_latest_phase2_resume_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    markers = sorted(
        checkpoint_dir.glob("checkpoint_step_*.ready.json"),
        key=_artifact_step,
        reverse=True,
    )
    if not markers:
        raise RuntimeError(
            f"No ready Phase 2 checkpoint found under {checkpoint_dir}"
        )
    latest_marker = markers[0]
    latest_step = _artifact_step(latest_marker)
    marker = _load_json_object(latest_marker)
    try:
        marker_step = int(marker["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Latest Phase 2 ready marker has invalid global_step: {latest_marker}"
        ) from exc
    if marker_step != latest_step:
        raise RuntimeError(
            "Latest Phase 2 ready marker step mismatch: "
            f"filename={latest_step}, marker={marker_step}"
        )
    checkpoint = Path(
        str(marker.get("checkpoint_path", ""))
    ).expanduser().resolve()
    if _artifact_step(checkpoint) != latest_step:
        raise RuntimeError(
            "Latest Phase 2 ready marker checkpoint step does not match "
            f"marker step {latest_step}: {checkpoint}"
        )
    validate_phase2_resume_bundle(checkpoint)
    return checkpoint


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read JSON evidence {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def validate_phase2_smoke_output(
    smoke_root: Path,
    *,
    expected_processes: int,
    expected_final_step: int,
    require_exact_resume: bool,
    require_inference: bool,
) -> dict[str, Any]:
    smoke_root = smoke_root.expanduser().resolve()
    checkpoint_dir = smoke_root / "checkpoints"
    final_checkpoint = find_latest_phase2_resume_checkpoint(checkpoint_dir)
    final_bundle = validate_phase2_resume_bundle(final_checkpoint)
    if final_bundle["global_step"] != expected_final_step:
        raise RuntimeError(
            "Phase 2 smoke final checkpoint step mismatch: "
            f"expected={expected_final_step}, actual={final_bundle['global_step']}"
        )
    if final_bundle["distributed_optimizer_world_size"] != expected_processes:
        raise RuntimeError(
            "Phase 2 smoke distributed optimizer world_size mismatch: "
            f"expected={expected_processes}, "
            f"actual={final_bundle['distributed_optimizer_world_size']}"
        )

    gradient_path = smoke_root / "phase2_gradient_audit.json"
    gradient = _load_json_object(gradient_path)
    if gradient.get("passed") is not True:
        raise RuntimeError("Phase 2 smoke gradient audit did not pass")
    if int(gradient.get("world_size", -1)) != expected_processes:
        raise RuntimeError("Phase 2 smoke gradient audit world size mismatch")
    parameters = gradient.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise RuntimeError("Phase 2 smoke gradient audit has no parameter records")
    invalid_parameters = [
        name
        for name, record in parameters.items()
        if not isinstance(record, dict)
        or record.get("requires_grad") is not True
        or record.get("grad_exists") is not True
        or record.get("grad_finite") is not True
    ]
    if invalid_parameters:
        raise RuntimeError(
            "Phase 2 smoke gradient parameter evidence is invalid: "
            f"{invalid_parameters[:20]}"
        )
    projection_records = [
        record
        for name, record in parameters.items()
        if name.startswith("feature_extractor.video_aggregate_embed.")
        or name.startswith("feature_extractor.aggregate_embed.")
    ]
    register_record = parameters.get("video_connector.learnable_registers")
    if not projection_records or not any(
        record.get("grad_nonzero") is True for record in projection_records
    ):
        raise RuntimeError("Phase 2 smoke projection gradient is zero")
    if (
        not isinstance(register_record, dict)
        or register_record.get("grad_nonzero") is not True
    ):
        raise RuntimeError("Phase 2 smoke learnable-register gradient is zero")
    frozen_gradients = gradient.get("frozen_module_gradients")
    if not isinstance(frozen_gradients, dict) or frozen_gradients != {
        "text_encoder": False,
        "vae_encoder": False,
        "audio_connector": False,
    }:
        raise RuntimeError("Phase 2 smoke frozen-module gradient evidence is invalid")
    groups = gradient.get("optimizer_groups")
    if (
        not isinstance(groups, list)
        or len(groups) != 2
        or not all(isinstance(group, dict) for group in groups)
        or [group.get("name") for group in groups]
        != ["dit_semantic", "conditioning_bridge"]
    ):
        raise RuntimeError("Phase 2 smoke optimizer group evidence is invalid")
    learning_rates = [
        float(group["learning_rate"])
        for group in groups
        if isinstance(group, dict)
    ]
    if learning_rates != [5.0e-6, 3.0e-6]:
        raise RuntimeError(
            f"Phase 2 smoke optimizer learning rates are invalid: {learning_rates}"
        )

    first_bundle: dict[str, Any] | None = None
    resume_audit: dict[str, Any] | None = None
    if require_exact_resume:
        first_checkpoint = checkpoint_dir / "model_weights_step_00001.safetensors"
        first_bundle = validate_phase2_resume_bundle(first_checkpoint)
        if first_bundle["distributed_optimizer_world_size"] != expected_processes:
            raise RuntimeError(
                "Phase 2 Stage A distributed optimizer world_size mismatch: "
                f"expected={expected_processes}, "
                f"actual={first_bundle['distributed_optimizer_world_size']}"
            )
        resume_path = smoke_root / "phase2_resume_runtime_audit.json"
        resume_audit = _load_json_object(resume_path)
        required_resume_values = {
            "initial_step": 1,
            "scheduler_last_epoch": 1,
            "sampler_task_schedule_cursor": 1,
            "sampler_microstep_in_optimizer_step": 0,
            "accelerator_state_restored": True,
            "distributed_optimizer_state_restored": True,
            "rng_state_restored": True,
            "world_size": expected_processes,
            "scheduler_restored": True,
            "sampler_restored": True,
            "optimizer_groups_restored": True,
            "optimizer_state_restored": True,
            "passed": True,
        }
        mismatches = {
            key: {"expected": expected, "actual": resume_audit.get(key)}
            for key, expected in required_resume_values.items()
            if resume_audit.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                f"Phase 2 exact-resume runtime evidence mismatch: {mismatches}"
            )
        optimizer_state = resume_audit.get("optimizer_state")
        if not isinstance(optimizer_state, dict):
            raise RuntimeError(
                "Phase 2 exact-resume audit is missing optimizer_state"
            )
        for group_name in ("dit_semantic", "conditioning_bridge"):
            group_state = optimizer_state.get(group_name)
            if not isinstance(group_state, dict):
                raise RuntimeError(
                    f"Phase 2 optimizer state is missing group {group_name}"
                )
            parameter_count = int(group_state.get("parameter_count", 0))
            state_count = int(group_state.get("parameters_with_state", 0))
            exp_avg_count = int(
                group_state.get("parameters_with_exp_avg", 0)
            )
            exp_avg_sq_count = int(
                group_state.get("parameters_with_exp_avg_sq", 0)
            )
            step_count = int(group_state.get("parameters_with_step", 0))
            if parameter_count <= 0:
                raise RuntimeError(
                    f"Phase 2 optimizer group {group_name} has no parameters"
                )
            if state_count <= 0:
                raise RuntimeError(
                    f"Phase 2 optimizer group {group_name} has no restored state"
                )
            if exp_avg_count != state_count:
                raise RuntimeError(
                    f"Phase 2 optimizer group {group_name} has incomplete exp_avg"
                )
            if exp_avg_sq_count != state_count:
                raise RuntimeError(
                    f"Phase 2 optimizer group {group_name} has incomplete exp_avg_sq"
                )
            if step_count != state_count:
                raise RuntimeError(
                    f"Phase 2 optimizer group {group_name} has incomplete state steps"
                )
            if group_state.get("exp_avg_finite") is not True:
                raise RuntimeError(
                    f"Phase 2 optimizer group {group_name} has invalid exp_avg"
                )
            if group_state.get("exp_avg_sq_finite") is not True:
                raise RuntimeError(
                    f"Phase 2 optimizer group {group_name} has invalid exp_avg_sq"
                )
            if (
                float(group_state.get("state_step_min", -1)) != 1.0
                or float(group_state.get("state_step_max", -1)) != 1.0
            ):
                raise RuntimeError(
                    f"Phase 2 optimizer group {group_name} has invalid state steps"
                )
        bridge_state = optimizer_state["conditioning_bridge"]
        if bridge_state.get("video_projection_state_restored") is not True:
            raise RuntimeError(
                "Phase 2 optimizer state is missing the video projection"
            )
        if bridge_state.get("learnable_registers_state_restored") is not True:
            raise RuntimeError(
                "Phase 2 optimizer state is missing learnable registers"
            )

    inference: dict[str, Any] = {}
    if require_inference:
        for mode in ("positive_ref", "latent_ref"):
            summary_path = smoke_root / "inference" / mode / "run_summary.json"
            summary = _load_json_object(summary_path)
            if (
                summary.get("guidance_mode") != mode
                or summary.get("success_count") != 1
                or summary.get("failure_count") != 0
            ):
                raise RuntimeError(
                    f"Phase 2 {mode} inference smoke evidence is invalid"
                )
            if Path(str(summary.get("checkpoint", ""))).resolve() != final_checkpoint:
                raise RuntimeError(
                    f"Phase 2 {mode} inference did not load the final checkpoint"
                )
            inference[mode] = {
                "run_summary": str(summary_path),
                "success_count": 1,
                "failure_count": 0,
            }

    is_exact_resume_smoke = require_exact_resume and expected_final_step == 2
    return {
        "passed": True,
        "world_size": expected_processes,
        "stage_a_step": 1 if is_exact_resume_smoke else expected_final_step,
        "stage_b_resume_initial_step": 1 if is_exact_resume_smoke else None,
        "stage_b_final_step": expected_final_step if is_exact_resume_smoke else None,
        "gradient_audit_passed": True,
        "checkpoint_step1_ready": (
            first_bundle is not None
            if is_exact_resume_smoke
            else expected_final_step == 1
        ),
        "checkpoint_step2_ready": (
            final_bundle["global_step"] == 2 if is_exact_resume_smoke else None
        ),
        "accelerator_state_restored": (
            bool(resume_audit["accelerator_state_restored"])
            if resume_audit is not None
            else None
        ),
        "distributed_optimizer_state_restored": (
            bool(resume_audit["distributed_optimizer_state_restored"])
            if resume_audit is not None
            else None
        ),
        "rng_state_restored": (
            bool(resume_audit["rng_state_restored"])
            if resume_audit is not None
            else None
        ),
        "scheduler_restored": (
            bool(resume_audit["scheduler_restored"])
            if resume_audit is not None
            else None
        ),
        "sampler_restored": (
            bool(resume_audit["sampler_restored"])
            if resume_audit is not None
            else None
        ),
        "optimizer_groups_restored": (
            bool(resume_audit["optimizer_groups_restored"])
            if resume_audit is not None
            else None
        ),
        "optimizer_state_restored": (
            bool(resume_audit["optimizer_state_restored"])
            if resume_audit is not None
            else None
        ),
        "bridge_strict_reload_passed": (
            set(inference) == {"positive_ref", "latent_ref"}
            if require_inference
            else None
        ),
        "positive_ref_inference_passed": (
            "positive_ref" in inference if require_inference else None
        ),
        "latent_ref_inference_passed": (
            "latent_ref" in inference if require_inference else None
        ),
        "expected_processes": expected_processes,
        "expected_final_step": expected_final_step,
        "gradient_audit": str(gradient_path),
        "gradient_parameter_count": len(parameters),
        "optimizer_group_names": ["dit_semantic", "conditioning_bridge"],
        "optimizer_learning_rates": learning_rates,
        "first_bundle": first_bundle,
        "final_bundle": final_bundle,
        "resume_runtime_audit": resume_audit,
        "inference": inference,
    }


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
    parser.add_argument("--config", type=Path)
    parser.add_argument("--accelerate-config", type=Path)
    parser.add_argument("--mode", choices=("start", "resume"))
    parser.add_argument("--expected-processes", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parameter-audit", type=Path)
    parser.add_argument("--find-latest-resume-checkpoint", type=Path)
    parser.add_argument("--validate-smoke-output", type=Path)
    parser.add_argument("--expected-final-step", type=int)
    parser.add_argument("--require-exact-resume", action="store_true")
    parser.add_argument("--require-inference", action="store_true")
    parser.add_argument("--result-output", type=Path)
    args = parser.parse_args()

    if args.find_latest_resume_checkpoint is not None:
        print(  # noqa: T201
            find_latest_phase2_resume_checkpoint(
                args.find_latest_resume_checkpoint
            )
        )
        return
    if args.validate_smoke_output is not None:
        if args.expected_final_step is None or args.result_output is None:
            parser.error(
                "--validate-smoke-output requires --expected-final-step and "
                "--result-output"
            )
        report = validate_phase2_smoke_output(
            args.validate_smoke_output,
            expected_processes=args.expected_processes,
            expected_final_step=args.expected_final_step,
            require_exact_resume=args.require_exact_resume,
            require_inference=args.require_inference,
        )
        result_path = assert_write_path_allowed(args.result_output)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201
        return
    required = {
        "--config": args.config,
        "--accelerate-config": args.accelerate_config,
        "--mode": args.mode,
        "--output": args.output,
        "--parameter-audit": args.parameter_audit,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    assert args.config is not None
    assert args.accelerate_config is not None
    assert args.mode is not None
    assert args.output is not None
    assert args.parameter_audit is not None
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
    else:
        validate_phase2_resume_bundle(checkpoint)

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
