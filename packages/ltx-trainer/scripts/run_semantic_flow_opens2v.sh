#!/usr/bin/env bash
set -euo pipefail

R2V_ROOT="/mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/semantic_flow_v2"

export TMPDIR="$R2V_ROOT/tmp"
export XDG_CACHE_HOME="$R2V_ROOT/cache/xdg"
export HF_HOME="$R2V_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$R2V_ROOT/cache/huggingface/transformers"
export TORCH_HOME="$R2V_ROOT/cache/torch"
export TRITON_CACHE_DIR="$R2V_ROOT/cache/triton"
export CUDA_CACHE_PATH="$R2V_ROOT/cache/cuda"
export UV_CACHE_DIR="$R2V_ROOT/cache/uv"
export WANDB_DIR="$R2V_ROOT/cache/wandb"
export PYTHONPYCACHEPREFIX="$R2V_ROOT/cache/pycache"

mkdir -p \
  "$TMPDIR" \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" \
  "$UV_CACHE_DIR" \
  "$WANDB_DIR" \
  "$PYTHONPYCACHEPREFIX" \
  "$R2V_ROOT/logs" \
  "$R2V_ROOT/checkpoints" \
  "$R2V_ROOT/train" \
  "$R2V_ROOT/manifests"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_CONFIG="$R2V_ROOT/manifests/multitask_online_480p121_opens2v.yaml"
TRAIN_CONFIG="$REPO_ROOT/packages/ltx-trainer/configs/semantic_flow_multitask_480p121.yaml"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$REPO_ROOT/packages/ltx-trainer/configs/accelerate_semantic_flow_fsdp_train_8gpu.yaml}"
FSDP_SMOKE_ACCELERATE_CONFIG="${FSDP_SMOKE_ACCELERATE_CONFIG:-$REPO_ROOT/packages/ltx-trainer/configs/accelerate_semantic_flow_fsdp_smoke_2gpu.yaml}"
SMOKE_ROOT="$R2V_ROOT/smoke"
INFERENCE_SMOKE_ROOT="/mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/inference/semantic_flow_v2_smoke"
RUNTIME_AUDIT="$R2V_ROOT/train/runtime_audit.json"
TRAIN_RUNTIME_AUDIT="$R2V_ROOT/train/runtime_audit_train.json"
RUNTIME_LOCK="$R2V_ROOT/train/runtime_lock.json"
SMOKE_SUCCESS_MARKER="$R2V_ROOT/train/semantic_flow_smoke_success.json"
FSDP_SMOKE_ROOT="$SMOKE_ROOT/fsdp_checkpoint"
FSDP_SMOKE_RESULT="$FSDP_SMOKE_ROOT/result.json"

semantic_flow_train_preflight() {
  local mode="$1"
  python3 - "$TRAIN_CONFIG" "$ACCELERATE_CONFIG" "$R2V_ROOT" "$RUNTIME_LOCK" "$mode" <<'PY'
import json
import os
import sys
from pathlib import Path

import yaml
from ltx_trainer.online_inference.runtime_lock import (
    collect_visible_cuda_hardware,
    validate_locked_gpu_hardware,
)

train_config_path = Path(sys.argv[1]).expanduser().resolve()
accelerate_config_path = Path(sys.argv[2]).expanduser().resolve()
r2v_root = Path(sys.argv[3]).expanduser()
runtime_lock_path = Path(sys.argv[4]).expanduser().resolve()
mode = sys.argv[5]
litengjie_root = Path("/mnt/workspace/litengjie")
errors = []


def path_value(raw):
    return Path(str(raw)).expanduser()


def require_file(label, raw):
    path = path_value(raw)
    if not path.is_file():
        errors.append(f"{label} is not a file: {path}")
    return path


def require_dir(label, raw):
    path = path_value(raw)
    if not path.is_dir():
        errors.append(f"{label} is not a directory: {path}")
    return path


def require_under_litengjie(label, raw):
    path = path_value(raw)
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(litengjie_root)
    except ValueError:
        errors.append(f"{label} must stay under {litengjie_root}: {path}")


if not train_config_path.is_file():
    errors.append(f"training config is missing: {train_config_path}")
if not accelerate_config_path.is_file():
    errors.append(f"accelerate config is missing: {accelerate_config_path}")
if errors:
    print(json.dumps({"ready": False, "errors": errors}, indent=2, sort_keys=True))
    sys.exit(1)

train_config = yaml.safe_load(train_config_path.read_text(encoding="utf-8")) or {}
accelerate_config = yaml.safe_load(accelerate_config_path.read_text(encoding="utf-8")) or {}
model_config = train_config.get("model") or {}
data_config = train_config.get("data") or {}
online_config = data_config.get("online_encoding") or {}

require_file("model.model_path", model_config.get("model_path"))
require_dir("model.text_encoder_path", model_config.get("text_encoder_path"))
require_file("data.train_data_config", data_config.get("train_data_config"))
manifest_path = require_file("data.manifest_path", data_config.get("manifest_path"))
require_file("data.manifest_path index", f"{manifest_path}.idx")

for label, raw in (
    ("R2V_ROOT", r2v_root),
    ("output_dir", train_config.get("output_dir")),
    ("runtime_reject_log_dir", online_config.get("runtime_reject_log_dir")),
):
    if raw is not None:
        require_under_litengjie(label, raw)

for env_name in (
    "TMPDIR",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "UV_CACHE_DIR",
    "WANDB_DIR",
    "PYTHONPYCACHEPREFIX",
):
    require_under_litengjie(env_name, os.environ[env_name])

fsdp_config = accelerate_config.get("fsdp_config") or {}
if accelerate_config.get("compute_environment") != "LOCAL_MACHINE":
    errors.append("Accelerate compute_environment must be LOCAL_MACHINE")
if accelerate_config.get("distributed_type") != "FSDP":
    errors.append("Accelerate distributed_type must be FSDP")
if accelerate_config.get("mixed_precision") != "bf16":
    errors.append("Accelerate mixed_precision must be bf16")
if int(accelerate_config.get("num_processes", 0)) < 4:
    errors.append("Real 22B semantic-flow smoke/training requires num_processes >= 4; 8 is recommended")
if fsdp_config.get("fsdp_version") != 1:
    errors.append("Accelerate fsdp_version must be 1")
if fsdp_config.get("fsdp_sharding_strategy") != "FULL_SHARD":
    errors.append("Accelerate fsdp_sharding_strategy must be FULL_SHARD")
if fsdp_config.get("fsdp_state_dict_type") != "FULL_STATE_DICT":
    errors.append("Accelerate fsdp_state_dict_type must be FULL_STATE_DICT")
if fsdp_config.get("fsdp_auto_wrap_policy") != "TRANSFORMER_BASED_WRAP":
    errors.append("Accelerate fsdp_auto_wrap_policy must be TRANSFORMER_BASED_WRAP")
if fsdp_config.get("fsdp_transformer_layer_cls_to_wrap") != "BasicAVTransformerBlock":
    errors.append("Accelerate must wrap BasicAVTransformerBlock")
if fsdp_config.get("fsdp_use_orig_params") is not True:
    errors.append("Accelerate fsdp_use_orig_params must be true")
if fsdp_config.get("fsdp_sync_module_states") is not True:
    errors.append("Accelerate fsdp_sync_module_states must be true")
if fsdp_config.get("fsdp_cpu_ram_efficient_loading") is not False:
    errors.append("Accelerate fsdp_cpu_ram_efficient_loading must be false")

num_processes = int(accelerate_config.get("num_processes", 0))
gpu_hardware = collect_visible_cuda_hardware()
if int(gpu_hardware["gpu_count"]) < num_processes:
    errors.append(
        f"Visible GPU count {gpu_hardware['gpu_count']} is smaller than accelerate num_processes {num_processes}"
    )
if mode == "train":
    try:
        runtime_lock = json.loads(runtime_lock_path.read_text(encoding="utf-8"))
        validate_locked_gpu_hardware(
            runtime_lock,
            gpu_hardware,
            num_processes=num_processes,
        )
    except Exception as exc:
        errors.append(f"runtime lock GPU validation failed: {type(exc).__name__}: {exc}")

report = {
    "ready": not errors,
    "accelerate_config": str(accelerate_config_path),
    "training_config": str(train_config_path),
    "gpu_hardware": gpu_hardware,
    "mode": mode,
    "errors": errors,
}
print(json.dumps(report, indent=2, sort_keys=True))
if errors:
    sys.exit(1)
PY
}

latest_smoke_checkpoint() {
  local output_dir="$1"
  local checkpoint
  checkpoint="$(find "$output_dir/checkpoints" -type f -name '*step_*.safetensors' | sort | tail -n 1)"
  if [[ -z "$checkpoint" ]]; then
    printf 'No smoke checkpoint found under %s/checkpoints\n' "$output_dir" >&2
    exit 1
  fi
  printf '%s\n' "$checkpoint"
}

case "${1:-}" in
  build-manifest)
    cp "$REPO_ROOT/packages/ltx-trainer/configs/multitask_online_480p121_opens2v.yaml" "$DATA_CONFIG"
    python "$REPO_ROOT/packages/ltx-trainer/scripts/build_multitask_online_manifest.py" \
      --train-data-config \
      "$DATA_CONFIG" \
      --output \
      "$R2V_ROOT/manifests/train_i2i_opens2v.jsonl" \
      --i2i-target-field video \
      --i2i-reference-field reference_images \
      --i2i-caption-field caption \
      --probe-workers 16 \
      --probe-batch-size 512
    ;;
  smoke)
    semantic_flow_train_preflight smoke
    python3 "$REPO_ROOT/packages/ltx-trainer/scripts/semantic_flow_runtime_audit.py" \
      --config "$TRAIN_CONFIG" \
      --accelerate-config "$ACCELERATE_CONFIG" \
      --output "$RUNTIME_AUDIT"
    accelerate launch --config_file "$FSDP_SMOKE_ACCELERATE_CONFIG" \
      "$REPO_ROOT/packages/ltx-trainer/scripts/smoke_semantic_flow_fsdp_checkpoint.py" \
      --output-dir "$FSDP_SMOKE_ROOT"
    accelerate launch --config_file "$ACCELERATE_CONFIG" \
      "$REPO_ROOT/packages/ltx-trainer/scripts/check_multitask_online_distributed.py" \
      --config "$TRAIN_CONFIG" \
      --task i2i \
      --output-dir "$SMOKE_ROOT/i2i"
    I2I_CHECKPOINT="$(latest_smoke_checkpoint "$SMOKE_ROOT/i2i")"
    accelerate launch --config_file "$ACCELERATE_CONFIG" \
      "$REPO_ROOT/packages/ltx-trainer/scripts/check_multitask_online_distributed.py" \
      --config "$TRAIN_CONFIG" \
      --task r2v \
      --init-checkpoint "$I2I_CHECKPOINT" \
      --output-dir "$SMOKE_ROOT/r2v"
    R2V_CHECKPOINT="$(latest_smoke_checkpoint "$SMOKE_ROOT/r2v")"
    python3 "$REPO_ROOT/packages/ltx-trainer/scripts/select_multitask_online_train_samples.py" \
      --manifest "$R2V_ROOT/manifests/train_i2i_opens2v.jsonl" \
      --output-dir "$INFERENCE_SMOKE_ROOT/selection" \
      --tasks i2i,r2v \
      --samples-per-task 1 \
      --overwrite
    python3 "$REPO_ROOT/packages/ltx-trainer/scripts/infer_multitask_online_train_samples.py" \
      --config "$TRAIN_CONFIG" \
      --samples "$INFERENCE_SMOKE_ROOT/selection/selected_samples.jsonl" \
      --output-root "$INFERENCE_SMOKE_ROOT/run/dry" \
      --latest-ready-dir "$SMOKE_ROOT/r2v/checkpoints" \
      --task both \
      --limit 2 \
      --dry-run \
      --overwrite
    python3 "$REPO_ROOT/packages/ltx-trainer/scripts/infer_multitask_online_train_samples.py" \
      --config "$TRAIN_CONFIG" \
      --samples "$INFERENCE_SMOKE_ROOT/selection/selected_samples.jsonl" \
      --output-root "$INFERENCE_SMOKE_ROOT/run/i2i" \
      --checkpoint "$I2I_CHECKPOINT" \
      --task i2i \
      --limit 1 \
      --num-inference-steps 2 \
      --no-dry-run \
      --overwrite
    python3 "$REPO_ROOT/packages/ltx-trainer/scripts/semantic_flow_runtime_audit.py" \
      --config "$TRAIN_CONFIG" \
      --accelerate-config "$ACCELERATE_CONFIG" \
      --output "$RUNTIME_AUDIT" \
      --runtime-lock "$RUNTIME_LOCK" \
      --fsdp-smoke-result "$FSDP_SMOKE_RESULT" \
      --refresh-runtime-lock
    CODE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    python3 "$REPO_ROOT/packages/ltx-trainer/scripts/semantic_flow_smoke_marker.py" write \
      --marker "$SMOKE_SUCCESS_MARKER" \
      --code-commit "$CODE_COMMIT" \
      --training-config "$TRAIN_CONFIG" \
      --accelerate-config "$ACCELERATE_CONFIG" \
      --i2i-checkpoint "$I2I_CHECKPOINT" \
      --r2v-checkpoint "$R2V_CHECKPOINT" \
      --runtime-audit "$RUNTIME_AUDIT" \
      --runtime-lock "$RUNTIME_LOCK" \
      --inference-summary "$INFERENCE_SMOKE_ROOT/run/i2i/run_summary.json"
    ;;
  train)
    semantic_flow_train_preflight train
    SKIP_SMOKE_GUARD=false
    for option in "${@:2}"; do
      case "$option" in
        --skip-smoke-guard)
          SKIP_SMOKE_GUARD=true
          ;;
        *)
          printf 'Unknown train option: %s\n' "$option" >&2
          exit 2
          ;;
      esac
    done
    if [[ "$SKIP_SMOKE_GUARD" == true ]]; then
      printf '%s\n' \
        "WARNING: --skip-smoke-guard bypasses validated checkpoints, inference artifacts, and runtime lock evidence."
    else
      CODE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
      python3 "$REPO_ROOT/packages/ltx-trainer/scripts/semantic_flow_smoke_marker.py" validate \
        --marker "$SMOKE_SUCCESS_MARKER" \
        --code-commit "$CODE_COMMIT" \
        --training-config "$TRAIN_CONFIG" \
        --accelerate-config "$ACCELERATE_CONFIG"
    fi
    python3 "$REPO_ROOT/packages/ltx-trainer/scripts/semantic_flow_runtime_audit.py" \
      --config "$TRAIN_CONFIG" \
      --accelerate-config "$ACCELERATE_CONFIG" \
      --output "$TRAIN_RUNTIME_AUDIT" \
      --runtime-lock "$RUNTIME_LOCK" \
      --fsdp-smoke-result "$FSDP_SMOKE_RESULT"
    accelerate launch --config_file "$ACCELERATE_CONFIG" \
      "$REPO_ROOT/packages/ltx-trainer/scripts/train.py" \
      "$TRAIN_CONFIG"
    ;;
  *)
    printf '%s\n' "Usage: $0 {build-manifest|smoke|train [--skip-smoke-guard]}"
    printf '%s\n' "Required distributed mode: FSDP FULL_SHARD with at least 4 processes; 8 is recommended."
    exit 2
    ;;
esac
