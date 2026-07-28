#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PHASE2_ROOT="/mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/semantic_flow_v2_phase2_parent14000"
PARENT_CHECKPOINT="/mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/semantic_flow_v2/train/checkpoints/model_weights_step_14000.safetensors"
BASE_CONFIG="$REPO_ROOT/packages/ltx-trainer/configs/semantic_flow_multitask_480p121_phase2.yaml"
TRAIN_ACCELERATE_CONFIG="$REPO_ROOT/packages/ltx-trainer/configs/accelerate_semantic_flow_fsdp_train_8gpu.yaml"
SMOKE_ACCELERATE_CONFIG="$REPO_ROOT/packages/ltx-trainer/configs/accelerate_semantic_flow_fsdp_smoke_2gpu.yaml"
RUNTIME_DIR="$PHASE2_ROOT/runtime"
START_AUDIT="$RUNTIME_DIR/phase2_start_audit.json"
RESUME_AUDIT="$RUNTIME_DIR/phase2_resume_audit.json"
PARAMETER_AUDIT="$RUNTIME_DIR/phase2_parameter_audit.jsonl"
CHECKER="$REPO_ROOT/packages/ltx-trainer/scripts/check_semantic_flow_phase2_runtime.py"
PYTHON="${PYTHON:-/mnt/workspace/litengjie/R2V-Next/.venv/bin/python}"
ACCELERATE="${ACCELERATE:-/mnt/workspace/litengjie/R2V-Next/.venv/bin/accelerate}"

if [[ ! -x "$PYTHON" ]]; then
  printf '%s\n' "Configured PYTHON is not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! -x "$ACCELERATE" ]]; then
  printf '%s\n' "Configured ACCELERATE is not executable: $ACCELERATE" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TMPDIR="${TMPDIR:-/mnt/workspace/litengjie/t}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PHASE2_ROOT/cache/xdg}"
export HF_HOME="${HF_HOME:-$PHASE2_ROOT/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PHASE2_ROOT/cache/huggingface/transformers}"
export TORCH_HOME="${TORCH_HOME:-$PHASE2_ROOT/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PHASE2_ROOT/cache/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$PHASE2_ROOT/cache/cuda}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PHASE2_ROOT/cache/uv}"
export WANDB_DIR="${WANDB_DIR:-$PHASE2_ROOT/cache/wandb}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$PHASE2_ROOT/cache/pycache}"

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
  "$RUNTIME_DIR" || exit $?

latest_phase2_checkpoint() {
  local checkpoint_dir="${1:-$PHASE2_ROOT/train/checkpoints}"
  "$PYTHON" "$CHECKER" --find-latest-resume-checkpoint "$checkpoint_dir"
}

write_runtime_config() {
  local source_config="$1"
  local destination_config="$2"
  local checkpoint="$3"
  local no_resume="$4"
  local steps="$5"
  local output_dir="$6"
  local interval="$7"
  "$PYTHON" - "$source_config" "$destination_config" "$checkpoint" "$no_resume" "$steps" "$output_dir" "$interval" <<'PY'
import sys
from pathlib import Path

import yaml

source, destination, checkpoint, no_resume, steps, output_dir, interval = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
config["model"]["load_checkpoint"] = checkpoint
config["checkpoints"]["no_resume"] = no_resume == "true"
config["checkpoints"]["interval"] = int(interval)
config["optimization"]["steps"] = int(steps)
config["output_dir"] = output_dir
path = Path(destination)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(path)
PY
}

run_preflight() {
  local config="$1"
  local mode="$2"
  local accelerate_config="$3"
  local expected_processes="$4"
  local audit="$5"
  "$PYTHON" "$CHECKER" \
    --config "$config" \
    --accelerate-config "$accelerate_config" \
    --mode "$mode" \
    --expected-processes "$expected_processes" \
    --output "$audit" \
    --parameter-audit "$PARAMETER_AUDIT"
}

launch_train() {
  local config="$1"
  local accelerate_config="$2"
  local smoke_audit="${3:-false}"
  local extra_args=()
  if [[ "$smoke_audit" == "true" ]]; then
    extra_args+=(--phase2-smoke-audit)
  fi
  "$ACCELERATE" launch --config_file "$accelerate_config" \
    "$REPO_ROOT/packages/ltx-trainer/scripts/train.py" \
    "$config" \
    --disable-progress-bars \
    "${extra_args[@]}"
}

run_phase2_inference_smoke() {
  local smoke_root="$1"
  local config="$2"
  local checkpoint="$3"
  local inference_root="$smoke_root/inference"
  local selection_root="$inference_root/selection"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
    "$REPO_ROOT/packages/ltx-trainer/scripts/select_multitask_online_train_samples.py" \
    --manifest "/mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/semantic_flow_v2/manifests/train_i2i_opens2v.jsonl" \
    --output-dir "$selection_root" \
    --tasks r2v \
    --samples-per-task 1 \
    --overwrite || exit $?
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
    "$REPO_ROOT/packages/ltx-trainer/scripts/infer_multitask_online_train_samples.py" \
    --config "$config" \
    --samples "$selection_root/selected_samples.jsonl" \
    --output-root "$inference_root/positive_ref" \
    --checkpoint "$checkpoint" \
    --task r2v \
    --limit 1 \
    --num-inference-steps 2 \
    --guidance-mode positive_ref \
    --guidance-scale 4 \
    --ref-guidance-scale 0 \
    --stg-scale 1 \
    --no-dry-run \
    --overwrite || exit $?
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
    "$REPO_ROOT/packages/ltx-trainer/scripts/infer_multitask_online_train_samples.py" \
    --config "$config" \
    --samples "$selection_root/selected_samples.jsonl" \
    --output-root "$inference_root/latent_ref" \
    --checkpoint "$checkpoint" \
    --task r2v \
    --limit 1 \
    --num-inference-steps 2 \
    --guidance-mode latent_ref \
    --guidance-scale 4 \
    --ref-guidance-scale 1 \
    --stg-scale 1 \
    --no-dry-run \
    --overwrite
}

validate_smoke_output() {
  local smoke_root="$1"
  local expected_processes="$2"
  local expected_final_step="$3"
  local result_output="$4"
  shift 4
  "$PYTHON" "$CHECKER" \
    --validate-smoke-output "$smoke_root" \
    --expected-processes "$expected_processes" \
    --expected-final-step "$expected_final_step" \
    --result-output "$result_output" \
    "$@"
}

case "${1:-}" in
  preflight)
    run_preflight "$BASE_CONFIG" start "$TRAIN_ACCELERATE_CONFIG" 8 "$START_AUDIT"
    ;;
  start)
    run_preflight "$BASE_CONFIG" start "$TRAIN_ACCELERATE_CONFIG" 8 "$START_AUDIT" || exit $?
    launch_train "$BASE_CONFIG" "$TRAIN_ACCELERATE_CONFIG" || exit $?
    ;;
  resume)
    RESUME_CHECKPOINT="$(latest_phase2_checkpoint)" || exit $?
    RESUME_CONFIG="$RUNTIME_DIR/semantic_flow_phase2_resume.yaml"
    write_runtime_config \
      "$BASE_CONFIG" \
      "$RESUME_CONFIG" \
      "$RESUME_CHECKPOINT" \
      false \
      16000 \
      "$PHASE2_ROOT/train" \
      1000 || exit $?
    run_preflight "$RESUME_CONFIG" resume "$TRAIN_ACCELERATE_CONFIG" 8 "$RESUME_AUDIT" || exit $?
    launch_train "$RESUME_CONFIG" "$TRAIN_ACCELERATE_CONFIG" || exit $?
    ;;
  smoke-2gpu)
    export CUDA_VISIBLE_DEVICES="${PHASE2_SMOKE_2GPU_DEVICES:-0,1}"
    SMOKE_ROOT="$PHASE2_ROOT/smoke/2gpu"
    SMOKE_CONFIG="$RUNTIME_DIR/semantic_flow_phase2_smoke_2gpu.yaml"
    write_runtime_config \
      "$BASE_CONFIG" \
      "$SMOKE_CONFIG" \
      "$PARENT_CHECKPOINT" \
      true \
      1 \
      "$SMOKE_ROOT" \
      1 || exit $?
    run_preflight \
      "$SMOKE_CONFIG" start "$SMOKE_ACCELERATE_CONFIG" 2 \
      "$RUNTIME_DIR/phase2_smoke_2gpu_audit.json" || exit $?
    launch_train "$SMOKE_CONFIG" "$SMOKE_ACCELERATE_CONFIG" true || exit $?
    validate_smoke_output \
      "$SMOKE_ROOT" \
      2 \
      1 \
      "$SMOKE_ROOT/phase2_smoke_2gpu_result.json" || exit $?
    ;;
  smoke-8gpu)
    export CUDA_VISIBLE_DEVICES="${PHASE2_SMOKE_8GPU_DEVICES:-0,1,2,3,4,5,6,7}"
    SMOKE_ROOT="$PHASE2_ROOT/smoke/8gpu"
    STAGE_A_CONFIG="$RUNTIME_DIR/semantic_flow_phase2_smoke_8gpu_stage_a.yaml"
    STAGE_B_CONFIG="$RUNTIME_DIR/semantic_flow_phase2_smoke_8gpu_stage_b.yaml"

    write_runtime_config \
      "$BASE_CONFIG" \
      "$STAGE_A_CONFIG" \
      "$PARENT_CHECKPOINT" \
      true \
      1 \
      "$SMOKE_ROOT" \
      1 || exit $?
    run_preflight \
      "$STAGE_A_CONFIG" start "$TRAIN_ACCELERATE_CONFIG" 8 \
      "$RUNTIME_DIR/phase2_smoke_8gpu_stage_a_audit.json" || exit $?
    launch_train "$STAGE_A_CONFIG" "$TRAIN_ACCELERATE_CONFIG" true || exit $?

    STAGE_A_CHECKPOINT="$(latest_phase2_checkpoint "$SMOKE_ROOT/checkpoints")" || exit $?
    if [[ "$(basename "$STAGE_A_CHECKPOINT")" != "model_weights_step_00001.safetensors" ]]; then
      printf '%s\n' "Phase 2 smoke Stage A did not publish step 1: $STAGE_A_CHECKPOINT" >&2
      exit 1
    fi
    write_runtime_config \
      "$BASE_CONFIG" \
      "$STAGE_B_CONFIG" \
      "$STAGE_A_CHECKPOINT" \
      false \
      2 \
      "$SMOKE_ROOT" \
      1 || exit $?
    run_preflight \
      "$STAGE_B_CONFIG" resume "$TRAIN_ACCELERATE_CONFIG" 8 \
      "$RUNTIME_DIR/phase2_smoke_8gpu_stage_b_audit.json" || exit $?
    launch_train "$STAGE_B_CONFIG" "$TRAIN_ACCELERATE_CONFIG" false || exit $?

    STAGE_B_CHECKPOINT="$(latest_phase2_checkpoint "$SMOKE_ROOT/checkpoints")" || exit $?
    if [[ "$(basename "$STAGE_B_CHECKPOINT")" != "model_weights_step_00002.safetensors" ]]; then
      printf '%s\n' "Phase 2 smoke Stage B did not publish step 2: $STAGE_B_CHECKPOINT" >&2
      exit 1
    fi
    run_phase2_inference_smoke \
      "$SMOKE_ROOT" \
      "$STAGE_B_CONFIG" \
      "$STAGE_B_CHECKPOINT" || exit $?
    validate_smoke_output \
      "$SMOKE_ROOT" \
      8 \
      2 \
      "$SMOKE_ROOT/phase2_smoke_8gpu_result.json" \
      --require-exact-resume \
      --require-inference
    ;;
  *)
    printf '%s\n' "Usage: $0 {preflight|start|resume|smoke-2gpu|smoke-8gpu}"
    exit 2
    ;;
esac
