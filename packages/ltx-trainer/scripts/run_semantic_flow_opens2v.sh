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
  train)
    : "${ACCELERATE_CONFIG:?Set ACCELERATE_CONFIG to an Accelerate FSDP FULL_SHARD config file.}"
    accelerate launch --config_file "$ACCELERATE_CONFIG" \
      "$REPO_ROOT/packages/ltx-trainer/scripts/train.py" \
      "$TRAIN_CONFIG"
    ;;
  *)
    printf '%s\n' "Usage: $0 {build-manifest|train}"
    printf '%s\n' "Required distributed mode: FSDP FULL_SHARD"
    printf '%s\n' "Plain DDP is unsupported for full 22B DiT semantic-flow training."
    exit 2
    ;;
esac
