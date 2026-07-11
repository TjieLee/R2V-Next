#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-configs/multiref_stage1_full_tokens_2048_2000.yaml}"
: "${CHECKPOINT:?Set CHECKPOINT to the Stage 1 safetensors checkpoint}"
: "${MANIFEST:?Set MANIFEST to the inference manifest}"
: "${PRECOMPUTED:?Set PRECOMPUTED to the .precomputed root}"
: "${OUTPUT:?Set OUTPUT to the inference output directory}"

NUM_GPUS="${NUM_GPUS:-8}"
mkdir -p "${OUTPUT}/logs"
PIDS=()

for ((GPU_ID = 0; GPU_ID < NUM_GPUS; GPU_ID++)); do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    uv run python scripts/infer_multiref_stage1_overfit.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --manifest "${MANIFEST}" \
        --precomputed-root "${PRECOMPUTED}" \
        --output-dir "${OUTPUT}" \
        --all-samples \
        --shard-index "${GPU_ID}" \
        --num-shards "${NUM_GPUS}" \
        --device cuda:0 \
        --seed 42 \
        --num-inference-steps 50 \
        --condition-mode full_siglip \
        --guidance-scale 2 \
        --negative-prompt "worst quality, low quality, blurry, temporal inconsistency, jittery motion, frozen motion, deformed objects, duplicate subjects, unreadable text, subtitles, watermark, logo, compression artifacts" \
        --cfg-negative-mode negative_prompt_no_siglip \
        --cfg-keep-ref-latents-in-negative \
        --ref-guidance-scale 2 \
        --siglip-guidance-scale 0 \
        --stg-scale 1 \
        --stg-blocks 28 \
        --stg-mode stg_v \
        --guidance-rescale 0.7 \
        --skip-existing \
        --continue-on-error \
        --no-copy-media \
        --decode-tile \
        > "${OUTPUT}/logs/shard_${GPU_ID}.log" 2>&1 &
    PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAILED=1
    fi
done

if [[ "$FAILED" -ne 0 ]]; then
    echo "One or more inference shards completed with failures."
    exit 1
fi

echo "All inference shards completed successfully."
