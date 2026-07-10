#!/usr/bin/env bash

set -Eeuo pipefail

# ==============================================================================
# 路径
# ==============================================================================
WORK_DIR="/mnt/workspace/litengjie/LTX-2/packages/ltx-trainer"

CONFIG="${WORK_DIR}/configs/multiref_stage1_visual_branch_2000.yaml"

CHECKPOINT="/mnt/workspace/litengjie/ltx2_multiref_stage1_visual_branch_2000/checkpoints/lora_weights_step_10000.safetensors"

MANIFEST="/mnt/workspace/litengjie/my_dataset/overfit_100/overfit_100.json"
PRECOMPUTED="/mnt/workspace/litengjie/my_dataset/overfit_100/.precomputed"

OUTPUT_DIR="/mnt/workspace/litengjie/my_dataset/overfit_2000/infer_stage1_cfg2_ref2_stg1_rescale0p7_test100_step80"
LOG_DIR="${OUTPUT_DIR}/logs"

# ==============================================================================
# GPU
# ==============================================================================
GPU_IDS=(0 1 2 3 4 5 6 7)
GPU_COUNT="${#GPU_IDS[@]}"

# 已存在非空视频时跳过，支持断点续跑
SKIP_EXISTING=1

# ==============================================================================
# 推理参数
# ==============================================================================
NUM_INFERENCE_STEPS=80
SEED=42

# N_R + 2.5 * (F - N_R)
GUIDANCE_SCALE=2

# 1.0 * (F - F_no_R)
REF_GUIDANCE_SCALE=2.0

# LTX-2.3 STG
STG_SCALE=1.0
STG_BLOCKS="28"

# LTX-2.3 guidance rescale
GUIDANCE_RESCALE=0.7

NEGATIVE_PROMPT="worst quality, low quality, inconsistent motion, blurry, jittery, distorted, deformed, artifacts"

# ==============================================================================
# Ctrl+C 清理后台任务
# ==============================================================================
cleanup() {
    trap - SIGINT SIGTERM

    echo
    echo "正在终止后台推理任务..."

    jobs -pr | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true

    exit 130
}

trap cleanup SIGINT SIGTERM

cd "$WORK_DIR"

# ==============================================================================
# 必要文件检查
# ==============================================================================
require_file() {
    if [[ ! -f "$1" ]]; then
        echo "错误：找不到文件：$1"
        exit 1
    fi
}

require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "错误：找不到目录：$1"
        exit 1
    fi
}

require_file "$CONFIG"
require_file "$CHECKPOINT"
require_file "$MANIFEST"
require_file "scripts/infer_multiref_stage1_overfit.py"

require_dir "$PRECOMPUTED"

for subdir in \
    latents \
    multi_reference_latents \
    vlm_conditions \
    gt_siglip_tokens
do
    require_dir "${PRECOMPUTED}/${subdir}"
done

# 检查当前代码是否包含新参数
if ! grep -q -- "--guidance-rescale" scripts/infer_multiref_stage1_overfit.py; then
    echo "错误：当前代码不支持 --guidance-rescale"
    exit 1
fi

if ! grep -q -- "--ref-guidance-scale" scripts/infer_multiref_stage1_overfit.py; then
    echo "错误：当前代码不支持 --ref-guidance-scale"
    exit 1
fi

# ==============================================================================
# 自动读取 manifest 样本数
# ==============================================================================
NUM_SAMPLES="$(
python - "$MANIFEST" <<'PY'
import csv
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
suffix = path.suffix.lower()

if suffix == ".json":
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        print(len(data))
    elif isinstance(data, dict):
        for key in ("samples", "data", "items"):
            rows = data.get(key)
            if isinstance(rows, list):
                print(len(rows))
                break
        else:
            raise SystemExit("JSON manifest 中没有找到样本列表")
    else:
        raise SystemExit("不支持的 JSON manifest 结构")

elif suffix == ".jsonl":
    print(
        sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    )

elif suffix == ".csv":
    with path.open("r", encoding="utf-8", newline="") as file:
        print(sum(1 for _ in csv.DictReader(file)))

else:
    raise SystemExit(f"不支持的 manifest 格式：{suffix}")
PY
)"

if [[ "$NUM_SAMPLES" -le 0 ]]; then
    echo "错误：manifest 中没有样本"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

rm -f "$LOG_DIR"/sample_*.ok
rm -f "$LOG_DIR"/sample_*.failed

echo "================================================================"
echo "Stage 1 多方向 Guidance 推理"
echo "================================================================"
echo "样本数：           $NUM_SAMPLES"
echo "GPU：              ${GPU_IDS[*]}"
echo "推理步数：         $NUM_INFERENCE_STEPS"
echo "CFG scale：        $GUIDANCE_SCALE"
echo "Ref scale：        $REF_GUIDANCE_SCALE"
echo "STG scale：        $STG_SCALE"
echo "STG block：        $STG_BLOCKS"
echo "Guidance rescale： $GUIDANCE_RESCALE"
echo "输出目录：         $OUTPUT_DIR"
echo
echo "公式："
echo "N_R + 2.5*(F-N_R) + 1.0*(F-F_no_R) + 1.0*(F-F_stg)"
echo "最后应用 guidance rescale=0.7"
echo "================================================================"

# ==============================================================================
# 单 GPU worker
# ==============================================================================
run_gpu() {
    local gpu_id="$1"
    local worker_index="$2"

    local sample_index
    local sample_log
    local output_video
    local status
    local failures=0

    echo "[GPU ${gpu_id}] worker 启动"

    for ((sample_index=worker_index; sample_index<NUM_SAMPLES; sample_index+=GPU_COUNT)); do
        sample_log="${LOG_DIR}/sample_${sample_index}.log"
        output_video="${OUTPUT_DIR}/sample_${sample_index}/generated_full_siglip.mp4"

        if [[ "$SKIP_EXISTING" == "1" && -s "$output_video" ]]; then
            touch "${LOG_DIR}/sample_${sample_index}.ok"
            echo "[GPU ${gpu_id}] 跳过已有 sample ${sample_index}"
            continue
        fi

        echo "[GPU ${gpu_id}] 开始 sample ${sample_index}"

        set +e

        CUDA_VISIBLE_DEVICES="$gpu_id" \
        PYTHONUNBUFFERED=1 \
        python scripts/infer_multiref_stage1_overfit.py \
            --config "$CONFIG" \
            --checkpoint "$CHECKPOINT" \
            --manifest "$MANIFEST" \
            --precomputed-root "$PRECOMPUTED" \
            --sample-index "$sample_index" \
            --output-dir "$OUTPUT_DIR" \
            --device cuda:0 \
            --seed "$SEED" \
            --num-inference-steps "$NUM_INFERENCE_STEPS" \
            --condition-mode full_siglip \
            --guidance-scale "$GUIDANCE_SCALE" \
            --negative-prompt "$NEGATIVE_PROMPT" \
            --cfg-negative-mode negative_prompt_no_siglip \
            --cfg-keep-ref-latents-in-negative \
            --ref-guidance-scale "$REF_GUIDANCE_SCALE" \
            --stg-scale "$STG_SCALE" \
            --stg-blocks "$STG_BLOCKS" \
            --stg-mode stg_v \
            --guidance-rescale "$GUIDANCE_RESCALE" \
            --decode-tile \
            > "$sample_log" 2>&1

        status=$?

        set -e

        if [[ "$status" -eq 0 && -s "$output_video" ]]; then
            touch "${LOG_DIR}/sample_${sample_index}.ok"
            echo "[GPU ${gpu_id}] 完成 sample ${sample_index}"
        else
            echo "$status" > "${LOG_DIR}/sample_${sample_index}.failed"
            echo "[GPU ${gpu_id}] sample ${sample_index} 失败，exit=${status}"
            echo "[GPU ${gpu_id}] 日志：${sample_log}"
            failures=$((failures + 1))
        fi
    done

    echo "[GPU ${gpu_id}] worker 结束，失败=${failures}"

    if [[ "$failures" -gt 0 ]]; then
        return 1
    fi
}

# ==============================================================================
# 启动 8 卡
# ==============================================================================
PIDS=()

for worker_index in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$worker_index]}"

    run_gpu "$gpu_id" "$worker_index" \
        > "${LOG_DIR}/gpu_${gpu_id}.log" 2>&1 &

    PIDS+=("$!")
done

WORKER_FAILURE=0

for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        WORKER_FAILURE=1
    fi
done

# ==============================================================================
# 结果统计
# ==============================================================================
SUCCESS_COUNT="$(
    find "$LOG_DIR" \
        -maxdepth 1 \
        -type f \
        -name 'sample_*.ok' \
        | wc -l \
        | tr -d ' '
)"

FAILED_COUNT="$(
    find "$LOG_DIR" \
        -maxdepth 1 \
        -type f \
        -name 'sample_*.failed' \
        | wc -l \
        | tr -d ' '
)"

VIDEO_COUNT="$(
    find "$OUTPUT_DIR" \
        -type f \
        -name 'generated_full_siglip.mp4' \
        | wc -l \
        | tr -d ' '
)"

echo
echo "================================================================"
echo "推理结束"
echo "================================================================"
echo "成功：       ${SUCCESS_COUNT}/${NUM_SAMPLES}"
echo "失败：       ${FAILED_COUNT}"
echo "视频数量：   ${VIDEO_COUNT}"
echo "输出目录：   ${OUTPUT_DIR}"
echo "日志目录：   ${LOG_DIR}"
echo "================================================================"

if [[ "$FAILED_COUNT" -gt 0 || "$WORKER_FAILURE" -ne 0 ]]; then
    echo
    echo "失败样本："

    for file in "$LOG_DIR"/sample_*.failed; do
        [[ -e "$file" ]] || continue
        basename "$file" .failed
    done

    exit 1
fi
