#!/bin/bash

set -u
set -o pipefail

# ==============================================================================
# Ctrl+C / SIGTERM：清理所有后台 GPU 推理任务
# ==============================================================================
cleanup() {
    echo -e "\n检测到终止信号，正在清理所有后台 GPU 推理任务..."
    jobs -pr | xargs -r kill
    wait
    exit 130
}

trap cleanup SIGINT SIGTERM

# ==============================================================================
# 工作目录
# ==============================================================================
cd /mnt/workspace/litengjie/LTX-2/packages/ltx-trainer || exit 1

# ==============================================================================
# 路径设置
# ==============================================================================
CONFIG="configs/multiref_stage1_visual_branch_2000.yaml"
CHECKPOINT="/mnt/workspace/litengjie/ltx2_multiref_stage1_visual_branch_2000/checkpoints/lora_weights_step_10000.safetensors"

MANIFEST="/mnt/workspace/litengjie/my_dataset/overfit_100/overfit_100.json"
PRECOMPUTED="/mnt/workspace/litengjie/my_dataset/overfit_100/.precomputed"

OUTPUT_DIR="/mnt/workspace/litengjie/my_dataset/overfit_2000/infer_stage1_visual_branch_test100_cfg3_stg1_neg_with_ref"
LOG_DIR="${OUTPUT_DIR}/logs"

# ==============================================================================
# 推理参数
# ==============================================================================
NUM_INFERENCE_STEPS=50
GUIDANCE_SCALE=3.0
STG_SCALE=1.0
STG_BLOCKS="29"
SEED=42

NEGATIVE_PROMPT="worst quality, low quality, inconsistent motion, blurry, jittery, distorted, deformed, artifacts"

# ==============================================================================
# 启动前检查
# ==============================================================================
if [ ! -f "$CONFIG" ]; then
    echo "错误：找不到配置文件：$CONFIG"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "错误：找不到 checkpoint：$CHECKPOINT"
    exit 1
fi

if [ ! -f "$MANIFEST" ]; then
    echo "错误：找不到 manifest：$MANIFEST"
    exit 1
fi

if [ ! -d "$PRECOMPUTED" ]; then
    echo "错误：找不到预计算目录：$PRECOMPUTED"
    exit 1
fi

# full_siglip 所需目录
for subdir in latents multi_reference_latents vlm_conditions gt_siglip_tokens; do
    if [ ! -d "$PRECOMPUTED/$subdir" ]; then
        echo "错误：缺少预计算目录：$PRECOMPUTED/$subdir"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Stage 1 visual branch 推理（negative + ref latents）"
echo "================================================================"
echo "配置文件：      $CONFIG"
echo "Checkpoint：    $CHECKPOINT"
echo "测试 Manifest： $MANIFEST"
echo "预计算目录：    $PRECOMPUTED"
echo "输出目录：      $OUTPUT_DIR"
echo "正分支：        full_siglip"
echo "负分支：        negative_prompt_no_siglip + keep_ref_latents"
echo "CFG：           $GUIDANCE_SCALE"
echo "STG：           $STG_SCALE"
echo "STG blocks：    $STG_BLOCKS"
echo "推理步数：      $NUM_INFERENCE_STEPS"
echo "================================================================"

# ==============================================================================
# 单 GPU 执行函数
# ==============================================================================
run_gpu() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3

    echo "[GPU $gpu_id] 启动，负责样本 $start_idx - $end_idx"

    for i in $(seq "$start_idx" "$end_idx"); do
        local sample_log="${LOG_DIR}/sample_${i}.log"

        echo "[GPU $gpu_id] 开始处理 sample $i"

        CUDA_VISIBLE_DEVICES="$gpu_id" \
        python scripts/infer_multiref_stage1_overfit.py \
            --config "$CONFIG" \
            --checkpoint "$CHECKPOINT" \
            --manifest "$MANIFEST" \
            --precomputed-root "$PRECOMPUTED" \
            --sample-index "$i" \
            --output-dir "$OUTPUT_DIR" \
            --device cuda:0 \
            --seed "$SEED" \
            --num-inference-steps "$NUM_INFERENCE_STEPS" \
            --condition-mode full_siglip \
            --guidance-scale "$GUIDANCE_SCALE" \
            --negative-prompt "$NEGATIVE_PROMPT" \
            --cfg-negative-mode negative_prompt_no_siglip \
            --cfg-keep-ref-latents-in-negative \
            --stg-scale "$STG_SCALE" \
            --stg-blocks "$STG_BLOCKS" \
            --stg-mode stg_v \
            --decode-tile \
            > "$sample_log" 2>&1

        status=$?

        if [ "$status" -eq 0 ]; then
            echo "[GPU $gpu_id] 完成 sample $i"
        else
            echo "[GPU $gpu_id] sample $i 失败，exit code=$status"
            echo "[GPU $gpu_id] 日志：$sample_log"
        fi
    done

    echo "[GPU $gpu_id] 全部任务完成"
}

# ==============================================================================
# 8 卡并行处理 100 条数据
# ==============================================================================
run_gpu 0 0 12 &
PID0=$!

run_gpu 1 13 25 &
PID1=$!

run_gpu 2 26 38 &
PID2=$!

run_gpu 3 39 51 &
PID3=$!

run_gpu 4 52 63 &
PID4=$!

run_gpu 5 64 75 &
PID5=$!

run_gpu 6 76 87 &
PID6=$!

run_gpu 7 88 99 &
PID7=$!

wait "$PID0"
wait "$PID1"
wait "$PID2"
wait "$PID3"
wait "$PID4"
wait "$PID5"
wait "$PID6"
wait "$PID7"

# ==============================================================================
# 结果检查
# ==============================================================================
SAMPLE_DIR_COUNT=$(
    find "$OUTPUT_DIR" \
        -maxdepth 1 \
        -type d \
        -name 'sample_*' \
        | wc -l
)

FAILED_LOGS=$(
    grep -lE \
        'Traceback|RuntimeError|CUDA out of memory|FileNotFoundError|ValueError|AssertionError' \
        "$LOG_DIR"/sample_*.log 2>/dev/null || true
)

if [ -n "$FAILED_LOGS" ]; then
    FAILED_COUNT=$(echo "$FAILED_LOGS" | wc -l)
else
    FAILED_COUNT=0
fi

echo "================================================================"
echo "全部推理任务已结束"
echo "生成的 sample 目录数量：$SAMPLE_DIR_COUNT"
echo "检测到失败日志数量：    $FAILED_COUNT"
echo "输出目录：              $OUTPUT_DIR"
echo "日志目录：              $LOG_DIR"
echo "================================================================"

if [ "$FAILED_COUNT" -gt 0 ]; then
    echo "失败样本日志："
    echo "$FAILED_LOGS"
    exit 1
fi

echo "所有任务已顺利完成。"
