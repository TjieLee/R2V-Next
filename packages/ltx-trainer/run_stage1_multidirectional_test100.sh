#!/bin/bash

set -u
set -o pipefail

# ==============================================================================
# Ctrl+C / SIGTERM：清理所有后台 GPU 任务
# ==============================================================================
cleanup() {
    echo -e "\n检测到终止信号，正在清理所有 GPU 推理任务..."
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
# 模型、数据与输出路径
# ==============================================================================
CONFIG="configs/multiref_stage1_visual_branch_2000.yaml"

CHECKPOINT="/mnt/workspace/litengjie/ltx2_multiref_stage1_visual_branch_2000/checkpoints/lora_weights_step_10000.safetensors"

MANIFEST="/mnt/workspace/litengjie/my_dataset/overfit_100/overfit_100.json"
PRECOMPUTED="/mnt/workspace/litengjie/my_dataset/overfit_100/.precomputed"

OUTPUT_DIR="/mnt/workspace/litengjie/my_dataset/overfit_2000/infer_stage1_visual_branch_test100"
LOG_DIR="${OUTPUT_DIR}/logs"

# ==============================================================================
# 推理参数
# ==============================================================================
GPU_COUNT=8
NUM_INFERENCE_STEPS=50
SEED=42

# N_R + CFG * (F - N_R)
GUIDANCE_SCALE=4.0

# 独立 reference latent guidance：F - F_no_R
REF_GUIDANCE_SCALE=4.0

# F - F_stg
STG_SCALE=1.0
STG_BLOCKS="29"

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

for subdir in \
    latents \
    multi_reference_latents \
    vlm_conditions \
    gt_siglip_tokens
do
    if [ ! -d "$PRECOMPUTED/$subdir" ]; then
        echo "错误：缺少预计算目录：$PRECOMPUTED/$subdir"
        exit 1
    fi
done

if ! grep -q -- "--ref-guidance-scale" scripts/infer_multiref_stage1_overfit.py; then
    echo "错误：本地推理脚本还没有 --ref-guidance-scale 参数。"
    echo "请先更新 GitHub 最新代码。"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 清理本次运行的状态文件，不删除已有视频
rm -f "$LOG_DIR"/*.ok "$LOG_DIR"/*.failed

# ==============================================================================
# 自动读取 manifest 样本数量
# ==============================================================================
NUM_SAMPLES=$(
python - "$MANIFEST" <<'PY'
import csv
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
suffix = path.suffix.lower()

if suffix == ".json":
    data = json.loads(path.read_text(encoding="utf-8"))
    print(len(data))
elif suffix == ".jsonl":
    rows = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(len(rows))
elif suffix == ".csv":
    with path.open("r", encoding="utf-8", newline="") as f:
        print(sum(1 for _ in csv.DictReader(f)))
else:
    raise SystemExit(f"不支持的 manifest 格式：{suffix}")
PY
)

if [ "$NUM_SAMPLES" -le 0 ]; then
    echo "错误：manifest 中没有样本。"
    exit 1
fi

echo "================================================================"
echo "Stage 1 多方向 Guidance 推理"
echo "================================================================"
echo "Git commit：       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "配置文件：         $CONFIG"
echo "Checkpoint：       $CHECKPOINT"
echo "Manifest：         $MANIFEST"
echo "样本数量：         $NUM_SAMPLES"
echo "预计算目录：       $PRECOMPUTED"
echo "输出目录：         $OUTPUT_DIR"
echo "GPU 数量：         $GPU_COUNT"
echo "推理步数：         $NUM_INFERENCE_STEPS"
echo "CFG scale：        $GUIDANCE_SCALE"
echo "Ref scale：        $REF_GUIDANCE_SCALE"
echo "STG scale：        $STG_SCALE"
echo "STG blocks：       $STG_BLOCKS"
echo
echo "正分支 F：         full text/VLM + SigLIP + ref latents"
echo "负分支 N_R：       negative prompt + ref latents"
echo "No-ref 分支：      full text/VLM + SigLIP，无 ref latents"
echo "公式："
echo "N_R + 2.5*(F-N_R) + 1.0*(F-F_no_R) + 0.5*(F-F_stg)"
echo "================================================================"

# ==============================================================================
# 单 GPU 工作函数
# 每张 GPU 交错处理样本，例如 GPU 0：0,8,16...
# ==============================================================================
run_gpu() {
    local gpu_id=$1
    local i
    local sample_log
    local output_video
    local status

    echo "[GPU $gpu_id] 工作进程启动"

    for ((i=gpu_id; i<NUM_SAMPLES; i+=GPU_COUNT)); do
        sample_log="${LOG_DIR}/sample_${i}.log"
        output_video="${OUTPUT_DIR}/sample_${i}/generated_full_siglip.mp4"

        echo "[GPU $gpu_id] 开始 sample $i"

        CUDA_VISIBLE_DEVICES="$gpu_id" \
        PYTHONUNBUFFERED=1 \
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
            --ref-guidance-scale "$REF_GUIDANCE_SCALE" \
            --stg-scale "$STG_SCALE" \
            --stg-blocks "$STG_BLOCKS" \
            --stg-mode stg_v \
            --decode-tile \
            > "$sample_log" 2>&1

        status=$?

        if [ "$status" -eq 0 ] && [ -s "$output_video" ]; then
            touch "${LOG_DIR}/sample_${i}.ok"
            echo "[GPU $gpu_id] 完成 sample $i"
        else
            echo "$status" > "${LOG_DIR}/sample_${i}.failed"
            echo "[GPU $gpu_id] sample $i 失败，exit code=$status"
            echo "[GPU $gpu_id] 日志：$sample_log"
        fi
    done

    echo "[GPU $gpu_id] 工作进程结束"
}

# ==============================================================================
# 启动 8 张 GPU
# ==============================================================================
PIDS=()

for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
    run_gpu "$gpu" > "${LOG_DIR}/gpu_${gpu}.log" 2>&1 &
    PIDS+=("$!")
done

# 等待所有 GPU 进程
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

# ==============================================================================
# 运行结果统计
# ==============================================================================
SUCCESS_COUNT=$(
    find "$LOG_DIR" \
        -maxdepth 1 \
        -type f \
        -name 'sample_*.ok' \
        | wc -l
)

FAILED_COUNT=$(
    find "$LOG_DIR" \
        -maxdepth 1 \
        -type f \
        -name 'sample_*.failed' \
        | wc -l
)

VIDEO_COUNT=$(
    find "$OUTPUT_DIR" \
        -type f \
        -name 'generated_full_siglip.mp4' \
        | wc -l
)

echo "================================================================"
echo "全部推理任务结束"
echo "成功样本：       $SUCCESS_COUNT / $NUM_SAMPLES"
echo "失败样本：       $FAILED_COUNT"
echo "生成视频数量：   $VIDEO_COUNT"
echo "输出目录：       $OUTPUT_DIR"
echo "日志目录：       $LOG_DIR"
echo "================================================================"

if [ "$FAILED_COUNT" -gt 0 ]; then
    echo "失败样本："

    for failure_file in "$LOG_DIR"/sample_*.failed; do
        [ -e "$failure_file" ] || continue
        basename "$failure_file" .failed
    done

    echo
    echo "查看失败原因："
    echo "tail -n 100 $LOG_DIR/sample_<编号>.log"
    exit 1
fi

echo "所有任务已顺利完成。"
