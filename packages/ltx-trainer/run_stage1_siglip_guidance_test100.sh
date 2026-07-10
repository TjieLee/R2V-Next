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

# ==============================================================================
# 可调参数
# ==============================================================================
# 运行示例：
# SIGLIP_GUIDANCE_SCALE=0   ./run_stage1_siglip_guidance_test100.sh
# SIGLIP_GUIDANCE_SCALE=0.5 ./run_stage1_siglip_guidance_test100.sh
# SIGLIP_GUIDANCE_SCALE=1   ./run_stage1_siglip_guidance_test100.sh
# SIGLIP_GUIDANCE_SCALE=2   ./run_stage1_siglip_guidance_test100.sh

SIGLIP_GUIDANCE_SCALE="${SIGLIP_GUIDANCE_SCALE:-1.0}"

# 可指定测试范围。END_INDEX=-1 表示运行到 manifest 最后一条。
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"

# 已存在非空视频时跳过
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# ==============================================================================
# 固定 Guidance 参数
# ==============================================================================
NUM_INFERENCE_STEPS=50
SEED=42

GUIDANCE_SCALE=2.0
REF_GUIDANCE_SCALE=2.0

STG_SCALE=1.0
STG_BLOCKS="28"

GUIDANCE_RESCALE=0.7

NEGATIVE_PROMPT="worst quality, low quality, inconsistent motion, blurry, jittery, distorted, deformed, artifacts"

GPU_IDS=(0 1 2 3 4 5 6 7)
GPU_COUNT="${#GPU_IDS[@]}"

# 将 1.5 转为 1p5，-1 转为 m1，用于目录名
SIGLIP_TAG="$(
python - "$SIGLIP_GUIDANCE_SCALE" <<'PY'
import sys

value = float(sys.argv[1])
text = f"{value:g}"
print(text.replace("-", "m").replace(".", "p"))
PY
)"

OUTPUT_DIR="/mnt/workspace/litengjie/my_dataset/overfit_2000/infer_stage1_cfg2_ref2_siglip${SIGLIP_TAG}_stg1_b28_rescale0p7_test100"
LOG_DIR="${OUTPUT_DIR}/logs"

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
# 路径检查
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

if ! grep -q -- "--siglip-guidance-scale" scripts/infer_multiref_stage1_overfit.py; then
    echo "错误：当前推理代码不支持 --siglip-guidance-scale"
    exit 1
fi

# ==============================================================================
# 自动读取 manifest 样本数量
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
    print(sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ))

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

if [[ "$END_INDEX" -lt 0 ]]; then
    END_INDEX=$((NUM_SAMPLES - 1))
fi

if [[ "$START_INDEX" -lt 0 || "$END_INDEX" -ge "$NUM_SAMPLES" || "$START_INDEX" -gt "$END_INDEX" ]]; then
    echo "错误：样本范围无效：${START_INDEX}-${END_INDEX}，manifest 数量=${NUM_SAMPLES}"
    exit 1
fi

RUN_SAMPLE_COUNT=$((END_INDEX - START_INDEX + 1))

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

rm -f "$LOG_DIR"/sample_*.ok
rm -f "$LOG_DIR"/sample_*.failed

echo "================================================================"
echo "Stage 1 Isolated SigLIP Guidance 推理"
echo "================================================================"
echo "样本范围：         ${START_INDEX}-${END_INDEX}"
echo "样本数量：         ${RUN_SAMPLE_COUNT}"
echo "GPU：              ${GPU_IDS[*]}"
echo "推理步数：         $NUM_INFERENCE_STEPS"
echo
echo "CFG scale：        $GUIDANCE_SCALE"
echo "Ref scale：        $REF_GUIDANCE_SCALE"
echo "SigLIP scale：     $SIGLIP_GUIDANCE_SCALE"
echo "STG scale：        $STG_SCALE"
echo "STG block：        $STG_BLOCKS"
echo "Guidance rescale： $GUIDANCE_RESCALE"
echo
echo "输出目录：         $OUTPUT_DIR"
echo
echo "公式："
echo "N_R"
echo "+ 2.0 * (F - N_R)"
echo "+ 2.0 * (F - F_no_R)"
echo "+ ${SIGLIP_GUIDANCE_SCALE} * (V - Z)"
echo "+ 1.0 * (F - F_stg)"
echo
echo "V = null_text + SigLIP + ref latents"
echo "Z = null_text + zero SigLIP + ref latents"
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

    for ((
        sample_index=START_INDEX + worker_index;
        sample_index<=END_INDEX;
        sample_index+=GPU_COUNT
    )); do
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
            --siglip-guidance-scale "$SIGLIP_GUIDANCE_SCALE" \
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
# 汇总
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
echo "成功：       ${SUCCESS_COUNT}/${RUN_SAMPLE_COUNT}"
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
