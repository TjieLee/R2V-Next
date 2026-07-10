#!/usr/bin/env bash

set -Eeuo pipefail

# ==============================================================================
# 路径
# ==============================================================================
WORK_DIR="/mnt/workspace/litengjie/LTX-2/packages/ltx-trainer"

CONFIG="${WORK_DIR}/configs/multiref_stage1_visual_branch_2000.yaml"

CKPT_ROOT="/mnt/workspace/litengjie/ltx2_multiref_stage1_visual_branch_2000/checkpoints"
BASELINE_CKPT="${CKPT_ROOT}/lora_weights_step_10000.safetensors"
DIAG_DIR="${CKPT_ROOT}/gate_diagnostics"

MANIFEST="/mnt/workspace/litengjie/my_dataset/overfit_100/overfit_100.json"
PRECOMPUTED="/mnt/workspace/litengjie/my_dataset/overfit_100/.precomputed"

# 默认只测试前 8 条
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-7}"

# 是否额外跑 residual_gate=0.1 的极端诊断
INCLUDE_EXTREME="${INCLUDE_EXTREME:-0}"

# 已存在视频时跳过
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# ==============================================================================
# 固定推理参数
# ==============================================================================
NUM_INFERENCE_STEPS=50
SEED=42

# 当前较好的 2 / 2 / 1
GUIDANCE_SCALE=2.0
REF_GUIDANCE_SCALE=2.0
STG_SCALE=1.0

STG_BLOCKS="28"
GUIDANCE_RESCALE="${GUIDANCE_RESCALE:-0.7}"

# Gate sweep 时必须关闭额外 isolated SigLIP guidance
SIGLIP_GUIDANCE_SCALE=0.0

NEGATIVE_PROMPT="worst quality, low quality, inconsistent motion, blurry, jittery, distorted, deformed, artifacts"

GPU_IDS=(0 1 2 3 4 5 6 7)
GPU_COUNT="${#GPU_IDS[@]}"

RESCALE_TAG="$(
python - "$GUIDANCE_RESCALE" <<'PY'
import sys
print(f"{float(sys.argv[1]):g}".replace("-", "m").replace(".", "p"))
PY
)"

OUTPUT_ROOT="/mnt/workspace/litengjie/my_dataset/overfit_2000/gate_sweep_cfg2_ref2_stg1_rescale${RESCALE_TAG}"

# ==============================================================================
# 要测试的 checkpoint
# ==============================================================================
VARIANT_NAMES=(
    "baseline"
    "visual_off"
    "query_only_vg0p1"
    "content_vg0p1_rg0p01"
)

VARIANT_CKPTS=(
    "$BASELINE_CKPT"
    "${DIAG_DIR}/visual_off.safetensors"
    "${DIAG_DIR}/query_only_vg0p1.safetensors"
    "${DIAG_DIR}/content_vg0p1_rg0p01.safetensors"
)

if [[ "$INCLUDE_EXTREME" == "1" ]]; then
    VARIANT_NAMES+=("content_vg0p1_rg0p1")
    VARIANT_CKPTS+=("${DIAG_DIR}/content_vg0p1_rg0p1.safetensors")
fi

# ==============================================================================
# Ctrl+C 清理
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
# 文件检查
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
require_file "$MANIFEST"
require_file "scripts/infer_multiref_stage1_overfit.py"

require_dir "$PRECOMPUTED"

for ckpt in "${VARIANT_CKPTS[@]}"; do
    require_file "$ckpt"
done

for subdir in \
    latents \
    multi_reference_latents \
    vlm_conditions \
    gt_siglip_tokens
do
    require_dir "${PRECOMPUTED}/${subdir}"
done

if [[ "$START_INDEX" -lt 0 || "$END_INDEX" -lt "$START_INDEX" ]]; then
    echo "错误：样本范围无效：${START_INDEX}-${END_INDEX}"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

echo "================================================================"
echo "Stage 1 Gate Sweep"
echo "================================================================"
echo "样本范围：          ${START_INDEX}-${END_INDEX}"
echo "GPU：               ${GPU_IDS[*]}"
echo "CFG：               $GUIDANCE_SCALE"
echo "Ref guidance：      $REF_GUIDANCE_SCALE"
echo "SigLIP guidance：   $SIGLIP_GUIDANCE_SCALE"
echo "STG：               $STG_SCALE"
echo "STG block：         $STG_BLOCKS"
echo "Guidance rescale：  $GUIDANCE_RESCALE"
echo "输出根目录：        $OUTPUT_ROOT"
echo "================================================================"

# ==============================================================================
# 运行一个 checkpoint variant
# ==============================================================================
run_variant() {
    local variant_name="$1"
    local checkpoint="$2"

    local output_dir="${OUTPUT_ROOT}/${variant_name}"
    local log_dir="${output_dir}/logs"

    mkdir -p "$output_dir" "$log_dir"

    rm -f "$log_dir"/sample_*.ok
    rm -f "$log_dir"/sample_*.failed

    echo
    echo "----------------------------------------------------------------"
    echo "开始 variant：$variant_name"
    echo "Checkpoint： $checkpoint"
    echo "输出目录：   $output_dir"
    echo "----------------------------------------------------------------"

    local pids=()

    for worker_index in "${!GPU_IDS[@]}"; do
        local gpu_id="${GPU_IDS[$worker_index]}"

        (
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
                sample_log="${log_dir}/sample_${sample_index}.log"
                output_video="${output_dir}/sample_${sample_index}/generated_full_siglip.mp4"

                if [[ "$SKIP_EXISTING" == "1" && -s "$output_video" ]]; then
                    touch "${log_dir}/sample_${sample_index}.ok"
                    echo "[GPU ${gpu_id}] 跳过已有 sample ${sample_index}"
                    continue
                fi

                echo "[GPU ${gpu_id}] 开始 sample ${sample_index}"

                set +e

                CUDA_VISIBLE_DEVICES="$gpu_id" \
                PYTHONUNBUFFERED=1 \
                python scripts/infer_multiref_stage1_overfit.py \
                    --config "$CONFIG" \
                    --checkpoint "$checkpoint" \
                    --manifest "$MANIFEST" \
                    --precomputed-root "$PRECOMPUTED" \
                    --sample-index "$sample_index" \
                    --output-dir "$output_dir" \
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
                    touch "${log_dir}/sample_${sample_index}.ok"
                    echo "[GPU ${gpu_id}] 完成 sample ${sample_index}"
                else
                    echo "$status" > "${log_dir}/sample_${sample_index}.failed"
                    echo "[GPU ${gpu_id}] sample ${sample_index} 失败，exit=${status}"
                    echo "[GPU ${gpu_id}] 日志：${sample_log}"
                    failures=$((failures + 1))
                fi
            done

            echo "[GPU ${gpu_id}] worker 结束，失败=${failures}"

            if [[ "$failures" -gt 0 ]]; then
                exit 1
            fi
        ) > "${log_dir}/gpu_${gpu_id}.log" 2>&1 &

        pids+=("$!")
    done

    local variant_failed=0

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            variant_failed=1
        fi
    done

    local success_count
    local failed_count

    success_count="$(
        find "$log_dir" \
            -maxdepth 1 \
            -type f \
            -name 'sample_*.ok' \
            | wc -l \
            | tr -d ' '
    )"

    failed_count="$(
        find "$log_dir" \
            -maxdepth 1 \
            -type f \
            -name 'sample_*.failed' \
            | wc -l \
            | tr -d ' '
    )"

    echo "variant=${variant_name}，成功=${success_count}，失败=${failed_count}"

    if [[ "$variant_failed" -ne 0 || "$failed_count" -gt 0 ]]; then
        return 1
    fi
}

# ==============================================================================
# 依次执行所有 variants
# ==============================================================================
OVERALL_FAILURE=0

for index in "${!VARIANT_NAMES[@]}"; do
    if ! run_variant \
        "${VARIANT_NAMES[$index]}" \
        "${VARIANT_CKPTS[$index]}"
    then
        OVERALL_FAILURE=1
    fi
done

echo
echo "================================================================"
echo "Gate sweep 完成"
echo "================================================================"

for variant_name in "${VARIANT_NAMES[@]}"; do
    echo "${variant_name}:"
    echo "  ${OUTPUT_ROOT}/${variant_name}"
done

echo "================================================================"

if [[ "$OVERALL_FAILURE" -ne 0 ]]; then
    echo "部分 variant 运行失败，请检查对应 logs 目录。"
    exit 1
fi
