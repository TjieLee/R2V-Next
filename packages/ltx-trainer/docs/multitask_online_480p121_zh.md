# I2I + R2V 在线训练（832x480 / 121 帧）

这条数据链路不再要求预先生成全量 VAE latent、GT SigLIP token 或 VLM condition。旧的
`PrecomputedDataset`、81 帧 YAML、checkpoint 和推理入口保持不变；只有 YAML 明确设置
`data.encoding_mode: online` 时才进入新链路。

## 固定语义

- I2I：target 是 1 张图，`frames=1`、`fps=1`，source image 进入 reference。
- R2V：target 严格为 121 帧、24 fps，短视频直接进入 reject 日志。
- VLM/SigLIP 视频索引固定为 `[0,17,34,51,69,86,103,120]`。
- I2I SigLIP 只计算真实 1 帧：256 个有效 token，补零到 2048；不会复制成 8 帧。
- Planner placeholder 始终是 2048，但 I2I `planner_output_mask` 只有前 256 个为真。
- 每个 optimizer step 的 4 个 accumulation microsteps 和 8 个 rank 使用同一 task。
- 30K step 精确包含 9000 个 I2I step 和 21000 个 R2V step。global batch 为 32，因此每阶段
  96 万次样本曝光；三个阶段都跑 30K 时总计 288 万次曝光。

## 1. 检查 schema

```bash
cd /mnt/workspace/litengjie/LTX-2/packages/ltx-trainer
mkdir -p /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/manifests

uv run python scripts/inspect_multitask_sources.py \
  --i2i-ann /mnt/workspace/liutao/data_process_one2x/r2i_train_data.jsonl \
  --r2v-ann /mnt/workspace/liutao/data_process_one2x/opens2v_train_data.parquet \
  --output /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/manifests/source_schema_report.json
```

I2I 字段不会在代码中猜测。先从报告确认 target、source/reference、instruction/caption 字段名，
再传给下一步。

## 2. 构建并验证确定性 manifest

```bash
cp configs/multitask_online_480p121_data.yaml \
  /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/manifests/multitask_480p121.yaml

uv run python scripts/build_multitask_online_manifest.py \
  --train-data-config /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/manifests/multitask_480p121.yaml \
  --output /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/manifests/train_unique.jsonl \
  --reject-output /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/logs/manifest_rejected.jsonl \
  --manifest-seed 42 \
  --i2i-target-field '<SOURCE_REPORT中的target字段>' \
  --i2i-reference-field '<SOURCE_REPORT中的source字段>' \
  --i2i-caption-field '<SOURCE_REPORT中的instruction字段>'

uv run python scripts/validate_multitask_online_manifest.py \
  /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/manifests/train_unique.jsonl
```

`/mnt/workspace/liutao/` 只读。manifest、reject、训练日志、cache 和 checkpoint 的写路径都会被
`assert_write_path_allowed()` 检查，必须位于 `/mnt/workspace/litengjie/`。

## 3. 训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 8 \
  scripts/train.py configs/multiref_stage1_multitask_online_480p121_full_tokens_30k.yaml \
  --disable-progress-bars
```

Stage 1 完成后确认以下 checkpoint 存在，再启动 Stage 2：

```text
/mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/stage1/checkpoints/lora_weights_step_30000.safetensors
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 8 \
  scripts/train.py configs/multiref_stage2_multitask_online_480p121_full_tokens_planner_30k.yaml \
  --disable-progress-bars

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 8 \
  scripts/train.py configs/multiref_stage3_multitask_online_480p121_joint_30k.yaml \
  --disable-progress-bars
```

Stage 2/3 YAML 已分别指向前一阶段的 step 30000 checkpoint。正式运行前应核对路径。在线 sampler
状态随 training state 保存，resume 后继续相同 task 和样本序列。某个 rank 解码失败时，全部 rank
同步丢弃该 microbatch，并从同 task 的确定性备用样本重试。

## 在线 batch 接口

GPU 编码完成后复用现有 strategy keys：`latents`、`multi_ref_latents`、`conditions`、
`vlm_conditions`、`text_conditions`、`cfg_text_conditions`、`gt_visual_tokens`，Stage 2/3 额外包含
`planner_vlm_inputs`。Stage 2/3 的 Gemma/Planner forward 仍在训练图内；VAE、SigLIP vision tower 和
multimodal projector 保持 frozen/eval。
