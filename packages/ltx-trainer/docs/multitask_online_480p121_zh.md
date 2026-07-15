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
  --summary-output /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/logs/manifest_summary.json \
  --manifest-seed 42 \
  --annotation-batch-size 4096 \
  --probe-workers 8 \
  --probe-batch-size 256 \
  --i2i-target-field '<SOURCE_REPORT中的target字段>' \
  --i2i-reference-field '<SOURCE_REPORT中的source字段>' \
  --i2i-caption-field '<SOURCE_REPORT中的instruction字段>'

uv run python scripts/validate_multitask_online_manifest.py \
  /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/manifests/train_unique.jsonl
```

`/mnt/workspace/liutao/` 只读。manifest、reject、训练日志、cache 和 checkpoint 的写路径都会被
`assert_write_path_allowed()` 检查，必须位于 `/mnt/workspace/litengjie/`。

builder 对 JSONL/CSV 逐行读取，对 Parquet 使用 PyArrow batch；JSON list 仍兼容但会发出内存警告。
accepted/rejected 文件逐行写入临时文件，完成 `flush + fsync` 后原子替换。去重键保存在输出目录下的
临时 SQLite 中，不在内存维护百万级 Python dict。成功后会同时得到：

```text
train_unique.jsonl
train_unique.jsonl.idx
manifest_rejected.jsonl
manifest_summary.json
```

`.idx` 只保存 byte offset 和 task id，`OnlineMultiTaskDataset` 按 offset 延迟读取单条 JSON；8 个 rank
及其 DataLoader worker 不会各自复制完整 manifest。summary 包含 raw/accepted/rejected/duplicate、
task/reject reason 统计、耗时和 builder 峰值 RSS。静态坏数据使用稳定 reason：`missing_target`、
`missing_reference`、`empty_caption`、`invalid_crop`、`invalid_face_cut`、`invalid_video_header`、
`insufficient_frames_for_121_at_24fps`。

## 3. Reference 和视频解码语义

- `reference_pixels_vae`：统一截断 reference 数量后，确定性 resize/center crop 到 832x480，再进 VAE。
- `reference_images_vlm`：默认保留原始 sRGB 尺寸、像素和顺序，直接交给 Gemma image processor。
- `vlm_reference_preprocess: target_crop` 仅用于显式消融；正式 YAML 默认是 `original`。
- R2V 默认 `video_decoder: pyav`，从目标区间前的 keyframe seek 后按 presentation order 解码；PTS
  不可靠时在同一容器中从头顺序建立精确 ordinal index。
- `decode_timeout_seconds: 120` 的超时会成为 `SampleLoadError`，所有 DDP rank 同步进入同 task retry。
- 目前仅开放 `encoder_device_policy: resident_cuda`。另外两种未完整实现的 policy 会在配置解析时失败，
  不会移动或复制 DDP 包裹的可训练 Gemma。

retry 候选会排除当前 optimizer step 的整个 normal block 和本 step 已成功消费的 retry。小数据 fallback
会明确 warning；若 retry 使用了已预取的 future normal sample，trainer 会在编码/forward 前识别并再次
确定性 retry，不推进 task schedule cursor。

## 4. 正式训练前门禁

先对三份 YAML 分别运行 CPU preflight：

```bash
for stage in \
  configs/multiref_stage1_multitask_online_480p121_full_tokens_30k.yaml \
  configs/multiref_stage2_multitask_online_480p121_full_tokens_planner_30k.yaml \
  configs/multiref_stage3_multitask_online_480p121_joint_30k.yaml; do
  uv run python scripts/check_multitask_online_training_ready.py "$stage" \
    --world-size 8 --samples-per-task 1
done
```

对应阶段必须输出 `READY_FOR_STAGE1/2/3=true`。然后运行真实模型 encode 和单卡一步训练：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/check_multitask_online_real_encode.py \
  --stage1-config configs/multiref_stage1_multitask_online_480p121_full_tokens_30k.yaml \
  --num-image-samples 1 --num-video-samples 1

CUDA_VISIBLE_DEVICES=0 uv run python scripts/check_multitask_online_stage1.py \
  --config configs/multiref_stage1_multitask_online_480p121_full_tokens_30k.yaml \
  --task i2i \
  --output-dir /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/smoke/stage1_i2i
```

Stage 1/2/3 wrapper 的参数一致，分别用对应 YAML 并对 `--task i2i`、`--task r2v` 各运行一次。
smoke 派生配置固定为 1 optimizer step、batch 1、accumulation 4、full-condition CFG off；它检查
scheduler 只前进一步、三项 Stage 2/3 loss finite、要求的模块产生非零 finite Adam moment，并保存
成对 checkpoint/training state。JSON 报告包含 decode、VAE、SigLIP、frozen condition、planner、DiT、
backward、optimizer、峰值显存和 reference shape。

最后运行真实两卡 DDP：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --config_file configs/accelerate/ddp.yaml --num_processes 2 \
  scripts/check_multitask_online_ddp.py \
  --stage1-config configs/multiref_stage1_multitask_online_480p121_full_tokens_30k.yaml \
  --stage2-config configs/multiref_stage2_multitask_online_480p121_full_tokens_planner_30k.yaml \
  --stage3-config configs/multiref_stage3_multitask_online_480p121_joint_30k.yaml
```

它会顺序覆盖三个阶段的 I2I/R2V 一步优化，验证真实 DDP condition encoding、Gemma/Planner/DiT、
checkpoint 保存和 scheduler 语义。只有这些门禁通过后才启动 30K。

## 5. 训练

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

## Stage 3-only 旧模型 warm-start

在线编码按任务选择 system prompt。I2I 使用
`gemma_multiref_image_edit_planner_system_prompt.txt`，明确生成单张 target image；R2V 继续使用原
`gemma_multiref_video_planner_system_prompt.txt`，其 chat serialization 保持不变。`conditions`、
`text_conditions` 和 `planner_vlm_inputs` 使用同一任务 prompt，batch 中
`task_system_prompt_id=0/1` 分别表示 image/video。

如果不准备先在新数据上重跑 Stage 1/2，可从完整旧 Stage 3 checkpoint 启动独立 joint warm-start：

```bash
uv run python scripts/check_multitask_online_training_ready.py \
  configs/multiref_stage3_multitask_online_480p121_joint_warmstart_old_stage3_30k.yaml \
  --world-size 8 --samples-per-task 2
```

报告必须包含 `initialization_mode=stage3_joint_warmstart`、
`strict_component_check_passed=true` 和 `starts_from_global_step=0`。该 YAML 的 `no_resume=true` 只加载
旧 Stage 3 模型权重，不加载旧 optimizer、scheduler 或 global step。

两卡只跑 Stage 3 的真实 DDP smoke：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 2 \
  scripts/check_multitask_online_ddp.py \
  --stages stage3 \
  --stage3-config configs/multiref_stage3_multitask_online_480p121_joint_warmstart_old_stage3_30k.yaml \
  --stage3-init-checkpoint /mnt/workspace/litengjie/ltx2_multiref_stage3_joint_full_tokens_planner_2048/checkpoints/lora_weights_step_02000.safetensors
```

需要验证三阶段依赖时可传 `--stages stage1,stage2,stage3 --chain-smoke-checkpoints`，脚本会把前一阶段
产生的 smoke checkpoint 注入下一阶段。正式 30K 前使用
`configs/multiref_stage3_multitask_online_480p121_joint_warmstart_benchmark_200.yaml` 完成八卡
200-step 门禁；它使用 interval 100 和独立 benchmark output_dir。

manifest index 现为 `LTXIDX02`，header 包含 manifest size、有效行数、SHA256 和 entry size。旧 v1、
hash/size 不匹配或截断索引都会 fail-fast 并要求重建；validator 还会逐行核对 offset/task。manifest
builder 在同一次构建的临时 SQLite 中按 path/size/mtime 缓存图片验证与视频 probe，不写入源数据目录。
