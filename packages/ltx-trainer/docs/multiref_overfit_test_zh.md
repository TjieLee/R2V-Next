# Multi-Reference / Planner 100 条 Overfit 测试流程

这份 README 专门用于大规模训练前的小规模验证。新增流程不改 trainer、strategy、CFG、`multi_reference_video.py`、`multi_reference_planner_stage2.py` 或数据格式。

## 核心目标

用固定 100 条样本验证：

- Stage 1 能用 `VLM text/reference context + thinking/register tokens + target-video GT SigLIP tokens + reference latents` 过拟合目标视频。
- Stage 2 能用 `VLM planner placeholders` 预测 visual tokens，并通过 MSE 对齐到 target-video GT SigLIP tokens。
- 生成视频应保留参考图主体身份，同时运动/内容接近 GT 视频。

## 新增文件

- `scripts/create_overfit_subset.py`：从原始 manifest 或 shard 目录抽样 100 条；可选创建只含这 100 条的 `.precomputed` 软链接根目录。
- `scripts/check_multiref_precomputed_dataset.py`：检查 `latents/`、`multi_reference_latents/`、`conditions/`、`vlm_conditions/`、`gt_siglip_tokens/`，Stage 2 还检查 `planner_vlm_inputs/`。
- `configs/multiref_stage1_overfit100.yaml`：Stage 1 overfit 配置。
- `configs/multiref_stage2_overfit100.yaml`：Stage 2 planner overfit 配置。
- `scripts/test_multiref_overfit_generation.py`：打包单条样本的 GT/ref/generated/metadata，可委托已有推理命令。
- `scripts/compare_overfit_generation.py`：生成 HTML 对比页。

## 重要说明：为什么要软链接 `.precomputed`

当前 `PrecomputedDataset` 训练时扫描 `data.preprocessed_data_root` 下的 `.pt` 文件，不读取 manifest。因此只生成 `overfit_100.json` 不会限制训练集。推荐创建一个 overfit 专用 `.precomputed` 根目录，里面只放 100 条样本对应 `.pt` 的软链接。软链接不复制大文件，也不改变数据格式。

## Overfit 默认关闭 CFG 和 validation

100 条 overfit 的默认目标是先验证 full-condition 主链路能不能拟合，所以两个 overfit config 都默认关闭 CFG dropout：`cfg_dropout_enabled: false`、`cfg_full_p: 1.0`，所有 drop 概率为 `0.0`。这可以避免 text/ref/all dropout 干扰你判断 Stage 1 和 Stage 2 的完整条件链路是否正确。

两个 overfit config 也默认关闭自动 validation：`validation.interval: null`、`validation.skip_initial_validation: true`。`generate_video: true` 会保留，但不会触发空 validation。生成视频应在训练检查通过后，用已有 validation output 或独立 inference command 单独打包。

## 0. 设置路径

```bash
cd /mnt/workspace/litengjie/LTX-2/packages/ltx-trainer

TRAIN_JSON=/mnt/workspace/litengjie/my_dataset/train.json
FULL_PRECOMP=/mnt/workspace/litengjie/my_dataset/.precomputed
OVERFIT_DIR=/mnt/workspace/litengjie/my_dataset/overfit_100
OVERFIT_JSON=$OVERFIT_DIR/overfit_100.json
OVERFIT_PRECOMP=$OVERFIT_DIR/.precomputed
MODEL=/mnt/workspace/litengjie/LTX-2/models/LTX-2.3/ltx-2.3-22b-dev.safetensors
GEMMA=/mnt/workspace/litengjie/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized
```

## 1. 抽取 100 条并创建 Stage 1 软链接数据根

```bash
python scripts/create_overfit_subset.py   --input-manifest $TRAIN_JSON   --output-manifest $OVERFIT_JSON   --num-samples 100   --seed 42   --video-column video   --link-precomputed   --precomputed-root $FULL_PRECOMP   --subset-precomputed-root $OVERFIT_PRECOMP   --precomputed-sources "latents,conditions,vlm_conditions,multi_reference_latents,gt_siglip_tokens"
```

成功时会看到：

```text
Selected 100 samples from N total samples
Output: /mnt/workspace/litengjie/my_dataset/overfit_100/overfit_100.json
```

## 2. 检查 Stage 1 数据

```bash
python scripts/check_multiref_precomputed_dataset.py   --manifest $OVERFIT_JSON   --precomputed-root $OVERFIT_PRECOMP   --planner-token-count 2048   --video-column video   --stage1-only
```

成功输出：

```text
Dataset check passed: 100/100 samples valid
```

如果 `precompute_gt_siglip_tokens.py` 的日志不是 `Detected 2048 GT visual tokens per sample`，把这里和 Stage 2 配置中的 `planner_token_count` 改成真实值。

## 3. Stage 1 overfit 训练

确认 `configs/multiref_stage1_overfit100.yaml` 中：

- `model.model_path = $MODEL`
- `model.text_encoder_path = $GEMMA`
- `data.preprocessed_data_root = $OVERFIT_PRECOMP`
- `output_dir = /mnt/workspace/litengjie/ltx2_multiref_overfit_stage1_100`

建议先单卡跑，方便看 loss。默认不会自动跑 validation，避免 overfit smoke test 被空 validation 或推理入口问题打断：

```bash
accelerate launch --num_processes 1 --mixed_precision bf16   scripts/train.py configs/multiref_stage1_overfit100.yaml
```

中间产物：

```text
/mnt/workspace/litengjie/ltx2_multiref_overfit_stage1_100/
  training_config.yaml
  checkpoints/
    lora_weights_step_00100.safetensors
    lora_weights_step_00200.safetensors
```

## 4. 生成 Stage 2 planner VLM inputs

```bash
python scripts/precompute_planner_vlm_inputs.py $OVERFIT_JSON   --text-encoder-path $GEMMA   --output-dir $OVERFIT_PRECOMP/planner_vlm_inputs   --video-column video   --caption-column caption   --reference-column reference_images   --max-ref-images 4   --planner-token-count 2048   --max-length 4096
```

新增产物：

```text
$OVERFIT_PRECOMP/planner_vlm_inputs/...
```

## 5. 检查 Stage 2 数据

```bash
python scripts/check_multiref_precomputed_dataset.py   --manifest $OVERFIT_JSON   --precomputed-root $OVERFIT_PRECOMP   --planner-token-count 2048   --video-column video   --stage2
```

## 6. Stage 2 overfit 训练

确认 `configs/multiref_stage2_overfit100.yaml` 中：

- `model.load_checkpoint` 指向 Stage 1 overfit 的 `checkpoints` 目录或具体 `.safetensors`。
- `data.preprocessed_data_root = $OVERFIT_PRECOMP`
- `training_strategy.planner_token_count = 2048`
- `training_strategy.train_gemma_backbone = false`

运行：

```bash
accelerate launch --num_processes 1 --mixed_precision bf16   scripts/train.py configs/multiref_stage2_overfit100.yaml
```

中间产物：

```text
/mnt/workspace/litengjie/ltx2_multiref_overfit_stage2_100/
  training_config.yaml
  checkpoints/
    lora_weights_step_00100.safetensors
    lora_weights_step_00200.safetensors
```

## 7. 打包生成结果

注意：`test_multiref_overfit_generation.py` 只是 packaging/delegation script，不是完整 multi-reference inference pipeline。它会打包 GT、reference、metadata，并可以复制已有 validation output 或委托你提供的独立 inference command。

如果没有独立 inference command，也没有 validation output，这一步只能完成训练检查，不能自动生成 `generated.mp4`。

如果 validation 已生成视频，用：

```bash
python scripts/test_multiref_overfit_generation.py   --checkpoint /mnt/workspace/litengjie/ltx2_multiref_overfit_stage2_100/checkpoints   --config configs/multiref_stage2_overfit100.yaml   --manifest $OVERFIT_JSON   --sample-index 0   --output-dir $OVERFIT_DIR/eval_stage2   --generated-video /path/to/validation/generated_sample_0.mp4
```

如果你已有独立 multi-reference 推理脚本，则委托它生成 `{generated}`：

```bash
python scripts/test_multiref_overfit_generation.py   --checkpoint /mnt/workspace/litengjie/ltx2_multiref_overfit_stage2_100/checkpoints   --config configs/multiref_stage2_overfit100.yaml   --manifest $OVERFIT_JSON   --sample-index 0   --output-dir $OVERFIT_DIR/eval_stage2   --generation-command 'python scripts/YOUR_INFER.py --config {config} --checkpoint {checkpoint} --manifest {manifest} --sample-index {sample_index} --output {generated}'
```

输出结构：

```text
$OVERFIT_DIR/eval_stage2/sample_0/
  generated.mp4
  gt.mp4
  ref_0.jpg
  ref_1.jpg
  metadata.json
```

## 8. 生成 HTML 对比页

```bash
python scripts/compare_overfit_generation.py   $OVERFIT_DIR/eval_stage2   --output-html $OVERFIT_DIR/eval_stage2/index.html
```

打开 `index.html` 后检查：参考身份是否保住，generated 的运动/场景是否接近 GT，Stage 2 的 visual tokens 是否真正起作用。

## 通过标准

100 条 overfit 不看泛化，只看能不能学进去。健康信号是：Stage 1 loss 明显下降；Stage 2 planner MSE/visual-token 对齐 loss 下降；生成结果同时记住参考身份和 GT 运动内容。
