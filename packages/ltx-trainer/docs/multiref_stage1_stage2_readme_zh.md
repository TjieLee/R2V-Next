# LTX-2 多参考图像 + 文本 + VLM Planner 训练说明

这版代码把 DiT 条件序列统一扩展为：

```text
text embedding tokens + thinking/register tokens + visual tokens
```

Stage 1 使用冻结 Gemma/SigLIP vision tower 和 multi-modal projector 生成的 GT visual tokens。Stage 2 使用 VLM language model + 固定数量 learnable planner placeholder tokens 预测 visual tokens，并用 MSE 对齐 Stage 1 的 GT SigLIP/projector visual tokens。`planner_token_count` 必须等于 `gt_siglip_tokens/*.pt` 中的 `num_visual_tokens`。

## 主要改动

- `ltx_core.multicond.visual_tokens`：新增冻结 SigLIP/projector visual token 提取、Gemma image-token scatter、固定数量 `VisualPlannerTokens`。
- `ltx_trainer.training_strategies.multi_reference_video`：Stage 1 在原 text features 后追加 `gt_siglip_tokens/visual_tokens`，再统一进入 LTX text connector。
- `ltx_trainer.training_strategies.multi_reference_planner_stage2`：Stage 2 不再把 planner hidden 压成 256 个任意 tokens，而是使用固定 `planner_token_count` 个 learnable placeholders；VLM 输出的同数量 tokens 直接替换 GT visual tokens 进入 DiT，并和 GT tokens 做 MSE。
- `scripts/precompute_gt_siglip_tokens.py`：从 `reference_images` 生成 `.precomputed/gt_siglip_tokens/`。
- `scripts/precompute_planner_vlm_inputs.py`：构建 system/user prompt，并在 token 序列末尾追加固定数量 planner placeholders。
- `configs/multiref_stage1_lora.yaml`、`configs/multiref_stage2_planner.yaml`：更新 Stage 1/2 配置。

## 目录结构

```text
/mnt/workspace/litengjie/my_dataset/.precomputed/
├── conditions/                 # text/thinking features, connector 前
├── latents/                    # target video VAE latents
├── multi_reference_latents/    # 参考图 VAE latents，用于 Stage 1 latent stream conditioning
├── gt_siglip_tokens/           # 冻结 SigLIP/projector GT visual tokens
└── planner_vlm_inputs/         # Stage 2 VLM 输入 + fixed planner placeholder mask
```

## 预处理顺序

1. 生成或确认 flat manifest：

```bash
python scripts/convert_phantom_manifest.py \
  /mnt/workspace/liutao/phantom_data/train_data_0202.json \
  --root-dir /mnt/workspace/liutao/phantom_data \
  --output-json /mnt/workspace/litengjie/my_dataset/train.json
```

每条样本需要包含：

```json
{
  "video": "/abs/path/video.mp4",
  "caption": "prompt text",
  "reference_images": ["/abs/path/ref_0.jpg"]
}
```

2. 生成 text conditions 和 target video latents：

```bash
accelerate launch --num_processes 8 --num_machines 1 \
  scripts/process_dataset.py /mnt/workspace/litengjie/my_dataset/train.json \
  --resolution-buckets "640x384x81" \
  --model-path /mnt/workspace/litengjie/LTX-2/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path /mnt/workspace/litengjie/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed \
  --video-column video \
  --caption-column caption \
  --skip-audio \
  --batch-size 4
```

如果你已经生成了 `conditions/`，只想补 `latents/`，运行：

```bash
accelerate launch --num_processes 8 --num_machines 1 \
  scripts/process_videos.py /mnt/workspace/litengjie/my_dataset/train.json \
  --resolution-buckets "640x384x81" \
  --model-path /mnt/workspace/litengjie/LTX-2/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/latents \
  --video-column video \
  --batch-size 1 \
  --device cuda
```

3. 生成 reference image latents：

```bash
python scripts/precompute_multiref_images.py /mnt/workspace/litengjie/my_dataset/train.json \
  --model-path /mnt/workspace/litengjie/LTX-2/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --target-latents-dir /mnt/workspace/litengjie/my_dataset/.precomputed/latents \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/multi_reference_latents \
  --media-column video \
  --reference-column reference_images \
  --device cuda
```

4. 生成 GT SigLIP/projector visual tokens：

```bash
python scripts/precompute_gt_siglip_tokens.py /mnt/workspace/litengjie/my_dataset/train.json \
  --text-encoder-path /mnt/workspace/litengjie/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/gt_siglip_tokens \
  --video-column video \
  --reference-column reference_images \
  --device cuda
```

脚本会打印检测到的 `num_visual_tokens`。把这个值填到 Stage 2 配置的：

```yaml
training_strategy:
  planner_token_count: <num_visual_tokens>
```

5. 生成 Stage 2 VLM 输入：

```bash
python scripts/precompute_planner_vlm_inputs.py /mnt/workspace/litengjie/my_dataset/train.json \
  --text-encoder-path /mnt/workspace/litengjie/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/planner_vlm_inputs \
  --video-column video \
  --caption-column caption \
  --reference-column reference_images \
  --planner-token-count <num_visual_tokens>
```

## Stage 1 训练

编辑 `configs/multiref_stage1_lora.yaml`：

- `model.model_path`：LTX-2 checkpoint。
- `model.text_encoder_path`：Gemma text encoder 目录。
- `data.preprocessed_data_root`：`/mnt/workspace/litengjie/my_dataset/.precomputed`。
- `training_strategy.gt_visual_tokens_dir`：默认 `gt_siglip_tokens`。
- `output_dir`：Stage 1 输出目录。

启动：

```bash
accelerate launch --num_processes 8 --num_machines 1 \
  scripts/train.py configs/multiref_stage1_lora.yaml
```

Stage 1 的 DiT 输入区别是：visual tokens 来自冻结 SigLIP/projector 的 GT tokens。

## Stage 2 训练

编辑 `configs/multiref_stage2_planner.yaml`：

- `model.load_checkpoint`：Stage 1 checkpoint。
- `training_strategy.planner_token_count`：必须等于 `gt_siglip_tokens` 的 `num_visual_tokens`。
- `training_strategy.train_vlm_language_model: true`：训练 Gemma language model。
- `training_strategy.freeze_vlm_vision_tower: true`：冻结 SigLIP。
- `training_strategy.freeze_vlm_multi_modal_projector: true`：冻结 image projection。
- `training_strategy.freeze_transformer: true`：默认冻结 Stage 1 DiT/LoRA。
- `training_strategy.train_text_connector: true`：训练 text connector。

启动：

```bash
accelerate launch --num_processes 8 --num_machines 1 \
  scripts/train.py configs/multiref_stage2_planner.yaml
```

Stage 2 的 DiT 输入区别是：visual tokens 来自 VLM + learnable planner placeholders 的预测结果。MSE 保证预测 tokens 和 Stage 1 使用的 GT SigLIP/projector tokens 位于同一 token 数量、同一特征空间。

## Stage 3 后续目标

Stage 3 应从 Stage 2 checkpoint 继续，联合训练：

- 多参考 latent conditioning。
- VLM predicted visual tokens。
- text/thinking tokens。
- DiT LoRA、planner tokens、Gemma language model、text connector 的可训练组合。

SigLIP vision tower 和 multi-modal projector 默认仍冻结。Stage 3 的重点是把 Stage 1 学会的 GT visual-token 使用方式，和 Stage 2 学会的 VLM predicted visual-token 分布联合微调到最终视频生成目标上。
