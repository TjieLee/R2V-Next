# LTX-2 多参考图像 + 文本 + VLM Planner 训练说明

这版代码把 DiT 条件序列统一扩展为：

```text
text embedding tokens + thinking/register tokens + visual tokens
```

Stage 1 使用冻结 Gemma/SigLIP vision tower 和 multi-modal projector 从 target video 采样帧生成的 GT visual tokens。Stage 2 使用 VLM language model + 固定数量 learnable planner placeholder tokens 预测 visual tokens，并用 MSE 对齐 Stage 1 的 target-video GT SigLIP/projector visual tokens。`planner_token_count` 必须等于 `gt_siglip_tokens/*.pt` 中的 `num_visual_tokens`。

## 不要混淆的视觉来源

这版实现里有三类视觉信息，来源和用途不同：

| 目录/数据 | 来源 | 进入哪里 | 作用 |
| --- | --- | --- | --- |
| `multi_reference_latents/` | `reference_images` 参考图 | DiT video latent stream，拼在 noisy target video latents 前面 | 多参考图的 VAE latent 条件 |
| `planner_vlm_inputs/` | `reference_images` 参考图 + system/user prompt | VLM/Gemma 输入 | 让 VLM planner 看参考图和文本，预测 visual planner tokens |
| `gt_siglip_tokens/` | `video` target video 的采样帧 | Stage 1 的 DiT condition tokens；Stage 2 的 MSE teacher | target-video GT SigLIP/projector visual tokens |

关键约束：

- `gt_siglip_tokens/` **只能从 target video 提取**，不能从 `reference_images` 提取。
- Stage 1 的 DiT condition 是 `text/thinking tokens + target-video GT SigLIP/projector tokens`。
- Stage 2 的 DiT condition 是 `text/thinking tokens + VLM predicted planner tokens`。
- Stage 2 的 MSE teacher 是同一条样本的 `target-video GT SigLIP/projector tokens`。
- `reference_images` 不作为 Stage 1 condition visual-token teacher，也不作为 Stage 2 MSE teacher；它们只进入 reference VAE latent stream 和 VLM 输入。

## 主要改动

- `ltx_core.multicond.visual_tokens`：新增冻结 SigLIP/projector visual token 提取、Gemma image-token scatter、固定数量 `VisualPlannerTokens`。
- `ltx_trainer.training_strategies.multi_reference_video`：Stage 1 在原 text features 后追加 `gt_siglip_tokens/visual_tokens`，再统一进入 LTX text connector。
- `ltx_trainer.training_strategies.multi_reference_planner_stage2`：Stage 2 不再把 planner hidden 压成 256 个任意 tokens，而是使用固定 `planner_token_count` 个 learnable placeholders；VLM 输出的同数量 tokens 直接替换 GT visual tokens 进入 DiT，并和 GT tokens 做 MSE。
- `scripts/precompute_gt_siglip_tokens.py`：从 target video 采样帧生成 `.precomputed/gt_siglip_tokens/`。
- `scripts/precompute_planner_vlm_inputs.py`：构建 system/user prompt，并在 token 序列末尾追加固定数量 planner placeholders。
- `configs/multiref_stage1_lora.yaml`、`configs/multiref_stage2_planner.yaml`：更新 Stage 1/2 配置。

## 目录结构

```text
/mnt/workspace/litengjie/my_dataset/.precomputed/
├── conditions/                 # text/thinking features, connector 前
├── latents/                    # target video VAE latents
├── multi_reference_latents/    # 参考图 VAE latents，用于 Stage 1 latent stream conditioning
├── gt_siglip_tokens/           # 冻结 target-video SigLIP/projector GT visual tokens
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

4. 生成 target-video GT SigLIP/projector visual tokens：

```bash
python scripts/precompute_gt_siglip_tokens.py /mnt/workspace/litengjie/my_dataset/train.json \
  --text-encoder-path /mnt/workspace/litengjie/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/gt_siglip_tokens \
  --video-column video \
  --num-sampled-frames 4 \
  --max-source-frames 81 \
  --expected-token-count 1024 \
  --device cuda
```

推荐固定从 target video 的前 `81` 个源帧内均匀抽 `4` 帧。Gemma/SigLIP 每帧产生 `256` 个 projected visual tokens，因此 `num_visual_tokens = 4 * 256 = 1024`。把这个值填到 Stage 2 配置的：

```yaml
training_strategy:
  planner_token_count: 1024
```

如果你改成 `--num-sampled-frames N`，则 `planner_token_count = N * 256`。`--sample-fps 6` 也可以用，但 token 数会随源视频 fps 和 `--max-source-frames` 变化；为了 Stage 2 固定 planner placeholders，推荐用 `--num-sampled-frames`。

5. 生成 Stage 2 VLM 输入：

```bash
python scripts/precompute_planner_vlm_inputs.py /mnt/workspace/litengjie/my_dataset/train.json \
  --text-encoder-path /mnt/workspace/litengjie/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/planner_vlm_inputs \
  --video-column video \
  --caption-column caption \
  --reference-column reference_images \
  --planner-token-count 1024 \
  --max-length 4096
```

这里的 `planner-token-count` 必须和 `gt_siglip_tokens` 的 `num_visual_tokens` 完全一致。最多 `4` 张 reference images 只影响 VLM 能看到多少参考图，以及 `multi_reference_latents/` 中有多少参考 latent；它不决定 MSE teacher token 数。

如果你已经生成的 `gt_siglip_tokens` 日志显示 `Detected 2048 GT visual tokens per sample`，这里就要改成：

```bash
  --planner-token-count 2048 \
  --max-length 4096
```

如果 caption 很长或参考图 token 占用较多，可以把 `--max-length` 提到 `8192`。

## 训练时的数据流

Stage 1 每条样本的数据流：

```text
reference_images
  -> precompute_multiref_images.py
  -> multi_reference_latents
  -> prepended clean reference latent tokens in DiT latent stream

target video
  -> precompute_gt_siglip_tokens.py
  -> gt_siglip_tokens.visual_tokens
  -> text connector input after text/thinking tokens
  -> DiT condition tokens
```

Stage 2 每条样本的数据流：

```text
reference_images + system prompt + user prompt + fixed planner placeholders
  -> VLM/Gemma language model
  -> predicted planner visual tokens
  -> text connector input after text/thinking tokens
  -> DiT condition tokens

target video
  -> gt_siglip_tokens.visual_tokens
  -> MSE teacher for predicted planner visual tokens
```

因此，Stage 1 和 Stage 2 的最大区别不是是否使用参考图，而是 DiT condition 里的 visual tokens 来源不同：

- Stage 1：使用 target-video GT SigLIP/projector tokens。
- Stage 2：使用 VLM + learnable planner placeholders 预测出来的 tokens，并用 target-video GT tokens 对齐。

## Negative RoPE

多参考图 VAE latents 会被拼到 target video latents 前面。参考图 latent token 的时间位置使用负向 T 维 RoPE：

```text
ref 1: T = -1 * reference_time_stride
ref 2: T = -2 * reference_time_stride
ref 3: T = -3 * reference_time_stride
ref 4: T = -4 * reference_time_stride
```

默认 `reference_time_stride: 1.0`，也就是 `-1, -2, -3, -4`。这个负向时间位置只作用在 reference VAE latent stream 上；`gt_siglip_tokens` 和 VLM predicted planner tokens 是 text connector condition sequence，不使用这套 reference latent RoPE。

## 预处理结果检查

检查 `gt_siglip_tokens` 是否确实来自 target video：

```bash
python - <<'PY'
import glob
import torch

p = glob.glob("/mnt/workspace/litengjie/my_dataset/.precomputed/gt_siglip_tokens/**/*.pt", recursive=True)[0]
x = torch.load(p, map_location="cpu")
print("file:", p)
print("visual_tokens:", tuple(x["visual_tokens"].shape))
print("num_visual_tokens:", int(x["num_visual_tokens"]))
print("tokens_per_frame:", int(x["tokens_per_frame"]))
print("sampled_frame_indices:", x["sampled_frame_indices"].tolist())
PY
```

推荐输出应类似：

```text
visual_tokens: (1024, D)
num_visual_tokens: 1024
tokens_per_frame: 256
sampled_frame_indices: [...]
```

路径应该镜像 `video` 字段的 target video 路径，而不是 reference image 路径。

## Stage 1 训练

编辑 `configs/multiref_stage1_lora.yaml`：

- `model.model_path`：LTX-2 checkpoint。
- `model.text_encoder_path`：Gemma text encoder 目录。
- `data.preprocessed_data_root`：`/mnt/workspace/litengjie/my_dataset/.precomputed`。
- `training_strategy.gt_visual_tokens_dir`：默认 `gt_siglip_tokens`。
- `training_strategy.visual_token_frame_stride`：默认 `1`。如果你已经按 `6fps` 编码，后期想按 `3fps` 用，可以设为 `2`，无需重跑 SigLIP。
- `output_dir`：Stage 1 输出目录。

启动：

```bash
accelerate launch --num_processes 8 --num_machines 1 \
  scripts/train.py configs/multiref_stage1_lora.yaml
```

Stage 1 的 DiT 输入区别是：visual tokens 来自 target video 的冻结 SigLIP/projector GT tokens。

## Stage 2 训练

编辑 `configs/multiref_stage2_planner.yaml`：

- `model.load_checkpoint`：Stage 1 checkpoint。
- `training_strategy.planner_token_count`：必须等于 `gt_siglip_tokens` 的 `num_visual_tokens`。
- `training_strategy.visual_token_frame_stride`：必须和 Stage 1 使用方式一致；如果 Stage 1 用 `2`，Stage 2 也用 `2`，并把 `planner_token_count` 改成降采样后的 token 数。
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

Stage 2 的 DiT 输入区别是：visual tokens 来自 VLM + learnable planner placeholders 的预测结果。MSE 保证预测 tokens 和 Stage 1 使用的 target-video GT SigLIP/projector tokens 位于同一 token 数量、同一特征空间。

## Stage 3 后续目标

Stage 3 应从 Stage 2 checkpoint 继续，联合训练：

- 多参考 latent conditioning。
- VLM predicted visual tokens。
- text/thinking tokens。
- DiT LoRA、planner tokens、Gemma language model、text connector 的可训练组合。

SigLIP vision tower 和 multi-modal projector 默认仍冻结。Stage 3 的重点是把 Stage 1 学会的 GT visual-token 使用方式，和 Stage 2 学会的 VLM predicted visual-token 分布联合微调到最终视频生成目标上。
