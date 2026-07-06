# LTX-2 多参考图像 + 文本 + VLM Planner 训练说明

这版代码把 DiT 条件序列统一扩展为：

```text
VLM context tokens + thinking/register tokens + visual tokens
```

其中 `VLM context tokens` 来自 `system prompt -> user prompt -> reference images` 的 Gemma/VLM 编码。Stage 1 在这段 context 后追加 target-video GT SigLIP/projector visual tokens。Stage 2/3 在同样的 source context 后追加 Baton-style target planning region：`<image_start> + <image_pad> * planner_token_count + <image_end>`。VLM 在 `<image_pad>` 位置输出 hidden states；planner bridge 使用 repeated LTX thinking/register tokens 作为 Q，使用这些 hidden states 作为 K/V，经过 zero-init cross-attention + zero-init FFN 生成 planner visual tokens，并用 MSE 对齐 target-video GT SigLIP/projector visual tokens。`planner_token_count` 必须等于 `gt_siglip_tokens/*.pt` 中的 `num_visual_tokens`。

## 不要混淆的视觉来源

这版实现里有三类视觉信息，来源和用途不同：

| 目录/数据 | 来源 | 进入哪里 | 作用 |
| --- | --- | --- | --- |
| `multi_reference_latents/` | `reference_images` 参考图 | DiT video latent stream，拼在 noisy target video latents 前面 | 多参考图的 VAE latent 条件 |
| `vlm_conditions/` | `system prompt + caption + reference_images` | Stage 1/2 的 DiT condition context，位于 visual tokens 前面 | 让 Stage 1 的基础条件已经看过参考图 |
| `planner_vlm_inputs/` | `reference_images` 参考图 + system/user prompt | VLM/Gemma 输入 | 让 VLM planner 看参考图和文本，预测 visual planner tokens |
| `gt_siglip_tokens/` | `video` target video 的采样帧 | Stage 1 的 DiT condition tokens；Stage 2 的 MSE teacher | target-video GT SigLIP/projector visual tokens |

关键约束：

- `gt_siglip_tokens/` **只能从 target video 提取**，不能从 `reference_images` 提取。
- Stage 1 的 DiT condition 是 `VLM(system + user + ref images) context tokens + target-video GT SigLIP/projector tokens`。
- Stage 2 的 DiT condition 是 `VLM(system + user + ref images) context tokens + VLM predicted planner tokens`。
- Stage 2 的 MSE teacher 是同一条样本的 `target-video GT SigLIP/projector tokens`。
- `reference_images` 不作为 Stage 1 condition visual-token teacher，也不作为 Stage 2 MSE teacher；它们只进入 reference VAE latent stream 和 VLM 输入。

## 主要改动

- `ltx_core.multicond.visual_tokens`：新增冻结 SigLIP/projector visual token 提取、Gemma image-token scatter、固定数量 `VisualPlannerTokens`。
- `ltx_trainer.training_strategies.multi_reference_video`：Stage 1 可通过 `conditions_dir` 读取 `vlm_conditions/`，再在 VLM context features 后追加 `gt_siglip_tokens/visual_tokens`，统一进入 LTX text connector。
- `ltx_trainer.training_strategies.multi_reference_planner_stage2`：Stage 2 使用固定 `planner_token_count` 个 `<image_pad>` target placeholders；VLM 的 placeholder hidden states 作为 K/V，repeated LTX thinking/register tokens 作为 Q，通过 zero-init cross-attention + zero-init FFN 生成 visual planner tokens，再替换 GT visual tokens 进入 DiT，并和 GT tokens 做 MSE。
- `scripts/precompute_gt_siglip_tokens.py`：从 target video 采样帧生成 `.precomputed/gt_siglip_tokens/`。
- `scripts/precompute_multiref_vlm_conditions.py`：为 Stage 1/2 生成 `system prompt -> user prompt -> reference image tokens` 的 VLM context conditions。
- `scripts/precompute_planner_vlm_inputs.py`：按 `system prompt -> user prompt -> reference image tokens -> <image_start> + <image_pad>*K + <image_end>` 构建 Stage 2 VLM 输入。
- `configs/multiref_stage1_lora.yaml`、`configs/multiref_stage2_planner.yaml`：更新 Stage 1/2 配置。

## 目录结构

```text
/mnt/workspace/litengjie/my_dataset/.precomputed/
├── conditions/                 # text/thinking features, connector 前
├── vlm_conditions/             # system + user + reference images 的 VLM context features
├── latents/                    # target video VAE latents
├── multi_reference_latents/    # 参考图 VAE latents，用于 Stage 1 latent stream conditioning
├── gt_siglip_tokens/           # 冻结 target-video SigLIP/projector GT visual tokens
└── planner_vlm_inputs/         # Stage 2 VLM 输入 + Baton-style target planner placeholder mask
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

3. 生成 Stage 1/2 使用的 VLM reference-image context conditions：

```bash
mkdir -p /mnt/workspace/litengjie/my_dataset/logs

for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i nohup python scripts/precompute_multiref_vlm_conditions.py \
    /mnt/workspace/litengjie/my_dataset/manifest_shards/train_shard_${i}.json \
    --model-path /mnt/workspace/litengjie/LTX-2/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --text-encoder-path /mnt/workspace/litengjie/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized \
    --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/vlm_conditions \
    --video-column video \
    --caption-column caption \
    --reference-column reference_images \
    --max-ref-images 4 \
    --max-length 8192 \
    --device cuda \
    > /mnt/workspace/litengjie/my_dataset/logs/vlm_conditions_${i}.log 2>&1 &
done
```

这个目录对应配置里的：

```yaml
training_strategy:
  conditions_dir: "vlm_conditions"
```

如果你临时想退回纯文本条件，可以把它改成 `conditions`，但这会让 Stage 1 的基础 context 看不到参考图。

4. 生成 reference image latents：

```bash
python scripts/precompute_multiref_images.py /mnt/workspace/litengjie/my_dataset/train.json \
  --model-path /mnt/workspace/litengjie/LTX-2/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --target-latents-dir /mnt/workspace/litengjie/my_dataset/.precomputed/latents \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/multi_reference_latents \
  --media-column video \
  --reference-column reference_images \
  --device cuda
```

5. 生成 target-video GT SigLIP/projector visual tokens：

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

推荐固定从 target video 的前 `81` 个源帧内均匀抽 `4` 帧。Gemma/SigLIP 每帧产生 `256` 个 projected visual tokens，因此 `num_visual_tokens = 4 * 256 = 1024`。如果你当前日志显示 `Detected 2048 GT visual tokens per sample`，说明这批数据每条样本是 `2048` 个 target-video GT visual tokens；后面的 planner token 数必须用 `2048`。

把实际检测到的值填到 Stage 2 配置的：

```yaml
training_strategy:
  planner_token_count: 2048
```

如果你改成 `--num-sampled-frames N`，则 `planner_token_count = N * 256`。`--sample-fps 6` 也可以用，但 token 数会随源视频 fps 和 `--max-source-frames` 变化；为了 Stage 2 固定 planner placeholders，推荐用 `--num-sampled-frames`。

6. 生成 Stage 2 VLM 输入：

```bash
mkdir -p /mnt/workspace/litengjie/my_dataset/logs

for i in 0 1 2 3 4 5 6 7; do
  nohup python scripts/precompute_planner_vlm_inputs.py \
    /mnt/workspace/litengjie/my_dataset/manifest_shards/train_shard_${i}.json \
    --text-encoder-path /mnt/workspace/litengjie/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized \
    --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/planner_vlm_inputs \
    --video-column video \
    --caption-column caption \
    --reference-column reference_images \
    --max-ref-images 4 \
    --planner-token-count 2048 \
    --max-length 8192 \
    > /mnt/workspace/litengjie/my_dataset/logs/planner_vlm_inputs_${i}.log 2>&1 &
done
```

这里的 `planner-token-count` 必须和 `gt_siglip_tokens` 的 `num_visual_tokens` 完全一致。最多 `4` 张 reference images 只影响 VLM 能看到多少参考图，以及 `multi_reference_latents/` 中有多少参考 latent；它不决定 MSE teacher token 数。

如果你重新按 `4` 帧固定采样生成 `1024` 个 GT tokens，这里就改成：

```bash
  --planner-token-count 1024
```

如果 caption 很长或参考图 token 占用较多，可以把 `--max-length` 提到 `8192`。

## VLM prompt 顺序和 system prompt

默认 system prompt 文件：

```text
packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_multiref_video_planner_system_prompt.txt
```

这个 prompt 明确告诉 VLM：reference images 是 source visual conditions，可能提供主体身份、外观、衣服、物体、风格、空间线索等；它们不是 target video 的首帧，除非用户显式说明。

Stage 1 的 `vlm_conditions/` 使用顺序：

```text
system prompt
-> User Raw Input Prompt: {caption}
-> Reference image 1: <image tokens>
-> Reference image 2: <image tokens>
-> ...
```

Stage 2 的 `planner_vlm_inputs/` 使用顺序：

```text
system prompt
-> User Raw Input Prompt: {caption}
-> Reference image 1: <image tokens>
-> Reference image 2: <image tokens>
-> ...
-> <image_start>
-> <image_pad> repeated planner_token_count times
-> <image_end>
```

这等价于 Bernini 公式里的 `MLLM(t, v_src_1, ..., v_src_N, v_tgt)`：文本 `t` 在前，参考图 `v_src` 在中间，target/planner visual slots `v_tgt` 在最后。这里的 `<image_start>/<image_pad>/<image_end>` 使用 Gemma 已有的图像特殊 token id，不扩 tokenizer，不 resize embedding。`planner_placeholder_mask` 只标记中间的 `planner_token_count` 个 `<image_pad>`；前后的 boundary tokens 只用于告诉 VLM 这是 target visual planning region。

VLM 内部仍沿用 Gemma language model 的 causal attention；因为 target planner region 位于序列最后，`<image_pad>` 可以 attend 到前面的 system/user/reference image tokens。VLM 输出后，代码提取 `<image_pad>` hidden states 作为 K/V，再用 repeated LTX thinking/register tokens 作为 Q，经过 zero-init cross-attention + zero-init FFN 得到最终 predicted visual planner tokens。也就是说，thinking/register tokens 不再用于初始化 `<image_pad>` embedding，而是替代 Baton 图里的 Learnable Video Query。

## 训练时的数据流

Stage 1 每条样本的数据流：

```text
system prompt + user prompt + reference_images
  -> precompute_multiref_vlm_conditions.py
  -> vlm_conditions
  -> DiT condition context before visual tokens

reference_images
  -> precompute_multiref_images.py
  -> multi_reference_latents
  -> prepended clean reference latent tokens in DiT latent stream

target video
  -> precompute_gt_siglip_tokens.py
  -> gt_siglip_tokens.visual_tokens
  -> text connector input after VLM context tokens
  -> DiT condition tokens
```

Stage 2 每条样本的数据流：

```text
reference_images + system prompt + user prompt + <image_start> + <image_pad>*K + <image_end>
  -> VLM/Gemma language model
  -> hidden states at <image_pad> positions as K/V
  -> zero-init cross-attention + zero-init FFN with repeated LTX thinking/register tokens as Q
  -> predicted planner visual tokens
  -> text connector input after VLM context tokens
  -> DiT condition tokens

target video
  -> gt_siglip_tokens.visual_tokens
  -> MSE teacher for predicted planner visual tokens
```

因此，Stage 1 和 Stage 2 的最大区别不是是否使用参考图，而是 DiT condition 里的 visual tokens 来源不同：

- Stage 1：使用 target-video GT SigLIP/projector tokens。
- Stage 2：使用 VLM `<image_pad>` hidden states + repeated LTX thinking/register-token queries 预测出来的 tokens，并用 target-video GT tokens 对齐。

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

固定 4 帧采样时推荐输出类似：

```text
visual_tokens: (1024, D)
num_visual_tokens: 1024
tokens_per_frame: 256
sampled_frame_indices: [...]
```

如果你当前日志显示 `2048`，则这里应是 `(2048, D)`，Stage 2 的 `planner_token_count` 也必须是 `2048`。

路径应该镜像 `video` 字段的 target video 路径，而不是 reference image 路径。

## Stage 1 训练

编辑 `configs/multiref_stage1_lora.yaml`：

- `model.model_path`：LTX-2 checkpoint。
- `model.text_encoder_path`：Gemma text encoder 目录。
- `data.preprocessed_data_root`：`/mnt/workspace/litengjie/my_dataset/.precomputed`。
- `training_strategy.conditions_dir`：默认 `vlm_conditions`，也就是 `system + user + reference images` 编码后的 VLM context。
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
- `training_strategy.conditions_dir`：保持和 Stage 1 一致，默认 `vlm_conditions`。
- `training_strategy.planner_token_count`：必须等于 `gt_siglip_tokens` 的 `num_visual_tokens`。
- `training_strategy.planner_cross_attention_heads`：zero-init planner cross-attention 的 head 数，默认 `16`。
- `training_strategy.planner_zero_init_cross_attention: true`：cross-attention 输出投影零初始化，初始时是 repeated thinking/register query 的 residual。
- `training_strategy.planner_ffn_multiplier: 4.0`：planner FFN 的 hidden dim 倍率。
- `training_strategy.planner_zero_init_ffn: true`：FFN 输出投影零初始化，初始不扰动 cross-attention 后的 residual。
- `training_strategy.planner_slot_encoding: true`：给 query slots 和 VLM placeholder hidden states 加可学习的 slot/type encoding，用于区分 token 位置和角色。
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

Stage 2 的 DiT 输入区别是：visual tokens 来自 VLM `<image_pad>` hidden states 与 repeated LTX thinking/register queries 的 cross-attention + FFN 输出。MSE 保证预测 tokens 和 Stage 1 使用的 target-video GT SigLIP/projector tokens 位于同一 token 数量、同一特征空间。

## Stage 3 后续目标

Stage 3 应从 Stage 2 checkpoint 继续，联合训练：

- 多参考 latent conditioning。
- VLM predicted visual tokens。
- text/thinking tokens。
- DiT LoRA、planner tokens、Gemma language model、text connector 的可训练组合。

SigLIP vision tower 和 multi-modal projector 默认仍冻结。Stage 3 的重点是把 Stage 1 学会的 GT visual-token 使用方式，和 Stage 2 学会的 VLM predicted visual-token 分布联合微调到最终视频生成目标上。
