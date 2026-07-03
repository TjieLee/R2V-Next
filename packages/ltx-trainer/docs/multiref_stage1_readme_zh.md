# 多参考图像 + 文本 Stage 1 训练说明

本次改动新增了一个 video-only 的 LTX-2 Stage 1 训练路径，用于支持 1-N 张参考图像 + 文本的可控视频生成训练。实现上保留 LTX-2 原有 transformer block、text connector 和 CFG 语义，只在数据 collate、reference token 打包、RoPE 位置和 training strategy 侧做增量改造。

## 代码改动

- 新增 `ltx_core.multicond` 工具模块：
  - `cfg_sampler.py`：为后续 factorized CFG 提供 per-sample 条件 drop mask。
  - `factorized_cfg.py`：提供 text/reference/planner 分支 CFG 组合辅助函数。
  - `rope_mask_builder.py`：把多参考图 clean tokens 拼到 target noisy tokens 前面，并为每个参考实体施加负时间轴 RoPE 偏移。
  - `planner_tokens.py`：为后续 Gemma VLM planner 阶段提供零初始化 residual 的 semantic query bridge。
- 新增 `MultiReferenceVideoStrategy`：
  - 读取 `latents/`、`conditions/`、`multi_reference_latents/`。
  - 将 clean reference latent tokens 拼接到 noisy target video tokens 前面。
  - reference tokens 的 timestep 为 `0`，不参与 loss。
  - reference 通过负时间 RoPE slot 区分：第 1 张参考图偏移 `-reference_time_stride`，第 2 张偏移 `-2 * reference_time_stride`，依此类推。
  - 直接返回 `audio=None`，屏蔽 audio 分支。
- 新增 `collate_precomputed_batch`，支持 batch 内动态 padding 变长参考图数量。
- 新增 `scripts/precompute_multiref_images.py`，把每条样本的 1-N 张参考图编码成 `multi_reference_latents/`。
- 新增 `configs/multiref_stage1_lora.yaml`，作为可直接改路径使用的 Stage 1 LoRA 配置。

## 预处理目录结构

```text
/mnt/workspace/litengjie/my_dataset/.precomputed/
├── latents/
├── conditions/
└── multi_reference_latents/
```

`multi_reference_latents/` 里的每个 `.pt` 文件与目标视频 latent 的相对路径保持一致，内容为：

```python
{
    "latents": Tensor[R, C, 1, H, W],
    "ref_valid_mask": BoolTensor[R],
    "num_refs": int,
    "num_frames": 1,
    "height": int,
    "width": int,
    "fps": 1.0,
}
```

其中 `R` 可以每条样本不同；padding 不写入缓存，而是在 DataLoader 当前 batch 内动态完成。

## 元数据格式

建议整理成 CSV/JSON/JSONL，至少包含：

```json
{
  "video": "videos/sample_001.mp4",
  "caption": "A wild boar rests on dry grass.",
  "reference_images": [
    "refs/sample_001/boar_ref.jpg",
    "refs/sample_001/background_ref.jpg"
  ]
}
```

对于你现在的 `stage1_dataset.py`，需要导出这三个概念：

- 目标视频路径 -> `video`，或训练时通过 `--media-column` 指定；
- 文本 prompt -> `caption`；
- cropped reference paths 列表 -> `reference_images`，或通过 `--reference-column` 指定。

## 数据预处理

1. 先按 LTX-2 原流程编码目标视频和文本：

```bash
cd /Users/litengjie.3/CodeSpace/JD-LTX/JD-LTX/packages/ltx-trainer

python scripts/process_dataset.py /mnt/workspace/litengjie/my_dataset/train.json \
  --resolution-buckets "640x384x81" \
  --model-path /path/to/ltx2.safetensors \
  --text-encoder-path /path/to/gemma-text-encoder \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed \
  --video-column video \
  --caption-column caption \
  --skip-audio
```

2. 再编码每条样本的 1-N 张参考图：

```bash
python scripts/precompute_multiref_images.py /mnt/workspace/litengjie/my_dataset/train.json \
  --model-path /path/to/ltx2.safetensors \
  --target-latents-dir /mnt/workspace/litengjie/my_dataset/.precomputed/latents \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/multi_reference_latents \
  --media-column video \
  --reference-column reference_images \
  --device cuda
```

如果还没有 target latents，也可以手动指定参考图分辨率：

```bash
python scripts/precompute_multiref_images.py train.json \
  --model-path /path/to/ltx2.safetensors \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/multi_reference_latents \
  --ref-resolution "640x384"
```

## 启动 Stage 1 训练

编辑 `configs/multiref_stage1_lora.yaml` 中的路径：

- `model.model_path`：本地 LTX-2 checkpoint。
- `model.text_encoder_path`：本地 Gemma text encoder 目录。
- `data.preprocessed_data_root`：包含 `latents/`、`conditions/`、`multi_reference_latents/` 的父目录。
- `output_dir`：必须写到 `/mnt/workspace/litengjie/...` 下。

启动训练：

```bash
cd /Users/litengjie.3/CodeSpace/JD-LTX/JD-LTX/packages/ltx-trainer
accelerate launch scripts/train.py configs/multiref_stage1_lora.yaml
```

## 路径约束

- 日志、缓存、ckpt、可视化结果和生成样例建议全部写到 `/mnt/workspace/litengjie`。
- 可以从 `/mnt/workspace/liutao` 读取已有数据，但不要写入。
- `preprocessed_data_root` 必须指向 `latents/`、`conditions/`、`multi_reference_latents/` 的父目录。
- `reference_latents_dir` 是相对 `preprocessed_data_root` 的目录名，默认是 `multi_reference_latents`。

## 当前范围

本次实现的是 Stage 1 renderer 适配：让 DiT 学会吃多参考图 clean latent canvas。Gemma VLM planner 相关的 query bridge、CFG sampler 和 factorized CFG 已经作为扩展点放入代码，但 Stage 2/3 的 planner 训练与推理 pipeline 还没有接入主训练循环。
