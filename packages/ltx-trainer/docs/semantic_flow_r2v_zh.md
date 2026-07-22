# R2V-Next 参考感知语义流架构

本实现使用一条端到端训练路径，不再包含旧的分阶段视觉规划器、固定长度视觉占位符、目标视觉对齐分支或额外的条件视觉 token。

## 条件与 teacher

- 冻结的 Gemma 前缀顺序为：system → 参考图 → user 指令。
- 参考图走 Gemma 原生 vision tower 和 projector；冻结的 LTX feature extractor/connector 生成 DiT cross-attention 条件。
- 训练时只从 GT 语义锚点帧提取原生 16×16、每帧 256 个 evidence token。
- 每帧初始化 8×8、共 64 个局部语义 query；每个 query 只读取对应的 2×2 evidence 区域。
- teacher mask 保证前缀看不到 GT suffix；evidence 可看前缀和同帧 evidence；query 可看前缀、局部 evidence 与自身。
- 冻结 Gemma/vision/projector/connector/VAE；训练完整 DiT、语义 query 初始化器、语义 encoder 和重建 decoder。

## 联合 flow matching

DiT 自注意力序列固定为：

```text
[clean reference latent tokens]
[noisy kept semantic tokens]
[noisy target video tokens]
```

参考 token 的 sigma、loss 和 velocity 恒为 0。语义与视频共享同一个 sigma，但使用独立噪声；语义 token 由单独的 velocity head 预测。训练损失分别记录视频 flow、语义 flow、局部 evidence 重建三项。

条件 dropout 为单次耦合采样：`full=0.70`、`drop_text=0.10`、`drop_reference_all=0.15`、`drop_all=0.05`。`drop_reference_all` 同时移除 Gemma 前缀参考图和 VAE 参考 latent、entity、mask、metadata。

## 数据路径

1. 第一版生产训练仅启用 I2I 与 OpenS2V；Phantom/VFR 暂不启用。
2. canonical manifest 保存 clip plan、原始 fps、source id 与 endpoint-inclusive 语义锚点。
3. 121 帧 R2V 按 10% 取 12 个语义锚点；I2I 固定为 `[0]`。
4. target 使用跨帧共享增强；reference 独立增强，同一增强结果同时送入 VLM 和 VAE。
5. source-aware DDP sampler 保证 30/70 I2I/R2V 和 R2V 数据源配比，并可按 cursor 精确恢复。

当前 online R2V manifest planning 仅支持 constant-frame-rate videos with source_fps >= 24。
Variable-frame-rate Phantom videos are not enabled in the first training version.

## 运行保护

完整 22B DiT semantic-flow 训练必须使用 Accelerate FSDP FULL_SHARD。Plain DDP 和单进程 full
semantic-flow 训练会在 trainer 初始化阶段被拒绝，以避免复制完整模型与 optimizer state 导致 OOM。

数据源与增强示例见 `configs/multitask_online_480p121_data.yaml`，训练配置见 `configs/semantic_flow_multitask_480p121.yaml`。

## 严格 no-GT 推理

推理 encoder 只接受文本和 1–4 张参考图。语义和视频均从独立噪声开始，参考 latent 保持干净；Euler ODE 在每一步为语义/视频使用同一个 sigma，并将参考 velocity 强制为 0。接口显式拒绝 target pixels、target latents、teacher evidence 等目标派生字段。

入口：

```bash
python scripts/infer_multitask_online_train_samples.py \
  --config configs/semantic_flow_multitask_480p121.yaml \
  --samples /path/to/selected_samples.jsonl \
  --checkpoint /path/to/model_weights_step_30000.safetensors \
  --output-root /path/to/inference_outputs
```

## 检查

```bash
python -m compileall src scripts tests
pytest -q tests/test_semantic_flow_architecture.py
pytest -q tests/test_phantom_adapter.py tests/test_phantom_manifest_builder.py
pytest -q tests/test_online_augmentation_determinism.py tests/test_online_source_sampling.py
```
