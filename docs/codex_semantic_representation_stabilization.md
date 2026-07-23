# Codex 任务：稳定 Semantic Representation，阻断 DiT 反向塑形并增加浅层教师对齐

> 仓库：`TjieLee/R2V-Next`  
> 基线分支：`main`  
> 基线提交：`02600906247957c38c0ea1b64e068c23cf0bc0bc`  
> 任务性质：模型结构与训练损失修改  
> 本文档是实现合同。除本文明确要求的内容外，不要扩展模型范围或重构无关代码。

---

## 1. 背景与问题

当前 semantic-flow 训练中，`SemanticEncoder` 输出的 `semantic_clean` 同时承担两种职责：

1. 作为 semantic reconstruction 的 latent；
2. 加噪后进入 DiT，参与 semantic flow 和 video flow 的联合训练。

现有实现中，DiT loss 可以通过 noisy semantic input 路径反向更新 `SemanticEncoder`。这产生一个危险的退化方向：如果 `semantic_clean` 的尺度或有效变化被压小，semantic flow 会变得更容易，而 semantic token 不一定继续保留足够语义。

当前 `SemanticEncoder` 末端还存在可学习的 `global_scale`，形成了直接的整体缩放通道。

本任务的目标是：

- 让 semantic representation 只由明确的冻结教师监督定义；
- 禁止 semantic flow 和 video flow 反向修改 `SemanticEncoder` 与 `SemanticQueryInitializer`；
- 保留原有 `reference → semantic → target` 联合 DiT forward；
- 增加一个浅层、逐 token 的教师对齐 head，避免现有 four-token reconstruction 成为唯一表征监督；
- 通过 semantic velocity head 的零初始化，减小新增 semantic 输出分支在训练初期对预训练 DiT 的冲击。

---

## 2. 必须保持不变的模型合同

不得修改以下合同：

- 目标视频规格：121 帧、24 FPS、832×480；
- R2V anchor 索引：`[0, 11, 22, 33, 44, 55, 65, 76, 87, 98, 109, 120]`；
- 每个 anchor：256 个 evidence token、64 个 semantic query token；
- DiT token 顺序：`reference → semantic → target`；
- semantic 与 target 共用同一个 sigma；
- reference timestep 为 0，reference velocity 为 0；
- semantic RoPE：`target_interpolated_8x8`；
- reference RoPE：`appended_time_shifted_width`；
- max reference images = 4；
- I2I/OpenS2V 多任务范围不变；
- Phantom 保持禁用；
- Full-DiT 训练不变；
- 冻结 Gemma、视觉塔、projector、connector 和 VAE；
- 不增加 attention-mask warmup，不增加 target→semantic curriculum；
- 不使用 Transformer alignment head；
- 不增加 variance/covariance loss；
- 不修改现有 semantic token dropout 规则。

---

## 3. 最终结构

训练时的数据流必须变为：

```text
Frozen visual evidence / Frozen Gemma
                │
                ▼
          query_hidden
                │
                ▼
        SemanticEncoder
                │
                ▼
         semantic_clean
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
reconstruction head   alignment head
        │                │
        ▼                ▼
reconstruction loss  alignment loss
        │                │
        └──────更新 SemanticQueryInitializer / SemanticEncoder / 两个监督 head

semantic_clean.detach()
        │
        ▼
semantic corruption
        │
        ▼
reference → semantic → target
        │
        ▼
       DiT
        │
        ▼
semantic flow + video flow
只更新 DiT，不更新 SemanticQueryInitializer / SemanticEncoder
```

推理路径不使用 reconstruction head 和 alignment head。

---

## 4. 修改一：DiT 路径对 `semantic_clean` 完整 stop-gradient

文件：

```text
packages/ltx-trainer/src/ltx_trainer/training_strategies/semantic_flow.py
```

在 `prepare_training_inputs()` 中，保留原始 `semantic_clean` 给 reconstruction 与 alignment 使用，但在任何 DiT 相关操作前创建：

```python
semantic_for_dit = semantic_clean.detach()
```

随后以下全部操作必须使用 `semantic_for_dit`，不能再使用带梯度的 `semantic_clean`：

- `torch.randn_like(...)` 的 semantic shape/dtype 参考；
- noisy semantic 构造；
- semantic flow target 构造；
- semantic token dropout 与 packing；
- 最终拼接进入 DiT 的 semantic token。

目标形式：

```python
semantic_for_dit = semantic_clean.detach()
semantic_noise = torch.randn_like(semantic_for_dit)

semantic_noisy_all = (
    (1.0 - sigma[:, None, None, None]) * semantic_for_dit
    + sigma[:, None, None, None] * semantic_noise
)

semantic_flow_target_all = semantic_noise - semantic_for_dit
```

必须保证：

- semantic flow loss 对 `SemanticEncoder` 无梯度；
- video flow 通过联合 self-attention 对 semantic input 的梯度，也不能回到 `SemanticEncoder`；
- reconstruction loss 与新增 alignment loss仍可更新 `SemanticQueryInitializer` 和 `SemanticEncoder`；
- DiT forward 仍然看到正常的非零 semantic token；
- 不对 semantic token 做输入值置零；
- 不使用随机 gradient mask。

---

## 5. 修改二：取消 `SemanticEncoder.global_scale`

文件：

```text
packages/ltx-core/src/ltx_core/multicond/semantic_tokens.py
```

当前 `SemanticEncoder.forward()` 使用可学习的 `global_scale` 乘以网络输出。本任务要求取消该乘法。

新 forward 应等价于：

```python
def forward(self, query_hidden: Tensor) -> Tensor:
    return self.network(query_hidden)
```

要求：

- 保留现有网络中的 RMSNorm，包括现有 affine 行为；
- 不把末端 RMSNorm 改成无 affine；
- 不增加固定 RMS 归一化；
- 不增加新的可学习 scalar gate；
- 新 checkpoint 不应再依赖 `global_scale`。

### 5.1 旧 checkpoint 兼容

当前仓库存在包含 `training_strategy.semantic_encoder.global_scale` 的旧 semantic-flow checkpoint。实现必须提供显式迁移，不允许因为删除参数而直接报 unexpected key。

推荐实现：

- 从新 `SemanticEncoder` 模块中删除 `global_scale` 参数；
- 从 `SINGLE_VALUE_CHECKPOINT_PARAMETERS` 中删除 `("semantic_encoder", "global_scale")`；
- 加载 `semantic_flow_v1` checkpoint 时，如果发现旧 `global_scale` key，显式丢弃并记录一次 warning；
- 加载新架构 checkpoint 时，不允许用该兼容逻辑掩盖其他 unexpected key；
- `position_gate` 仍保持 shape `[1]`，相关兼容逻辑不变。

不要把旧 `global_scale` 保留为仍参与 forward 的冻结参数。

---

## 6. 修改三：新增浅层 `SemanticAlignmentHead`

文件：

```text
packages/ltx-core/src/ltx_core/multicond/semantic_tokens.py
```

新增逐 semantic token 独立工作的浅层 MLP head。

建议结构：

```python
class SemanticAlignmentHead(nn.Module):
    def __init__(
        self,
        semantic_dim: int,
        gemma_dim: int,
        *,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.RMSNorm(semantic_dim, elementwise_affine=True),
            nn.Linear(semantic_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, gemma_dim),
        )

    def forward(self, semantic_latent: Tensor) -> Tensor:
        return self.network(semantic_latent)
```

输入输出：

```text
输入： [B, F_anchor, 64, C_semantic]
输出： [B, F_anchor, 64, D_gemma]
```

其中：

- `C_semantic` 来自 `transformer.patchify_proj.in_features`，通常是视频 VAE latent token channel 维度；
- `D_gemma` 来自冻结 Gemma input embedding / hidden dimension，当前环境为 3840；
- 不得硬编码 128 或 3840，必须运行时读取；
- 不使用 Transformer；
- 不允许不同 semantic token在 alignment head 中互相通信；
- 不在 teacher target 一侧增加 learnable projector。

将该类加入相应导出与测试。

---

## 7. 修改四：构造 pooled contextual evidence teacher target

沿用现有：

```python
gather_local_evidence(evidence_hidden)
```

其输出为：

```text
[B, F_anchor, 64, 4, D_gemma]
```

现有 four-token reconstruction target保持不变：

```python
reconstruction_target = gather_local_evidence(evidence_hidden).detach()
```

新增 alignment target：

```python
alignment_target = reconstruction_target.mean(dim=-2).detach()
```

形状：

```text
[B, F_anchor, 64, D_gemma]
```

新增 prediction：

```python
alignment_prediction = alignment_head(semantic_clean)
```

要求：

- 使用 Gemma 最后一层 contextualized `evidence_hidden`；
- 使用每个 semantic token对应的局部 2×2、共 4 个 evidence hidden 的均值；
- teacher target 必须 detach；
- 不额外提取原始 SigLIP hidden；
- 不新增第二套 raw evidence alignment；
- alignment 只在训练时计算。

`build_semantic_teacher_outputs()` 应返回：

```text
alignment_prediction
alignment_target
```

命名可按现有风格调整，但语义必须清晰。

---

## 8. 修改五：新增 cosine alignment loss

文件：

```text
packages/ltx-core/src/ltx_core/multicond/semantic_tokens.py
```

新增函数，例如：

```python
def semantic_alignment_loss(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(...)

    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.detach().float(), dim=-1)
    loss = 1.0 - (prediction * target).sum(dim=-1)
    return loss.flatten(1).mean(dim=1)
```

要求：

- 返回 per-sample shape `[B]`，与现有 loss 风格一致；
- cosine 计算使用 float32，避免 bf16 数值问题；
- target 必须 detach；
- 对 prediction/target shape 做严格校验；
- 不加入额外 MSE、InfoNCE、batch negatives 或 temperature。

---

## 9. 修改六：策略模块注册与配置

文件：

```text
packages/ltx-trainer/src/ltx_trainer/training_strategies/semantic_flow.py
packages/ltx-trainer/src/ltx_trainer/config.py
packages/ltx-trainer/configs/semantic_flow_multitask_480p121.yaml
```

### 9.1 新配置

在 `SemanticFlowConfig` 增加：

```python
semantic_alignment_hidden_dim: int = Field(default=1024, ge=1)
semantic_alignment_weight: float = Field(default=1.0, ge=0.0)
```

生产 YAML 增加：

```yaml
training_strategy:
  semantic_alignment_hidden_dim: 1024
  semantic_alignment_weight: 1.0
```

现有权重保持：

```yaml
video_flow_weight: 1.0
semantic_flow_weight: 1.0
semantic_reconstruction_weight: 1.0
```

### 9.2 模块生命周期

在 `SemanticFlowStrategy` 中新增成员：

```text
_semantic_alignment_head
```

在 `attach_models()` 中使用运行时 `semantic_dim` 与 `gemma_dim` 构造。

在以下接口中完整注册：

- `get_trainable_modules()`；
- `set_trainable_modules()`；
- checkpoint save/load；
- FSDP prepare；
- trainable parameter统计。

新 checkpoint必须包含：

```text
training_strategy.semantic_alignment_head.*
```

---

## 10. 修改七：总损失

需要扩展 `ModelInputs` 或等价结构，携带：

```text
semantic_alignment_prediction
semantic_alignment_target
```

`compute_loss()` 中增加：

```python
alignment_loss = semantic_alignment_loss(
    inputs.semantic_alignment_prediction,
    inputs.semantic_alignment_target,
)
```

总损失变为：

```text
L_total
= video_flow_weight × L_video_flow
+ semantic_flow_weight × L_semantic_flow
+ semantic_reconstruction_weight × L_reconstruction
+ semantic_alignment_weight × L_alignment
```

要求：

- 不改变现有 video、semantic 和 reconstruction loss 定义；
- alignment loss 以 per-sample `[B]` 形式参与总和；
- 新增 metric：

```text
train/loss_semantic_alignment
```

普通 `--disable-progress-bars` 日志增加：

```text
Alignment: <value>
```

当前日志仍然只代表梯度累积边界上的最后一个 microbatch；本任务不修改该日志聚合口径。

---

## 11. 修改八：semantic velocity head 零初始化

文件：

```text
packages/ltx-core/src/ltx_core/model/transformer/model.py
```

在 `enable_semantic_flow_conditioning()` 首次构造：

```text
semantic_norm_out
semantic_proj_out
```

之后，对 `semantic_proj_out` 执行：

```python
nn.init.zeros_(self.semantic_proj_out.weight)
n.init.zeros_(self.semantic_proj_out.bias)
```

目的：

- 初始 semantic velocity 输出为 0；
- 第一批 semantic flow 梯度优先更新新的 output head；
- 在 head 权重仍为 0 时，semantic loss 对共享 DiT hidden 的反向梯度为 0；
- 随着 head 更新，semantic flow再逐步进入共享 DiT。

要求：

- 仅在模块首次创建时零初始化；
- 如果 semantic head 已存在，函数的 early return 行为不能重新清零参数；
- 从 checkpoint 加载时，checkpoint 权重必须覆盖初始化值；
- 不零初始化 `SemanticEncoder`；
- 不把 `semantic_clean` 置零；
- 不增加 attention mask warmup。

---

## 12. Checkpoint 架构版本与迁移

当前 metadata 使用：

```text
architecture = semantic_flow_v1
```

本次结构增加新 trainable module并删除 `global_scale`，必须升级为：

```text
architecture = semantic_flow_v2
```

建议新增 metadata：

```text
semantic_encoder_dit_gradient = detached
semantic_alignment_target = pooled_contextual_local_2x2
semantic_alignment_head = tokenwise_mlp
semantic_velocity_head_init = zero
```

### 12.1 v1 → v2 warm migration

允许显式加载 `semantic_flow_v1` 权重作为 warm start：

- 加载已有 DiT、query initializer、semantic encoder、reconstruction decoder；
- 丢弃旧 `semantic_encoder.global_scale`；
- 新 alignment head 使用默认初始化；
- 已有旧 semantic velocity head权重按 checkpoint加载，不能被重新清零；
- 打印明确 warning，说明 alignment head是新初始化且 representation contract 已变化。

### 12.2 v2 strict load

加载 `semantic_flow_v2` checkpoint 时：

- alignment head必须完整存在；
- 不允许缺 key；
- 不允许出现旧 `global_scale`；
- 继续保持 strict shape validation。

### 12.3 训练恢复语义

不要把旧 v1 production run 当作完全精确 resume继续训练。

原因：

- representation gradient contract已变化；
- 新增 alignment head；
- 当前 minimal training state 本就不含完整 AdamW moments。

从 v1 权重启动 v2 时应视为 warm start，使用新的 output dir和新的训练日志。

---

## 13. 需要修改或关注的主要文件

至少检查并按需修改：

```text
packages/ltx-core/src/ltx_core/multicond/semantic_tokens.py
packages/ltx-core/src/ltx_core/model/transformer/model.py
packages/ltx-trainer/src/ltx_trainer/training_strategies/semantic_flow.py
packages/ltx-trainer/src/ltx_trainer/training_strategies/base_strategy.py
packages/ltx-trainer/src/ltx_trainer/config.py
packages/ltx-trainer/src/ltx_trainer/trainer.py
packages/ltx-trainer/configs/semantic_flow_multitask_480p121.yaml
```

并更新相关测试文件。不要因为路径列表存在而强制修改没有必要的文件。

---

## 14. 单元测试要求

Codex 在 Mac 上至少完成以下 CPU/静态测试。

### 14.1 Alignment shape 与 loss

- 构造 `semantic=[B,F,64,C_sem]`；
- 构造 `evidence_hidden=[B,F,256,D_gemma]`；
- 验证 pooled target shape 为 `[B,F,64,D_gemma]`；
- 验证 head prediction shape一致；
- 验证 loss 返回 `[B]`；
- prediction/target shape不一致时必须报错；
- prediction 与 target 相同方向时 loss接近 0。

### 14.2 DiT loss 不回传 SemanticEncoder

构造最小可运行 strategy/model mock：

- 仅对 video flow + semantic flow反向；
- 确认 `SemanticEncoder` 参数无梯度；
- 确认 `SemanticQueryInitializer` 参数无来自 DiT 的梯度；
- 确认 DiT 参数有梯度。

不能只检查 `semantic_flow_target.detach()`；必须覆盖 noisy semantic input 路径和 video-flow 联合 attention 路径。

### 14.3 Representation loss 正常回传

- 对 reconstruction + alignment loss反向；
- 确认 `SemanticEncoder` 有非零有限梯度；
- 确认 alignment head有非零有限梯度；
- 在可构造的测试条件下，确认 query initializer有梯度。

### 14.4 `global_scale` 已取消

- 新 `SemanticEncoder.state_dict()` 不包含 `global_scale`；
- forward 不依赖任何 scalar gate；
- `position_gate` 仍为 shape `[1]`。

### 14.5 semantic velocity head 零初始化

- 首次 `enable_semantic_flow_conditioning()` 后，`semantic_proj_out.weight/bias` 全零；
- 一次非零 semantic loss backward后，`semantic_proj_out` 梯度非零且有限；
- 第一次 backward时，共享 hidden/input 的 semantic-head梯度符合零初始化预期；
- 再次调用 enable函数不能重新清零已有 head。

### 14.6 Checkpoint v1/v2

- v1 checkpoint包含旧 `global_scale` 且缺 alignment head时，可按 warm migration加载；
- warning明确；
- v2 checkpoint缺 alignment head时严格失败；
- v2 checkpoint包含旧 `global_scale` 时严格失败；
- v2 save/load roundtrip保持 alignment head参数一致；
- metadata architecture为 `semantic_flow_v2`。

### 14.7 配置与生产合同

- 生产配置解析成功；
- alignment hidden dim = 1024；
- alignment weight = 1.0；
- 其他 anchors、frames、fps、loss weights、FSDP配置未被改变。

---

## 15. Codex 本地验证

Codex 只能在 Mac 上完成代码和基础验证，不得声称完成真实 H200/FSDP训练验证。

至少执行：

```bash
python -m compileall packages/ltx-core/src packages/ltx-trainer/src packages/ltx-trainer/scripts
```

运行所有与以下关键词相关的单元测试：

```text
semantic_flow
semantic_tokens
checkpoint
config
transformer semantic conditioning
```

并执行：

```bash
git diff --check
```

如果本机依赖限制导致部分测试无法运行，必须在提交说明中列出：

- 已运行的测试；
- 未运行的测试；
- 未运行的具体原因；
- 需要服务器验证的项目。

---

## 16. 服务器侧后续验证要求

代码提交后，服务器上必须重新运行 smoke；旧 smoke marker因 commit/config hash变化应视为失效。

至少需要：

1. CPU/轻量 preflight；
2. 2 卡 FSDP checkpoint roundtrip；
3. 2 卡 Accelerate multi-model prepare；
4. 7 卡 I2I one-step；
5. 7 卡 R2V one-step；
6. checkpoint save/load；
7. GPU 7 strict-no-GT inference。

短训练重点观察：

```text
Total
Video flow
Semantic flow
Reconstruction
Alignment
semantic latent RMS
LR
Time/Step
```

预期：

- semantic flow不再通过压缩 SemanticEncoder输出获得捷径；
- alignment loss可正常下降；
- reconstruction loss继续提供细粒度监督；
- semantic velocity head从零输出开始学习；
- video flow无 NaN/Inf；
- query initializer、semantic encoder只接收 representation losses梯度；
- DiT接收 flow losses梯度。

---

## 17. 明确非目标

本任务不要实现：

- attention mask warmup；
- target→semantic逐步开放；
- semantic token输入 gate；
- Transformer alignment/reconstruction decoder；
- 原始 SigLIP feature alignment；
- 多教师 alignment；
- contrastive negatives；
- variance/covariance/VICReg 类损失；
- EMA teacher；
- 修改 semantic dropout；
- 修改数据采样比例；
- 修改 checkpoint间隔与保留数量；
- 修改日志累积口径；
- 修改 reference/semantic/target顺序；
- 修改 RoPE布局；
- 修改 anchor或视频规格。

---

## 18. 完成标准

只有满足以下条件才算完成：

- `semantic_clean` 进入 DiT 前完整 detach；
- DiT flow losses无法更新 SemanticEncoder/QueryInitializer；
- reconstruction 与 alignment仍可更新 SemanticEncoder/QueryInitializer；
- `global_scale` 不再存在于新模型 forward/state dict；
- 新增逐 token浅层 alignment head；
- alignment target为 4 个局部 contextualized evidence hidden 的平均；
- alignment 使用 float32 cosine loss；
- 总 loss新增 alignment项，默认权重1.0；
- semantic velocity output head首次构造时严格零初始化；
- 新 checkpoint为 `semantic_flow_v2`；
- v1 warm migration与v2 strict load均有测试；
- 所有相关单元测试通过；
- 未修改本文禁止变更的模型合同；
- Codex提交说明诚实区分本地验证与服务器验证。
