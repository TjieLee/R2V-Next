# LTX-2 Multi-Reference Image + Text + VLM Planner Training

This branch does not remove LTX-2's original 128 thinking/register tokens. The implementation has two layers:

```text
pre-connector features: VLM context tokens + visual tokens
post-connector DiT context: VLM context tokens + LTX thinking/register tokens + visual tokens
```

`VLM context tokens` are encoded from `system prompt -> user prompt -> reference images`. In Stage 1, target-video GT SigLIP/projector visual tokens are first appended after that context in the pre-connector feature sequence; the features then still pass through the native LTX `Embeddings1DConnector`, which injects/reuses the 128 learnable thinking/register tokens. Stage 2/3 append a Baton-style target planning region after the same source context: `<image_start> + <image_pad> * planner_token_count + <image_end>`. The VLM produces hidden states at the `<image_pad>` positions; the planner bridge uses repeated LTX thinking/register tokens as Q and those hidden states as K/V, then applies zero-init cross-attention + zero-init FFN to produce planner visual tokens, which are aligned to target-video GT SigLIP/projector visual tokens with MSE. `planner_token_count` must exactly match `num_visual_tokens` in `gt_siglip_tokens/*.pt`.

## Visual Sources That Must Not Be Confused

This implementation uses three different visual data streams:

| Directory/data | Source | Where it goes | Role |
| --- | --- | --- | --- |
| `multi_reference_latents/` | `reference_images` | DiT video latent stream, prepended before noisy target video latents | Multi-reference VAE latent conditioning |
| `vlm_conditions/` | `system prompt + caption + reference_images` | Stage 1/2 DiT condition context before visual tokens | Makes the base Stage 1 context see reference images |
| `planner_vlm_inputs/` | `reference_images` plus system/user prompt | VLM/Gemma input | Lets the VLM planner see the reference images and text before predicting planner tokens |
| `gt_siglip_tokens/` | sampled frames from `video`, the target video | Stage 1 DiT condition tokens; Stage 2 MSE teacher | Target-video GT SigLIP/projector visual tokens |

Hard requirements:

- `gt_siglip_tokens/` must be extracted from the target video only, never from `reference_images`.
- Stage 1 DiT conditioning is `VLM(system + user + ref images) context tokens + target-video GT SigLIP/projector tokens`.
- Stage 2 DiT conditioning is `VLM(system + user + ref images) context tokens + VLM-predicted planner tokens`.
- The Stage 2 MSE teacher is the same sample's target-video GT SigLIP/projector tokens.
- `reference_images` are not the Stage 1 condition visual-token teacher and are not the Stage 2 MSE teacher; they only enter the reference VAE latent stream and the VLM input.

## Main Changes

- `ltx_core.multicond.visual_tokens`: frozen SigLIP/projector token extraction, Gemma image-token scatter, and fixed-count `VisualPlannerTokens`.
- `ltx_trainer.training_strategies.multi_reference_video`: Stage 1 can read `vlm_conditions/` through `conditions_dir`, then append `gt_siglip_tokens/visual_tokens` after the VLM context features before the LTX text connector.
- `ltx_trainer.training_strategies.multi_reference_planner_stage2`: Stage 2 uses fixed `planner_token_count` `<image_pad>` target placeholders. Their VLM hidden states act as K/V, repeated LTX thinking/register tokens act as Q, and a zero-init cross-attention + zero-init FFN bridge produces visual planner tokens that replace GT visual tokens in the DiT condition sequence and align to GT visual tokens with MSE.
- `ltx_core.multicond.cfg_sampler` plus the multi-reference training strategies: per-sample CFG condition dropout with default `full/drop_text/drop_ref/drop_all = 0.7/0.1/0.1/0.1`; legacy `drop_planner` is only a compatibility alias for `drop_all/null`.
- `scripts/precompute_gt_siglip_tokens.py`: builds `.precomputed/gt_siglip_tokens/` from sampled target-video frames.
- `scripts/precompute_multiref_vlm_conditions.py`: builds Stage 1/2 `system prompt -> user prompt -> reference image tokens` VLM context conditions.
- `scripts/precompute_planner_vlm_inputs.py`: builds Stage 2 VLM inputs in `system prompt -> user prompt -> reference image tokens -> <image_start> + <image_pad>*K + <image_end>` order.
- `configs/multiref_stage1_lora.yaml` and `configs/multiref_stage2_planner.yaml`: Stage 1/2 configs.

## CFG Training Dropout

Stage 1 and Stage 2 both support training-time condition dropout for factorized CFG. Each sample uses exactly one mode:

| Mode | Default probability | What changes |
| --- | ---: | --- |
| `full` | `0.7` | Keep text/VLM context, reference latents, and visual planner tokens |
| `drop_text` | `0.1` | Remove text conditions. Stage 1/2 DiT context uses `cfg_ref_only_conditions_dir` when available; otherwise it falls back to zeroing the mixed VLM context. Stage 2 online VLM masks `text_token_mask` while keeping reference image tokens and planner slots |
| `drop_ref` | `0.1` | Remove image/visual conditions: set DiT reference latent `ref_valid_mask` to false; swap VLM context to text-only `conditions/` when `cfg_text_conditions_dir` is available; mask reference image tokens in Stage 2 online VLM; zero Stage 1 GT SigLIP tokens and Stage 2 predicted planner visual tokens |
| `drop_all` / `null` | `0.1` | Remove all conditions: zero text/VLM context, mask reference latents, mask both text tokens and reference image tokens in Stage 2 online VLM, and zero GT/predicted visual tokens |

For Stage 2, `drop_ref` and `drop_all/null` samples are excluded from the planner MSE and only train the flow branch. `full` and `drop_text` samples still align predicted planner tokens to target-video GT SigLIP/projector tokens. There is no independent `drop_planner_only` branch; `cfg_drop_planner_p` is only the old name for `cfg_drop_all_p/null`.

## Preprocessed Layout

```text
/mnt/workspace/litengjie/my_dataset/.precomputed/
├── conditions/                 # pre-connector text/thinking features
├── vlm_conditions/             # VLM context features from system + user + reference images
├── latents/                    # target video VAE latents
├── multi_reference_latents/    # reference-image VAE latents for Stage 1 latent-stream conditioning
├── gt_siglip_tokens/           # frozen target-video SigLIP/projector GT visual tokens
└── planner_vlm_inputs/         # Stage 2 VLM inputs + Baton-style target planner placeholder masks
```

## Precompute Order

1. Convert the Phantom manifest:

```bash
python scripts/convert_phantom_manifest.py \
  /mnt/workspace/liutao/phantom_data/train_data_0202.json \
  --root-dir /mnt/workspace/liutao/phantom_data \
  --output-json /mnt/workspace/litengjie/my_dataset/train.json
```

Each row should contain:

```json
{
  "video": "/abs/path/video.mp4",
  "caption": "prompt text",
  "reference_images": ["/abs/path/ref_0.jpg"]
}
```

2. Build text conditions and target video latents:

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

If `conditions/` already exists and you only need `latents/`:

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

3. Build Stage 1/2 VLM reference-image context conditions:

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
    --max-length 4096 \
    --device cuda \
    > /mnt/workspace/litengjie/my_dataset/logs/vlm_conditions_${i}.log 2>&1 &
done
```

This directory is selected by:

```yaml
training_strategy:
  conditions_dir: "vlm_conditions"
```

You can temporarily set it back to `conditions` for text-only conditioning, but then the Stage 1 base context will not see reference images.

4. Build reference-image VAE latents:

```bash
python scripts/precompute_multiref_images.py /mnt/workspace/litengjie/my_dataset/train.json \
  --model-path /mnt/workspace/litengjie/LTX-2/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --target-latents-dir /mnt/workspace/litengjie/my_dataset/.precomputed/latents \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/multi_reference_latents \
  --media-column video \
  --reference-column reference_images \
  --device cuda
```

5. Build target-video GT SigLIP/projector visual tokens:

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

The recommended setting samples `4` frames uniformly from the target video's first `81` source frames. Gemma/SigLIP produces `256` projected visual tokens per sampled frame, so `num_visual_tokens = 4 * 256 = 1024`. If your current log says `Detected 2048 GT visual tokens per sample`, then this dataset has `2048` target-video GT visual tokens per sample and the planner token count below must also be `2048`.

Use the actually detected value in:

```yaml
training_strategy:
  planner_token_count: 2048
```

If you change this to `--num-sampled-frames N`, then `planner_token_count = N * 256`. `--sample-fps 6` is also supported, but the token count then depends on the source video fps and `--max-source-frames`; for fixed Stage 2 planner placeholders, prefer `--num-sampled-frames`.

6. Build Stage 2 VLM inputs:

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
    --max-length 4096 \
    > /mnt/workspace/litengjie/my_dataset/logs/planner_vlm_inputs_${i}.log 2>&1 &
done
```

`planner-token-count` must exactly match `num_visual_tokens` in `gt_siglip_tokens`. The maximum of `4` reference images only controls how many reference images the VLM can see and how many reference latents are stored in `multi_reference_latents/`; it does not define the MSE teacher token count.

If you regenerate GT tokens with fixed `4`-frame sampling and get `1024` GT tokens, use:

```bash
  --planner-token-count 1024
```

`precompute_planner_vlm_inputs.py` reserves `planner_token_count + 2` tokens for the target planner region:

```text
source_max_length = max_length - planner_token_count - 2
```

With `--max-length 4096` and `--planner-token-count 2048`, the system/user/ref-image source side gets at most `2046` tokens. For very long captions or heavier reference-image prompts, increase `--max-length` to `8192`. If the reference image token count does not equal `num_ref_images * 256`, the script skips the sample and tells you to increase `--max-length`, reduce `--planner-token-count`, reduce `--max-ref-images`, or shorten the caption.

New `planner_vlm_inputs/*.pt` files contain:

```text
planner_placeholder_mask    # target <image_pad> positions
planner_boundary_mask       # target <image_start>/<image_end>
planner_region_mask         # placeholder + boundary
ref_visual_token_mask       # pure source reference image-pad tokens
ref_image_region_mask       # source reference image boundaries + image-pad tokens
gt_image_token_mask         # legacy alias for ref_image_region_mask
text_token_mask             # system/user/template text tokens, excluding ref/planner regions
source_max_length
ref_visual_token_count
num_ref_images
```

## VLM Prompt Order And System Prompt

Default system prompt file:

```text
packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_multiref_video_planner_system_prompt.txt
```

This prompt tells the VLM that reference images are source visual conditions. They may provide subject identity, appearance, clothing, objects, style, spatial cues, or other visual constraints. They are not the first frame of the target video unless the user explicitly says so.

Stage 1 `vlm_conditions/` order:

```text
system prompt
-> User Raw Input Prompt: {caption}
-> Reference image 1: <image tokens>
-> Reference image 2: <image tokens>
-> ...
```

Stage 2 `planner_vlm_inputs/` order:

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

This matches the Bernini-style sequence `MLLM(t, v_src_1, ..., v_src_N, v_tgt)`: text `t` first, source reference visuals `v_src` in the middle, and target/planner visual slots `v_tgt` at the end. `<image_start>/<image_pad>/<image_end>` use Gemma's existing image special token ids, so there is no tokenizer extension or embedding resize. `planner_placeholder_mask` marks only the middle `planner_token_count` `<image_pad>` tokens; the boundary tokens only mark this as the target visual planning region.

Inside the VLM, Gemma still uses its standard causal language-model attention. Because the target planner region is at the sequence tail, `<image_pad>` tokens can attend to previous system/user/reference-image tokens. After the VLM forward pass, the code extracts hidden states at `<image_pad>` positions as K/V, then uses repeated LTX thinking/register tokens as Q in a zero-init cross-attention + zero-init FFN bridge to produce final predicted visual planner tokens. In other words, thinking/register tokens are not used to initialize `<image_pad>` embeddings; they replace the Learnable Video Query in the Baton-style VA-planner.

## Training Data Flow

Stage 1 per-sample flow:

```text
system prompt + user prompt + reference_images
  -> precompute_multiref_vlm_conditions.py
  -> vlm_conditions
  -> DiT condition context before visual tokens

reference_images
  -> precompute_multiref_images.py
  -> multi_reference_latents
  -> prepended clean reference latent tokens in the DiT latent stream

target video
  -> precompute_gt_siglip_tokens.py
  -> gt_siglip_tokens.visual_tokens
  -> text connector input after VLM context tokens
  -> DiT condition tokens
```

Stage 2 per-sample flow:

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

Therefore, the main Stage 1 vs Stage 2 difference is not whether reference images are used; it is where the DiT condition visual tokens come from:

- Stage 1: target-video GT SigLIP/projector tokens.
- Stage 2: VLM `<image_pad>` hidden states plus repeated LTX thinking/register-token queries, aligned to the target-video GT tokens.

## Negative RoPE

Reference-image VAE latents are prepended before target video latents. Reference latent tokens use negative temporal RoPE positions:

```text
ref 1: T = -1 * reference_time_stride
ref 2: T = -2 * reference_time_stride
ref 3: T = -3 * reference_time_stride
ref 4: T = -4 * reference_time_stride
```

The default `reference_time_stride: 1.0` gives `-1, -2, -3, -4`. This negative temporal position is only applied to the reference VAE latent stream. `gt_siglip_tokens` and VLM-predicted planner tokens are text-connector condition sequence tokens and do not use this reference-latent RoPE scheme.

## Precompute Sanity Check

Check that `gt_siglip_tokens` were extracted from target videos:

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

For fixed `4`-frame sampling, the output should look like:

```text
visual_tokens: (1024, D)
num_visual_tokens: 1024
tokens_per_frame: 256
sampled_frame_indices: [...]
```

If your current log says `2048`, this should be `(2048, D)`, and Stage 2 `planner_token_count` must be `2048`.

The output path should mirror the `video` target-video path, not a reference-image path.

## Stage 1 Training

Edit `configs/multiref_stage1_lora.yaml`:

- `model.model_path`: LTX-2 checkpoint.
- `model.text_encoder_path`: Gemma text encoder directory.
- `data.preprocessed_data_root`: `/mnt/workspace/litengjie/my_dataset/.precomputed`.
- `training_strategy.conditions_dir`: defaults to `vlm_conditions`, the VLM context encoded from `system + user + reference images`.
- `training_strategy.gt_visual_tokens_dir`: defaults to `gt_siglip_tokens`.
- `training_strategy.visual_token_frame_stride`: defaults to `1`. If you encoded target-video tokens at `6fps` and later want to use them as `3fps`, set it to `2` without rerunning SigLIP.
- `training_strategy.cfg_dropout_enabled: true`: enables CFG condition dropout during training.
- `training_strategy.cfg_full_p/drop_text_p/drop_ref_p/drop_all_p`: defaults to `0.7/0.1/0.1/0.1`; old `cfg_drop_planner_p` configs still load, but it is only a `drop_all/null` alias.
- `training_strategy.cfg_text_conditions_dir: "conditions"`: when `drop_ref` is sampled, use text-only conditions instead of `vlm_conditions`, so reference-image semantics do not remain in the VLM context.
- `training_strategy.cfg_ref_only_conditions_dir: null`: optional ref-only VLM conditions for `drop_text`; when unset, `drop_text` falls back to zeroing the mixed VLM context.
- `output_dir`: Stage 1 output directory.

Run:

```bash
accelerate launch --num_processes 8 --num_machines 1 \
  scripts/train.py configs/multiref_stage1_lora.yaml
```

Stage 1 feeds target-video GT SigLIP/projector visual tokens to the DiT condition path.

## Stage 2 Training

Edit `configs/multiref_stage2_planner.yaml`:

- `model.load_checkpoint`: Stage 1 checkpoint.
- `training_strategy.conditions_dir`: keep this consistent with Stage 1, default `vlm_conditions`.
- `training_strategy.planner_token_count`: must equal `num_visual_tokens` in `gt_siglip_tokens`.
- `training_strategy.cfg_dropout_enabled: true`: enables CFG condition dropout during training.
- `training_strategy.cfg_full_p/drop_text_p/drop_ref_p/drop_all_p`: defaults to `0.7/0.1/0.1/0.1`; old `cfg_drop_planner_p` configs still load, but it is only a `drop_all/null` alias.
- `training_strategy.cfg_text_conditions_dir: "conditions"`: when `drop_ref` is sampled, use text-only conditions instead of `vlm_conditions`.
- `training_strategy.cfg_ref_only_conditions_dir: null`: optional ref-only VLM conditions for `drop_text`; when unset, `drop_text` falls back to zeroing the mixed VLM context.
- `training_strategy.planner_cross_attention_heads`: number of heads in the zero-init planner cross-attention bridge, default `16`.
- `training_strategy.planner_zero_init_cross_attention: true`: zero-initialize the cross-attention output projection, so initialization is a residual over repeated thinking/register queries.
- `training_strategy.planner_ffn_multiplier: 4.0`: hidden-dim multiplier for the planner FFN.
- `training_strategy.planner_zero_init_ffn: true`: zero-initialize the FFN output projection, so initialization does not perturb the post-cross-attention residual.
- `training_strategy.planner_slot_encoding: true`: add trainable slot/type encodings to repeated LTX thinking/register query slots and VLM placeholder hidden states.
- `training_strategy.planner_slot_init_std: 1e-4`: initialize per-slot encodings with a tiny random offset so repeated thinking/register tokens are not perfectly periodic at step 0; set this to `0.0` to recover all-zero initialization.
- `training_strategy.planner_slot_init_seed: 0`: local random seed for slot encoding initialization; set to `null` to use the current torch RNG.
- `training_strategy.visual_token_frame_stride`: must match Stage 1. If Stage 1 uses `2`, Stage 2 also uses `2`, and `planner_token_count` must be the downsampled token count.
- `training_strategy.train_gemma_backbone: false`: freeze the Gemma language-model backbone by default and train only lightweight modules such as the planner bridge, LTX registers, and text connector. Set this to `true` only when you explicitly want Gemma joint fine-tuning.
- `training_strategy.freeze_vlm_vision_tower: true`: freeze SigLIP.
- `training_strategy.freeze_vlm_multi_modal_projector: true`: freeze the image projection.
- `training_strategy.freeze_transformer: true`: freeze the Stage 1 DiT/LoRA by default.
- `training_strategy.train_text_connector: true`: train the text connector.

Run:

```bash
accelerate launch --num_processes 8 --num_machines 1 \
  scripts/train.py configs/multiref_stage2_planner.yaml
```

Stage 2 feeds the cross-attention + FFN output from VLM `<image_pad>` hidden states and repeated LTX thinking/register queries to the DiT condition path. The MSE loss keeps those predicted tokens aligned with the exact target-video GT SigLIP/projector token count and feature space learned by Stage 1.

## Stage 3 Target

Stage 3 should continue from the Stage 2 checkpoint and jointly fine-tune:

- multi-reference latent conditioning,
- VLM-predicted visual tokens,
- text/thinking tokens,
- the selected DiT LoRA modules, planner tokens, text connector, and optionally the Gemma language-model backbone.

The SigLIP vision tower and multi-modal projector should remain frozen by default. Gemma can also stay frozen with `train_gemma_backbone: false`, so Stage 3 only trains register tokens, the planner bridge, MLP/connector modules, and DiT LoRA. If memory and data volume allow it, set `train_gemma_backbone: true` as an optional joint fine-tuning mode. The purpose is to jointly adapt the Stage 1 GT-token renderer and the Stage 2 predicted-token planner to the final controllable video generation objective.
