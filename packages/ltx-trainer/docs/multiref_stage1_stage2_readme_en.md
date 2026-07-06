# LTX-2 Multi-Reference Image + Text + VLM Planner Training

This branch extends the DiT conditioning sequence to:

```text
text embedding tokens + thinking/register tokens + visual tokens
```

Stage 1 uses target-video GT visual tokens produced by the frozen Gemma/SigLIP vision tower and frozen multi-modal projector from sampled target-video frames. Stage 2 replaces those GT visual tokens with VLM-predicted visual tokens from a fixed number of learnable planner placeholders, and aligns them to the Stage 1 target-video GT SigLIP/projector tokens with MSE. `planner_token_count` must exactly match `num_visual_tokens` in `gt_siglip_tokens/*.pt`.

## Visual Sources That Must Not Be Confused

This implementation uses three different visual data streams:

| Directory/data | Source | Where it goes | Role |
| --- | --- | --- | --- |
| `multi_reference_latents/` | `reference_images` | DiT video latent stream, prepended before noisy target video latents | Multi-reference VAE latent conditioning |
| `planner_vlm_inputs/` | `reference_images` plus system/user prompt | VLM/Gemma input | Lets the VLM planner see the reference images and text before predicting planner tokens |
| `gt_siglip_tokens/` | sampled frames from `video`, the target video | Stage 1 DiT condition tokens; Stage 2 MSE teacher | Target-video GT SigLIP/projector visual tokens |

Hard requirements:

- `gt_siglip_tokens/` must be extracted from the target video only, never from `reference_images`.
- Stage 1 DiT conditioning is `text/thinking tokens + target-video GT SigLIP/projector tokens`.
- Stage 2 DiT conditioning is `text/thinking tokens + VLM-predicted planner tokens`.
- The Stage 2 MSE teacher is the same sample's target-video GT SigLIP/projector tokens.
- `reference_images` are not the Stage 1 condition visual-token teacher and are not the Stage 2 MSE teacher; they only enter the reference VAE latent stream and the VLM input.

## Main Changes

- `ltx_core.multicond.visual_tokens`: frozen SigLIP/projector token extraction, Gemma image-token scatter, and fixed-count `VisualPlannerTokens`.
- `ltx_trainer.training_strategies.multi_reference_video`: Stage 1 appends `gt_siglip_tokens/visual_tokens` after the normal text features before the LTX text connector.
- `ltx_trainer.training_strategies.multi_reference_planner_stage2`: Stage 2 uses fixed `planner_token_count` learnable placeholders. The VLM hidden states at those placeholder positions become the predicted visual tokens; the same tokens are injected into the DiT condition sequence and aligned to GT visual tokens with MSE.
- `scripts/precompute_gt_siglip_tokens.py`: builds `.precomputed/gt_siglip_tokens/` from sampled target-video frames.
- `scripts/precompute_planner_vlm_inputs.py`: builds system/user VLM inputs and appends a fixed planner placeholder mask.
- `configs/multiref_stage1_lora.yaml` and `configs/multiref_stage2_planner.yaml`: Stage 1/2 configs.

## Preprocessed Layout

```text
/mnt/workspace/litengjie/my_dataset/.precomputed/
├── conditions/                 # pre-connector text/thinking features
├── latents/                    # target video VAE latents
├── multi_reference_latents/    # reference-image VAE latents for Stage 1 latent-stream conditioning
├── gt_siglip_tokens/           # frozen target-video SigLIP/projector GT visual tokens
└── planner_vlm_inputs/         # Stage 2 VLM inputs + fixed planner placeholder masks
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

3. Build reference-image VAE latents:

```bash
python scripts/precompute_multiref_images.py /mnt/workspace/litengjie/my_dataset/train.json \
  --model-path /mnt/workspace/litengjie/LTX-2/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --target-latents-dir /mnt/workspace/litengjie/my_dataset/.precomputed/latents \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/multi_reference_latents \
  --media-column video \
  --reference-column reference_images \
  --device cuda
```

4. Build target-video GT SigLIP/projector visual tokens:

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

The recommended setting samples `4` frames uniformly from the target video's first `81` source frames. Gemma/SigLIP produces `256` projected visual tokens per sampled frame, so `num_visual_tokens = 4 * 256 = 1024`. Use that value in:

```yaml
training_strategy:
  planner_token_count: 1024
```

If you change this to `--num-sampled-frames N`, then `planner_token_count = N * 256`. `--sample-fps 6` is also supported, but the token count then depends on the source video fps and `--max-source-frames`; for fixed Stage 2 planner placeholders, prefer `--num-sampled-frames`.

5. Build Stage 2 VLM inputs:

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

`planner-token-count` must exactly match `num_visual_tokens` in `gt_siglip_tokens`. The maximum of `4` reference images only controls how many reference images the VLM can see and how many reference latents are stored in `multi_reference_latents/`; it does not define the MSE teacher token count.

If your existing `gt_siglip_tokens` log says `Detected 2048 GT visual tokens per sample`, use:

```bash
  --planner-token-count 2048 \
  --max-length 4096
```

For very long captions or heavier reference-image prompts, increase `--max-length` to `8192`.

## Training Data Flow

Stage 1 per-sample flow:

```text
reference_images
  -> precompute_multiref_images.py
  -> multi_reference_latents
  -> prepended clean reference latent tokens in the DiT latent stream

target video
  -> precompute_gt_siglip_tokens.py
  -> gt_siglip_tokens.visual_tokens
  -> text connector input after text/thinking tokens
  -> DiT condition tokens
```

Stage 2 per-sample flow:

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

Therefore, the main Stage 1 vs Stage 2 difference is not whether reference images are used; it is where the DiT condition visual tokens come from:

- Stage 1: target-video GT SigLIP/projector tokens.
- Stage 2: VLM + learnable planner placeholder predicted tokens, aligned to the target-video GT tokens.

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

Recommended output should look like:

```text
visual_tokens: (1024, D)
num_visual_tokens: 1024
tokens_per_frame: 256
sampled_frame_indices: [...]
```

The output path should mirror the `video` target-video path, not a reference-image path.

## Stage 1 Training

Edit `configs/multiref_stage1_lora.yaml`:

- `model.model_path`: LTX-2 checkpoint.
- `model.text_encoder_path`: Gemma text encoder directory.
- `data.preprocessed_data_root`: `/mnt/workspace/litengjie/my_dataset/.precomputed`.
- `training_strategy.gt_visual_tokens_dir`: defaults to `gt_siglip_tokens`.
- `training_strategy.visual_token_frame_stride`: defaults to `1`. If you encoded target-video tokens at `6fps` and later want to use them as `3fps`, set it to `2` without rerunning SigLIP.
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
- `training_strategy.planner_token_count`: must equal `num_visual_tokens` in `gt_siglip_tokens`.
- `training_strategy.visual_token_frame_stride`: must match Stage 1. If Stage 1 uses `2`, Stage 2 also uses `2`, and `planner_token_count` must be the downsampled token count.
- `training_strategy.train_vlm_language_model: true`: train the Gemma language model.
- `training_strategy.freeze_vlm_vision_tower: true`: freeze SigLIP.
- `training_strategy.freeze_vlm_multi_modal_projector: true`: freeze the image projection.
- `training_strategy.freeze_transformer: true`: freeze the Stage 1 DiT/LoRA by default.
- `training_strategy.train_text_connector: true`: train the text connector.

Run:

```bash
accelerate launch --num_processes 8 --num_machines 1 \
  scripts/train.py configs/multiref_stage2_planner.yaml
```

Stage 2 feeds VLM-predicted visual tokens to the DiT condition path. The MSE loss keeps those predicted tokens aligned with the exact target-video GT SigLIP/projector token count and feature space learned by Stage 1.

## Stage 3 Target

Stage 3 should continue from the Stage 2 checkpoint and jointly fine-tune:

- multi-reference latent conditioning,
- VLM-predicted visual tokens,
- text/thinking tokens,
- the selected DiT LoRA modules, planner tokens, Gemma language model, and text connector.

The SigLIP vision tower and multi-modal projector should remain frozen by default. The purpose is to jointly adapt the Stage 1 GT-token renderer and the Stage 2 predicted-token planner to the final controllable video generation objective.
