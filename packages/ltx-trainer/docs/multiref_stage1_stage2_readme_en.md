# LTX-2 Multi-Reference Image + Text + VLM Planner Training

This branch extends the DiT conditioning sequence to:

```text
text embedding tokens + thinking/register tokens + visual tokens
```

Stage 1 uses GT visual tokens produced by the frozen Gemma/SigLIP vision tower and frozen multi-modal projector. Stage 2 replaces those GT visual tokens with VLM-predicted visual tokens from a fixed number of learnable planner placeholders, and aligns them to the Stage 1 GT SigLIP/projector tokens with MSE. `planner_token_count` must exactly match `num_visual_tokens` in `gt_siglip_tokens/*.pt`.

## Main Changes

- `ltx_core.multicond.visual_tokens`: frozen SigLIP/projector token extraction, Gemma image-token scatter, and fixed-count `VisualPlannerTokens`.
- `ltx_trainer.training_strategies.multi_reference_video`: Stage 1 appends `gt_siglip_tokens/visual_tokens` after the normal text features before the LTX text connector.
- `ltx_trainer.training_strategies.multi_reference_planner_stage2`: Stage 2 uses fixed `planner_token_count` learnable placeholders. The VLM hidden states at those placeholder positions become the predicted visual tokens; the same tokens are injected into the DiT condition sequence and aligned to GT visual tokens with MSE.
- `scripts/precompute_gt_siglip_tokens.py`: builds `.precomputed/gt_siglip_tokens/` from `reference_images`.
- `scripts/precompute_planner_vlm_inputs.py`: builds system/user VLM inputs and appends a fixed planner placeholder mask.
- `configs/multiref_stage1_lora.yaml` and `configs/multiref_stage2_planner.yaml`: Stage 1/2 configs.

## Preprocessed Layout

```text
/mnt/workspace/litengjie/my_dataset/.precomputed/
├── conditions/                 # pre-connector text/thinking features
├── latents/                    # target video VAE latents
├── multi_reference_latents/    # reference-image VAE latents for Stage 1 latent-stream conditioning
├── gt_siglip_tokens/           # frozen SigLIP/projector GT visual tokens
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

4. Build GT SigLIP/projector visual tokens:

```bash
python scripts/precompute_gt_siglip_tokens.py /mnt/workspace/litengjie/my_dataset/train.json \
  --text-encoder-path /mnt/workspace/litengjie/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/gt_siglip_tokens \
  --video-column video \
  --reference-column reference_images \
  --device cuda
```

The script logs the detected `num_visual_tokens`. Use that value in:

```yaml
training_strategy:
  planner_token_count: <num_visual_tokens>
```

5. Build Stage 2 VLM inputs:

```bash
python scripts/precompute_planner_vlm_inputs.py /mnt/workspace/litengjie/my_dataset/train.json \
  --text-encoder-path /mnt/workspace/litengjie/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/planner_vlm_inputs \
  --video-column video \
  --caption-column caption \
  --reference-column reference_images \
  --planner-token-count <num_visual_tokens>
```

## Stage 1 Training

Edit `configs/multiref_stage1_lora.yaml`:

- `model.model_path`: LTX-2 checkpoint.
- `model.text_encoder_path`: Gemma text encoder directory.
- `data.preprocessed_data_root`: `/mnt/workspace/litengjie/my_dataset/.precomputed`.
- `training_strategy.gt_visual_tokens_dir`: defaults to `gt_siglip_tokens`.
- `output_dir`: Stage 1 output directory.

Run:

```bash
accelerate launch --num_processes 8 --num_machines 1 \
  scripts/train.py configs/multiref_stage1_lora.yaml
```

Stage 1 feeds GT SigLIP/projector visual tokens to the DiT condition path.

## Stage 2 Training

Edit `configs/multiref_stage2_planner.yaml`:

- `model.load_checkpoint`: Stage 1 checkpoint.
- `training_strategy.planner_token_count`: must equal `num_visual_tokens` in `gt_siglip_tokens`.
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

Stage 2 feeds VLM-predicted visual tokens to the DiT condition path. The MSE loss keeps those predicted tokens aligned with the exact GT SigLIP/projector token count and feature space learned by Stage 1.

## Stage 3 Target

Stage 3 should continue from the Stage 2 checkpoint and jointly fine-tune:

- multi-reference latent conditioning,
- VLM-predicted visual tokens,
- text/thinking tokens,
- the selected DiT LoRA modules, planner tokens, Gemma language model, and text connector.

The SigLIP vision tower and multi-modal projector should remain frozen by default. The purpose is to jointly adapt the Stage 1 GT-token renderer and the Stage 2 predicted-token planner to the final controllable video generation objective.
