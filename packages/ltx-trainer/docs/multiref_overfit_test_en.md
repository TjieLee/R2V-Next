# Multi-Reference / Planner 100-Sample Overfit Test

This README is dedicated to the small-scale validation workflow before large-scale training. It does not modify the trainer, strategies, CFG logic, `multi_reference_video.py`, `multi_reference_planner_stage2.py`, or the data format.

## Goals

Use a fixed 100-sample subset to verify that:

- Stage 1 can overfit target videos from `VLM text/reference context + thinking/register tokens + target-video GT SigLIP tokens + reference latents`.
- Stage 2 can predict visual tokens from `VLM planner placeholders` and align them to target-video GT SigLIP tokens with MSE.
- Generated videos preserve reference identity and match GT motion/content on the overfit set.

## Added Files

- `scripts/create_overfit_subset.py`: samples 100 rows from the original manifest or shard directory; optionally creates a symlinked `.precomputed` subset root.
- `scripts/check_multiref_precomputed_dataset.py`: checks `latents/`, `multi_reference_latents/`, `conditions/`, `vlm_conditions/`, `gt_siglip_tokens/`, and for Stage 2 also `planner_vlm_inputs/`.
- `configs/multiref_stage1_overfit100.yaml`: Stage 1 overfit config.
- `configs/multiref_stage2_overfit100.yaml`: Stage 2 planner overfit config.
- `scripts/test_multiref_overfit_generation.py`: packages GT/reference/generated/metadata for one sample and can delegate to an existing inference command.
- `scripts/compare_overfit_generation.py`: builds an HTML comparison page.

## Why the Symlinked `.precomputed` Root Matters

The current `PrecomputedDataset` scans `.pt` files under `data.preprocessed_data_root`; it does not read the manifest during training. Therefore, `overfit_100.json` alone does not restrict training to 100 samples.

Use a dedicated overfit `.precomputed` root containing symlinks for only the selected samples. This does not copy large files and does not alter the data format.

## 0. Paths

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

## 1. Select 100 Samples and Build the Stage 1 Symlink Root

```bash
python scripts/create_overfit_subset.py   --input-manifest $TRAIN_JSON   --output-manifest $OVERFIT_JSON   --num-samples 100   --seed 42   --video-column video   --link-precomputed   --precomputed-root $FULL_PRECOMP   --subset-precomputed-root $OVERFIT_PRECOMP   --precomputed-sources "latents,conditions,vlm_conditions,multi_reference_latents,gt_siglip_tokens"
```

Expected log:

```text
Selected 100 samples from N total samples
Output: /mnt/workspace/litengjie/my_dataset/overfit_100/overfit_100.json
```

## 2. Check Stage 1 Data

```bash
python scripts/check_multiref_precomputed_dataset.py   --manifest $OVERFIT_JSON   --precomputed-root $OVERFIT_PRECOMP   --planner-token-count 2048   --video-column video   --stage1-only
```

Expected success:

```text
Dataset check passed: 100/100 samples valid
```

If `precompute_gt_siglip_tokens.py` did not log `Detected 2048 GT visual tokens per sample`, replace `2048` here and in the Stage 2 config with the detected value.

## 3. Stage 1 Overfit Training

Check `configs/multiref_stage1_overfit100.yaml`:

- `model.model_path = $MODEL`
- `model.text_encoder_path = $GEMMA`
- `data.preprocessed_data_root = $OVERFIT_PRECOMP`
- `output_dir = /mnt/workspace/litengjie/ltx2_multiref_overfit_stage1_100`

Start with one GPU for easier debugging:

```bash
accelerate launch --num_processes 1 --mixed_precision bf16   scripts/train.py configs/multiref_stage1_overfit100.yaml
```

Outputs:

```text
/mnt/workspace/litengjie/ltx2_multiref_overfit_stage1_100/
  training_config.yaml
  checkpoints/
    lora_weights_step_00100.safetensors
    lora_weights_step_00200.safetensors
```

## 4. Build Stage 2 Planner VLM Inputs

```bash
python scripts/precompute_planner_vlm_inputs.py $OVERFIT_JSON   --text-encoder-path $GEMMA   --output-dir $OVERFIT_PRECOMP/planner_vlm_inputs   --video-column video   --caption-column caption   --reference-column reference_images   --max-ref-images 4   --planner-token-count 2048   --max-length 4096
```

New artifact:

```text
$OVERFIT_PRECOMP/planner_vlm_inputs/...
```

## 5. Check Stage 2 Data

```bash
python scripts/check_multiref_precomputed_dataset.py   --manifest $OVERFIT_JSON   --precomputed-root $OVERFIT_PRECOMP   --planner-token-count 2048   --video-column video   --stage2
```

## 6. Stage 2 Overfit Training

Check `configs/multiref_stage2_overfit100.yaml`:

- `model.load_checkpoint` points to the Stage 1 overfit `checkpoints` directory or a specific `.safetensors` file.
- `data.preprocessed_data_root = $OVERFIT_PRECOMP`
- `training_strategy.planner_token_count = 2048`
- `training_strategy.train_gemma_backbone = false`

Run:

```bash
accelerate launch --num_processes 1 --mixed_precision bf16   scripts/train.py configs/multiref_stage2_overfit100.yaml
```

Outputs:

```text
/mnt/workspace/litengjie/ltx2_multiref_overfit_stage2_100/
  training_config.yaml
  checkpoints/
    lora_weights_step_00100.safetensors
    lora_weights_step_00200.safetensors
```

## 7. Package One Generated Sample

If validation already produced a video:

```bash
python scripts/test_multiref_overfit_generation.py   --checkpoint /mnt/workspace/litengjie/ltx2_multiref_overfit_stage2_100/checkpoints   --config configs/multiref_stage2_overfit100.yaml   --manifest $OVERFIT_JSON   --sample-index 0   --output-dir $OVERFIT_DIR/eval_stage2   --generated-video /path/to/validation/generated_sample_0.mp4
```

If you have a standalone multi-reference inference script, delegate to it:

```bash
python scripts/test_multiref_overfit_generation.py   --checkpoint /mnt/workspace/litengjie/ltx2_multiref_overfit_stage2_100/checkpoints   --config configs/multiref_stage2_overfit100.yaml   --manifest $OVERFIT_JSON   --sample-index 0   --output-dir $OVERFIT_DIR/eval_stage2   --generation-command 'python scripts/YOUR_INFER.py --config {config} --checkpoint {checkpoint} --manifest {manifest} --sample-index {sample_index} --output {generated}'
```

Output layout:

```text
$OVERFIT_DIR/eval_stage2/sample_0/
  generated.mp4
  gt.mp4
  ref_0.jpg
  ref_1.jpg
  metadata.json
```

## 8. Build the HTML Report

```bash
python scripts/compare_overfit_generation.py   $OVERFIT_DIR/eval_stage2   --output-html $OVERFIT_DIR/eval_stage2/index.html
```

Open `index.html` and check whether reference identity is preserved, whether generated motion/scene content matches GT, and whether Stage 2 visual tokens are actually effective.

## Pass Criteria

This is a memorization/plumbing test, not a generalization test. Healthy signals: Stage 1 loss drops clearly; Stage 2 planner MSE or visual-token alignment loss drops; generated videos visibly memorize both reference identity and GT motion/content.
