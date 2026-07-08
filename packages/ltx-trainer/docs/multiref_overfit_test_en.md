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

## CFG and Validation Are Disabled by Default

The 100-sample overfit configs are meant to verify whether the full-condition path can fit first. Therefore both overfit configs default to `cfg_dropout_enabled: false`, `cfg_full_p: 1.0`, and all drop probabilities set to `0.0`. This keeps text/ref/all dropout from obscuring whether the Stage 1 and Stage 2 full-condition pipeline is wired correctly.

Automatic validation is also disabled by default: `validation.interval: null` and `validation.skip_initial_validation: true`. `generate_video: true` is preserved, but it will not trigger empty validation. Generate videos after the training check by packaging an existing validation output or delegating to a standalone inference command.



## Conditioning Dimensions and Checkpoint Compatibility

The LTX text conditioning path has two stages: `feature_extractor` first turns Gemma/VLM hidden states into connector-input features, then `video_connector` / text connector produces the final DiT cross-attention condition features. The name `vlm_conditions/video_prompt_embeds` is therefore slightly misleading in this pipeline: it stores feature-extractor outputs before the connector, usually with dimension 4096, not final DiT conditions.

Stage 1 keeps visual-token concatenation before the video connector. Raw target-video GT SigLIP/Gemma-projector tokens are 3840-dimensional; `training_strategy.visual_token_projection` maps them to the connector-input 4096-dimensional space before concatenating them with feature-extracted text/reference features. The combined sequence then enters `embeddings_processor.video_connector`, where connector padding slots are handled as learnable thinking/register tokens. With `train_text_connector: true`, Stage 1 trains and checkpoints `embeddings_processor.video_connector.*`; it does not train the feature extractor.

Stage 2 planner/Q-former now predicts raw SigLIP-space tokens. Gemma planner placeholder hidden states `[B,K,3840]` feed newly initialized 3840-dimensional learned planner query tokens, producing `[B,K,3840]`; the MSE is computed only against raw GT SigLIP tokens `[B,K,3840]`. The predicted tokens are projected through the Stage 1 `visual_token_projection` (`3840 -> 4096`) only before appending them to the Stage 1 condition sequence. The default path does not use 4096-dimensional connector registers as planner queries; `use_connector_register_queries: true` is legacy-only.

Checkpoint compatibility: old Stage 1 checkpoints without `embeddings_processor.*` still load and use the base LTX connector weights. New Stage 1 checkpoints with `train_text_connector: true` include `embeddings_processor.video_connector.*`. When resuming or inferencing from old checkpoints, LoRA `rank`, `alpha`, and `target_modules` must match the original training config; shape mismatches raise an error instead of being silently ignored.

Inspect checkpoint contents:

```bash
python - <<'PY'
from safetensors.torch import load_file

p = "/path/to/lora_weights_step_XXXXX.safetensors"
sd = load_file(p)

for prefix in [
    "diffusion_model.",
    "training_strategy.",
    "embeddings_processor.",
    "text_encoder.",
]:
    keys = [k for k in sd if k.startswith(prefix)]
    print(prefix, len(keys))
    for k in keys[:10]:
        print(" ", k, tuple(sd[k].shape), sd[k].dtype)
PY
```

An old Stage 1 checkpoint usually has `diffusion_model.* > 0`, `training_strategy.*` containing at least the visual projection, `embeddings_processor.* = 0`, and `text_encoder.* = 0`. A new Stage 1 checkpoint with `train_text_connector: true` should have `embeddings_processor.* > 0`, especially `embeddings_processor.video_connector.*`.

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

Start with one GPU for easier loss debugging. Validation is not triggered by default, so the smoke test is not blocked by an empty validation setup or missing inference entrypoint:

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

`test_multiref_overfit_generation.py` is only a packaging/delegation script, not a complete multi-reference inference pipeline. It packages GT, references, and metadata, and can either copy an existing validation output or delegate to a standalone inference command you provide.

If you do not have a standalone inference command or validation output, this workflow can complete the training check only; it cannot automatically create `generated.mp4`.

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

## 7A. Stage 1 Teacher-Forcing In-Domain Inference

`scripts/infer_multiref_stage1_overfit.py` is a real Stage 1 multi-reference teacher-forcing inference entrypoint for checking a Stage 1 checkpoint. It reads `vlm_conditions/`, `multi_reference_latents/`, `gt_siglip_tokens/`, and target `latents/`, loads the Stage 1 checkpoint, and generates `generated.mp4` with target-video GT SigLIP tokens as visual conditions.

It is still not the final Stage 2 planner inference path: it does not run Gemma/VLM online, does not use planner placeholders, and does not predict visual tokens. It uses precomputed target-video GT SigLIP tokens, so use it to verify that Stage 1 learned to consume the full condition path, not as the deployable final inference endpoint.

Example:

```bash
python scripts/infer_multiref_stage1_overfit.py \
  --config configs/multiref_stage1_overfit100.yaml \
  --checkpoint /mnt/workspace/litengjie/ltx2_multiref_overfit_stage1_100/checkpoints/lora_weights_step_00500.safetensors \
  --manifest $OVERFIT_JSON \
  --precomputed-root $OVERFIT_PRECOMP \
  --sample-index 0 \
  --output-dir $OVERFIT_DIR/eval_stage1_teacher \
  --device cuda:0 \
  --num-inference-steps 50 \
  --video-column video \
  --caption-column caption \
  --reference-column reference_images
```

Output layout:

```text
$OVERFIT_DIR/eval_stage1_teacher/sample_0/
  generated.mp4
  gt.mp4
  ref_0.jpg
  ref_1.jpg
  metadata.json
```

`metadata.json` records the original VLM condition shape, raw GT SigLIP token shape, the pre-connector shape after appending GT tokens, and the final condition shape sent to the DiT after the LTX text connector/register tokens. CFG, negative prompts, ValidationRunner, and online VLM are not used.

## 8. Build the HTML Report

```bash
python scripts/compare_overfit_generation.py   $OVERFIT_DIR/eval_stage2   --output-html $OVERFIT_DIR/eval_stage2/index.html
```

Open `index.html` and check whether reference identity is preserved, whether generated motion/scene content matches GT, and whether Stage 2 visual tokens are actually effective.

## Pass Criteria

This is a memorization/plumbing test, not a generalization test. Healthy signals: Stage 1 loss drops clearly; Stage 2 planner MSE or visual-token alignment loss drops; generated videos visibly memorize both reference identity and GT motion/content.
