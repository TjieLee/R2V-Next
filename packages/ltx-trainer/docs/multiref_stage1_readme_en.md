# Multi-Reference Image + Text Stage 1 Training

This update adds a video-only Stage 1 path for controllable LTX-2 training with 1-N reference images plus text. It keeps the LTX-2 transformer blocks, text connector, and CFG semantics intact. The new code only changes data collation, reference-token packing, RoPE positions, and the training strategy.

## What Changed

- Added `ltx_core.multicond` utilities:
  - `cfg_sampler.py`: per-sample condition-drop mode masks for later factorized CFG work.
  - `factorized_cfg.py`: optional helper for combining text/reference/planner CFG branches.
  - `rope_mask_builder.py`: packs multi-reference clean tokens before target noisy tokens and applies negative temporal RoPE offsets per reference entity.
  - `planner_tokens.py`: a zero-initialized semantic query bridge for future Gemma VLM planner stages.
- Added `MultiReferenceVideoStrategy` under `ltx_trainer.training_strategies`.
  - Loads `latents/`, `conditions/`, and `multi_reference_latents/`.
  - Prepends clean reference latent tokens to noisy target video tokens.
  - Assigns reference tokens timestep `0` and excludes them from loss.
  - Uses negative temporal RoPE slots for references: ref 1 is shifted by `-reference_time_stride`, ref 2 by `-2 * reference_time_stride`, etc.
  - Disables audio by returning `audio=None`.
- Added dynamic batch padding for variable reference counts in `collate_precomputed_batch`.
- Added `scripts/precompute_multiref_images.py` to encode 1-N reference images per sample into `multi_reference_latents/`.
- Added `configs/multiref_stage1_lora.yaml` as the ready-to-edit Stage 1 config.

## Expected Preprocessed Layout

```text
/mnt/workspace/litengjie/my_dataset/.precomputed/
├── latents/
├── conditions/
└── multi_reference_latents/
```

Each file in `multi_reference_latents/` mirrors the target video latent path and contains:

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

`R` can vary per sample. Padding happens inside the DataLoader, not in the cache.

## Metadata Format

Use a CSV/JSON/JSONL file with at least:

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

For your cleaned `stage1_dataset.py` data, export the same three concepts:

- target video path -> `video` or pass `--media-column`
- text prompt -> `caption`
- list of cropped reference paths -> `reference_images` or pass `--reference-column`

If you are using the raw Phantom JSON shown in `stage1_dataset.py`, convert it first:

```bash
python scripts/convert_phantom_manifest.py \
  /mnt/workspace/liutao/phantom_data/train_data_0202.json \
  --root-dir /mnt/workspace/liutao/phantom_data \
  --output-json /mnt/workspace/litengjie/my_dataset/train.json
```

This maps `video_path -> video`, `metadata.video_caption -> caption`, and
`cropped_ref_paths -> reference_images`.

## Precompute Data

1. Encode target videos and text as usual:

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

2. Encode 1-N reference images:

```bash
python scripts/precompute_multiref_images.py /mnt/workspace/litengjie/my_dataset/train.json \
  --model-path /path/to/ltx2.safetensors \
  --target-latents-dir /mnt/workspace/litengjie/my_dataset/.precomputed/latents \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/multi_reference_latents \
  --media-column video \
  --reference-column reference_images \
  --device cuda
```

If target latents are not available yet, provide a fixed reference image resolution instead:

```bash
python scripts/precompute_multiref_images.py train.json \
  --model-path /path/to/ltx2.safetensors \
  --output-dir /mnt/workspace/litengjie/my_dataset/.precomputed/multi_reference_latents \
  --ref-resolution "640x384"
```

## Train Stage 1

Edit these paths in `configs/multiref_stage1_lora.yaml`:

- `model.model_path`: local LTX-2 checkpoint.
- `model.text_encoder_path`: local Gemma text encoder directory.
- `data.preprocessed_data_root`: directory containing `latents/`, `conditions/`, and `multi_reference_latents/`.
- `output_dir`: write under `/mnt/workspace/litengjie/...`.

Run:

```bash
cd /Users/litengjie.3/CodeSpace/JD-LTX/JD-LTX/packages/ltx-trainer
accelerate launch scripts/train.py configs/multiref_stage1_lora.yaml
```

## Important Path Rules

- Write logs, caches, checkpoints, and generated samples under `/mnt/workspace/litengjie`.
- It is safe to read source data from `/mnt/workspace/liutao`, but do not write there.
- `preprocessed_data_root` must point to the parent of `latents/`, `conditions/`, and `multi_reference_latents/`.
- `reference_latents_dir` is relative to `preprocessed_data_root`; default is `multi_reference_latents`.

## Current Scope

This implements the Stage 1 renderer adaptation path. The Gemma VLM planner modules are scaffolded through `SemanticQueryBridge`, `cfg_sampler`, and `factorized_cfg`, but Stage 2/3 planner training and inference pipelines are intentionally not wired into the main trainer yet.
