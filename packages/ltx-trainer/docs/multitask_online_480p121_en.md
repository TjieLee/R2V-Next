# I2I + R2V Online Training (832x480 / 121 Frames)

This path removes the requirement to precompute every VAE latent, GT SigLIP token, and VLM condition.
Legacy `PrecomputedDataset`, 81-frame YAML files, checkpoints, and inference commands are unchanged. The new
path is selected only by `data.encoding_mode: online`.

## Fixed contract

- I2I uses one real target image at 1 fps; source images are references.
- R2V uses exactly 121 target frames at 24 fps. Short clips are rejected.
- Video VLM/SigLIP indices are `[0,17,34,51,69,86,103,120]`.
- I2I runs SigLIP on one frame only: 256 valid tokens, zero-padded to 2048 without repetition.
- Planner placeholders always have capacity 2048; the I2I `planner_output_mask` enables only the first 256.
- All four accumulation microsteps on all eight ranks use one task per optimizer step.
- A 30K run has exactly 9K I2I and 21K R2V optimizer steps. With global batch 32, that is 960K exposures per
  stage (288K I2I and 672K R2V), or 2.88M across three independent 30K stages.

## Prepare the deterministic manifest

Run `scripts/inspect_multitask_sources.py` first. I2I column names are deliberately not guessed. Use the
reported target, source/reference, and instruction/caption names when invoking the builder.

```bash
cd /mnt/workspace/litengjie/LTX-2/packages/ltx-trainer
mkdir -p /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/manifests

uv run python scripts/inspect_multitask_sources.py \
  --i2i-ann /mnt/workspace/liutao/data_process_one2x/r2i_train_data.jsonl \
  --r2v-ann /mnt/workspace/liutao/data_process_one2x/opens2v_train_data.parquet \
  --output /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/manifests/source_schema_report.json

cp configs/multitask_online_480p121_data.yaml \
  /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/manifests/multitask_480p121.yaml

uv run python scripts/build_multitask_online_manifest.py \
  --train-data-config /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/manifests/multitask_480p121.yaml \
  --output /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/manifests/train_unique.jsonl \
  --reject-output /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/logs/manifest_rejected.jsonl \
  --manifest-seed 42 \
  --i2i-target-field '<target column from report>' \
  --i2i-reference-field '<source column from report>' \
  --i2i-caption-field '<instruction column from report>'

uv run python scripts/validate_multitask_online_manifest.py \
  /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/manifests/train_unique.jsonl
```

`/mnt/workspace/liutao/` is read-only. Every generated manifest, reject log, runtime log, cache, and training
output is guarded by `assert_write_path_allowed()` and must stay under `/mnt/workspace/litengjie/`.

## Train Stage 1, Stage 2, and Stage 3

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 8 \
  scripts/train.py configs/multiref_stage1_multitask_online_480p121_full_tokens_30k.yaml \
  --disable-progress-bars

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 8 \
  scripts/train.py configs/multiref_stage2_multitask_online_480p121_full_tokens_planner_30k.yaml \
  --disable-progress-bars

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 8 \
  scripts/train.py configs/multiref_stage3_multitask_online_480p121_joint_30k.yaml \
  --disable-progress-bars
```

Stage 2 and Stage 3 point to the preceding stage's step-30000 checkpoint. Verify those paths before launching.
The sampler state is included in training state, so exact resume continues the same task and sample sequence.
A load failure on any rank causes all ranks to discard the microbatch and deterministically retry another sample
from the same task.

The GPU encoder emits the existing strategy interface: `latents`, `multi_ref_latents`, `conditions`,
`vlm_conditions`, `text_conditions`, `cfg_text_conditions`, and `gt_visual_tokens`; Stage 2/3 also receive
`planner_vlm_inputs`. Gemma/Planner remains in the Stage 2/3 gradient graph, while VAE, SigLIP vision, and the
multimodal projector remain frozen and in eval mode.
