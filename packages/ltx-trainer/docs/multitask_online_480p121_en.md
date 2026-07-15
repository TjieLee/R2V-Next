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
  --summary-output /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/logs/manifest_summary.json \
  --manifest-seed 42 \
  --annotation-batch-size 4096 \
  --probe-workers 8 \
  --probe-batch-size 256 \
  --i2i-target-field '<target column from report>' \
  --i2i-reference-field '<source column from report>' \
  --i2i-caption-field '<instruction column from report>'

uv run python scripts/validate_multitask_online_manifest.py \
  /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/manifests/train_unique.jsonl
```

`/mnt/workspace/liutao/` is read-only. Every generated manifest, reject log, runtime log, cache, and training
output is guarded by `assert_write_path_allowed()` and must stay under `/mnt/workspace/litengjie/`.

The builder streams JSONL/CSV rows and PyArrow Parquet batches. JSON lists remain compatible but emit an
in-memory warning. Accepted and rejected records are written line by line, flushed, fsynced, and atomically
renamed. Deduplication uses a temporary SQLite database under the allowed output root instead of a million-row
Python dictionary. A successful build produces:

```text
train_unique.jsonl
train_unique.jsonl.idx
manifest_rejected.jsonl
manifest_summary.json
```

The compact `.idx` stores only byte offsets and task ids. `OnlineMultiTaskDataset` seeks to one JSON row on
demand, so every rank and DataLoader worker does not retain a full manifest copy. The summary records row/task/
reject counts, duplicates, elapsed time, and peak builder RSS. Static rejects use stable reasons:
`missing_target`, `missing_reference`, `empty_caption`, `invalid_crop`, `invalid_face_cut`,
`invalid_video_header`, and `insufficient_frames_for_121_at_24fps`.

## Reference and video decode semantics

- `reference_pixels_vae` is deterministically resized/cropped to 832x480 after a shared reference-count trim.
- `reference_images_vlm` preserves original sRGB pixels, dimensions, aspect ratio, and ordering for Gemma.
- `vlm_reference_preprocess: target_crop` is an explicit ablation; production YAML defaults to `original`.
- R2V defaults to `video_decoder: pyav`: it seeks before the requested region and decodes presentation order;
  unreliable PTS falls back to exact sequential ordinal indexing in the same container.
- `decode_timeout_seconds: 120` becomes `SampleLoadError`, then all DDP ranks enter synchronized same-task retry.
- Only `encoder_device_policy: resident_cuda` is currently public. Incomplete policies fail during config parsing
  rather than moving or copying a DDP-wrapped trainable Gemma module.

Retry candidates exclude the complete normal block reserved for the current optimizer step and all successful
retry samples in that step. A constrained small-dataset fallback warns explicitly. If it consumes a prefetched
future normal slot, the trainer detects that collision before encode/forward and deterministically retries again
without advancing the task cursor.

## Preflight and real smoke gates

Run the CPU preflight for all three YAML files first:

```bash
for stage in \
  configs/multiref_stage1_multitask_online_480p121_full_tokens_30k.yaml \
  configs/multiref_stage2_multitask_online_480p121_full_tokens_planner_30k.yaml \
  configs/multiref_stage3_multitask_online_480p121_joint_30k.yaml; do
  uv run python scripts/check_multitask_online_training_ready.py "$stage" \
    --world-size 8 --samples-per-task 1
done
```

The matching `READY_FOR_STAGE1/2/3` field must be true. Then run real-model encode and a one-step single-GPU
training smoke:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/check_multitask_online_real_encode.py \
  --stage1-config configs/multiref_stage1_multitask_online_480p121_full_tokens_30k.yaml \
  --num-image-samples 1 --num-video-samples 1

CUDA_VISIBLE_DEVICES=0 uv run python scripts/check_multitask_online_stage1.py \
  --config configs/multiref_stage1_multitask_online_480p121_full_tokens_30k.yaml \
  --task i2i \
  --output-dir /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/smoke/stage1_i2i
```

The Stage 1/2/3 wrappers share this interface. Run both `i2i` and `r2v` with each matching YAML. Their derived
configs use one optimizer step, local batch 1, accumulation 4, and full-condition CFG disabled. They verify one
scheduler step, finite Stage 2/3 flow/MSE/NTP losses, nonzero finite Adam moments for required trainable modules,
and paired checkpoint/training-state output. JSON reports include decode, VAE, SigLIP, frozen condition, planner,
DiT, backward, optimizer, peak VRAM, and reference shapes.

Finish with the real two-rank DDP smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --config_file configs/accelerate/ddp.yaml --num_processes 2 \
  scripts/check_multitask_online_ddp.py \
  --stage1-config configs/multiref_stage1_multitask_online_480p121_full_tokens_30k.yaml \
  --stage2-config configs/multiref_stage2_multitask_online_480p121_full_tokens_planner_30k.yaml \
  --stage3-config configs/multiref_stage3_multitask_online_480p121_joint_30k.yaml
```

It covers one I2I and one R2V optimizer step for all three stages using real DDP condition encoding,
Gemma/Planner/DiT, checkpoint saving, and global-step scheduler semantics. Launch 30K only after these gates pass.

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

## Stage 3-only warm-start from the previous joint model

Online encoding now selects a system prompt by task. I2I uses
`gemma_multiref_image_edit_planner_system_prompt.txt` and describes one target image. R2V keeps the existing
`gemma_multiref_video_planner_system_prompt.txt` and its chat serialization unchanged. `conditions`,
`text_conditions`, and `planner_vlm_inputs` share the same task prompt; batch metadata uses
`task_system_prompt_id=0/1` for image/video.

To skip new-data Stage 1/2 and initialize directly from the complete previous Stage 3 checkpoint, run:

```bash
uv run python scripts/check_multitask_online_training_ready.py \
  configs/multiref_stage3_multitask_online_480p121_joint_warmstart_old_stage3_30k.yaml \
  --world-size 8 --samples-per-task 2
```

The report must contain `initialization_mode=stage3_joint_warmstart`,
`strict_component_check_passed=true`, and `starts_from_global_step=0`. The YAML uses `no_resume=true`, so it
loads the previous Stage 3 model weights but not its optimizer, scheduler, or global step.

Run the real two-rank Stage 3-only DDP smoke with:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 2 \
  scripts/check_multitask_online_ddp.py \
  --stages stage3 \
  --stage3-config configs/multiref_stage3_multitask_online_480p121_joint_warmstart_old_stage3_30k.yaml \
  --stage3-init-checkpoint /mnt/workspace/litengjie/ltx2_multiref_stage3_joint_full_tokens_planner_2048/checkpoints/lora_weights_step_02000.safetensors
```

For a three-stage dependency smoke, pass `--stages stage1,stage2,stage3 --chain-smoke-checkpoints`; each smoke
checkpoint is injected into the next stage. Before the formal 30K run, use
`configs/multiref_stage3_multitask_online_480p121_joint_warmstart_benchmark_200.yaml` for the eight-GPU,
200-step gate. It uses interval 100 and a separate benchmark output directory.

Manifest indexes now use `LTXIDX02`; the header stores manifest size, nonblank row count, SHA256, and entry size.
Legacy v1, hash/size mismatches, and truncation fail fast with a rebuild instruction, while the validator also
checks every offset/task pair. During one build, the temporary SQLite database caches image validation and video
probing by path/size/mtime, including failures, without writing into the source-data directory.
