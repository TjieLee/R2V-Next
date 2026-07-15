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

Finish with real two-rank DDP smoke runs. One launch may contain only one stage, one task, and one Trainer. This
example covers Stage 1 I2I; start a fresh launch for every other stage/task pair:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --config_file configs/accelerate/ddp.yaml --num_processes 2 \
  scripts/check_multitask_online_ddp.py \
  --config configs/multiref_stage1_multitask_online_480p121_full_tokens_30k.yaml \
  --task i2i \
  --output-dir /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/ddp_smoke/stage1_i2i
```

That launch executes one optimizer step and checks real DDP condition encoding, the selected model path,
checkpoint saving, and global-step scheduler semantics. Do not create multiple Trainers inside one process group.
Launch formal training only after every required independent gate passes.

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
  --world-size 7 --samples-per-task 2
```

The report must contain `initialization_mode=stage3_joint_warmstart`,
`strict_component_check_passed=true`, and `starts_from_global_step=0`. The YAML uses `no_resume=true`, so it
loads the previous Stage 3 model weights but not its optimizer, scheduler, or global step.

The report also reads `train_unique_summary.json` and prints task row counts, planned exposures, coverage ratios,
and estimated repeats. Seven ranks give `1 x 4 x 7 = 28` effective global samples. The 30K config therefore
produces 840,000 exposures. A separately chosen 34,286-step run would approximate the previous eight-rank
960,000 exposures; preflight reports these semantics without changing the configured schedule.

Run I2I and R2V as two independent two-rank Stage 3 launches:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 2 \
  scripts/check_multitask_online_ddp.py \
  --config configs/multiref_stage3_multitask_online_480p121_joint_warmstart_old_stage3_30k.yaml \
  --task i2i \
  --init-checkpoint /mnt/workspace/litengjie/ltx2_multiref_stage3_joint_full_tokens_planner_2048/checkpoints/lora_weights_step_02000.safetensors \
  --output-dir /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/ddp_smoke/stage3_i2i

# Use a second launch with r2v/stage3_r2v for --task/--output-dir.
```

`scripts/run_multitask_online_ddp_smoke_matrix.py` can create both launches with separate `subprocess.run()`
calls and fresh Accelerate process groups. Before formal training, use
`configs/multiref_stage3_multitask_online_480p121_joint_warmstart_benchmark_200_7gpu.yaml` for the seven-GPU,
200-step gate. It keeps `lr=1e-5`, temporarily uses one worker and prefetch 1, and executes 60 I2I steps
(1,680 samples) plus 140 R2V steps (3,920 samples), totaling 5,600. `cpu_transform_chunk_frames=4` prevents the 121-frame spatial transform
from materializing a full-video float32 tensor.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 PYTORCH_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
uv run accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 7 \
  scripts/train.py \
  configs/multiref_stage3_multitask_online_480p121_joint_warmstart_benchmark_200_7gpu.yaml \
  --disable-progress-bars
```

Stage 3 checkpoints are validated in a same-directory temporary file. A
`checkpoint_step_XXXXX.ready.json` marker is published only after both weights and matching training state are
atomically visible. The GPU 7 watcher consumes only markers and hard-links each checkpoint before inference, so
`keep_last_n` cleanup cannot race an open checkpoint. `--command-template` is required:

```bash
CUDA_VISIBLE_DEVICES=7 uv run python scripts/watch_stage3_checkpoints_and_infer.py \
  --checkpoint-dir /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/stage3_warmstart_old_stage3/checkpoints \
  --output-root /mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/checkpoint_inference \
  --gpu 7 --poll-seconds 30 --min-step 2500 --step-stride 2500 \
  --command-template 'uv run python <inference_script> --checkpoint {checkpoint} --output-dir {output_dir}'
```

Use separate fixed validation manifests and commands for R2V and I2I. I2I must emit a single image rather than a
copied pseudo-video. Each step's success or failure is persisted in `watcher_state.json`; failures never propagate
to training.

After the 200-step gate passes, run the formal seven-GPU 30K job by switching the training config back to
`multiref_stage3_multitask_online_480p121_joint_warmstart_old_stage3_30k.yaml` while retaining
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6` and `--num_processes 7`. Do not include GPU 7 in training.

Manifest indexes now use `LTXIDX02`; the header stores manifest size, nonblank row count, SHA256, and entry size.
Legacy v1, hash/size mismatches, and truncation fail fast with a rebuild instruction, while the validator also
checks every offset/task pair. During one build, the temporary SQLite database caches image validation and video
probing by path/size/mtime, including failures, without writing into the source-data directory.
