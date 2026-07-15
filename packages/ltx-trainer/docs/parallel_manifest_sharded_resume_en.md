# Parallel, Sharded, Resumable Multi-Task Manifests

This workflow changes manifest construction only. Stage 1/2/3 training, checkpoints, CFG, and inference remain
unchanged. The legacy `scripts/build_multitask_online_manifest.py` entry point keeps its current behavior. The
new workflow writes an independent shard tree, so separate I2I and R2V processes can run concurrently.

## Path contract

- `/mnt/workspace/liutao/` and `/mnt/workspace/jiangyuxiang2/` are read-only.
- Shards, JSONL offset indexes, SQLite files, progress files, rejects, and final manifests must be written under
  `/mnt/workspace/litengjie/`.
- The new builder neither consumes nor removes legacy `.tmp.<pid>` files. They are not valid shards.
- Do not point a legacy builder and the new merge command at the same final `train_unique.jsonl`.

```bash
cd /mnt/workspace/litengjie/LTX-2/packages/ltx-trainer

ROOT=/mnt/workspace/litengjie/jd_ltx_multitask_online_480p121
CONFIG=${ROOT}/manifests/multitask_480p121.yaml
SHARDS=${ROOT}/manifests/build_shards

mkdir -p "${ROOT}/manifests" "${ROOT}/logs"
```

## Two-stage R2V filtering

R2V starts with media-free Stage A validation: required fields, non-empty text, finite and valid crop,
valid `face_cut`, at least one parseable reference, and immediate rejection when the face-cut span is below
121. Only Stage A survivors enter Stage B: video header probing, source FPS/frame/crop checks, construction of
an exact 121-frame plan, and finally reference-image validation. A failed video plan never stats or decodes a
reference image.

Run Stage A alone to estimate the number of rows that really require video probing:

```bash
uv run python scripts/build_multitask_manifest_shards.py \
  --train-data-config "${CONFIG}" \
  --tasks r2v \
  --shard-root "${SHARDS}" \
  --prefilter-only
```

This writes `r2v_prefilter.accepted.jsonl`, `r2v_prefilter.rejected.jsonl`, and
`r2v_prefilter_summary.json` without opening or statting target videos or reference images.

## Concurrent builds

Terminal 1 builds I2I. Target and reference images run bounded parallel `stat`, Pillow verification, and
EXIF-safe dimension reads. Reference order is preserved.

```bash
nohup uv run python scripts/build_multitask_manifest_shards.py \
  --train-data-config "${CONFIG}" \
  --tasks i2i \
  --shard-root "${SHARDS}" \
  --shards-per-task 16 \
  --image-workers 64 \
  --video-workers 1 \
  --max-in-flight 512 \
  --annotation-batch-size 4096 \
  --resume-build \
  --progress-interval-seconds 10 \
  --manifest-seed 42 \
  --i2i-target-field image \
  --i2i-reference-field edit_image \
  --i2i-caption-field prompt \
  > "${ROOT}/logs/build_i2i.log" 2>&1 &

echo $! > "${ROOT}/logs/build_i2i.pid"
```

Terminal 2 builds R2V concurrently. The default is a persistent PyAV probe pool: each process handles many
videos and is recycled after `max_tasks_per_worker`. A task exceeding 60 seconds terminates and replaces only
its assigned worker; other workers continue.

```bash
nohup uv run python scripts/build_multitask_manifest_shards.py \
  --train-data-config "${CONFIG}" \
  --tasks r2v \
  --shard-root "${SHARDS}" \
  --shards-per-task 16 \
  --image-workers 1 \
  --video-workers 32 \
  --video-probe-mode persistent \
  --video-probe-max-tasks-per-worker 1000 \
  --max-in-flight 256 \
  --annotation-batch-size 4096 \
  --probe-timeout-seconds 60 \
  --resume-build \
  --progress-interval-seconds 10 \
  --manifest-seed 42 \
  > "${ROOT}/logs/build_r2v.log" 2>&1 &

echo $! > "${ROOT}/logs/build_r2v.pid"
```

`--video-probe-mode isolated` retains the legacy one-child-per-video implementation for regression testing and
pathological-file diagnosis. Use the default `persistent` mode for production. The task SQLite cache and
in-flight Future deduplication sit in front of the pool, so concurrent occurrences of one path submit one probe;
successful and failed probes are reusable after resume.

Conservative defaults are `image_workers=16`, `video_workers=8`, `shards_per_task=8`, and
`max_in_flight=256`. The 64/32 values above are a first server trial for roughly 192 CPU cores. Do not start at
192/192; observe rows/s, iowait, and storage latency for five to ten minutes first.

## Artifacts and progress

```text
build_shards/
├── build_metadata.json
├── indexes/<fingerprint>.jsonl_offsets.idx
├── build_i2i.progress.json
├── build_r2v.progress.json
├── i2i/
│   ├── media_cache.sqlite
│   ├── shard_00000.accepted.jsonl
│   ├── shard_00000.rejected.jsonl
│   ├── shard_00000.summary.json
│   └── shard_00000.done.json
└── r2v/
    └── ...
```

```bash
watch -n 10 "cat ${SHARDS}/build_i2i.progress.json"
watch -n 10 "cat ${SHARDS}/build_r2v.progress.json"

tail -f "${ROOT}/logs/build_i2i.log"
tail -f "${ROOT}/logs/build_r2v.log"
```

Progress includes processed/total, accepted/rejected, current and average rate, ETA, in-flight rows, and stage.
R2V separately reports image/video cache hit/miss counts, annotation prefilter rejects, rows sent to probing,
probe submission/success/timeout/invalid counts, worker starts/restarts, and started/skipped reference validation.

JSONL sources receive one byte-offset index, after which every shard seeks directly to its contiguous row range.
Parquet uses metadata row counts and row-group/batch range reads. Workers may finish out of order, but records are
emitted in source-row order and the combined future/reorder buffer is bounded by `max_in_flight`.

## Resume and locks

Each shard publishes `.done.json` last. `--resume-build` reuses a shard only when all of the following hold:

1. task, shard id, and source range match;
2. source path/size/mtime fingerprints match;
3. semantic build settings, including seed, field mapping, timeout, shard count, and R2V filter version, match;
4. accepted, rejected, and summary artifacts exist and their SHA256 values match.

Any failure cleans and rebuilds that shard only. Other completed shards and legacy builder temporary files are
left untouched.

`shard_00000.lock` prevents two processes from owning one shard. Active locks fail fast. Stale locks are not
taken over automatically; after confirming no builder is active, recovery must be explicit:

```bash
uv run python scripts/build_multitask_manifest_shards.py \
  ... \
  --resume-build \
  --recover-stale-locks
```

Each task has a persistent `media_cache.sqlite`, keyed by kind/path/size/mtime. Resume reuses valid entries, a
changed file naturally misses, and a corrupt cache is rebuilt without invalidating completed shards.

`video_probe_mode`, `video_workers`, `max_in_flight`, and `video_probe_max_tasks_per_worker` are performance
settings and do not enter the semantic fingerprint. This R2V filter reorder has its own semantic version, so old
R2V shards are revalidated while I2I fingerprints remain unchanged. Resume cleans and rebuilds an unfinished old
R2V shard without a `.done.json`; keep completed I2I shards.

## Deterministic merge

After all I2I and R2V shards are complete:

```bash
uv run python scripts/merge_multitask_manifest_shards.py \
  --shard-root "${SHARDS}" \
  --tasks i2i,r2v \
  --output "${ROOT}/manifests/train_unique.jsonl" \
  --reject-output "${ROOT}/logs/manifest_rejected.jsonl" \
  --summary-output "${ROOT}/manifests/train_unique_summary.json" \
  --require-all-shards
```

The merge order is fixed as I2I, R2V, dataset config order, then source row index. Internal `_build_*` fields are
removed. A temporary SQLite database enforces global `sample_key` uniqueness: identical key/plan rows are
skipped, while the same key with a different plan fails immediately. The merger rebuilds and strictly validates
the `LTXIDX02` SHA256, offsets, and task ids on temporary files before atomically publishing:

```text
train_unique.jsonl
train_unique.jsonl.idx
manifest_rejected.jsonl
train_unique_summary.json
```

Run the existing preflight last:

```bash
uv run python scripts/validate_multitask_online_manifest.py \
  "${ROOT}/manifests/train_unique.jsonl"
```

## Small-scale trial

Before a full build, cap source rows to verify fields, throughput, reject reasons, and cache behavior:

```bash
uv run python scripts/build_multitask_manifest_shards.py \
  --train-data-config "${CONFIG}" \
  --tasks i2i \
  --shard-root "${ROOT}/manifests/build_shards_smoke" \
  --shards-per-task 4 \
  --image-workers 16 \
  --max-in-flight 128 \
  --max-samples-per-task 5000 \
  --no-resume-build
```

Benchmark on the target server and storage: first 50,000 rows with `--prefilter-only`, then 5,000 rows with full
media validation. Try R2V 16/32/48 workers and max-tasks 500/1000. Record rows/s, elapsed time, peak RSS, actual
probe submissions, worker starts/restarts, saved reference validations, and iowait. Do not scan all production
data merely for a benchmark.

This optimization does not restore the old reader's variable 41-to-120-frame behavior. Every accepted R2V row
still requires exactly 121 strictly increasing, unique 24 FPS source indices inside
`[face_cut_start, face_cut_end)`; source FPS below 24 is rejected explicitly.
