# 多任务 Manifest 并行分片与断点续建

该流程只改造 manifest 构建，不改变 Stage 1/2/3 训练、checkpoint、CFG 或推理逻辑。旧入口
`scripts/build_multitask_online_manifest.py` 保持原行为；新流程使用独立 shard 目录，因此 I2I 和 R2V
可以由两个进程同时构建。

## 路径约束

- `/mnt/workspace/liutao/` 和 `/mnt/workspace/jiangyuxiang2/` 只读。
- shard、JSONL offset index、SQLite、progress、reject 和最终 manifest 必须写到
  `/mnt/workspace/litengjie/`。
- 新 builder 不读取或删除旧 builder 的 `.tmp.<pid>`。旧 `.tmp` 不是新 shard，不能参与 merge。
- 不要让旧 builder 和新 merge 写同一个最终 `train_unique.jsonl`。

```bash
cd /mnt/workspace/litengjie/LTX-2/packages/ltx-trainer

ROOT=/mnt/workspace/litengjie/jd_ltx_multitask_online_480p121
CONFIG=${ROOT}/manifests/multitask_480p121.yaml
SHARDS=${ROOT}/manifests/build_shards

mkdir -p "${ROOT}/manifests" "${ROOT}/logs"
```

## 并行构建

终端 1 构建 I2I。每行的 target/reference 图片会并行执行 `stat`、Pillow verify 和 EXIF-safe
尺寸读取；reference 顺序不会改变。

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

终端 2 同时构建 R2V。每个 PyAV header probe 在独立子进程中运行；超过 60 秒会终止该子进程，
记录 `video_probe_timeout` 后继续，不会卡死整个 worker pool。

```bash
nohup uv run python scripts/build_multitask_manifest_shards.py \
  --train-data-config "${CONFIG}" \
  --tasks r2v \
  --shard-root "${SHARDS}" \
  --shards-per-task 16 \
  --image-workers 1 \
  --video-workers 32 \
  --max-in-flight 256 \
  --annotation-batch-size 4096 \
  --probe-timeout-seconds 60 \
  --resume-build \
  --progress-interval-seconds 10 \
  --manifest-seed 42 \
  > "${ROOT}/logs/build_r2v.log" 2>&1 &

echo $! > "${ROOT}/logs/build_r2v.pid"
```

保守默认值是 `image_workers=16`、`video_workers=8`、`shards_per_task=8`、
`max_in_flight=256`。上面的 64/32 是 192 核服务器的首轮建议，不要直接设为 192/192；先观察
5 到 10 分钟 rows/s、iowait 和存储延迟。

## 中间产物和进度

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

progress JSON 包含 processed/total、accepted/rejected、当前和平均吞吐、ETA、in-flight、缓存命中，
R2V 还包含 probe success/timeout/invalid header。

JSONL source 首次构建一次 byte-offset index，之后每个 shard 直接 seek 自己的连续行区间；Parquet
使用 metadata row count 和 row-group/batch 范围读取。worker 可以乱序完成，但 accepted/rejected 始终按
source row index 输出，等待重排的数据量受 `max_in_flight` 限制。

## Resume 和锁

每个 shard 最后才原子发布 `.done.json`。`--resume-build` 只复用同时满足以下条件的 shard：

1. done marker 的 task、shard id、source range 正确；
2. source path/size/mtime fingerprint 未变化；
3. manifest seed、字段映射、超时和分片配置的 build fingerprint 未变化；
4. accepted/rejected/summary 文件存在且 SHA256 匹配。

任一校验失败只清理并重跑该 shard，不删除其他完成 shard，也不碰旧 builder 临时文件。

`shard_00000.lock` 防止两个进程处理同一 shard。活跃锁会 fail-fast。死进程留下的 stale lock 默认也不
自动接管，确认没有活跃 builder 后才能显式使用：

```bash
uv run python scripts/build_multitask_manifest_shards.py \
  ... \
  --resume-build \
  --recover-stale-locks
```

每个 task 的 `media_cache.sqlite` 按 kind/path/size/mtime 缓存验证结果，resume 可复用；文件变化会自然
miss。缓存损坏时会单独重建，不会使已完成 shard 失效。

## 确定性合并

I2I 和 R2V 的所有 shard 完成后运行：

```bash
uv run python scripts/merge_multitask_manifest_shards.py \
  --shard-root "${SHARDS}" \
  --tasks i2i,r2v \
  --output "${ROOT}/manifests/train_unique.jsonl" \
  --reject-output "${ROOT}/logs/manifest_rejected.jsonl" \
  --summary-output "${ROOT}/manifests/train_unique_summary.json" \
  --require-all-shards
```

merge 按 I2I、R2V、dataset 配置顺序、source row index 的固定顺序流式读取，删除 `_build_*` 内部字段，
使用临时 SQLite 做全局 `sample_key` 去重：相同 key/plan 跳过，相同 key 但不同 plan 立即失败。它会在
临时 manifest 上重建并严格验证 `LTXIDX02` 的 SHA256、offset 和 task id，随后原子发布：

```text
train_unique.jsonl
train_unique.jsonl.idx
manifest_rejected.jsonl
train_unique_summary.json
```

最后执行现有 preflight：

```bash
uv run python scripts/validate_multitask_online_manifest.py \
  "${ROOT}/manifests/train_unique.jsonl"
```

## 小规模试跑

正式全量前先限制每个 task 的 source 行数，检查字段、吞吐、reject reason 和缓存：

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

性能基准必须在目标服务器和真实存储上完成。建议矩阵为 I2I 16/32/64 workers、R2V 8/16/32
workers；记录 rows/s、elapsed、peak RSS、cache hit rate 和 iowait。不要为了测试扫描全部正式数据。
