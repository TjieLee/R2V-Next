"""Benchmark production-shaped multitask manifest builder subsets.

The benchmark is intentionally a thin subprocess harness around
``build_multitask_online_manifest.py`` so each worker-count run measures the
same CLI entrypoint used in production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ALLOWED_OUTPUT_ROOT = Path("/mnt/workspace/litengjie")
DEFAULT_OUTPUT_ROOT = ALLOWED_OUTPUT_ROOT / "manifest_benchmark_v7_1"
DEFAULT_BASELINE_REF = "2a6779e61c3ecb791f84144b2f2a4e8bb0e9cf6e"


@dataclass(frozen=True)
class _BuilderTarget:
    label: str
    script: Path
    pythonpath_root: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_litengjie_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    allowed = ALLOWED_OUTPUT_ROOT.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"Benchmark output must be under {allowed}: {resolved}")
    return resolved


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), list):
        raise ValueError(f"Config must contain a datasets list: {path}")
    return payload


def _subset_config(
    source_config: dict[str, Any],
    *,
    i2i_rows: int,
    r2v_rows: int,
) -> dict[str, Any]:
    subset: dict[str, Any] = {
        key: value
        for key, value in source_config.items()
        if key != "datasets"
    }
    selected: list[dict[str, Any]] = []
    selected_i2i = False
    selected_r2v = False
    for raw_dataset in source_config["datasets"]:
        if not isinstance(raw_dataset, dict):
            raise ValueError("Each dataset entry must be a mapping")
        dataset = dict(raw_dataset)
        task = str(dataset.get("task", ""))
        dataset_type = str(dataset.get("dataset_type", ""))
        if task == "i2i" and not selected_i2i:
            dataset["max_samples"] = min(int(dataset.get("max_samples", i2i_rows)), i2i_rows)
            selected.append(dataset)
            selected_i2i = True
            continue
        if task == "r2v" and dataset_type == "OpenS2VDataset" and not selected_r2v:
            dataset["max_samples"] = min(int(dataset.get("max_samples", r2v_rows)), r2v_rows)
            selected.append(dataset)
            selected_r2v = True
    if not selected_i2i:
        raise ValueError("No i2i dataset found for benchmark subset")
    if not selected_r2v:
        raise ValueError("No OpenS2VDataset r2v dataset found for benchmark subset")
    subset["datasets"] = selected
    return subset


def _write_subset_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _child_rusage() -> resource.struct_rusage:
    return resource.getrusage(resource.RUSAGE_CHILDREN)


def _rss_gb_from_rusage(usage: resource.struct_rusage) -> float:
    peak = float(usage.ru_maxrss)
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024.0
    return peak_bytes / (1024.0**3)


def _run_builder(
    target: _BuilderTarget,
    *,
    subset_config: Path,
    output_dir: Path,
    workers: int,
    media_batch_size: int,
    manifest_seed: int,
    annotation_batch_size: int,
    i2i_target_field: str,
    i2i_reference_field: str,
    i2i_caption_field: str,
    i2i_crop_field: str | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "accepted.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    summary_path = output_dir / "summary.json"
    stdout_path = output_dir / "stdout.json"
    stderr_path = output_dir / "stderr.log"
    command = [
        sys.executable,
        str(target.script),
        "--train-data-config",
        str(subset_config),
        "--output",
        str(accepted_path),
        "--reject-output",
        str(rejected_path),
        "--summary-output",
        str(summary_path),
        "--manifest-seed",
        str(manifest_seed),
        "--annotation-batch-size",
        str(annotation_batch_size),
        "--media-workers",
        str(workers),
        "--media-batch-size",
        str(media_batch_size),
        "--progress-interval-seconds",
        "10",
        "--progress-every-rows",
        "10000",
        "--i2i-target-field",
        i2i_target_field,
        "--i2i-reference-field",
        i2i_reference_field,
        "--i2i-caption-field",
        i2i_caption_field,
    ]
    if i2i_crop_field is not None:
        command.extend(["--i2i-crop-field", i2i_crop_field])

    env = dict(os.environ)
    pythonpath_parts = [str(target.pythonpath_root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(str(env["PYTHONPATH"]))
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    before_usage = _child_rusage()
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            check=False,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    elapsed = time.perf_counter() - started
    after_usage = _child_rusage()
    user_cpu = after_usage.ru_utime - before_usage.ru_utime
    system_cpu = after_usage.ru_stime - before_usage.ru_stime
    cpu_count = os.cpu_count() or 1
    cpu_utilization = ((user_cpu + system_cpu) / max(elapsed, 1e-9)) / cpu_count

    if completed.returncode != 0:
        return {
            "label": target.label,
            "workers": workers,
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "label": target.label,
        "workers": workers,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "rows_per_second": summary.get("raw_rows", 0) / max(elapsed, 1e-9),
        "builder_elapsed_seconds": summary.get("elapsed_seconds"),
        "builder_rows_per_second": summary.get("rows_per_second"),
        "cpu_utilization": cpu_utilization,
        "peak_rss_gb": max(float(summary.get("peak_rss_gb", 0.0)), _rss_gb_from_rusage(after_usage)),
        "signature_calls": summary.get("signature_submitted"),
        "video_probes": summary.get("video_probe_submitted"),
        "image_probes": summary.get("image_probe_submitted"),
        "reference_probes_skipped_due_video_reject": summary.get(
            "reference_probes_skipped_due_video_reject"
        ),
        "cache_hits": summary.get("media_validation_cache_hits"),
        "accepted": summary.get("accepted_rows"),
        "rejected": summary.get("rejected_rows"),
        "accepted_sha256": _sha256_file(accepted_path),
        "rejected_sha256": _sha256_file(rejected_path),
        "idx_sha256": _sha256_file(Path(f"{accepted_path}.idx")),
        "summary_path": str(summary_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _build_targets(args: argparse.Namespace, repo_root: Path) -> list[_BuilderTarget]:
    current_script = repo_root / "packages" / "ltx-trainer" / "scripts" / "build_multitask_online_manifest.py"
    current_pythonpath = repo_root / "packages" / "ltx-trainer" / "src"
    targets = [
        _BuilderTarget(
            label="v7.1",
            script=current_script,
            pythonpath_root=current_pythonpath,
        )
    ]
    if args.baseline_builder_script is not None:
        baseline_script = Path(args.baseline_builder_script).expanduser().resolve()
        baseline_root = Path(args.baseline_pythonpath_root).expanduser().resolve()
        targets.insert(
            0,
            _BuilderTarget(
                label=f"baseline-{args.baseline_ref}",
                script=baseline_script,
                pythonpath_root=baseline_root,
            ),
        )
    return targets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data-config", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--i2i-rows", type=int, default=20_000)
    parser.add_argument("--r2v-rows", type=int, default=20_000)
    parser.add_argument("--workers", type=int, action="append")
    parser.add_argument("--media-batch-size", type=int, default=2048)
    parser.add_argument("--annotation-batch-size", type=int, default=4096)
    parser.add_argument("--manifest-seed", type=int, default=42)
    parser.add_argument("--i2i-target-field", required=True)
    parser.add_argument("--i2i-reference-field", required=True)
    parser.add_argument("--i2i-caption-field", required=True)
    parser.add_argument("--i2i-crop-field")
    parser.add_argument("--baseline-ref", default=DEFAULT_BASELINE_REF)
    parser.add_argument("--baseline-builder-script")
    parser.add_argument("--baseline-pythonpath-root")
    args = parser.parse_args()
    if args.workers is None:
        args.workers = [1, 16, 32]
    return args


def main() -> None:
    args = _parse_args()
    if (args.baseline_builder_script is None) != (args.baseline_pythonpath_root is None):
        raise ValueError("--baseline-builder-script and --baseline-pythonpath-root must be provided together")
    repo_root = Path(__file__).resolve().parents[3]
    output_root = _assert_litengjie_output_root(Path(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    source_config = _load_config(Path(args.train_data_config).expanduser().resolve())
    subset = _subset_config(source_config, i2i_rows=args.i2i_rows, r2v_rows=args.r2v_rows)
    subset_config = output_root / "benchmark_subset.yaml"
    _write_subset_config(subset_config, subset)

    results: list[dict[str, Any]] = []
    for target in _build_targets(args, repo_root):
        for workers in args.workers:
            result = _run_builder(
                target,
                subset_config=subset_config,
                output_dir=output_root / target.label / f"workers_{workers}",
                workers=workers,
                media_batch_size=args.media_batch_size,
                manifest_seed=args.manifest_seed,
                annotation_batch_size=args.annotation_batch_size,
                i2i_target_field=args.i2i_target_field,
                i2i_reference_field=args.i2i_reference_field,
                i2i_caption_field=args.i2i_caption_field,
                i2i_crop_field=args.i2i_crop_field,
            )
            results.append(result)

    v71_successes = [item for item in results if item["label"] == "v7.1" and item["returncode"] == 0]
    determinism = {
        "accepted_jsonl_bytes_identical": len({item["accepted_sha256"] for item in v71_successes}) <= 1,
        "rejected_jsonl_bytes_identical": len({item["rejected_sha256"] for item in v71_successes}) <= 1,
        "idx_bytes_identical": len({item["idx_sha256"] for item in v71_successes}) <= 1,
    }
    payload = {
        "output_root": str(output_root),
        "subset_config": str(subset_config),
        "i2i_rows": args.i2i_rows,
        "r2v_rows": args.r2v_rows,
        "media_batch_size": args.media_batch_size,
        "workers": args.workers,
        "results": results,
        "v7_1_worker_determinism": determinism,
    }
    report_path = output_root / "benchmark_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
