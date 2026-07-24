"""Batch selection and execution helpers for targetless external R2V evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ltx_trainer.online_inference.checkpoint_runtime import OnlineInferenceRuntime
from ltx_trainer.online_inference.external_eval_schema import (
    load_normalized_manifest,
    stable_sample_seed,
)
from ltx_trainer.online_inference.output_artifacts import output_is_complete
from ltx_trainer.online_inference.runner import run_online_sample
from ltx_trainer.online_inference.semantic_guidance import SemanticGuidanceConfig


def select_external_records(
    manifest: str | Path,
    *,
    dataset_name: str,
    ids: Iterable[str] = (),
    id_prefix: str | None = None,
    start_index: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Select one dataset deterministically without touching referenced target media."""
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    records = load_normalized_manifest(manifest)
    unexpected = sorted(
        {str(record["dataset_name"]) for record in records if record["dataset_name"] != dataset_name}
    )
    if unexpected:
        raise ValueError(
            f"Manifest invocation is restricted to dataset {dataset_name!r}; "
            f"found other datasets {unexpected}"
        )
    requested_ids = set(ids)
    selected = [
        record
        for record in records
        if (not requested_ids or str(record["source_record_id"]) in requested_ids)
        and (id_prefix is None or str(record["source_record_id"]).startswith(id_prefix))
    ]
    if requested_ids:
        found = {str(record["source_record_id"]) for record in selected}
        missing = sorted(requested_ids - found)
        if missing:
            raise ValueError(f"Requested external evaluation IDs were not found: {missing}")
    selected = selected[start_index:]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No external evaluation records matched the selection")
    return selected


def external_sample_dir(output_root: Path, sample: dict[str, Any]) -> Path:
    return output_root / str(sample["dataset_name"]) / str(sample["output_id"])


def run_external_records(
    *,
    runtime: OnlineInferenceRuntime,
    records: list[dict[str, Any]],
    output_root: Path,
    base_seed: int,
    num_inference_steps: int,
    guidance: SemanticGuidanceConfig,
    negative_prompt: str | None,
    dry_run: bool,
    resume: bool,
    overwrite_incomplete: bool,
    decode_tile: bool,
    code_commit: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run targetless records in manifest order with stable per-record seeds."""
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sample in records:
        sample_dir = external_sample_dir(output_root, sample)
        complete = output_is_complete(sample_dir)
        if complete and resume:
            results.append(
                {
                    "status": "skipped_existing",
                    "sample_dir": str(sample_dir),
                    "sample_key": sample["sample_key"],
                }
            )
            continue
        if complete and not resume:
            failures.append(
                {
                    "sample_key": sample["sample_key"],
                    "source_record_id": sample["source_record_id"],
                    "error_type": "CompleteOutputExists",
                    "message": f"Complete output already exists: {sample_dir}; use --resume",
                }
            )
            continue
        if sample_dir.exists() and any(sample_dir.iterdir()) and not overwrite_incomplete:
            failures.append(
                {
                    "sample_key": sample["sample_key"],
                    "source_record_id": sample["source_record_id"],
                    "error_type": "IncompleteOutputExists",
                    "message": (
                        f"Incomplete output already exists: {sample_dir}; "
                        "use --overwrite-incomplete"
                    ),
                }
            )
            continue
        seed = stable_sample_seed(
            base_seed,
            str(sample["dataset_name"]),
            str(sample["source_record_id"]),
        )
        try:
            results.append(
                run_online_sample(
                    runtime=runtime,
                    sample=sample,
                    output_root=output_root,
                    dry_run=dry_run,
                    overwrite=True,
                    seed=seed,
                    num_inference_steps=num_inference_steps,
                    decode_tile=decode_tile,
                    guidance=guidance,
                    negative_prompt=negative_prompt,
                    external_eval=True,
                    code_commit=code_commit,
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(
                {
                    "sample_key": sample["sample_key"],
                    "source_record_id": sample["source_record_id"],
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return results, failures


__all__ = [
    "external_sample_dir",
    "run_external_records",
    "select_external_records",
]
