#!/usr/bin/env python3
"""Normalize and preflight external reference-only R2V evaluation JSON files."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ltx_trainer.online_inference.external_eval_schema import (
    DATASET_SCHEMAS,
    build_preflight_report,
    crop_rows_csv,
    manifest_jsonl,
    normalize_external_dataset,
)
from ltx_trainer.online_inference.read_only_sources import ReadOnlySourcePolicy

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)
console = Console()
DEFAULT_WRITABLE_ROOT = Path("/mnt/workspace/litengjie")


def prepare_external_dataset(
    *,
    dataset_schema: str,
    input_json: str | Path,
    output_manifest: str | Path,
    report_dir: str | Path,
    writable_root: str | Path = DEFAULT_WRITABLE_ROOT,
) -> dict[str, object]:
    source_json = Path(input_json).expanduser().resolve()
    roots = (source_json.parent,) if dataset_schema == "opens2v_open_domain" else ()
    policy = ReadOnlySourcePolicy(
        writable_root=Path(writable_root),
        allowed_roots=roots,
        allowed_files=frozenset({source_json}),
    )
    source_before = policy.snapshot(source_json)
    payload = policy.read_json(source_json)
    actual_count = len(payload) if isinstance(payload, (list, dict)) else 0
    records, schema_errors = normalize_external_dataset(
        dataset_schema=dataset_schema,
        input_json=source_json,
        payload=payload,
    )
    policy = policy.with_allowed_files(
        [reference for record in records for reference in record["reference_paths"]]
    )
    report, crop_rows = build_preflight_report(
        records,
        policy=policy,
        schema_errors=schema_errors,
    )
    source_after = policy.snapshot(source_json)
    report.update(
        {
            "record_count": actual_count,
            "actual_count": actual_count,
            "normalized_record_count": len(records),
            "dataset_schema": dataset_schema,
            "input_json": str(source_json),
            "output_manifest": str(Path(output_manifest).expanduser().resolve()),
            "source_json_before": source_before.__dict__,
            "source_json_after": source_after.__dict__,
            "source_json_unchanged": source_before == source_after,
            "read_only_source_policy": {
                "allowed_roots": [str(path) for path in policy.allowed_roots],
                "explicit_allowed_file_count": len(policy.allowed_files),
                "writable_root": str(policy.writable_root),
            },
        }
    )
    report_root = policy.ensure_directory(report_dir)
    policy.atomic_write_json(report_root / "report.json", report)
    policy.atomic_write_text(report_root / "crop_risk.csv", crop_rows_csv(crop_rows))
    if not report["preflight_passed"]:
        raise RuntimeError(
            f"Preflight failed for {dataset_schema}; see {report_root / 'report.json'}"
        )
    policy.atomic_write_text(output_manifest, manifest_jsonl(records))
    return report


@app.command()
def main(
    dataset_schema: str = typer.Option(..., "--dataset-schema"),
    input_json: str = typer.Option(..., "--input-json"),
    output_manifest: str = typer.Option(..., "--output-manifest"),
    report_dir: str = typer.Option(..., "--report-dir"),
) -> None:
    if dataset_schema not in DATASET_SCHEMAS:
        raise typer.BadParameter(
            f"--dataset-schema must be one of {sorted(DATASET_SCHEMAS)}"
        )
    report = prepare_external_dataset(
        dataset_schema=dataset_schema,
        input_json=input_json,
        output_manifest=output_manifest,
        report_dir=report_dir,
        writable_root=DEFAULT_WRITABLE_ROOT,
    )
    console.print_json(
        data={
            "dataset_schema": dataset_schema,
            "actual_count": report["actual_count"],
            "reference_count_distribution": report["reference_count_distribution"],
            "preflight_passed": report["preflight_passed"],
            "output_manifest": str(Path(output_manifest).expanduser().resolve()),
            "report_dir": str(Path(report_dir).expanduser().resolve()),
        }
    )


if __name__ == "__main__":
    app()
