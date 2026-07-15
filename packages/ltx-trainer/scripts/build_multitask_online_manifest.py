"""Build a deterministic I2I/R2V online manifest from raw annotations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer

from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_data.manifest import (
    ManifestReject,
    build_i2i_record,
    build_r2v_record,
    deduplicate_records,
    load_multitask_data_config,
    read_annotation_rows,
)
from ltx_trainer.online_data.path_safety import assert_write_path_allowed

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    tmp_path.replace(path)


@app.command()
def main(  # noqa: PLR0913
    train_data_config: str = typer.Option(..., "--train-data-config"),
    output: str = typer.Option(..., "--output"),
    reject_output: str | None = typer.Option(None, "--reject-output"),
    manifest_seed: int = typer.Option(42, "--manifest-seed"),
    i2i_target_field: str = typer.Option(..., "--i2i-target-field"),
    i2i_reference_field: str = typer.Option(..., "--i2i-reference-field"),
    i2i_caption_field: str = typer.Option(..., "--i2i-caption-field"),
    i2i_crop_field: str | None = typer.Option(None, "--i2i-crop-field"),
) -> None:
    output_path = assert_write_path_allowed(output)
    reject_path = assert_write_path_allowed(
        reject_output or str(output_path.with_name(output_path.stem + "_rejected.jsonl"))
    )
    config = load_multitask_data_config(train_data_config)
    records: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []

    for dataset in config["datasets"]:
        if not isinstance(dataset, dict):
            raise ValueError("Each datasets entry must be a mapping")
        task = str(dataset.get("task", ""))
        dataset_name = str(dataset.get("name", task))
        annotation_path = dataset.get("ann_path") or dataset.get("parquet") or dataset.get("path")
        if annotation_path is None:
            raise ValueError(f"Dataset {dataset_name!r} has no ann_path/parquet/path")
        rows = read_annotation_rows(annotation_path)
        available_columns = sorted({key for row in rows for key in row})
        if task == IMAGE_TASK:
            required_i2i_fields = {i2i_target_field, i2i_reference_field, i2i_caption_field}
            if i2i_crop_field is not None:
                required_i2i_fields.add(i2i_crop_field)
            missing_fields = sorted(required_i2i_fields - set(available_columns))
            if missing_fields:
                raise ValueError(
                    f"I2I schema mismatch; missing adapter fields {missing_fields}. "
                    f"Available columns: {available_columns}"
                )
        elif task == VIDEO_TASK:
            required_r2v_fields = {"video_path", "text", "crop", "face_cut", "ref_images"}
            missing_fields = sorted(required_r2v_fields - set(available_columns))
            if missing_fields:
                raise ValueError(
                    f"OpenS2V schema mismatch; missing fields {missing_fields}. "
                    f"Available columns: {available_columns}"
                )
        max_samples = dataset.get("max_samples")
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        data_root = dataset.get("data_root", config.get("data_root"))
        for row_index, row in enumerate(rows):
            try:
                if task == IMAGE_TASK:
                    record = build_i2i_record(
                        row,
                        dataset_name=dataset_name,
                        data_root=data_root,
                        target_field=i2i_target_field,
                        reference_field=i2i_reference_field,
                        caption_field=i2i_caption_field,
                        crop_field=i2i_crop_field,
                    )
                elif task == VIDEO_TASK:
                    record = build_r2v_record(
                        row,
                        dataset_name=dataset_name,
                        data_root=data_root,
                        manifest_seed=manifest_seed,
                    )
                else:
                    raise ValueError(f"Unsupported task {task!r} in dataset {dataset_name!r}")
                records.append(record)
            except Exception as exc:
                reason = exc.reason if isinstance(exc, ManifestReject) else type(exc).__name__
                rejects.append(
                    {
                        "dataset_name": dataset_name,
                        "row_index": row_index,
                        "task": task,
                        "reason": reason,
                        "message": str(exc),
                    }
                )

    unique_records = deduplicate_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reject_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(output_path, unique_records)
    _atomic_write_jsonl(reject_path, rejects)
    task_counts = {
        task: sum(record["task"] == task for record in unique_records)
        for task in (IMAGE_TASK, VIDEO_TASK)
    }
    if any(count == 0 for count in task_counts.values()):
        raise RuntimeError(
            f"Built manifest does not contain both tasks: {task_counts}. "
            f"Inspect rejects at {reject_path}."
        )
    typer.echo(
        f"Wrote {len(unique_records)} deterministic samples {task_counts} to {output_path}; "
        f"{len(rejects)} rejects to {reject_path}"
    )


if __name__ == "__main__":
    app()
