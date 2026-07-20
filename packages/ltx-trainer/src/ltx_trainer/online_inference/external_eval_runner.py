"""Persistent targetless Stage 3 runner for normalized external R2V records."""

from __future__ import annotations

import html
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image

from ltx_trainer.online_inference.checkpoint_runtime import OnlineInferenceRuntime
from ltx_trainer.online_inference.external_eval_schema import (
    center_crop_risk,
    stable_sample_seed,
)
from ltx_trainer.online_inference.output_artifacts import output_is_complete
from ltx_trainer.online_inference.raw_condition_encoder import (
    encode_external_reference_only_conditions,
)
from ltx_trainer.online_inference.read_only_sources import ReadOnlySourcePolicy
from ltx_trainer.online_inference.runner import run_online_sample


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def filter_external_records(
    records: list[dict[str, Any]],
    *,
    dataset_name: str | None = None,
    record_ids: set[str] | None = None,
    id_prefixes: tuple[str, ...] = (),
    start_index: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    selected = [
        record
        for record in records
        if (dataset_name is None or record["dataset_name"] == dataset_name)
        and (record_ids is None or record["source_record_id"] in record_ids)
        and (
            not id_prefixes
            or any(record["source_record_id"].startswith(prefix) for prefix in id_prefixes)
        )
    ]
    selected = selected[start_index:]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        selected = selected[:limit]
    return selected


def dataset_output_root(
    output_root: str | Path,
    dataset_name: str,
    policy: ReadOnlySourcePolicy,
) -> Path:
    return policy.assert_write_path(Path(output_root).expanduser().resolve() / dataset_name)


def external_sample_dir(dataset_root: Path, output_id: str, policy: ReadOnlySourcePolicy) -> Path:
    return policy.assert_write_path(dataset_root / "samples" / output_id)


def _dry_run_complete(sample_dir: Path) -> bool:
    marker = sample_dir / "dry_run.success.json"
    metadata = sample_dir / "dry_run.json"
    return marker.is_file() and metadata.is_file() and metadata.stat().st_size > 0


def classify_sample_output(sample_dir: Path, *, dry_run: bool) -> str:
    if not sample_dir.exists():
        return "missing"
    if _dry_run_complete(sample_dir) if dry_run else output_is_complete(sample_dir):
        return "complete"
    return "incomplete"


def source_crop_risks(
    sample: dict[str, Any],
    *,
    policy: ReadOnlySourcePolicy,
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    for index, value in enumerate(sample["reference_paths"]):
        path = policy.assert_read_path(value)
        with Image.open(path) as image:
            width, height = image.size
        risks.append({"reference_index": index, **center_crop_risk(width, height)})
    return risks


def _finalize_flat_export(
    *,
    sample: dict[str, Any],
    sample_dir: Path,
    dataset_root: Path,
    policy: ReadOnlySourcePolicy,
    replace_existing: bool = False,
) -> tuple[Path, str]:
    generated = policy.assert_write_path(sample_dir / "generated.mp4")
    flat_video = policy.assert_write_path(
        dataset_root / "Generated_Videos" / f"{sample['output_id']}.mp4"
    )
    if not replace_existing and flat_video.is_file() and flat_video.stat().st_size > 0:
        mode = "existing"
    else:
        mode = policy.link_or_copy(generated, flat_video)
    metadata_path = policy.assert_write_path(sample_dir / "metadata.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["flat_export_path"] = str(flat_video)
        metadata["flat_export_mode"] = mode
        policy.atomic_write_json(metadata_path, metadata)
    return flat_video, mode


def run_external_sample(
    *,
    runtime: OnlineInferenceRuntime,
    sample: dict[str, Any],
    output_root: Path,
    policy: ReadOnlySourcePolicy,
    dry_run: bool,
    resume: bool,
    overwrite_incomplete: bool,
    base_seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    ref_guidance_scale: float,
    vision_guidance_scale: float,
    ref_guidance_mode: str,
    guidance_rescale: float,
    stg_scale: float,
    stg_blocks: list[int] | None,
    decode_tile: bool,
    code_commit: str | None,
) -> dict[str, Any]:  # noqa: PLR0913
    dataset_root = dataset_output_root(output_root, sample["dataset_name"], policy)
    sample_dir = external_sample_dir(dataset_root, sample["output_id"], policy)
    state = classify_sample_output(sample_dir, dry_run=dry_run)
    if state == "complete":
        if resume:
            flat_path = None
            flat_mode = None
            if not dry_run:
                flat_path, flat_mode = _finalize_flat_export(
                    sample=sample,
                    sample_dir=sample_dir,
                    dataset_root=dataset_root,
                    policy=policy,
                )
            return {
                "status": "skipped_existing",
                "sample_dir": str(sample_dir),
                "source_record_id": sample["source_record_id"],
                "output_id": sample["output_id"],
                "flat_export_path": str(flat_path) if flat_path else None,
                "flat_export_mode": flat_mode,
            }
        raise FileExistsError(f"Complete output exists; pass --resume to skip it: {sample_dir}")
    if state == "incomplete" and not overwrite_incomplete:
        raise FileExistsError(
            f"Incomplete output exists; pass --overwrite-incomplete to retry: {sample_dir}"
        )

    for reference in sample["reference_paths"]:
        policy.assert_read_path(reference)
    sample_seed = stable_sample_seed(
        base_seed,
        sample["dataset_name"],
        sample["source_record_id"],
    )
    crop_risks = source_crop_risks(sample, policy=policy)
    metadata_overrides = {
        "dataset_name": sample["dataset_name"],
        "source_json": sample["source_json"],
        "source_record_id": sample["source_record_id"],
        "output_id": sample["output_id"],
        "dataset_metadata": sample["dataset_metadata"],
        "seed_mode": "stable-id",
        "base_seed": base_seed,
        "has_target": False,
        "reference_target_alias_check": "not_applicable_no_target",
        "target_open_count": 0,
        "uses_target_latents": False,
        "uses_gt_siglip_tokens": False,
        "crop_risks": crop_risks,
    }
    result = run_online_sample(
        runtime=runtime,
        sample=sample,
        output_root=dataset_root,
        dry_run=dry_run,
        overwrite=state == "incomplete" and overwrite_incomplete,
        seed=sample_seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        ref_guidance_scale=ref_guidance_scale,
        vision_guidance_scale=vision_guidance_scale,
        ref_guidance_mode=ref_guidance_mode,
        guidance_rescale=guidance_rescale,
        stg_scale=stg_scale,
        stg_blocks=stg_blocks,
        decode_tile=decode_tile,
        code_commit=code_commit,
        condition_encoder=encode_external_reference_only_conditions,
        sample_dir_override=sample_dir,
        metadata_overrides=metadata_overrides,
        save_mid_frame=True,
    )
    result.update(
        {
            "source_record_id": sample["source_record_id"],
            "output_id": sample["output_id"],
            "seed": sample_seed,
        }
    )
    if result["status"] == "success":
        flat_video, flat_mode = _finalize_flat_export(
            sample=sample,
            sample_dir=sample_dir,
            dataset_root=dataset_root,
            policy=policy,
            replace_existing=True,
        )
        result["flat_export_mode"] = flat_mode
        result["flat_export_path"] = str(flat_video)
    return result


def _sample_gallery_entry(sample: dict[str, Any], dataset_root: Path) -> str:
    sample_dir = Path("samples") / sample["output_id"]
    metadata_path = dataset_root / sample_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reference_images = "".join(
        f'<img src="{html.escape(str(sample_dir / f"reference_{index:02d}.png"))}" '
        f'alt="reference {index}">'
        for index in range(len(sample["reference_paths"]))
    )
    crop_flags = sorted(
        {
            key
            for risk in metadata.get("crop_risks", [])
            for key in (
                "severe_crop_risk",
                "vertical_crop_risk",
                "horizontal_crop_risk",
                "square_to_wide_risk",
                "portrait_to_wide_risk",
            )
            if risk.get(key)
        }
    )
    video_path = sample_dir / "generated.mp4"
    return f"""
<article>
  <h2>{html.escape(sample['source_record_id'])}</h2>
  <p>{html.escape(sample['caption'])}</p>
  <div class="references">{reference_images}</div>
  <video controls preload="metadata" src="{html.escape(str(video_path))}"></video>
  <div class="frames">
    <img src="{html.escape(str(sample_dir / 'generated_first.png'))}" alt="first frame">
    <img src="{html.escape(str(sample_dir / 'generated_mid.png'))}" alt="middle frame">
    <img src="{html.escape(str(sample_dir / 'generated_contact_sheet.png'))}" alt="contact sheet">
    <img src="{html.escape(str(sample_dir / 'generated_last.png'))}" alt="last frame">
  </div>
  <p>seed={metadata.get('seed', 'pending')} refs={len(sample['reference_paths'])}
     elapsed={metadata.get('elapsed_seconds', 'pending')}
     crop={html.escape(', '.join(crop_flags) or 'none')}</p>
  <a href="{html.escape(str(sample_dir / 'metadata.json'))}">metadata</a>
  <label>review <select data-review="{html.escape(sample['source_record_id'])}">
    <option value="">unreviewed</option><option value="pass">pass</option>
    <option value="fail">fail</option></select></label>
</article>"""


def write_static_gallery(
    *,
    records: list[dict[str, Any]],
    dataset_root: Path,
    policy: ReadOnlySourcePolicy,
) -> Path:
    entries = "\n".join(_sample_gallery_entry(sample, dataset_root) for sample in records)
    title = records[0]["dataset_name"] if records else dataset_root.name
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font:14px system-ui;margin:24px;background:#f5f5f5;color:#171717}}
article{{background:white;border:1px solid #ddd;margin:0 0 20px;padding:16px;max-width:1180px}}
h2{{font-size:18px;margin:0 0 8px}} video{{width:min(832px,100%);display:block;margin:12px 0}}
.references,.frames{{display:flex;gap:8px;overflow:auto}} .references img{{height:160px}}
.frames img{{height:150px;object-fit:contain;background:#111}}
</style></head><body><h1>{html.escape(title)}</h1>
<button onclick="downloadReviews('json')">Export JSON</button>
<button onclick="downloadReviews('csv')">Export CSV</button>{entries}
<script>
function reviews(){{return [...document.querySelectorAll('[data-review]')].map(x=>({{
  source_record_id:x.dataset.review,label:x.value
}})).filter(x=>x.label);}}
function downloadReviews(format){{
  const rows=reviews(); let body, type;
  if(format==='csv'){{body='source_record_id,label\\n'+rows.map(x=>
    '"'+x.source_record_id.replaceAll('"','""')+'","'+x.label+'"').join('\\n');type='text/csv';}}
  else{{body=JSON.stringify(rows,null,2);type='application/json';}}
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([body],{{type}}));
  a.download='human_reviews.'+format;a.click();URL.revokeObjectURL(a.href);
}}
</script></body></html>"""
    return policy.atomic_write_text(dataset_root / "index.html", document)


def incremental_summary(
    *,
    dataset_root: Path,
    policy: ReadOnlySourcePolicy,
    run_metadata: dict[str, Any],
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    started: float,
) -> Path:
    summary = {
        **run_metadata,
        "updated_at_unix": time.time(),
        "elapsed_seconds": time.perf_counter() - started,
        "result_count": len(results),
        "success_count": sum(result.get("status") == "success" for result in results),
        "dry_run_success_count": sum(
            result.get("status") == "dry_run_success" for result in results
        ),
        "skipped_existing_count": sum(
            result.get("status") == "skipped_existing" for result in results
        ),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
    }
    return policy.atomic_write_json(dataset_root / "run_summary.json", summary)


__all__ = [
    "classify_sample_output",
    "dataset_output_root",
    "external_sample_dir",
    "filter_external_records",
    "git_commit",
    "incremental_summary",
    "run_external_sample",
    "source_crop_risks",
    "write_static_gallery",
]
