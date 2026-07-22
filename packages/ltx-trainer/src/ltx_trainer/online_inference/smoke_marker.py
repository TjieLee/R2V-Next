"""Create and validate semantic-flow smoke success markers."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ltx_trainer.online_inference.output_artifacts import atomic_write_json


class SemanticFlowSmokeMarkerError(RuntimeError):
    """A semantic-flow smoke marker or its evidence is invalid."""


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticFlowSmokeMarkerError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SemanticFlowSmokeMarkerError(f"Expected a JSON object: {path}")
    return payload


def _require_file(label: str, path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise SemanticFlowSmokeMarkerError(f"{label} is missing: {resolved}")
    return resolved


def validate_non_dry_run_i2i_summary(path: str | Path) -> dict[str, Any]:
    summary_path = _require_file("non-dry-run inference summary", path)
    summary = _load_json_object(summary_path)
    results = summary.get("results")
    if summary.get("strict_no_gt") is not True:
        raise SemanticFlowSmokeMarkerError("Inference summary did not enforce strict_no_gt")
    if summary.get("success_count") != 1 or summary.get("failure_count") != 0:
        raise SemanticFlowSmokeMarkerError("Non-dry-run I2I smoke must have exactly one success and no failures")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise SemanticFlowSmokeMarkerError("Non-dry-run I2I smoke must contain exactly one result")

    result = results[0]
    checks = result.get("strict_no_gt_checks")
    required_values = {
        "status": "success",
        "task": "i2i",
        "dry_run": False,
        "num_inference_steps": 2,
        "reference_velocity": 0.0,
        "shared_semantic_video_sigma": True,
        "semantic_latent_finite": True,
        "video_latent_finite": True,
        "decoded_image_finite": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": result.get(key)}
        for key, expected in required_values.items()
        if result.get(key) != expected
    }
    if mismatches:
        raise SemanticFlowSmokeMarkerError(f"Non-dry-run I2I smoke metadata mismatch: {mismatches}")
    if not isinstance(checks, dict) or checks.get("target_path_passed_to_condition_encoder") is not False:
        raise SemanticFlowSmokeMarkerError("Non-dry-run I2I smoke failed strict-no-GT metadata checks")

    output_shape = result.get("output_shape")
    if not isinstance(output_shape, list) or len(output_shape) != 4:
        raise SemanticFlowSmokeMarkerError(f"Invalid decoded image shape: {output_shape}")
    if output_shape[0] != 1 or output_shape[1] != 3 or any(int(value) <= 0 for value in output_shape):
        raise SemanticFlowSmokeMarkerError(f"Decoded I2I shape must be [1,3,H,W], got {output_shape}")

    sample_dir = Path(str(result.get("sample_dir", ""))).expanduser().resolve()
    generated = _require_file("generated I2I PNG", sample_dir / "generated.png")
    success_path = _require_file("I2I success.json", sample_dir / "success.json")
    metadata_path = _require_file("I2I metadata.json", sample_dir / "metadata.json")
    if generated.stat().st_size <= 0:
        raise SemanticFlowSmokeMarkerError(f"Generated I2I PNG is empty: {generated}")
    if _load_json_object(success_path).get("status") != "success":
        raise SemanticFlowSmokeMarkerError("I2I success.json does not report success")
    metadata = _load_json_object(metadata_path)
    for key, expected in required_values.items():
        if key == "status":
            continue
        if metadata.get(key) != expected:
            raise SemanticFlowSmokeMarkerError(
                f"I2I metadata field {key} changed: expected {expected!r}, got {metadata.get(key)!r}"
            )
    return result


def write_semantic_flow_smoke_marker(
    marker_path: str | Path,
    *,
    code_commit: str,
    training_config_path: str | Path,
    accelerate_config_path: str | Path,
    i2i_checkpoint_path: str | Path,
    r2v_checkpoint_path: str | Path,
    runtime_audit_path: str | Path,
    runtime_lock_path: str | Path,
    inference_summary_path: str | Path,
) -> dict[str, Any]:
    training_config = _require_file("training config", training_config_path)
    accelerate_config = _require_file("accelerate config", accelerate_config_path)
    i2i_checkpoint = _require_file("I2I checkpoint", i2i_checkpoint_path)
    r2v_checkpoint = _require_file("R2V checkpoint", r2v_checkpoint_path)
    runtime_audit = _require_file("runtime audit", runtime_audit_path)
    runtime_lock = _require_file("runtime lock", runtime_lock_path)
    inference_summary = _require_file("non-dry-run inference summary", inference_summary_path)
    inference_result = validate_non_dry_run_i2i_summary(inference_summary)
    inference_sample_dir = Path(str(inference_result["sample_dir"])).expanduser().resolve()
    generated_png = _require_file("generated I2I PNG", inference_sample_dir / "generated.png")
    inference_success_json = _require_file("I2I success.json", inference_sample_dir / "success.json")
    inference_metadata_json = _require_file("I2I metadata.json", inference_sample_dir / "metadata.json")
    i2i_checkpoint_sha256 = sha256_file(i2i_checkpoint)
    if inference_result.get("code_commit") != code_commit:
        raise SemanticFlowSmokeMarkerError("Non-dry-run inference was produced by a different code commit")
    if Path(str(inference_result.get("checkpoint", ""))).expanduser().resolve() != i2i_checkpoint:
        raise SemanticFlowSmokeMarkerError("Non-dry-run inference did not use the I2I smoke checkpoint")
    if inference_result.get("checkpoint_sha256") != i2i_checkpoint_sha256:
        raise SemanticFlowSmokeMarkerError("Non-dry-run inference checkpoint SHA256 does not match the I2I checkpoint")

    payload = {
        "architecture": "semantic_flow_v1",
        "completed_at_unix": time.time(),
        "code_commit": code_commit,
        "training_config": str(training_config),
        "training_config_sha256": sha256_file(training_config),
        "accelerate_config": str(accelerate_config),
        "accelerate_config_sha256": sha256_file(accelerate_config),
        "i2i_checkpoint": str(i2i_checkpoint),
        "i2i_checkpoint_sha256": i2i_checkpoint_sha256,
        "r2v_checkpoint": str(r2v_checkpoint),
        "r2v_checkpoint_sha256": sha256_file(r2v_checkpoint),
        "runtime_audit": str(runtime_audit),
        "runtime_audit_sha256": sha256_file(runtime_audit),
        "runtime_lock": str(runtime_lock),
        "runtime_lock_sha256": sha256_file(runtime_lock),
        "non_dry_run_inference_summary": str(inference_summary),
        "non_dry_run_inference_summary_sha256": sha256_file(inference_summary),
        "generated_png": str(generated_png),
        "generated_png_sha256": sha256_file(generated_png),
        "inference_success_json": str(inference_success_json),
        "inference_success_json_sha256": sha256_file(inference_success_json),
        "inference_metadata_json": str(inference_metadata_json),
        "inference_metadata_json_sha256": sha256_file(inference_metadata_json),
        "non_dry_run_inference_passed": True,
    }
    atomic_write_json(Path(marker_path).expanduser().resolve(), payload)
    return payload


def validate_semantic_flow_smoke_marker(
    marker_path: str | Path,
    *,
    code_commit: str,
    training_config_path: str | Path,
    accelerate_config_path: str | Path,
) -> dict[str, Any]:
    marker = _load_json_object(_require_file("semantic-flow smoke marker", marker_path))
    if marker.get("non_dry_run_inference_passed") is not True:
        raise SemanticFlowSmokeMarkerError("Smoke marker does not include a passing non-dry-run inference")
    if marker.get("code_commit") != code_commit:
        raise SemanticFlowSmokeMarkerError(
            f"Smoke marker code commit changed: {marker.get('code_commit')} != {code_commit}"
        )

    current_files = {
        "training_config": _require_file("training config", training_config_path),
        "accelerate_config": _require_file("accelerate config", accelerate_config_path),
        "i2i_checkpoint": _require_file("I2I checkpoint", str(marker.get("i2i_checkpoint", ""))),
        "r2v_checkpoint": _require_file("R2V checkpoint", str(marker.get("r2v_checkpoint", ""))),
        "runtime_audit": _require_file("runtime audit", str(marker.get("runtime_audit", ""))),
    }
    errors = []
    for label, path in current_files.items():
        expected_path = marker.get(label)
        if label in {"training_config", "accelerate_config"} and expected_path != str(path):
            errors.append(f"{label} path changed: {expected_path} != {path}")
        expected_sha = marker.get(f"{label}_sha256")
        actual_sha = sha256_file(path)
        if expected_sha != actual_sha:
            errors.append(f"{label} SHA256 changed: {expected_sha} != {actual_sha}")
    if errors:
        raise SemanticFlowSmokeMarkerError("; ".join(errors))
    return marker


__all__ = [
    "SemanticFlowSmokeMarkerError",
    "sha256_file",
    "validate_non_dry_run_i2i_summary",
    "validate_semantic_flow_smoke_marker",
    "write_semantic_flow_smoke_marker",
]
