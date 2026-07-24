"""Write-path policy shared by online selection and generation CLIs."""

from __future__ import annotations

from pathlib import Path

from ltx_trainer.online_data.path_safety import assert_write_path_allowed

ONLINE_INFERENCE_ROOT = Path(
    "/mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/inference"
)
EXTERNAL_EVAL_ROOT = Path("/mnt/workspace/litengjie")


def assert_online_inference_output_path(path: str | Path) -> Path:
    resolved = assert_write_path_allowed(path)
    try:
        resolved.relative_to(ONLINE_INFERENCE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Online sample inference outputs must stay under {ONLINE_INFERENCE_ROOT}: {resolved}"
        ) from exc
    return resolved


def assert_external_eval_output_path(path: str | Path) -> Path:
    resolved = assert_write_path_allowed(path)
    try:
        resolved.relative_to(EXTERNAL_EVAL_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"External evaluation outputs must stay under {EXTERNAL_EVAL_ROOT}: {resolved}"
        ) from exc
    return resolved
