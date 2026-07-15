"""Watch atomically published Stage 3 checkpoints and run isolated inference."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import typer

from ltx_trainer.online_data.path_safety import assert_write_path_allowed

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

DEFAULT_CHECKPOINT_DIR = (
    "/mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/"
    "stage3_warmstart_old_stage3/checkpoints"
)
DEFAULT_OUTPUT_ROOT = (
    "/mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/checkpoint_inference"
)


def _reject_forbidden_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    forbidden = Path("/mnt/workspace/liutao")
    if resolved == forbidden or forbidden in resolved.parents:
        raise ValueError(f"Watcher must not read or write the forbidden path: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(f"{path}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "checkpoints": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("checkpoints"), dict):
        raise ValueError(f"Invalid watcher state: {path}")
    return payload


def _load_ready_marker(marker_path: Path) -> dict[str, Any]:
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    step = int(payload["global_step"])
    expected_name = f"checkpoint_step_{step:05d}.ready.json"
    if marker_path.name != expected_name:
        raise ValueError(
            f"Ready marker filename/global_step mismatch: {marker_path.name} != {expected_name}"
        )
    checkpoint_path = _reject_forbidden_path(payload["checkpoint_path"])
    training_state_path = _reject_forbidden_path(payload["training_state_path"])
    config_path = _reject_forbidden_path(payload["config_path"])
    if not checkpoint_path.name.endswith(f"_weights_step_{step:05d}.safetensors"):
        raise ValueError(f"Ready marker has an unexpected checkpoint filename: {checkpoint_path.name}")
    if training_state_path.name != f"training_state_step_{step:05d}.pt":
        raise ValueError(f"Ready marker has an unexpected training-state filename: {training_state_path.name}")
    if not checkpoint_path.is_file() or not training_state_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            "Ready checkpoint publication is incomplete: "
            f"weights={checkpoint_path}, state={training_state_path}, config={config_path}"
        )
    if checkpoint_path.stat().st_size != int(payload["checkpoint_size_bytes"]):
        raise ValueError(f"Checkpoint size no longer matches marker: {checkpoint_path}")
    if str(payload.get("metadata_global_step")) != str(step):
        raise ValueError(f"Ready marker metadata_global_step does not match step {step}")
    if payload.get("metadata_training_phase") != "stage3":
        raise ValueError("Watcher accepts only training_phase=stage3 ready markers")
    return payload


def _stage_checkpoint(marker: dict[str, Any], staged_dir: Path) -> Path:
    source = _reject_forbidden_path(marker["checkpoint_path"])
    step = int(marker["global_step"])
    expected_sha256 = str(marker["checkpoint_sha256"])
    destination = staged_dir / f"step_{step:05d}.safetensors"
    if destination.is_file():
        if _sha256_file(destination) != expected_sha256:
            raise ValueError(f"Existing staged checkpoint checksum mismatch: {destination}")
        return destination

    temporary = Path(f"{destination}.tmp.{os.getpid()}")
    try:
        try:
            os.link(source, temporary)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copyfile(source, temporary)
            descriptor = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        actual_sha256 = _sha256_file(temporary)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Staged checkpoint checksum mismatch: expected={expected_sha256}, actual={actual_sha256}"
            )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _format_command(
    template: str,
    *,
    checkpoint: Path,
    step: int,
    output_dir: Path,
    gpu: int,
) -> list[str]:
    try:
        rendered = template.format(
            checkpoint=str(checkpoint),
            step=step,
            output_dir=str(output_dir),
            gpu=gpu,
        )
    except KeyError as exc:
        raise ValueError(f"Unknown command-template placeholder: {exc.args[0]}") from exc
    if "/mnt/workspace/liutao" in rendered:
        raise ValueError("Inference command must not read or write /mnt/workspace/liutao")
    command = shlex.split(rendered)
    if not command:
        raise ValueError("--command-template rendered an empty command")
    return command


def _eligible_markers(
    checkpoint_dir: Path,
    *,
    processed_steps: set[int],
    min_step: int,
    step_stride: int,
) -> list[Path]:
    eligible: list[tuple[int, Path]] = []
    for marker_path in checkpoint_dir.glob("checkpoint_step_*.ready.json"):
        try:
            step = int(marker_path.name.removeprefix("checkpoint_step_").removesuffix(".ready.json"))
        except ValueError:
            continue
        if step in processed_steps or step < min_step or step % step_stride != 0:
            continue
        eligible.append((step, marker_path))
    return [path for _, path in sorted(eligible)]


@app.command()
def main(  # noqa: PLR0912, PLR0913, PLR0915
    checkpoint_dir: str = typer.Option(DEFAULT_CHECKPOINT_DIR, "--checkpoint-dir"),
    output_root: str = typer.Option(DEFAULT_OUTPUT_ROOT, "--output-root"),
    state_path: str | None = typer.Option(None, "--state-path"),
    gpu: int = typer.Option(7, "--gpu"),
    poll_seconds: float = typer.Option(30.0, "--poll-seconds"),
    min_step: int = typer.Option(0, "--min-step"),
    step_stride: int = typer.Option(1, "--step-stride"),
    max_checkpoints: int | None = typer.Option(None, "--max-checkpoints"),
    command_template: str = typer.Option(..., "--command-template"),
) -> None:
    if gpu < 0 or poll_seconds <= 0 or min_step < 0 or step_stride <= 0:
        raise typer.BadParameter("gpu/min-step must be non-negative and poll-seconds/step-stride positive")
    if max_checkpoints is not None and max_checkpoints <= 0:
        raise typer.BadParameter("--max-checkpoints must be positive")

    checkpoints = _reject_forbidden_path(checkpoint_dir)
    if not checkpoints.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoints}")
    root = assert_write_path_allowed(output_root)
    root.mkdir(parents=True, exist_ok=True)
    staged_dir = root / "staged_checkpoints"
    staged_dir.mkdir(parents=True, exist_ok=True)
    watcher_state_path = assert_write_path_allowed(state_path or root / "watcher_state.json")
    watcher_state_path.parent.mkdir(parents=True, exist_ok=True)
    state = _load_state(watcher_state_path)
    attempted = 0

    while max_checkpoints is None or attempted < max_checkpoints:
        processed_steps = {int(step) for step in state["checkpoints"]}
        markers = _eligible_markers(
            checkpoints,
            processed_steps=processed_steps,
            min_step=min_step,
            step_stride=step_stride,
        )
        if not markers:
            time.sleep(poll_seconds)
            continue
        remaining = None if max_checkpoints is None else max_checkpoints - attempted
        selected_markers = markers if remaining is None else markers[:remaining]
        staged_jobs: list[tuple[Path, dict[str, Any], Path, float]] = []
        for marker_path in selected_markers:
            if max_checkpoints is not None and attempted >= max_checkpoints:
                break
            attempted += 1
            step = int(marker_path.name.removeprefix("checkpoint_step_").removesuffix(".ready.json"))
            started_at = time.time()
            try:
                marker = _load_ready_marker(marker_path)
                staged_checkpoint = _stage_checkpoint(marker, staged_dir)
                staged_jobs.append((marker_path, marker, staged_checkpoint, started_at))
            except Exception as exc:
                output_dir = root / f"step_{step:05d}"
                output_dir.mkdir(parents=True, exist_ok=True)
                state["checkpoints"][str(step)] = {
                    "step": step,
                    "ready_marker": str(marker_path),
                    "started_at_unix": started_at,
                    "finished_at_unix": time.time(),
                    "status": "failed",
                    "error": str(exc),
                    "output_dir": str(output_dir),
                }
                _atomic_write_json(watcher_state_path, state)
                typer.echo(f"Checkpoint step {step} staging failed: {exc}", err=True)

        for marker_path, marker, staged_checkpoint, started_at in staged_jobs:
            step = int(marker["global_step"])
            output_dir = root / f"step_{step:05d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            result: dict[str, Any] = {
                "step": step,
                "ready_marker": str(marker_path),
                "started_at_unix": started_at,
            }
            try:
                command = _format_command(
                    command_template,
                    checkpoint=staged_checkpoint,
                    step=step,
                    output_dir=output_dir,
                    gpu=gpu,
                )
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                environment["TOKENIZERS_PARALLELISM"] = "false"
                environment["OMP_NUM_THREADS"] = "4"
                completed = subprocess.run(command, check=False, env=environment)
                if completed.returncode != 0:
                    raise RuntimeError(f"Inference command exited with code {completed.returncode}")
                result.update(
                    {
                        "status": "success",
                        "staged_checkpoint": str(staged_checkpoint),
                        "output_dir": str(output_dir),
                        "command": command,
                    }
                )
            except Exception as exc:
                result.update({"status": "failed", "error": str(exc), "output_dir": str(output_dir)})
                typer.echo(f"Checkpoint step {step} inference failed: {exc}", err=True)
            result["finished_at_unix"] = time.time()
            state["checkpoints"][str(step)] = result
            _atomic_write_json(watcher_state_path, state)


if __name__ == "__main__":
    app()
