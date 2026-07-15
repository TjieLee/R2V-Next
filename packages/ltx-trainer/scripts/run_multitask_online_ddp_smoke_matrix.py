"""Launch each online Stage 3 smoke case in a fresh Accelerate process group."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import typer

from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_data.path_safety import assert_write_path_allowed

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


def _parse_tasks(value: str) -> list[str]:
    tasks = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(tasks) - {IMAGE_TASK, VIDEO_TASK})
    if not tasks or invalid:
        raise typer.BadParameter("--tasks must be a comma-separated subset of i2i,r2v")
    if len(tasks) != len(set(tasks)):
        raise typer.BadParameter("--tasks must not contain duplicates")
    return tasks


@app.command()
def main(
    config: str = typer.Option(..., "--config"),
    init_checkpoint: str | None = typer.Option(None, "--init-checkpoint"),
    output_root: str = typer.Option(
        "/mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/ddp_smoke",
        "--output-root",
    ),
    tasks: str = typer.Option("i2i,r2v", "--tasks"),
    num_processes: int = typer.Option(2, "--num-processes", min=2),
    gpu_devices: str = typer.Option("0,1", "--gpu-devices"),
    accelerate_config: str = typer.Option("configs/accelerate/ddp.yaml", "--accelerate-config"),
    accelerate_executable: str = typer.Option("accelerate", "--accelerate-executable"),
) -> None:
    if num_processes != 2:
        raise typer.BadParameter("Stage 3 DDP smoke matrix requires --num-processes 2")
    root = assert_write_path_allowed(output_root)
    root.mkdir(parents=True, exist_ok=True)
    worker_script = Path(__file__).with_name("check_multitask_online_ddp.py")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu_devices
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment.setdefault("OMP_NUM_THREADS", "4")

    for task in _parse_tasks(tasks):
        command = [
            accelerate_executable,
            "launch",
            "--config_file",
            accelerate_config,
            "--num_processes",
            str(num_processes),
            str(worker_script),
            "--config",
            config,
            "--task",
            task,
            "--output-dir",
            str(root / f"stage3_{task}"),
        ]
        if init_checkpoint is not None:
            command.extend(["--init-checkpoint", init_checkpoint])
        typer.echo(f"Launching isolated Stage 3 {task} DDP smoke")
        subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    app()
