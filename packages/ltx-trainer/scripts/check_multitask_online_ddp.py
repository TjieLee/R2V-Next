"""Run real two-rank Stage 1/2/3 online training smoke tests."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
import torch.distributed as dist
import typer

from ltx_trainer.online_data.smoke import run_one_step_training_smoke

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


@app.command()
def main(
    stage1_config: str = typer.Option(..., "--stage1-config"),
    stage2_config: str = typer.Option(..., "--stage2-config"),
    stage3_config: str = typer.Option(..., "--stage3-config"),
    output_root: str = typer.Option(
        "/mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/ddp_smoke",
        "--output-root",
    ),
) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size != 2:
        raise RuntimeError(f"DDP smoke requires exactly two processes, got {world_size}")
    launcher_pid = os.getppid()
    run_root = Path(output_root) / f"run_{launcher_pid}"
    configs = {
        "stage1": stage1_config,
        "stage2": stage2_config,
        "stage3": stage3_config,
    }
    for stage, config in configs.items():
        for task in ("i2i", "r2v"):
            output_dir = run_root / stage / task
            run_one_step_training_smoke(config, task=task, output_dir=output_dir)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if rank == 0:
        typer.echo(
            f"Two-rank online DDP smoke completed for all stages/tasks; output={run_root}"
        )


if __name__ == "__main__":
    app()
