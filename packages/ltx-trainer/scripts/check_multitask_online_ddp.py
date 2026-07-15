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
_STAGE_ORDER = ("stage1", "stage2", "stage3")


def _parse_stages(value: str) -> list[str]:
    requested = [stage.strip().lower() for stage in value.split(",") if stage.strip()]
    invalid = sorted(set(requested) - set(_STAGE_ORDER))
    if not requested or invalid:
        raise typer.BadParameter(
            "--stages must be a comma-separated subset of stage1,stage2,stage3"
        )
    if len(requested) != len(set(requested)):
        raise typer.BadParameter("--stages must not contain duplicates")
    return [stage for stage in _STAGE_ORDER if stage in requested]


@app.command()
def main(
    stages: str = typer.Option("stage1,stage2,stage3", "--stages"),
    stage1_config: str | None = typer.Option(None, "--stage1-config"),
    stage2_config: str | None = typer.Option(None, "--stage2-config"),
    stage3_config: str | None = typer.Option(None, "--stage3-config"),
    stage2_init_checkpoint: str | None = typer.Option(None, "--stage2-init-checkpoint"),
    stage3_init_checkpoint: str | None = typer.Option(None, "--stage3-init-checkpoint"),
    chain_smoke_checkpoints: bool = typer.Option(False, "--chain-smoke-checkpoints"),
    output_root: str = typer.Option(
        "/mnt/workspace/litengjie/jd_ltx_multitask_online_480p121/ddp_smoke",
        "--output-root",
    ),
) -> None:
    selected_stages = _parse_stages(stages)
    configs = {
        "stage1": stage1_config,
        "stage2": stage2_config,
        "stage3": stage3_config,
    }
    missing_configs = [stage for stage in selected_stages if configs[stage] is None]
    if missing_configs:
        raise typer.BadParameter(f"Missing config options for selected stages: {missing_configs}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size != 2:
        raise RuntimeError(f"DDP smoke requires exactly two processes, got {world_size}")
    launcher_pid = os.getppid()
    run_root = Path(output_root) / f"run_{launcher_pid}"
    explicit_initializers = {
        "stage2": stage2_init_checkpoint,
        "stage3": stage3_init_checkpoint,
    }
    for task in ("i2i", "r2v"):
        checkpoints_by_stage: dict[str, str] = {}
        for stage in selected_stages:
            init_checkpoint = explicit_initializers.get(stage)
            if chain_smoke_checkpoints and stage == "stage2" and "stage1" in checkpoints_by_stage:
                init_checkpoint = checkpoints_by_stage["stage1"]
            if chain_smoke_checkpoints and stage == "stage3" and "stage2" in checkpoints_by_stage:
                init_checkpoint = checkpoints_by_stage["stage2"]
            output_dir = run_root / stage / task
            report = run_one_step_training_smoke(
                configs[stage],
                task=task,
                output_dir=output_dir,
                init_checkpoint=init_checkpoint,
            )
            checkpoints_by_stage[stage] = str(report["checkpoint"])
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if rank == 0:
        typer.echo(
            f"Two-rank online DDP smoke completed for stages={selected_stages}; output={run_root}"
        )


if __name__ == "__main__":
    app()
