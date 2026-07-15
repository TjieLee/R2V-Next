"""Run a real one-optimizer-step Stage 3 online training smoke."""

from __future__ import annotations

from typing import Literal

import typer

from ltx_trainer.online_data.smoke import run_one_step_training_smoke

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


@app.command()
def main(
    config: str = typer.Option(..., "--config"),
    task: Literal["i2i", "r2v"] = typer.Option(..., "--task"),
    output_dir: str = typer.Option(..., "--output-dir"),
) -> None:
    run_one_step_training_smoke(config, task=task, output_dir=output_dir)


if __name__ == "__main__":
    app()
