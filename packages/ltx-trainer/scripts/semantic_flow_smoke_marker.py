#!/usr/bin/env python3
"""Write or validate a semantic-flow smoke success marker."""

from __future__ import annotations

import json

import typer

from ltx_trainer.online_inference.smoke_marker import (
    validate_semantic_flow_smoke_marker,
    write_semantic_flow_smoke_marker,
)

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


@app.command()
def write(
    marker: str = typer.Option(..., "--marker"),
    code_commit: str = typer.Option(..., "--code-commit"),
    training_config: str = typer.Option(..., "--training-config"),
    accelerate_config: str = typer.Option(..., "--accelerate-config"),
    i2i_checkpoint: str = typer.Option(..., "--i2i-checkpoint"),
    r2v_checkpoint: str = typer.Option(..., "--r2v-checkpoint"),
    runtime_audit: str = typer.Option(..., "--runtime-audit"),
    runtime_lock: str = typer.Option(..., "--runtime-lock"),
    inference_summary: str = typer.Option(..., "--inference-summary"),
) -> None:
    payload = write_semantic_flow_smoke_marker(
        marker,
        code_commit=code_commit,
        training_config_path=training_config,
        accelerate_config_path=accelerate_config,
        i2i_checkpoint_path=i2i_checkpoint,
        r2v_checkpoint_path=r2v_checkpoint,
        runtime_audit_path=runtime_audit,
        runtime_lock_path=runtime_lock,
        inference_summary_path=inference_summary,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def validate(
    marker: str = typer.Option(..., "--marker"),
    code_commit: str = typer.Option(..., "--code-commit"),
    training_config: str = typer.Option(..., "--training-config"),
    accelerate_config: str = typer.Option(..., "--accelerate-config"),
) -> None:
    payload = validate_semantic_flow_smoke_marker(
        marker,
        code_commit=code_commit,
        training_config_path=training_config,
        accelerate_config_path=accelerate_config,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
