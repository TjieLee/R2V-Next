"""Inspect I2I/OpenS2V annotation schemas without encoding media tensors."""

from __future__ import annotations

import json

import typer

from ltx_trainer.online_data.manifest import inspect_annotation
from ltx_trainer.online_data.path_safety import assert_write_path_allowed

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


@app.command()
def main(
    i2i_ann: str = typer.Option(..., "--i2i-ann"),
    r2v_ann: str = typer.Option(..., "--r2v-ann"),
    output: str = typer.Option(..., "--output"),
) -> None:
    output_path = assert_write_path_allowed(output)
    report = {
        "i2i": inspect_annotation(i2i_ann),
        "r2v": inspect_annotation(r2v_ann),
        "adapter_contract": {
            "i2i": "Pass explicit target/reference/caption fields to the manifest builder after inspection.",
            "r2v": ["video_path", "text", "crop", "face_cut", "ref_images"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"Wrote source schema report to {output_path}")


if __name__ == "__main__":
    app()
