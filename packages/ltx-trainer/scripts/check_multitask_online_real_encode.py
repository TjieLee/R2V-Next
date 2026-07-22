"""Run real frozen VAE/Gemma online encoding for minimal I2I/R2V samples."""

from __future__ import annotations

import typer

from ltx_trainer.online_data.smoke import run_real_encode_check

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


@app.command()
def main(
    config: str = typer.Option(..., "--config"),
    num_image_samples: int = typer.Option(1, "--num-image-samples", min=0),
    num_video_samples: int = typer.Option(1, "--num-video-samples", min=0),
) -> None:
    if num_image_samples + num_video_samples <= 0:
        raise typer.BadParameter("At least one image or video sample must be requested")
    run_real_encode_check(
        config,
        num_image_samples=num_image_samples,
        num_video_samples=num_video_samples,
    )


if __name__ == "__main__":
    app()
