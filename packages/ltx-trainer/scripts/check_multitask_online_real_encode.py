"""Run real VAE/Gemma/SigLIP online encoding for minimal I2I/R2V samples."""

from __future__ import annotations

import typer

from ltx_trainer.online_data.smoke import run_real_encode_check

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


@app.command()
def main(
    stage1_config: str = typer.Option(..., "--stage1-config"),
    num_image_samples: int = typer.Option(1, "--num-image-samples", min=1),
    num_video_samples: int = typer.Option(1, "--num-video-samples", min=1),
) -> None:
    run_real_encode_check(
        stage1_config,
        num_image_samples=num_image_samples,
        num_video_samples=num_video_samples,
    )


if __name__ == "__main__":
    app()
