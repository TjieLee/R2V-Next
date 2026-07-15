"""Strict-no-GT bridge from selected raw references to planner conditions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import Tensor

from ltx_trainer.online_data.constants import (
    IMAGE_FPS,
    IMAGE_NUM_FRAMES,
    IMAGE_TASK,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VIDEO_FPS,
    VIDEO_NUM_FRAMES,
    VIDEO_TASK,
)
from ltx_trainer.online_data.media_decoder import decode_image_rgb
from ltx_trainer.online_data.online_batch_encoder import OnlineBatchEncoder
from ltx_trainer.online_data.transforms import deterministic_resize_center_crop


@dataclass(frozen=True)
class RawReferenceInputs:
    reference_pixels_vae: list[Tensor]
    reference_images_vlm: list[Tensor]
    reference_paths: list[Path]


class RawReferenceLoadError(RuntimeError):
    """A sample-local reference media failure that may be skipped safely."""


def validate_selected_geometry(sample: dict[str, Any]) -> None:
    task = str(sample.get("task"))
    if task == IMAGE_TASK:
        expected = (TARGET_WIDTH, TARGET_HEIGHT, IMAGE_NUM_FRAMES, IMAGE_FPS)
    elif task == VIDEO_TASK:
        expected = (TARGET_WIDTH, TARGET_HEIGHT, VIDEO_NUM_FRAMES, VIDEO_FPS)
    else:
        raise ValueError(f"Unsupported selected task {task!r}")
    actual = (
        int(sample.get("width", -1)),
        int(sample.get("height", -1)),
        int(sample.get("num_frames", -1)),
        float(sample.get("fps", -1.0)),
    )
    if actual != expected:
        raise ValueError(f"Selected {task} geometry must be {expected}, got {actual}")


def load_reference_inputs(
    sample: dict[str, Any],
    *,
    vlm_reference_preprocess: str,
    chunk_frames: int = 4,
) -> RawReferenceInputs:
    """Decode references only. This function intentionally has no target-path argument."""
    validate_selected_geometry(sample)
    paths = [Path(value).expanduser().resolve() for value in sample.get("reference_paths", [])]
    if not 1 <= len(paths) <= 4:
        raise ValueError(f"Strict-no-GT inference requires 1..4 references, got {len(paths)}")
    originals: list[Tensor] = []
    for path in paths:
        try:
            originals.append(decode_image_rgb(path))
        except (OSError, ValueError, RuntimeError) as exc:
            raise RawReferenceLoadError(f"Could not decode reference image {path}: {exc}") from exc
    vae_references = [
        deterministic_resize_center_crop(
            image.unsqueeze(0),
            target_height=TARGET_HEIGHT,
            target_width=TARGET_WIDTH,
            chunk_frames=chunk_frames,
        )[0]
        for image in originals
    ]
    if vlm_reference_preprocess == "original":
        vlm_references = originals
    elif vlm_reference_preprocess == "target_crop":
        vlm_references = vae_references
    else:
        raise ValueError(
            "vlm_reference_preprocess must be 'original' or 'target_crop', "
            f"got {vlm_reference_preprocess!r}"
        )
    return RawReferenceInputs(
        reference_pixels_vae=vae_references,
        reference_images_vlm=vlm_references,
        reference_paths=paths,
    )


def encode_selected_sample_conditions(
    encoder: OnlineBatchEncoder,
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Create online Stage 3 conditions without touching target media or target-derived tensors."""
    references = load_reference_inputs(
        sample,
        vlm_reference_preprocess=encoder.config.vlm_reference_preprocess,
        chunk_frames=encoder.config.cpu_transform_chunk_frames,
    )
    conditions = encoder.encode_inference_conditions_from_references(
        task=str(sample["task"]),
        caption=str(sample["caption"]),
        reference_pixels_vae=references.reference_pixels_vae,
        reference_images_vlm=references.reference_images_vlm,
        width=int(sample["width"]),
        height=int(sample["height"]),
        num_frames=int(sample["num_frames"]),
        fps=float(sample["fps"]),
    )
    conditions["reference_metadata"] = {
        **conditions["reference_metadata"],
        "reference_paths": [str(path) for path in references.reference_paths],
    }
    return conditions
