"""Strict-no-GT bridge from raw references to semantic-flow conditions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from ltx_trainer.online_data.transforms import deterministic_resize_center_crop
from ltx_trainer.online_inference.media_identity import (
    RawReferenceLoadError,
    TargetReferenceAliasError,
    assert_references_do_not_alias_target,
)
from ltx_trainer.online_inference.semantic_guidance import SemanticGuidanceConfig

if TYPE_CHECKING:
    from ltx_trainer.online_data.online_batch_encoder import OnlineBatchEncoder


@dataclass(frozen=True)
class RawReferenceInputs:
    reference_pixels_vae: list[Tensor]
    reference_images_vlm: list[Tensor]
    reference_paths: list[Path]


def validate_selected_geometry(sample: dict[str, Any]) -> None:
    task = str(sample.get("task"))
    if task == IMAGE_TASK:
        expected = (TARGET_WIDTH, TARGET_HEIGHT, IMAGE_NUM_FRAMES, IMAGE_FPS)
    elif task == VIDEO_TASK:
        expected = (TARGET_WIDTH, TARGET_HEIGHT, VIDEO_NUM_FRAMES, VIDEO_FPS)
    else:
        raise RawReferenceLoadError(f"Unsupported selected task {task!r}")
    actual = (
        int(sample.get("width", -1)),
        int(sample.get("height", -1)),
        int(sample.get("num_frames", -1)),
        float(sample.get("fps", -1.0)),
    )
    if actual != expected:
        raise RawReferenceLoadError(
            f"Selected {task} geometry must be {expected}, got {actual}"
        )


def load_reference_inputs(
    sample: dict[str, Any],
    *,
    vlm_reference_preprocess: str,
    chunk_frames: int = 4,
) -> RawReferenceInputs:
    """Decode references only. This function intentionally has no target-path argument."""
    validate_selected_geometry(sample)
    paths = [Path(value).expanduser().absolute() for value in sample.get("reference_paths", [])]
    if not 1 <= len(paths) <= 4:
        raise RawReferenceLoadError(
            f"Strict-no-GT inference requires 1..4 references, got {len(paths)}"
        )
    target_path = sample.get("target_path")
    if not target_path:
        raise RawReferenceLoadError("Strict-no-GT alias validation requires target_path metadata")
    # This metadata-only identity check runs before any reference pixels are opened.
    assert_references_do_not_alias_target(paths, str(target_path))
    originals: list[Tensor] = []
    for path in paths:
        try:
            originals.append(decode_image_rgb(path))
        except (OSError, ValueError, RuntimeError) as exc:
            raise RawReferenceLoadError(
                f"Could not decode reference image {path}: {exc}"
            ) from exc
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
        raise RawReferenceLoadError(
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
    *,
    guidance: SemanticGuidanceConfig | None = None,
    negative_prompt: str | None = None,
) -> dict[str, Any]:
    """Create online semantic-flow conditions without touching target-derived tensors."""
    guidance = guidance or SemanticGuidanceConfig(
        guidance_scale=1.0,
        ref_guidance_scale=0.0,
        guidance_rescale=0.0,
    )
    references = load_reference_inputs(
        sample,
        vlm_reference_preprocess=encoder.config.vlm_reference_preprocess,
        chunk_frames=encoder.config.cpu_transform_chunk_frames,
    )
    conditions = encoder.encode_inference_guidance_bundle_from_references(
        task=str(sample["task"]),
        positive_prompt=str(sample["caption"]),
        negative_prompt=negative_prompt,
        need_negative=guidance.need_negative,
        need_no_reference=guidance.need_reference,
        reference_pixels_vae=references.reference_pixels_vae,
        reference_images_vlm=references.reference_images_vlm,
        width=int(sample["width"]),
        height=int(sample["height"]),
        num_frames=int(sample["num_frames"]),
        fps=float(sample["fps"]),
    )
    conditions["conditions"] = conditions["positive_conditions"]
    conditions["reference_metadata"] = {
        **conditions["reference_metadata"],
        "reference_paths": [str(path) for path in references.reference_paths],
        "reference_paths_used": [str(path) for path in references.reference_paths],
        "original_reference_paths": list(
            sample.get("original_reference_paths", sample.get("reference_paths", []))
        ),
        "reference_export_modes": list(sample.get("reference_export_modes", [])),
    }
    conditions["strict_no_gt_checks"] = {
        "reference_target_alias_check": "passed",
        "target_path_passed_to_condition_encoder": False,
        "target_path_passed_to_denoiser": False,
        "uses_target_latents": False,
        "uses_gt_teacher_evidence": False,
    }
    return conditions


def encode_external_reference_only_conditions(
    encoder: OnlineBatchEncoder,
    sample: dict[str, Any],
    *,
    guidance: SemanticGuidanceConfig | None = None,
    negative_prompt: str | None = None,
) -> dict[str, Any]:
    """Encode an external benchmark sample that has references and no target by design."""
    guidance = guidance or SemanticGuidanceConfig(
        guidance_scale=1.0,
        ref_guidance_scale=0.0,
        guidance_rescale=0.0,
    )
    validate_selected_geometry(sample)
    paths = [
        Path(value).expanduser().resolve()
        for value in sample.get("reference_paths", [])
    ]
    if not 1 <= len(paths) <= 4:
        raise RawReferenceLoadError(
            f"External reference-only inference requires 1..4 references, got {len(paths)}"
        )
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
            chunk_frames=encoder.config.cpu_transform_chunk_frames,
        )[0]
        for image in originals
    ]
    if encoder.config.vlm_reference_preprocess == "original":
        vlm_references = originals
    elif encoder.config.vlm_reference_preprocess == "target_crop":
        vlm_references = vae_references
    else:
        raise RawReferenceLoadError(
            "vlm_reference_preprocess must be 'original' or 'target_crop', "
            f"got {encoder.config.vlm_reference_preprocess!r}"
        )
    conditions = encoder.encode_inference_guidance_bundle_from_references(
        task=str(sample["task"]),
        positive_prompt=str(sample["caption"]),
        negative_prompt=negative_prompt,
        need_negative=guidance.need_negative,
        need_no_reference=guidance.need_reference,
        reference_pixels_vae=vae_references,
        reference_images_vlm=vlm_references,
        width=int(sample["width"]),
        height=int(sample["height"]),
        num_frames=int(sample["num_frames"]),
        fps=float(sample["fps"]),
    )
    conditions["conditions"] = conditions["positive_conditions"]
    conditions["reference_metadata"] = {
        **conditions["reference_metadata"],
        "reference_paths": [str(path) for path in paths],
        "reference_paths_used": [str(path) for path in paths],
        "original_reference_paths": list(
            sample.get("original_reference_paths", sample.get("reference_paths", []))
        ),
    }
    conditions["strict_no_gt_checks"] = {
        "strict_no_gt": True,
        "has_target": False,
        "reference_target_alias_check": "not_applicable_no_target",
        "target_open_count": 0,
        "target_path_passed_to_condition_encoder": False,
        "target_path_passed_to_denoiser": False,
        "uses_target_latents": False,
        "uses_gt_teacher_evidence": False,
    }
    return conditions


__all__ = [
    "RawReferenceInputs",
    "RawReferenceLoadError",
    "TargetReferenceAliasError",
    "encode_external_reference_only_conditions",
    "encode_selected_sample_conditions",
    "load_reference_inputs",
    "validate_selected_geometry",
]
