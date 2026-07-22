"""Deterministic, resume-stable transforms for online image and video samples."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


@dataclass(frozen=True)
class OnlineAugmentationConfig:
    enabled: bool = True
    target_crop_scale: tuple[float, float] = (0.90, 1.00)
    target_flip_probability: float = 0.50
    target_color_jitter_probability: float = 0.20
    target_color_jitter_strength: float = 0.10
    reference_crop_scale: tuple[float, float] = (0.80, 1.00)
    reference_flip_probability: float = 0.50
    reference_color_jitter_probability: float = 0.30
    reference_color_jitter_strength: float = 0.10
    reference_blur_probability: float = 0.05
    reference_jpeg_probability: float = 0.10
    reference_erasing_probability: float = 0.05

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "OnlineAugmentationConfig":
        if value is None:
            return cls()
        target_value = value.get("target", {})
        reference_value = value.get("reference", {})
        if not isinstance(target_value, Mapping) or not isinstance(reference_value, Mapping):
            raise ValueError("online_augmentation target/reference sections must be mappings")

        def scale(section: Mapping[str, object], key: str, default: tuple[float, float]) -> tuple[float, float]:
            raw = section.get(key, default)
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ValueError(f"online_augmentation {key} must contain two values")
            return float(raw[0]), float(raw[1])

        config = cls(
            enabled=bool(value.get("enabled", True)),
            target_crop_scale=scale(target_value, "random_resized_crop_scale", cls.target_crop_scale),
            target_flip_probability=float(
                target_value.get("horizontal_flip_probability", cls.target_flip_probability)
            ),
            target_color_jitter_probability=float(
                target_value.get("color_jitter_probability", cls.target_color_jitter_probability)
            ),
            target_color_jitter_strength=float(
                target_value.get("color_jitter_strength", cls.target_color_jitter_strength)
            ),
            reference_crop_scale=scale(
                reference_value, "random_resized_crop_scale", cls.reference_crop_scale
            ),
            reference_flip_probability=float(
                reference_value.get("horizontal_flip_probability", cls.reference_flip_probability)
            ),
            reference_color_jitter_probability=float(
                reference_value.get("color_jitter_probability", cls.reference_color_jitter_probability)
            ),
            reference_color_jitter_strength=float(
                reference_value.get("color_jitter_strength", cls.reference_color_jitter_strength)
            ),
            reference_blur_probability=float(reference_value.get("blur_probability", cls.reference_blur_probability)),
            reference_jpeg_probability=float(reference_value.get("jpeg_probability", cls.reference_jpeg_probability)),
            reference_erasing_probability=float(
                reference_value.get("random_erasing_probability", cls.reference_erasing_probability)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for name, bounds in (
            ("target_crop_scale", self.target_crop_scale),
            ("reference_crop_scale", self.reference_crop_scale),
        ):
            if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1] <= 1:
                raise ValueError(f"{name} must satisfy 0 < min <= max <= 1, got {bounds}")
        probabilities = {
            "target_flip_probability": self.target_flip_probability,
            "target_color_jitter_probability": self.target_color_jitter_probability,
            "reference_flip_probability": self.reference_flip_probability,
            "reference_color_jitter_probability": self.reference_color_jitter_probability,
            "reference_blur_probability": self.reference_blur_probability,
            "reference_jpeg_probability": self.reference_jpeg_probability,
            "reference_erasing_probability": self.reference_erasing_probability,
        }
        for name, value in probabilities.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1], got {value}")
        for name, value in (
            ("target_color_jitter_strength", self.target_color_jitter_strength),
            ("reference_color_jitter_strength", self.reference_color_jitter_strength),
        ):
            if not 0 <= value <= 0.1:
                raise ValueError(f"{name} must be in [0,0.1], got {value}")


def augmentation_seed(
    *,
    global_seed: int,
    optimizer_step: int,
    microstep: int,
    sample_key: str,
    reference_index: int = -1,
) -> int:
    """Derive a CPU generator seed without consulting process or worker RNG state."""
    payload = (
        f"{int(global_seed)}:{int(optimizer_step)}:{int(microstep)}:"
        f"{sample_key}:{int(reference_index)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def normalize_crop_xyxy(
    crop: Sequence[int | float] | None,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if crop is None:
        return 0, 0, width, height
    if len(crop) != 4:
        raise ValueError(f"crop_xyxy must contain four values, got {crop}")
    x0, y0, x1, y1 = (int(round(float(value))) for value in crop)
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


def _validate_frames(frames: Tensor, *, chunk_frames: int) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be [T,H,W,3], got {tuple(frames.shape)}")
    if frames.dtype != torch.uint8:
        raise ValueError(f"frames must be uint8, got {frames.dtype}")
    if not 1 <= chunk_frames <= 16:
        raise ValueError(f"chunk_frames must be in [1,16], got {chunk_frames}")


def _resize_crop_chunks(
    frames: Tensor,
    *,
    crop: tuple[int, int, int, int],
    target_height: int,
    target_width: int,
    flip: bool,
    brightness: float,
    contrast: float,
    saturation: float,
    chunk_frames: int,
) -> Tensor:
    x0, y0, x1, y1 = crop
    output = torch.empty(
        (frames.shape[0], target_height, target_width, 3),
        dtype=torch.uint8,
        device=frames.device,
    )
    for start in range(0, frames.shape[0], chunk_frames):
        stop = min(start + chunk_frames, frames.shape[0])
        chunk = frames[start:stop, y0:y1, x0:x1].permute(0, 3, 1, 2).float() / 255.0
        chunk = F.interpolate(
            chunk,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        if flip:
            chunk = chunk.flip(-1)
        if brightness != 1:
            chunk = chunk * brightness
        if contrast != 1:
            mean = chunk.mean(dim=(-2, -1), keepdim=True)
            chunk = (chunk - mean) * contrast + mean
        if saturation != 1:
            gray = chunk.mean(dim=1, keepdim=True)
            chunk = (chunk - gray) * saturation + gray
        output[start:stop].copy_(
            chunk.mul_(255).round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        )
    return output


def deterministic_resize_center_crop(
    frames: Tensor,
    *,
    target_height: int,
    target_width: int,
    crop_xyxy: Sequence[int | float] | None = None,
    chunk_frames: int = 4,
) -> Tensor:
    """Transform uint8 ``[T,H,W,C]`` frames with bounded float-memory use."""
    _validate_frames(frames, chunk_frames=chunk_frames)
    _, source_height, source_width, _ = frames.shape
    x0, y0, x1, y1 = normalize_crop_xyxy(crop_xyxy, width=source_width, height=source_height)
    crop_height, crop_width = y1 - y0, x1 - x0
    target_ratio = target_width / target_height
    crop_ratio = crop_width / crop_height
    if crop_ratio > target_ratio:
        adjusted_width = max(1, round(crop_height * target_ratio))
        x0 += (crop_width - adjusted_width) // 2
        x1 = x0 + adjusted_width
    elif crop_ratio < target_ratio:
        adjusted_height = max(1, round(crop_width / target_ratio))
        y0 += (crop_height - adjusted_height) // 2
        y1 = y0 + adjusted_height
    return _resize_crop_chunks(
        frames,
        crop=(x0, y0, x1, y1),
        target_height=target_height,
        target_width=target_width,
        flip=False,
        brightness=1,
        contrast=1,
        saturation=1,
        chunk_frames=chunk_frames,
    )


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator).item())


def _bernoulli(generator: torch.Generator, probability: float) -> bool:
    return float(torch.rand((), generator=generator).item()) < probability


def _random_resized_crop(
    *,
    height: int,
    width: int,
    target_ratio: float,
    scale_bounds: tuple[float, float],
    generator: torch.Generator,
) -> tuple[int, int, int, int]:
    scale = _uniform(generator, scale_bounds[0], scale_bounds[1])
    crop_area = max(1.0, height * width * scale)
    crop_width = min(width, max(1, round((crop_area * target_ratio) ** 0.5)))
    crop_height = min(height, max(1, round(crop_width / target_ratio)))
    if crop_height > height:
        crop_height = height
        crop_width = min(width, max(1, round(crop_height * target_ratio)))
    max_top = height - crop_height
    max_left = width - crop_width
    top = int(torch.randint(max_top + 1, (), generator=generator).item()) if max_top else 0
    left = int(torch.randint(max_left + 1, (), generator=generator).item()) if max_left else 0
    return left, top, left + crop_width, top + crop_height


def _jitter_parameters(
    generator: torch.Generator,
    *,
    probability: float,
    strength: float,
) -> tuple[float, float, float]:
    if not _bernoulli(generator, probability):
        return 1, 1, 1
    return tuple(_uniform(generator, 1 - strength, 1 + strength) for _ in range(3))  # type: ignore[return-value]


def augment_target_frames(
    frames: Tensor,
    *,
    seed: int,
    config: OnlineAugmentationConfig,
    target_height: int,
    target_width: int,
    chunk_frames: int = 4,
) -> Tensor:
    """Apply one shared spatial/color draw to every frame in a target clip."""
    _validate_frames(frames, chunk_frames=chunk_frames)
    config.validate()
    if not config.enabled:
        return frames.clone()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    crop = _random_resized_crop(
        height=int(frames.shape[1]),
        width=int(frames.shape[2]),
        target_ratio=target_width / target_height,
        scale_bounds=config.target_crop_scale,
        generator=generator,
    )
    brightness, contrast, saturation = _jitter_parameters(
        generator,
        probability=config.target_color_jitter_probability,
        strength=config.target_color_jitter_strength,
    )
    return _resize_crop_chunks(
        frames,
        crop=crop,
        target_height=target_height,
        target_width=target_width,
        flip=_bernoulli(generator, config.target_flip_probability),
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        chunk_frames=chunk_frames,
    )


def _gaussian_blur(image: Tensor) -> Tensor:
    kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=image.device)
    kernel = (kernel[:, None] * kernel[None, :]) / 256.0
    weight = kernel.expand(3, 1, 5, 5)
    chw = image.permute(2, 0, 1).float().unsqueeze(0)
    blurred = F.conv2d(F.pad(chw, (2, 2, 2, 2), mode="reflect"), weight, groups=3)
    return blurred[0].round_().clamp_(0, 255).to(torch.uint8).permute(1, 2, 0)


def _jpeg_degrade(image: Tensor, *, quality: int) -> Tensor:
    pil_image = Image.fromarray(image.cpu().numpy(), mode="RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        raw = bytearray(decoded.convert("RGB").tobytes())
        restored = torch.frombuffer(raw, dtype=torch.uint8).clone().reshape(image.shape)
    return restored.to(image.device)


def _random_erasing(image: Tensor, *, generator: torch.Generator) -> Tensor:
    height, width, _ = image.shape
    area_ratio = _uniform(generator, 0.02, 0.08)
    aspect = _uniform(generator, 0.5, 2.0)
    erase_height = min(height, max(1, round((height * width * area_ratio / aspect) ** 0.5)))
    erase_width = min(width, max(1, round(erase_height * aspect)))
    top = int(torch.randint(height - erase_height + 1, (), generator=generator).item())
    left = int(torch.randint(width - erase_width + 1, (), generator=generator).item())
    output = image.clone()
    fill = image.float().mean(dim=(0, 1)).round().to(torch.uint8)
    output[top : top + erase_height, left : left + erase_width] = fill
    return output


def augment_reference_image(
    image: Tensor,
    *,
    seed: int,
    config: OnlineAugmentationConfig,
    target_height: int,
    target_width: int,
) -> Tensor:
    """Independently augment one reference and return the shared VLM/VAE pixels."""
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != torch.uint8:
        raise ValueError(f"reference must be uint8 [H,W,3], got {tuple(image.shape)} {image.dtype}")
    config.validate()
    if not config.enabled:
        return deterministic_resize_center_crop(
            image.unsqueeze(0),
            target_height=target_height,
            target_width=target_width,
            chunk_frames=1,
        )[0]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    crop = _random_resized_crop(
        height=int(image.shape[0]),
        width=int(image.shape[1]),
        target_ratio=target_width / target_height,
        scale_bounds=config.reference_crop_scale,
        generator=generator,
    )
    brightness, contrast, saturation = _jitter_parameters(
        generator,
        probability=config.reference_color_jitter_probability,
        strength=config.reference_color_jitter_strength,
    )
    output = _resize_crop_chunks(
        image.unsqueeze(0),
        crop=crop,
        target_height=target_height,
        target_width=target_width,
        flip=_bernoulli(generator, config.reference_flip_probability),
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        chunk_frames=1,
    )[0]
    if _bernoulli(generator, config.reference_blur_probability):
        output = _gaussian_blur(output)
    if _bernoulli(generator, config.reference_jpeg_probability):
        quality = int(torch.randint(65, 96, (), generator=generator).item())
        output = _jpeg_degrade(output, quality=quality)
    if _bernoulli(generator, config.reference_erasing_probability):
        output = _random_erasing(output, generator=generator)
    return output
