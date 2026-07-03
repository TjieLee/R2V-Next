"""Multi-reference image/video conditioning strategy for video-only training.

This strategy is the Stage 1 renderer path from the project requirements:
clean reference latents are prepended to noisy target video latents, reference
tokens receive timestep 0 and no loss, target tokens keep the normal flow
matching target, and reference entities are separated by negative temporal RoPE
offsets.
"""

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_core.multicond.rope_mask_builder import build_multiref_sequence
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class MultiReferenceVideoConfig(TrainingStrategyConfigBase):
    """Stage 1 multi-reference video-only conditioning."""

    name: Literal["multi_reference_video"] = "multi_reference_video"

    first_frame_conditioning_p: float = Field(
        default=0.0,
        description="Optional probability of making the target first frame clean and no-loss.",
        ge=0.0,
        le=1.0,
    )

    reference_latents_dir: str = Field(
        default="multi_reference_latents",
        description="Directory containing per-sample stacked reference latents.",
    )

    max_ref_images_per_sample: int | None = Field(
        default=None,
        description="Optional cap applied after loading. None keeps all references in each sample.",
        ge=1,
    )

    reference_time_stride: float = Field(
        default=1.0,
        description="Negative temporal RoPE spacing between reference entities.",
        gt=0.0,
    )

    def get_data_sources(self) -> dict[str, str]:
        return {
            "latents": "latents",
            "conditions": "conditions",
            self.reference_latents_dir: "multi_ref_latents",
        }


class MultiReferenceVideoStrategy(TrainingStrategy):
    """Video-only multi-reference strategy with target-only flow matching loss."""

    config: MultiReferenceVideoConfig

    def __init__(self, config: MultiReferenceVideoConfig):
        super().__init__(config)
        self.reference_spatial_scale_factor: int | None = None

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        latents = batch["latents"]
        target_latents = latents["latents"]

        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        ref_data = batch["multi_ref_latents"]
        ref_latents = self._normalize_reference_latents(ref_data["latents"])
        ref_valid_mask = self._get_reference_valid_mask(ref_data, ref_latents)

        if self.config.max_ref_images_per_sample is not None:
            max_refs = self.config.max_ref_images_per_sample
            ref_latents = ref_latents[:, :max_refs]
            ref_valid_mask = ref_valid_mask[:, :max_refs]

        batch_size, num_refs, _channels, ref_frames, ref_height, ref_width = ref_latents.shape
        device = target_latents.device

        target_tokens = self._video_patchifier.patchify(target_latents)
        ref_tokens = self._video_patchifier.patchify(ref_latents.reshape(batch_size * num_refs, *ref_latents.shape[2:]))
        ref_tokens = ref_tokens.reshape(batch_size, num_refs, ref_tokens.shape[1], ref_tokens.shape[2])

        target_seq_len = target_tokens.shape[1]

        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        sigmas = timestep_sampler.sample_for(target_tokens)
        noise = torch.randn_like(target_tokens)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_target = (1 - sigmas_expanded) * target_tokens + sigmas_expanded * noise

        target_conditioning_mask_expanded = target_conditioning_mask.unsqueeze(-1)
        noisy_target = torch.where(target_conditioning_mask_expanded, target_tokens, noisy_target)

        targets = noise - target_tokens
        target_timesteps = self._create_per_token_timesteps(target_conditioning_mask, sigmas.squeeze())
        target_loss_mask = ~target_conditioning_mask

        target_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
        )

        ref_fps = self._first_scalar(ref_data.get("fps"), default=1.0)
        ref_positions = self._get_video_positions(
            num_frames=ref_frames,
            height=ref_height,
            width=ref_width,
            batch_size=batch_size * num_refs,
            fps=ref_fps,
            device=device,
        )
        ref_positions = ref_positions.reshape(batch_size, num_refs, *ref_positions.shape[1:])
        ref_positions = self._scale_reference_positions(ref_positions, height, width, ref_height, ref_width)

        packed = build_multiref_sequence(
            ref_tokens=ref_tokens,
            ref_positions=ref_positions,
            ref_valid_mask=ref_valid_mask,
            target_tokens=noisy_target,
            target_positions=target_positions,
            target_timesteps=target_timesteps,
            target_loss_mask=target_loss_mask,
            reference_time_stride=self.config.reference_time_stride,
        )

        conditions = batch["conditions"]
        prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        video_modality = Modality(
            enabled=True,
            latent=packed.latents,
            sigma=sigmas,
            timesteps=packed.timesteps,
            positions=packed.positions,
            context=prompt_embeds,
            context_mask=prompt_attention_mask,
            attention_mask=packed.attention_mask,
        )

        return ModelInputs(
            video=video_modality,
            audio=None,
            video_targets=targets,
            audio_targets=None,
            video_loss_mask=packed.loss_mask,
            audio_loss_mask=None,
        )

    def compute_loss(
        self,
        video_pred: Tensor,
        _audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        target_len = inputs.video_targets.shape[1]
        target_pred = video_pred[:, -target_len:, :]
        target_loss_mask = inputs.video_loss_mask[:, -target_len:]

        loss = (target_pred - inputs.video_targets).pow(2)
        loss_mask = target_loss_mask.unsqueeze(-1).float()
        masked = loss.mul(loss_mask)
        return masked.mean(dim=[-2, -1]) / loss_mask.mean(dim=[-2, -1]).clamp(min=1e-8)

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "conditioning": "multi_reference_video",
            "reference_time_stride": self.config.reference_time_stride,
        }
        if self.reference_spatial_scale_factor is not None:
            metadata["reference_spatial_scale_factor"] = self.reference_spatial_scale_factor
        return metadata

    @staticmethod
    def _normalize_reference_latents(latents: Tensor) -> Tensor:
        if latents.ndim == 5:
            return latents.unsqueeze(1)
        if latents.ndim == 6:
            return latents
        raise ValueError(f"Reference latents must be [B,C,F,H,W] or [B,R,C,F,H,W], got {tuple(latents.shape)}")

    @staticmethod
    def _get_reference_valid_mask(ref_data: dict[str, Any], ref_latents: Tensor) -> Tensor:
        batch_size, num_refs = ref_latents.shape[:2]
        device = ref_latents.device

        mask = ref_data.get("ref_valid_mask")
        if mask is not None:
            return mask.to(device=device, dtype=torch.bool)

        num_refs_tensor = ref_data.get("num_refs")
        if num_refs_tensor is not None:
            arange = torch.arange(num_refs, device=device).unsqueeze(0)
            return arange < num_refs_tensor.to(device=device).view(batch_size, 1)

        return torch.ones(batch_size, num_refs, dtype=torch.bool, device=device)

    @staticmethod
    def _first_scalar(value: Any, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, Tensor):
            return value.flatten()[0].item()
        return float(value)

    def _scale_reference_positions(
        self,
        ref_positions: Tensor,
        target_height: int,
        target_width: int,
        ref_height: int,
        ref_width: int,
    ) -> Tensor:
        if target_height == ref_height and target_width == ref_width:
            if self.reference_spatial_scale_factor is None:
                self.reference_spatial_scale_factor = 1
            return ref_positions

        if target_height % ref_height != 0 or target_width % ref_width != 0:
            raise ValueError(
                f"Target latent size ({target_height}x{target_width}) must be an integer multiple of "
                f"reference latent size ({ref_height}x{ref_width})"
            )

        scale_h = target_height // ref_height
        scale_w = target_width // ref_width
        if scale_h != scale_w:
            raise ValueError(f"Reference spatial scale must be uniform, got h={scale_h}, w={scale_w}")

        if self.reference_spatial_scale_factor is None:
            self.reference_spatial_scale_factor = scale_h
        elif self.reference_spatial_scale_factor != scale_h:
            raise ValueError(
                f"Inconsistent reference scale factor: expected {self.reference_spatial_scale_factor}, got {scale_h}"
            )

        ref_positions = ref_positions.clone()
        ref_positions[:, :, 1, ...] *= scale_h
        ref_positions[:, :, 2, ...] *= scale_w
        return ref_positions
