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
from torch import Tensor, nn

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

    gt_visual_tokens_dir: str | None = Field(
        default="gt_siglip_tokens",
        description=(
            "Directory containing frozen target-video SigLIP/projector visual tokens. Set to null to disable the "
            "Stage 1 text-condition expansion path."
        ),
    )

    visual_token_key: str = Field(
        default="visual_tokens",
        description="Tensor key in gt_visual_tokens_dir files. Shape per sample: [K, D].",
    )

    visual_token_mask_key: str = Field(
        default="visual_token_mask",
        description="Optional bool mask key in gt_visual_tokens_dir files. Shape per sample: [K].",
    )

    visual_tokens_per_frame_key: str = Field(
        default="tokens_per_frame",
        description="Optional scalar key for target-video SigLIP tokens per sampled frame.",
    )

    visual_token_frame_stride: int = Field(
        default=1,
        description=(
            "Optional post-encoding temporal downsampling for target-video SigLIP tokens. "
            "For example, stride=2 uses every other sampled frame without rerunning SigLIP."
        ),
        ge=1,
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
        data_sources = {
            "latents": "latents",
            "conditions": "conditions",
            self.reference_latents_dir: "multi_ref_latents",
        }
        if self.gt_visual_tokens_dir is not None:
            data_sources[self.gt_visual_tokens_dir] = "gt_visual_tokens"
        return data_sources


class MultiReferenceVideoStrategy(TrainingStrategy):
    """Video-only multi-reference strategy with target-only flow matching loss."""

    config: MultiReferenceVideoConfig

    def __init__(self, config: MultiReferenceVideoConfig):
        super().__init__(config)
        self.reference_spatial_scale_factor: int | None = None
        self._connector_register_count: int | None = None

    def attach_models(
        self,
        *,
        transformer: nn.Module,
        embeddings_processor: nn.Module,
        text_encoder: nn.Module | None = None,
    ) -> None:
        del transformer, text_encoder
        video_connector = embeddings_processor.video_connector
        self._connector_register_count = getattr(video_connector, "num_learnable_registers", None)

    def prepare_conditions(self, batch: dict[str, Any], conditions: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.gt_visual_tokens_dir is None or "gt_visual_tokens" not in batch:
            return conditions
        visual_tokens, visual_mask = self._load_condition_visual_tokens(batch["gt_visual_tokens"], conditions)
        return self._append_visual_tokens_to_conditions(conditions, visual_tokens, visual_mask)

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

    def _load_condition_visual_tokens(
        self,
        visual_data: dict[str, Any],
        conditions: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        video_feature_key = "video_prompt_embeds" if "video_prompt_embeds" in conditions else "prompt_embeds"
        video_features = conditions[video_feature_key]
        tokens = visual_data[self.config.visual_token_key].to(device=video_features.device, dtype=video_features.dtype)
        if tokens.ndim != 3:
            raise ValueError(f"GT visual tokens must be [B,K,D], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != video_features.shape[-1]:
            raise ValueError(
                f"GT visual token dim {tokens.shape[-1]} does not match prompt feature dim {video_features.shape[-1]}"
            )

        mask = visual_data.get(self.config.visual_token_mask_key)
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            mask = mask.to(device=tokens.device, dtype=torch.bool)
        if mask.shape != tokens.shape[:2]:
            raise ValueError(f"GT visual token mask must be [B,K], got {tuple(mask.shape)} for tokens {tuple(tokens.shape)}")
        tokens, mask = self._downsample_visual_tokens_by_frame(tokens, mask, visual_data)
        return tokens, mask

    def _downsample_visual_tokens_by_frame(
        self,
        tokens: Tensor,
        mask: Tensor,
        visual_data: dict[str, Any],
    ) -> tuple[Tensor, Tensor]:
        if self.config.visual_token_frame_stride == 1:
            return tokens, mask

        tokens_per_frame = visual_data.get(self.config.visual_tokens_per_frame_key)
        if tokens_per_frame is None:
            raise ValueError(
                "visual_token_frame_stride > 1 requires tokens_per_frame metadata in gt visual-token files."
            )
        if isinstance(tokens_per_frame, Tensor):
            if not torch.all(tokens_per_frame == tokens_per_frame.flatten()[0]):
                raise ValueError(f"Mixed tokens_per_frame in batch: {tokens_per_frame.tolist()}")
            tokens_per_frame = int(tokens_per_frame.flatten()[0].item())
        else:
            tokens_per_frame = int(tokens_per_frame)

        if tokens.shape[1] % tokens_per_frame != 0:
            raise ValueError(
                f"Visual token count {tokens.shape[1]} is not divisible by tokens_per_frame={tokens_per_frame}"
            )

        num_frames = tokens.shape[1] // tokens_per_frame
        stride = self.config.visual_token_frame_stride
        tokens = tokens.reshape(tokens.shape[0], num_frames, tokens_per_frame, tokens.shape[-1])[:, ::stride]
        mask = mask.reshape(mask.shape[0], num_frames, tokens_per_frame)[:, ::stride]
        return tokens.reshape(tokens.shape[0], -1, tokens.shape[-1]), mask.reshape(mask.shape[0], -1)

    def _append_visual_tokens_to_conditions(
        self,
        conditions: dict[str, Tensor],
        visual_tokens: Tensor,
        visual_mask: Tensor,
    ) -> dict[str, Tensor]:
        conditions = dict(conditions)
        video_feature_key = "video_prompt_embeds" if "video_prompt_embeds" in conditions else "prompt_embeds"
        video_features = conditions[video_feature_key]
        conditions[video_feature_key] = torch.cat([video_features, visual_tokens.to(dtype=video_features.dtype)], dim=1)

        audio_features = conditions.get("audio_prompt_embeds")
        if audio_features is not None:
            audio_pad = torch.zeros(
                audio_features.shape[0],
                visual_tokens.shape[1],
                audio_features.shape[-1],
                dtype=audio_features.dtype,
                device=audio_features.device,
            )
            conditions["audio_prompt_embeds"] = torch.cat([audio_features, audio_pad], dim=1)

        prompt_mask = conditions["prompt_attention_mask"].to(device=visual_tokens.device, dtype=torch.bool)
        conditions["prompt_attention_mask"] = torch.cat([prompt_mask, visual_mask], dim=1)
        return self._pad_conditions_to_connector_multiple(conditions)

    def _pad_conditions_to_connector_multiple(self, conditions: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self._connector_register_count:
            return conditions
        video_feature_key = "video_prompt_embeds" if "video_prompt_embeds" in conditions else "prompt_embeds"
        seq_len = conditions[video_feature_key].shape[1]
        remainder = seq_len % self._connector_register_count
        if remainder == 0:
            return conditions

        pad_len = self._connector_register_count - remainder
        for key in ("video_prompt_embeds", "prompt_embeds", "audio_prompt_embeds"):
            features = conditions.get(key)
            if features is None:
                continue
            pad = torch.zeros(
                features.shape[0],
                pad_len,
                features.shape[-1],
                dtype=features.dtype,
                device=features.device,
            )
            conditions[key] = torch.cat([features, pad], dim=1)

        mask = conditions["prompt_attention_mask"]
        mask_pad = torch.zeros(mask.shape[0], pad_len, dtype=mask.dtype, device=mask.device)
        conditions["prompt_attention_mask"] = torch.cat([mask, mask_pad], dim=1)
        return conditions

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
