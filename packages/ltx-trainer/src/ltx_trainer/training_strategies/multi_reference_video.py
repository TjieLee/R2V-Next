"""Multi-reference image/video conditioning strategy for video-only training.

This strategy is the Stage 1 renderer path from the project requirements:
clean reference latents are prepended to noisy target video latents, reference
tokens receive timestep 0 and no loss, target tokens keep the normal flow
matching target, and reference entities are separated by negative temporal RoPE
offsets.
"""

import copy
import math
from typing import Any, Literal

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn

from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.multicond.cfg_sampler import CFGModeBatch, sample_cfg_modes
from ltx_core.multicond.rope_mask_builder import build_multiref_sequence
from ltx_core.multicond.visual_tokens import Visual3DResampler, Visual3DTokenEncoder
from ltx_core.text_encoders.gemma import convert_to_additive_mask
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

    conditions_dir: str = Field(
        default="conditions",
        description=(
            "Directory containing connector-input text/VLM condition features. "
            "Use vlm_conditions for Stage 1 text-plus-reference-image VLM context."
        ),
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

    visual_token_source_dim: int | None = Field(
        default=None,
        description=(
            "Optional source dim of raw GT SigLIP/projector visual tokens before appending to LTX condition "
            "features. Set with visual_token_target_dim to create a trainable projection, e.g. 3840 -> 4096."
        ),
        ge=1,
    )

    visual_token_target_dim: int | None = Field(
        default=None,
        description=(
            "Optional target dim of visual condition tokens after projection. Must match video_prompt_embeds / "
            "LTX connector input dim, e.g. 4096."
        ),
        ge=1,
    )

    train_text_connector: bool = Field(
        default=False,
        description=(
            "Train the LTX video text connector on top of precomputed feature-extractor outputs. "
            "When true, embeddings_processor.video_connector is optimized and checkpointed."
        ),
    )

    visual_branch_enabled: bool = Field(
        default=True,
        description="Enable independent post-connector target-video SigLIP visual branch.",
    )

    visual_context_mode: Literal["qformer_512", "full_tokens_3d_sa"] = Field(
        default="qformer_512",
        description="Visual context architecture. The default preserves legacy Q-former checkpoints.",
    )

    visual_full_sa_num_heads: int = Field(default=32, ge=1)
    visual_full_sa_depth: int = Field(default=1, ge=1)
    visual_full_sa_ffn_multiplier: float = Field(default=2.0, gt=0.0)
    visual_full_sa_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    visual_full_sa_residual_init_gain: float = Field(default=0.1, gt=0.0)
    visual_full_sa_use_middle_positions: bool = Field(default=True)
    visual_context_expected_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional strict expected visual token count. Use 2048 for 8 frames * 16 * 16 tokens."
        ),
    )

    visual_context_spatial_grid: int = Field(
        default=8,
        description=(
            "Spatial grid size for compressed visual context per sampled frame. "
            "With 8 frames and grid=8, this yields 512 visual context tokens."
        ),
        ge=1,
    )

    visual_context_frame_stride: int = Field(
        default=1,
        description="Temporal stride applied to sampled SigLIP frames before building compressed visual context.",
        ge=1,
    )

    visual_context_max_tokens: int = Field(
        default=512,
        description="Maximum query tokens reserved by the visual 3D resampler.",
        ge=1,
    )

    visual_resampler_num_heads: int = Field(default=16, ge=1)
    visual_resampler_depth: int = Field(default=1, ge=1)
    visual_resampler_ffn_multiplier: float = Field(default=4.0, gt=0.0)
    visual_resampler_dropout: float = Field(default=0.0, ge=0.0)
    visual_resampler_zero_init_output: bool = Field(default=True)
    visual_resampler_gate_init: float = Field(default=1.0e-3)

    visual_connector_enabled: bool = Field(
        default=True,
        description="Use a trainable visual connector initialized from embeddings_processor.video_connector.",
    )

    visual_gate_init: float = Field(
        default=1.0e-2,
        description="Initial scalar gate for visual_context before concatenating to DiT context.",
    )

    cfg_dropout_enabled: bool = Field(
        default=False,
        description=(
            "Enable per-sample condition dropout for factorized CFG training. "
            "Modes are mutually exclusive: full, drop_text, drop_siglip, drop_ref_latents, drop_all/null."
        ),
    )

    cfg_full_p: float = Field(
        default=0.7,
        description="Probability of keeping all conditions during CFG-dropout training.",
        ge=0.0,
    )

    cfg_drop_text_p: float = Field(
        default=0.1,
        description="Probability of zeroing text/VLM context features while keeping sequence shape.",
        ge=0.0,
    )

    cfg_drop_siglip_p: float = Field(
        default=0.15,
        description="Probability of dropping only the target-video SigLIP visual branch.",
        ge=0.0,
    )

    cfg_drop_ref_latents_p: float = Field(
        default=0.05,
        description="Probability of dropping only DiT packed reference latent tokens.",
        ge=0.0,
    )

    cfg_drop_ref_p: float = Field(
        default=0.0,
        description=(
            "Legacy alias. If nonzero in an old config that does not set cfg_drop_siglip_p or "
            "cfg_drop_ref_latents_p, it is treated as cfg_drop_siglip_p only."
        ),
        ge=0.0,
    )

    cfg_drop_all_p: float | None = Field(
        default=None,
        description=(
            "Probability of the null/drop_all branch. If unset, cfg_drop_planner_p is used as a deprecated alias."
        ),
        ge=0.0,
    )

    cfg_drop_planner_p: float = Field(
        default=0.1,
        description=(
            "Deprecated alias for cfg_drop_all_p / null branch. This is not a planner-only branch: it zeros "
            "text/VLM context, drops reference latents, and zeros appended GT/predicted visual condition tokens."
        ),
        ge=0.0,
    )

    cfg_text_conditions_dir: str | None = Field(
        default="conditions",
        description=(
            "Text-only condition directory used for drop_ref. Keep this as the Stage 0/standard "
            "conditions directory when conditions_dir points to vlm_conditions."
        ),
    )

    cfg_ref_only_conditions_dir: str | None = Field(
        default=None,
        description=(
            "Optional reference-only VLM condition directory used for drop_text. If unset, drop_text falls back "
            "to zeroing mixed text/reference VLM context because precomputed vlm_conditions cannot be separated."
        ),
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

    @model_validator(mode="after")
    def _validate_visual_token_projection_dims(self) -> "MultiReferenceVideoConfig":
        has_source = self.visual_token_source_dim is not None
        has_target = self.visual_token_target_dim is not None
        if has_source != has_target:
            raise ValueError(
                "visual_token_source_dim and visual_token_target_dim must be set together, "
                "e.g. 3840 and 4096, or both left unset to require matching dims."
            )
        if self.visual_context_mode == "full_tokens_3d_sa":
            if self.visual_connector_enabled:
                raise ValueError(
                    "visual_connector_enabled must be false when visual_context_mode='full_tokens_3d_sa'"
                )
            if self.visual_token_target_dim is None:
                raise ValueError(
                    "visual_token_target_dim must be set when visual_context_mode='full_tokens_3d_sa'"
                )
            if self.visual_token_target_dim % self.visual_full_sa_num_heads != 0:
                raise ValueError(
                    f"visual_token_target_dim={self.visual_token_target_dim} must be divisible by "
                    f"visual_full_sa_num_heads={self.visual_full_sa_num_heads}"
                )
            head_dim = self.visual_token_target_dim // self.visual_full_sa_num_heads
            if head_dim % 2 != 0:
                raise ValueError(
                    "full_tokens_3d_sa requires even head_dim for split RoPE, got "
                    f"target_dim={self.visual_token_target_dim}, num_heads={self.visual_full_sa_num_heads}, "
                    f"head_dim={head_dim}"
                )
        return self

    def get_data_sources(self) -> dict[str, str]:
        data_sources = {
            "latents": "latents",
            self.conditions_dir: "conditions",
            self.reference_latents_dir: "multi_ref_latents",
        }
        if self.gt_visual_tokens_dir is not None:
            data_sources[self.gt_visual_tokens_dir] = "gt_visual_tokens"
        if (
            self.cfg_dropout_enabled
            and self.cfg_drop_ref_p > 0
            and self.cfg_text_conditions_dir is not None
            and self.cfg_text_conditions_dir != self.conditions_dir
        ):
            data_sources[self.cfg_text_conditions_dir] = "cfg_text_conditions"
        if (
            self.cfg_dropout_enabled
            and self.cfg_drop_text_p > 0
            and self.cfg_ref_only_conditions_dir is not None
            and self.cfg_ref_only_conditions_dir != self.conditions_dir
        ):
            data_sources[self.cfg_ref_only_conditions_dir] = "cfg_ref_only_conditions"
        return data_sources


class ScalarParameterModule(nn.Module):
    """Small wrapper so a single Parameter can be optimizer/checkpoint managed."""

    def __init__(self, value: nn.Parameter):
        super().__init__()
        self.value = value

    def forward(self) -> Tensor:
        return self.value

    def get_value(self) -> Tensor:
        return self.value


class MultiReferenceVideoStrategy(TrainingStrategy):
    """Video-only multi-reference strategy with target-only flow matching loss."""

    config: MultiReferenceVideoConfig

    def __init__(self, config: MultiReferenceVideoConfig):
        super().__init__(config)
        self.reference_spatial_scale_factor: int | None = None
        self._connector_register_count: int | None = None
        self._visual_token_projection: nn.Module | None = None
        self._visual_token_source_dim: int | None = None
        self._visual_token_target_dim: int | None = None
        self._visual_resampler: Visual3DResampler | None = None
        self._visual_full_encoder: Visual3DTokenEncoder | None = None
        self._visual_connector: nn.Module | None = None
        self._visual_gate: ScalarParameterModule | None = None
        self._last_visual_context_shape: list[int] | None = None

    @staticmethod
    def _unwrap_strategy_module(module: nn.Module) -> nn.Module:
        return getattr(module, "module", module)

    def attach_models(
        self,
        *,
        transformer: nn.Module,
        embeddings_processor: nn.Module,
        text_encoder: nn.Module | None = None,
    ) -> None:
        del text_encoder
        video_connector = embeddings_processor.video_connector
        self._connector_register_count = getattr(video_connector, "num_learnable_registers", None)
        self._init_visual_token_projection(video_connector)
        self._init_visual_branch(transformer, video_connector)

    def _init_visual_branch(self, transformer: nn.Module, video_connector: nn.Module) -> None:
        self._visual_resampler = None
        self._visual_full_encoder = None
        self._visual_connector = None
        self._visual_gate = None
        if not self.config.visual_branch_enabled:
            return

        param = next(video_connector.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        dtype = param.dtype if param is not None and param.is_floating_point() else torch.float32
        connector_dim = getattr(video_connector, "inner_dim", None)
        if connector_dim is None:
            connector_dim = param.shape[-1] if param is not None and param.ndim > 0 else self.config.visual_token_target_dim
        if connector_dim is None:
            raise ValueError("Cannot infer visual branch connector dimension from embeddings_processor.video_connector")

        if self.config.visual_context_mode == "full_tokens_3d_sa":
            if int(connector_dim) != self.config.visual_token_target_dim:
                raise ValueError(
                    "full_tokens_3d_sa connector dimension must match visual_token_target_dim, got "
                    f"connector_dim={connector_dim}, visual_token_target_dim={self.config.visual_token_target_dim}"
                )
            self._visual_full_encoder = Visual3DTokenEncoder(
                dim=int(connector_dim),
                num_heads=self.config.visual_full_sa_num_heads,
                depth=self.config.visual_full_sa_depth,
                ffn_multiplier=self.config.visual_full_sa_ffn_multiplier,
                dropout=self.config.visual_full_sa_dropout,
                residual_init_gain=self.config.visual_full_sa_residual_init_gain,
                positional_embedding_theta=getattr(transformer, "positional_embedding_theta", 10000.0),
                positional_embedding_max_pos=getattr(
                    transformer,
                    "positional_embedding_max_pos",
                    [20, 2048, 2048],
                ),
                rope_type=getattr(transformer, "rope_type", LTXRopeType.SPLIT),
                use_middle_positions=self.config.visual_full_sa_use_middle_positions,
            ).to(device=device, dtype=dtype)
            return

        self._visual_resampler = Visual3DResampler(
            dim=int(connector_dim),
            max_query_tokens=self.config.visual_context_max_tokens,
            num_heads=self.config.visual_resampler_num_heads,
            depth=self.config.visual_resampler_depth,
            ffn_multiplier=self.config.visual_resampler_ffn_multiplier,
            dropout=self.config.visual_resampler_dropout,
            zero_init_output=self.config.visual_resampler_zero_init_output,
            gate_init=self.config.visual_resampler_gate_init,
            positional_embedding_theta=getattr(transformer, "positional_embedding_theta", 10000.0),
            positional_embedding_max_pos=getattr(transformer, "positional_embedding_max_pos", [20, 2048, 2048]),
            rope_type=getattr(transformer, "rope_type", LTXRopeType.SPLIT),
        ).to(device=device, dtype=dtype)

        if self.config.visual_connector_enabled:
            self._visual_connector = copy.deepcopy(video_connector).to(device=device, dtype=dtype)
            self._visual_connector.requires_grad_(True)

        gate = nn.Parameter(torch.tensor(float(self.config.visual_gate_init), device=device, dtype=torch.float32))
        self._visual_gate = ScalarParameterModule(gate)

    def _init_visual_token_projection(self, video_connector: nn.Module) -> None:
        self._visual_token_projection = None
        self._visual_token_source_dim = self.config.visual_token_source_dim
        self._visual_token_target_dim = self.config.visual_token_target_dim
        if self._visual_token_source_dim is None or self._visual_token_target_dim is None:
            return
        if self._visual_token_source_dim == self._visual_token_target_dim:
            return

        projection = nn.Linear(self._visual_token_source_dim, self._visual_token_target_dim, bias=False)
        with torch.no_grad():
            projection.weight.zero_()
            dim = min(self._visual_token_source_dim, self._visual_token_target_dim)
            eye = torch.eye(dim, dtype=projection.weight.dtype, device=projection.weight.device)
            projection.weight[:dim, :dim].copy_(eye)

        param = next(video_connector.parameters(), None)
        if param is not None:
            projection = projection.to(device=param.device, dtype=param.dtype)
        self._visual_token_projection = projection

    def train_embeddings_processor(self) -> bool:
        return self.config.train_text_connector

    def get_trainable_modules(self) -> dict[str, nn.Module]:
        modules: dict[str, nn.Module] = {}
        if self._visual_token_projection is not None:
            modules["visual_token_projection"] = self._visual_token_projection
        if self._visual_resampler is not None:
            modules["visual_resampler"] = self._visual_resampler
        if self._visual_full_encoder is not None:
            modules["visual_full_encoder"] = self._visual_full_encoder
        if self._visual_connector is not None:
            modules["visual_connector"] = self._visual_connector
        if self._visual_gate is not None:
            modules["visual_gate"] = self._visual_gate
        return modules

    def set_trainable_modules(self, modules: dict[str, nn.Module]) -> None:
        if "visual_token_projection" in modules:
            self._visual_token_projection = modules["visual_token_projection"]
        if "visual_resampler" in modules:
            self._visual_resampler = modules["visual_resampler"]
        if "visual_full_encoder" in modules:
            self._visual_full_encoder = modules["visual_full_encoder"]
        if "visual_connector" in modules:
            self._visual_connector = modules["visual_connector"]
        if "visual_gate" in modules:
            self._visual_gate = modules["visual_gate"]

    def load_extra_checkpoint_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        has_legacy_visual_state = any(
            key.startswith(
                (
                    "training_strategy.visual_resampler.",
                    "training_strategy.visual_connector.",
                    "training_strategy.visual_gate.",
                )
            )
            for key in state_dict
        )
        has_full_visual_state = any(
            key.startswith("training_strategy.visual_full_encoder.") for key in state_dict
        )
        if self.config.visual_context_mode == "full_tokens_3d_sa" and has_legacy_visual_state:
            raise ValueError(
                "Cannot load legacy Q-former visual_resampler/visual_connector/visual_gate weights into "
                "visual_context_mode='full_tokens_3d_sa'. Start the new visual branch from scratch."
            )
        if self.config.visual_context_mode == "qformer_512" and has_full_visual_state:
            raise ValueError(
                "Cannot load visual_full_encoder weights into visual_context_mode='qformer_512'. "
                "Use the checkpoint's matching visual_context_mode."
            )
        for name, module in self.get_trainable_modules().items():
            prefix = f"training_strategy.{name}."
            module_state = {key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)}
            if module_state:
                module.load_state_dict(module_state, strict=True)
            elif name == "visual_token_projection":
                logger.warning(
                    "visual_token_projection not found in checkpoint; using initialized "
                    f"{self._visual_token_source_dim}->{self._visual_token_target_dim} adapter."
                )
            else:
                logger.warning(f"{name} not found in checkpoint; using initialized Stage 1 visual-branch weights.")

    def prepare_conditions(self, batch: dict[str, Any], conditions: dict[str, Tensor]) -> dict[str, Tensor]:
        conditions = self._apply_cfg_preconnector_context_switch(batch, conditions)
        return self._pad_conditions_to_connector_multiple(conditions)

    def postprocess_conditions_after_connector(
        self,
        batch: dict[str, Any],
        conditions: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        conditions = self._apply_cfg_postconnector_text_dropout(batch, conditions)
        if not self.config.visual_branch_enabled:
            return conditions
        if self.config.gt_visual_tokens_dir is None or "gt_visual_tokens" not in batch:
            return conditions

        visual_context, visual_mask = self._build_visual_context_after_connector(batch, conditions)
        visual_context, visual_mask = self._apply_cfg_visual_dropout_after_connector(batch, visual_context, visual_mask)
        return self._append_postconnector_visual_context(conditions, visual_context, visual_mask)

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
        if fps is None:
            fps_for_positions: float | Tensor = float(DEFAULT_FPS)
        else:
            fps_for_positions = fps.to(device=target_latents.device, dtype=torch.float32).flatten()

        ref_data = batch["multi_ref_latents"]
        ref_latents = self._normalize_reference_latents(ref_data["latents"])
        ref_valid_mask = self._get_reference_valid_mask(ref_data, ref_latents)

        if self.config.max_ref_images_per_sample is not None:
            max_refs = self.config.max_ref_images_per_sample
            ref_latents = ref_latents[:, :max_refs]
            ref_valid_mask = ref_valid_mask[:, :max_refs]

        batch_size, num_refs, _channels, ref_frames, ref_height, ref_width = ref_latents.shape
        device = target_latents.device
        ref_valid_mask = self._apply_cfg_reference_dropout(batch, ref_valid_mask, device=device)

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
            fps=fps_for_positions,
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

    def _cfg_probabilities(self) -> dict[str, float]:
        explicit_fields = getattr(self.config, "model_fields_set", set())
        uses_legacy_ref = (
            self.config.cfg_drop_ref_p > 0
            and "cfg_drop_siglip_p" not in explicit_fields
            and "cfg_drop_ref_latents_p" not in explicit_fields
        )
        drop_siglip = self.config.cfg_drop_ref_p if uses_legacy_ref else self.config.cfg_drop_siglip_p
        drop_ref_latents = 0.0 if uses_legacy_ref else self.config.cfg_drop_ref_latents_p
        return {
            "full": self.config.cfg_full_p,
            "drop_text": self.config.cfg_drop_text_p,
            "drop_siglip": drop_siglip,
            "drop_ref_latents": drop_ref_latents,
            "drop_all": self._cfg_drop_all_probability(),
        }

    def _cfg_drop_all_probability(self) -> float:
        return self.config.cfg_drop_all_p if self.config.cfg_drop_all_p is not None else self.config.cfg_drop_planner_p

    def _get_or_sample_cfg_modes(
        self,
        batch: dict[str, Any],
        batch_size: int,
        device: torch.device,
    ) -> CFGModeBatch | None:
        if not self.config.cfg_dropout_enabled:
            return None

        existing = batch.get("_cfg_modes")
        if isinstance(existing, CFGModeBatch):
            if existing.mode_id.device != device:
                existing = CFGModeBatch(
                    mode_id=existing.mode_id.to(device=device),
                    drop_text=existing.drop_text.to(device=device),
                    drop_siglip=existing.drop_siglip.to(device=device),
                    drop_ref_latents=existing.drop_ref_latents.to(device=device),
                    drop_all=existing.drop_all.to(device=device),
                    keep_full=existing.keep_full.to(device=device),
                )
                batch["_cfg_modes"] = existing
            return existing

        modes = sample_cfg_modes(
            batch_size=batch_size,
            probs=self._cfg_probabilities(),
            device=device,
        )
        batch["_cfg_modes"] = modes
        return modes

    def _apply_cfg_preconnector_context_switch(
        self,
        batch: dict[str, Any],
        conditions: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        del batch
        # Split CFG keeps the VLM/reference-image context intact before the
        # connector. Text/null dropout is applied after the connector; reference
        # latent dropout is applied only to the DiT packed latent stream.
        return conditions

    def _apply_cfg_reference_dropout(
        self,
        batch: dict[str, Any],
        ref_valid_mask: Tensor,
        *,
        device: torch.device,
    ) -> Tensor:
        if not self.config.cfg_dropout_enabled:
            return ref_valid_mask
        drop_reference = self._cfg_drop_ref_latents_mask(batch, batch_size=ref_valid_mask.shape[0], device=device)
        if drop_reference is None or not torch.any(drop_reference):
            return ref_valid_mask
        ref_valid_mask = ref_valid_mask.clone()
        ref_valid_mask[drop_reference] = False
        return ref_valid_mask

    def _apply_cfg_planner_dropout(
        self,
        batch: dict[str, Any],
        visual_tokens: Tensor,
        visual_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Zero predicted/GT visual condition tokens for drop_siglip and drop_all/null."""

        if not self.config.cfg_dropout_enabled:
            return visual_tokens, visual_mask
        drop_visual = self._cfg_drop_visual_mask(batch, batch_size=visual_tokens.shape[0], device=visual_tokens.device)
        if drop_visual is None or not torch.any(drop_visual):
            return visual_tokens, visual_mask
        visual_tokens = visual_tokens.clone()
        visual_tokens[drop_visual] = 0
        return visual_tokens, visual_mask

    def _cfg_drop_visual_mask(
        self,
        batch: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor | None:
        modes = self._get_or_sample_cfg_modes(batch, batch_size, device)
        if modes is None:
            return None
        return modes.drop_siglip | modes.drop_all

    def _cfg_drop_planner_mask(
        self,
        batch: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor | None:
        return self._cfg_drop_visual_mask(batch, batch_size=batch_size, device=device)

    def _cfg_drop_all_mask(
        self,
        batch: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor | None:
        modes = self._get_or_sample_cfg_modes(batch, batch_size, device)
        if modes is None:
            return None
        return modes.drop_all

    def _cfg_drop_ref_latents_mask(
        self,
        batch: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor | None:
        modes = self._get_or_sample_cfg_modes(batch, batch_size, device)
        if modes is None:
            return None
        return modes.drop_ref_latents | modes.drop_all

    def _cfg_drop_ref_mask(
        self,
        batch: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor | None:
        return self._cfg_drop_ref_latents_mask(batch, batch_size=batch_size, device=device)

    def _cfg_drop_text_mask(
        self,
        batch: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor | None:
        modes = self._get_or_sample_cfg_modes(batch, batch_size, device)
        if modes is None:
            return None
        return modes.drop_text | modes.drop_all

    def _apply_cfg_postconnector_text_dropout(
        self,
        batch: dict[str, Any],
        conditions: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        if not self.config.cfg_dropout_enabled:
            return conditions
        key = "video_prompt_embeds" if "video_prompt_embeds" in conditions else "prompt_embeds"
        features = conditions[key]
        drop_text = self._cfg_drop_text_mask(batch, batch_size=features.shape[0], device=features.device)
        if drop_text is None or not torch.any(drop_text):
            return conditions

        out = dict(conditions)
        out[key] = features.clone()
        out[key][drop_text] = 0
        if out.get("audio_prompt_embeds") is not None:
            audio_features = out["audio_prompt_embeds"].clone()
            audio_features[drop_text] = 0
            out["audio_prompt_embeds"] = audio_features
        return out

    def _build_visual_context_after_connector(
        self,
        batch: dict[str, Any],
        conditions: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        key = "video_prompt_embeds" if "video_prompt_embeds" in conditions else "prompt_embeds"
        features = conditions[key]
        raw_tokens, raw_mask = self._load_raw_condition_visual_tokens(
            batch["gt_visual_tokens"],
            device=features.device,
            dtype=features.dtype,
        )
        projected_tokens = self._project_visual_tokens(raw_tokens, target_dim=features.shape[-1])
        token_positions = self._build_visual_token_positions(
            batch["gt_visual_tokens"],
            batch["latents"],
            token_count=projected_tokens.shape[1],
            device=projected_tokens.device,
            dtype=torch.float32,
        )
        if self.config.visual_context_mode == "full_tokens_3d_sa":
            if self._visual_full_encoder is None:
                raise RuntimeError(
                    "visual_context_mode='full_tokens_3d_sa' requires initialized Visual3DTokenEncoder"
                )
            self._validate_full_visual_token_layout(
                batch["gt_visual_tokens"],
                token_count=projected_tokens.shape[1],
            )
            visual_context, visual_mask = self._visual_full_encoder(
                tokens=projected_tokens,
                token_positions=token_positions,
                token_mask=raw_mask,
            )
        else:
            if self._visual_resampler is None:
                raise RuntimeError("visual_branch_enabled=True requires initialized Visual3DResampler.")
            query_positions = self._build_visual_query_positions(
                batch["gt_visual_tokens"],
                batch["latents"],
                token_count=projected_tokens.shape[1],
                device=projected_tokens.device,
                dtype=torch.float32,
            )
            visual_context, visual_mask = self._visual_resampler(
                tokens=projected_tokens,
                token_positions=token_positions,
                token_mask=raw_mask,
                query_positions=query_positions,
            )
        if self._visual_connector is not None:
            visual_context, visual_mask = self._run_visual_connector(visual_context, visual_mask)
        if self._visual_gate is not None:
            gate_module = self._unwrap_strategy_module(self._visual_gate)
            gate = gate_module.get_value().to(device=visual_context.device, dtype=visual_context.dtype)
            visual_context = visual_context * gate

        shape = list(visual_context.shape)
        self._last_visual_context_shape = shape
        batch["_visual_context_shape"] = shape
        batch["_visual_context_token_count"] = int(visual_context.shape[1])
        return visual_context, visual_mask

    def _validate_full_visual_token_layout(self, visual_data: dict[str, Any], *, token_count: int) -> None:
        expected_tokens = self.config.visual_context_expected_tokens
        if expected_tokens is not None and token_count != expected_tokens:
            raise ValueError(
                f"Full visual context expected {expected_tokens} tokens, got {token_count}"
            )
        tokens_per_frame = self._visual_tokens_per_frame(visual_data, token_count=token_count)
        frame_count = token_count // tokens_per_frame
        spatial_grid = math.isqrt(tokens_per_frame)
        if spatial_grid * spatial_grid != tokens_per_frame:
            raise ValueError(
                f"Full visual context tokens_per_frame={tokens_per_frame} must form a square spatial grid"
            )
        if expected_tokens == 2048 and (tokens_per_frame, frame_count, spatial_grid) != (256, 8, 16):
            raise ValueError(
                "Expected 2048-token layout as 8 frames * 16 * 16 tokens, got "
                f"frame_count={frame_count}, spatial_grid={spatial_grid}, tokens_per_frame={tokens_per_frame}"
            )
        sampled_frame_mask = visual_data.get("sampled_frame_mask")
        valid_frame_counts = visual_data.get("num_valid_vlm_frames")
        visual_token_mask = visual_data.get(self.config.visual_token_mask_key)
        if sampled_frame_mask is not None:
            sampled_frame_mask = sampled_frame_mask.to(dtype=torch.bool)
            if sampled_frame_mask.ndim == 1:
                sampled_frame_mask = sampled_frame_mask.unsqueeze(0)
            if sampled_frame_mask.shape[1] != frame_count:
                raise ValueError(
                    f"sampled_frame_mask must have {frame_count} frame slots, got {tuple(sampled_frame_mask.shape)}"
                )
            if valid_frame_counts is not None:
                valid_frame_counts = valid_frame_counts.to(dtype=torch.long).flatten()
                if not torch.equal(
                    sampled_frame_mask.sum(dim=1),
                    valid_frame_counts.to(device=sampled_frame_mask.device),
                ):
                    raise ValueError("sampled_frame_mask does not match num_valid_vlm_frames")
            if visual_token_mask is not None:
                token_mask = visual_token_mask.to(dtype=torch.bool)
                if token_mask.ndim == 1:
                    token_mask = token_mask.unsqueeze(0)
                expected_mask = sampled_frame_mask.repeat_interleave(tokens_per_frame, dim=1)
                if token_mask.shape != expected_mask.shape or not torch.equal(
                    token_mask.to(device=expected_mask.device), expected_mask
                ):
                    raise ValueError("visual_token_mask must match sampled_frame_mask expanded by tokens_per_frame")

    def _run_visual_connector(self, visual_context: Tensor, visual_mask: Tensor) -> tuple[Tensor, Tensor]:
        if self._visual_connector is None:
            return visual_context, visual_mask
        visual_context, visual_mask = self._pad_visual_context_to_connector_multiple(visual_context, visual_mask)
        additive_mask = convert_to_additive_mask(visual_mask.to(device=visual_context.device), visual_context.dtype)
        connected_context, connected_mask = self._visual_connector(visual_context, additive_mask)
        if connected_mask is None:
            return connected_context, visual_mask
        if connected_mask.ndim == 4:
            binary_mask = connected_mask[:, 0, 0, :] >= 0
        elif connected_mask.ndim == 2:
            binary_mask = connected_mask.to(device=connected_context.device, dtype=torch.bool)
        else:
            raise ValueError(f"Visual connector returned unsupported mask shape {tuple(connected_mask.shape)}")
        return connected_context, binary_mask.to(device=connected_context.device, dtype=torch.bool)

    def _pad_visual_context_to_connector_multiple(self, visual_context: Tensor, visual_mask: Tensor) -> tuple[Tensor, Tensor]:
        register_count = getattr(self._visual_connector, "num_learnable_registers", None) or self._connector_register_count
        if not register_count:
            return visual_context, visual_mask
        remainder = visual_context.shape[1] % int(register_count)
        if remainder == 0:
            return visual_context, visual_mask
        pad_len = int(register_count) - remainder
        context_pad = torch.zeros(
            visual_context.shape[0],
            pad_len,
            visual_context.shape[-1],
            dtype=visual_context.dtype,
            device=visual_context.device,
        )
        mask_pad = torch.zeros(visual_mask.shape[0], pad_len, dtype=torch.bool, device=visual_mask.device)
        return torch.cat([visual_context, context_pad], dim=1), torch.cat([visual_mask, mask_pad], dim=1)

    def _apply_cfg_visual_dropout_after_connector(
        self,
        batch: dict[str, Any],
        visual_context: Tensor,
        visual_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not self.config.cfg_dropout_enabled:
            return visual_context, visual_mask
        drop_visual = self._cfg_drop_visual_mask(batch, batch_size=visual_context.shape[0], device=visual_context.device)
        if drop_visual is None or not torch.any(drop_visual):
            return visual_context, visual_mask
        visual_context = visual_context.clone()
        visual_context[drop_visual] = 0
        return visual_context, visual_mask

    def _append_postconnector_visual_context(
        self,
        conditions: dict[str, Tensor],
        visual_context: Tensor,
        visual_mask: Tensor,
    ) -> dict[str, Tensor]:
        out = dict(conditions)
        key = "video_prompt_embeds" if "video_prompt_embeds" in out else "prompt_embeds"
        text_context = out[key]
        out[key] = torch.cat([text_context, visual_context.to(device=text_context.device, dtype=text_context.dtype)], dim=1)

        audio_context = out.get("audio_prompt_embeds")
        if audio_context is not None:
            audio_pad = torch.zeros(
                audio_context.shape[0],
                visual_context.shape[1],
                audio_context.shape[-1],
                dtype=audio_context.dtype,
                device=audio_context.device,
            )
            out["audio_prompt_embeds"] = torch.cat([audio_context, audio_pad], dim=1)

        prompt_mask = out["prompt_attention_mask"].to(device=visual_mask.device, dtype=torch.long)
        visual_mask = visual_mask.to(device=prompt_mask.device, dtype=torch.long)
        out["prompt_attention_mask"] = torch.cat([prompt_mask, visual_mask], dim=1)
        return out

    def _build_visual_token_positions(
        self,
        visual_data: dict[str, Any],
        latents_data: dict[str, Any],
        *,
        token_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        tokens_per_frame = self._visual_tokens_per_frame(visual_data, token_count=token_count)
        if token_count % tokens_per_frame != 0:
            raise ValueError(f"Visual token count {token_count} is not divisible by tokens_per_frame={tokens_per_frame}")
        frame_count = token_count // tokens_per_frame
        spatial_grid = int(math.isqrt(tokens_per_frame))
        if spatial_grid * spatial_grid != tokens_per_frame:
            raise ValueError(f"tokens_per_frame={tokens_per_frame} must be a perfect square for 3D visual positions")
        times = self._visual_sample_times(
            visual_data,
            frame_count=frame_count,
            batch_size=self._visual_batch_size(visual_data, latents_data),
            device=device,
            dtype=dtype,
        )
        height, width = self._visual_target_hw(latents_data, device=device, dtype=dtype)
        return self._make_visual_positions(times, height=height, width=width, spatial_grid=spatial_grid, dtype=dtype)

    def _build_visual_query_positions(
        self,
        visual_data: dict[str, Any],
        latents_data: dict[str, Any],
        *,
        token_count: int | None = None,
        output_grid: int | None = None,
        frame_stride: int | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        output_grid = int(output_grid or self.config.visual_context_spatial_grid)
        frame_stride = int(frame_stride or self.config.visual_context_frame_stride)
        if token_count is None:
            token_count = self._infer_visual_token_count(visual_data)
        tokens_per_frame = self._visual_tokens_per_frame(visual_data, token_count=token_count)
        if token_count % tokens_per_frame != 0:
            raise ValueError(f"Visual token count {token_count} is not divisible by tokens_per_frame={tokens_per_frame}")
        frame_count = token_count // tokens_per_frame
        times = self._visual_sample_times(
            visual_data,
            frame_count=frame_count,
            batch_size=self._visual_batch_size(visual_data, latents_data),
            device=device,
            dtype=dtype,
        )[:, ::frame_stride]
        if times.shape[1] == 0:
            raise ValueError("visual_context_frame_stride selected zero query frames")
        query_count = times.shape[1] * output_grid * output_grid
        if query_count > self.config.visual_context_max_tokens:
            raise ValueError(
                f"Visual query token count {query_count} exceeds visual_context_max_tokens="
                f"{self.config.visual_context_max_tokens}. Increase the max or reduce grid/frame count."
            )
        height, width = self._visual_target_hw(latents_data, device=device, dtype=dtype)
        return self._make_visual_positions(times, height=height, width=width, spatial_grid=output_grid, dtype=dtype)

    def _visual_tokens_per_frame(self, visual_data: dict[str, Any], *, token_count: int) -> int:
        value = visual_data.get(self.config.visual_tokens_per_frame_key)
        if value is None:
            raise ValueError("GT visual-token metadata must include tokens_per_frame to build 3D positions.")
        if isinstance(value, Tensor):
            flat = value.flatten()
            if flat.numel() == 0:
                raise ValueError("tokens_per_frame tensor is empty")
            if not torch.all(flat == flat[0]):
                raise ValueError(f"Mixed tokens_per_frame in batch: {flat.tolist()}")
            value = int(flat[0].item())
        else:
            value = int(value)
        if value <= 0:
            raise ValueError(f"tokens_per_frame must be positive, got {value}")
        if token_count % value != 0:
            raise ValueError(f"Visual token count {token_count} is not divisible by tokens_per_frame={value}")
        return value

    def _visual_sample_times(
        self,
        visual_data: dict[str, Any],
        *,
        frame_count: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        sampled = visual_data.get("sampled_frame_indices")
        if sampled is None:
            sampled_indices = torch.arange(frame_count, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
        else:
            sampled_indices = sampled.to(device=device, dtype=dtype)
            if sampled_indices.ndim == 1:
                sampled_indices = sampled_indices.unsqueeze(0)
            if sampled_indices.shape[0] == 1 and batch_size > 1:
                sampled_indices = sampled_indices.expand(batch_size, -1)
            if sampled_indices.shape[0] != batch_size:
                raise ValueError(
                    f"sampled_frame_indices batch {sampled_indices.shape[0]} does not match batch_size={batch_size}"
                )
            if sampled_indices.shape[1] != frame_count:
                strided = sampled_indices[:, :: self.config.visual_token_frame_stride]
                if strided.shape[1] < frame_count:
                    raise ValueError(
                        f"sampled_frame_indices has {sampled_indices.shape[1]} frames; cannot match visual frame_count="
                        f"{frame_count} after stride={self.config.visual_token_frame_stride}"
                    )
                sampled_indices = strided[:, :frame_count]

        sampled_frame_mask = visual_data.get("sampled_frame_mask")
        if sampled_frame_mask is not None:
            sampled_frame_mask = sampled_frame_mask.to(device=device, dtype=torch.bool)
            if sampled_frame_mask.ndim == 1:
                sampled_frame_mask = sampled_frame_mask.unsqueeze(0)
            if sampled_frame_mask.shape != sampled_indices.shape:
                raise ValueError(
                    f"sampled_frame_mask shape {tuple(sampled_frame_mask.shape)} does not match "
                    f"sampled_frame_indices {tuple(sampled_indices.shape)}"
                )
            sampled_indices = sampled_indices.masked_fill(~sampled_frame_mask, 0)

        fps = visual_data.get("source_fps")
        if fps is None:
            fps_values = torch.full((batch_size,), float(DEFAULT_FPS), device=device, dtype=dtype)
        else:
            fps_values = fps.to(device=device, dtype=dtype).flatten()
            if fps_values.numel() == 1 and batch_size > 1:
                fps_values = fps_values.expand(batch_size)
            if fps_values.numel() != batch_size:
                raise ValueError(f"source_fps has {fps_values.numel()} values, expected {batch_size}")
        fps_values = fps_values.clamp(min=1.0e-6)
        return sampled_indices[:, :frame_count] / fps_values[:, None]

    def _visual_target_hw(
        self,
        latents_data: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        height = latents_data.get("height")
        width = latents_data.get("width")
        if height is None or width is None:
            raise ValueError("latents metadata must include height and width to build visual positions")
        height_tensor = height.to(device=device, dtype=dtype).flatten() if isinstance(height, Tensor) else torch.tensor([height], device=device, dtype=dtype)
        width_tensor = width.to(device=device, dtype=dtype).flatten() if isinstance(width, Tensor) else torch.tensor([width], device=device, dtype=dtype)
        return height_tensor, width_tensor

    def _visual_batch_size(self, visual_data: dict[str, Any], latents_data: dict[str, Any]) -> int:
        tokens = visual_data.get(self.config.visual_token_key)
        if isinstance(tokens, Tensor) and tokens.ndim >= 3:
            return int(tokens.shape[0])
        height = latents_data.get("height")
        if isinstance(height, Tensor):
            return int(height.flatten().numel())
        return 1

    def _infer_visual_token_count(self, visual_data: dict[str, Any]) -> int:
        tokens = visual_data.get(self.config.visual_token_key)
        if not isinstance(tokens, Tensor) or tokens.ndim < 2:
            raise ValueError("Cannot infer visual token count without a visual_tokens tensor")
        return int(tokens.shape[-2])

    @staticmethod
    def _make_visual_positions(
        times: Tensor,
        *,
        height: Tensor,
        width: Tensor,
        spatial_grid: int,
        dtype: torch.dtype,
    ) -> Tensor:
        batch_size, frame_count = times.shape
        device = times.device
        if height.numel() == 1 and batch_size > 1:
            height = height.expand(batch_size)
        if width.numel() == 1 and batch_size > 1:
            width = width.expand(batch_size)
        if height.numel() != batch_size or width.numel() != batch_size:
            raise ValueError(f"height/width metadata must have 1 or {batch_size} values")

        grid = torch.arange(spatial_grid, device=device, dtype=dtype) + 0.5
        y_unit, x_unit = torch.meshgrid(grid / spatial_grid, grid / spatial_grid, indexing="ij")
        y_unit = y_unit.reshape(-1)
        x_unit = x_unit.reshape(-1)

        y_coords = height[:, None, None] * y_unit[None, None, :]
        x_coords = width[:, None, None] * x_unit[None, None, :]
        t_coords = times[:, :, None].expand(batch_size, frame_count, spatial_grid * spatial_grid)
        y_coords = y_coords.expand(batch_size, frame_count, spatial_grid * spatial_grid)
        x_coords = x_coords.expand(batch_size, frame_count, spatial_grid * spatial_grid)

        coords = torch.stack(
            [
                t_coords.reshape(batch_size, -1),
                y_coords.reshape(batch_size, -1),
                x_coords.reshape(batch_size, -1),
            ],
            dim=1,
        ).to(dtype=dtype)
        return torch.stack([coords, coords], dim=-1)

    @classmethod
    def _select_condition_rows(
        cls,
        *,
        primary: dict[str, Tensor],
        alternate: dict[str, Tensor],
        use_alternate: Tensor,
    ) -> dict[str, Tensor]:
        out = dict(primary)
        for key in ("video_prompt_embeds", "prompt_embeds", "audio_prompt_embeds", "prompt_attention_mask"):
            primary_value = out.get(key)
            alternate_value = alternate.get(key)
            if primary_value is None or alternate_value is None:
                continue

            alternate_value = alternate_value.to(device=primary_value.device, dtype=primary_value.dtype)
            max_len = max(primary_value.shape[1], alternate_value.shape[1])
            primary_value = cls._pad_sequence_axis(primary_value, max_len)
            alternate_value = cls._pad_sequence_axis(alternate_value, max_len)

            selector = use_alternate.to(device=primary_value.device, dtype=torch.bool)
            selector = selector.view(selector.shape[0], *([1] * (primary_value.ndim - 1)))
            out[key] = torch.where(selector, alternate_value, primary_value)
        return out

    @staticmethod
    def _zero_condition_feature_rows(conditions: dict[str, Tensor], row_mask: Tensor) -> dict[str, Tensor]:
        out = dict(conditions)
        for key in ("video_prompt_embeds", "prompt_embeds", "audio_prompt_embeds"):
            features = out.get(key)
            if features is None:
                continue
            mask = row_mask.to(device=features.device, dtype=torch.bool)
            features = features.clone()
            features[mask] = 0
            out[key] = features
        return out

    @staticmethod
    def _pad_sequence_axis(value: Tensor, target_len: int) -> Tensor:
        if value.shape[1] == target_len:
            return value
        if value.shape[1] > target_len:
            return value[:, :target_len]
        pad_shape = (value.shape[0], target_len - value.shape[1], *value.shape[2:])
        pad = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
        return torch.cat([value, pad], dim=1)

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "conditioning": "multi_reference_video",
            "reference_time_stride": self.config.reference_time_stride,
            "cfg_dropout_enabled": self.config.cfg_dropout_enabled,
            "cfg_full_p": self.config.cfg_full_p,
            "cfg_drop_text_p": self.config.cfg_drop_text_p,
            "cfg_drop_siglip_p": self._cfg_probabilities()["drop_siglip"],
            "cfg_drop_ref_latents_p": self._cfg_probabilities()["drop_ref_latents"],
            "cfg_drop_ref_p": self.config.cfg_drop_ref_p,
            "cfg_drop_ref_p_is_legacy_drop_siglip_alias": True,
            "cfg_drop_all_p": self._cfg_drop_all_probability(),
            "cfg_drop_planner_p": self.config.cfg_drop_planner_p,
            "cfg_drop_planner_p_is_legacy_drop_all_alias": True,
            "cfg_text_conditions_dir": self.config.cfg_text_conditions_dir,
            "cfg_ref_only_conditions_dir": self.config.cfg_ref_only_conditions_dir,
            "visual_token_source_dim": self.config.visual_token_source_dim,
            "visual_token_target_dim": self.config.visual_token_target_dim,
            "visual_token_projection_enabled": self._visual_token_projection is not None,
            "train_text_connector": self.config.train_text_connector,
            "visual_branch_enabled": self.config.visual_branch_enabled,
            "visual_context_mode": self.config.visual_context_mode,
            "visual_context_expected_tokens": self.config.visual_context_expected_tokens,
            "visual_context_spatial_grid": self.config.visual_context_spatial_grid,
            "visual_context_frame_stride": self.config.visual_context_frame_stride,
            "visual_context_token_count": self._last_visual_context_shape[1] if self._last_visual_context_shape else None,
            "visual_resampler_depth": self.config.visual_resampler_depth,
            "visual_resampler_num_heads": self.config.visual_resampler_num_heads,
            "visual_connector_enabled": self.config.visual_connector_enabled,
            "visual_gate_init": self.config.visual_gate_init,
            "visual_full_sa_num_heads": self.config.visual_full_sa_num_heads,
            "visual_full_sa_depth": self.config.visual_full_sa_depth,
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

    def _load_raw_condition_visual_tokens(
        self,
        visual_data: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        tokens = visual_data[self.config.visual_token_key].to(device=device, dtype=dtype)
        if tokens.ndim != 3:
            raise ValueError(f"GT visual tokens must be [B,K,D], got {tuple(tokens.shape)}")

        mask = visual_data.get(self.config.visual_token_mask_key)
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            mask = mask.to(device=tokens.device, dtype=torch.bool)
        if mask.shape != tokens.shape[:2]:
            raise ValueError(f"GT visual token mask must be [B,K], got {tuple(mask.shape)} for tokens {tuple(tokens.shape)}")
        return self._downsample_visual_tokens_by_frame(tokens, mask, visual_data)

    def _load_condition_visual_tokens(
        self,
        visual_data: dict[str, Any],
        conditions: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        video_feature_key = "video_prompt_embeds" if "video_prompt_embeds" in conditions else "prompt_embeds"
        video_features = conditions[video_feature_key]
        tokens, mask = self._load_raw_condition_visual_tokens(
            visual_data,
            device=video_features.device,
            dtype=video_features.dtype,
        )
        tokens = self._project_visual_tokens(tokens, target_dim=video_features.shape[-1])
        return tokens, mask

    def _project_visual_tokens(self, tokens: Tensor, target_dim: int) -> Tensor:
        source_dim = tokens.shape[-1]
        if source_dim == target_dim:
            return tokens

        if self._visual_token_projection is None:
            raise ValueError(
                f"GT visual token dim {source_dim} does not match prompt feature dim {target_dim}. "
                "Set training_strategy.visual_token_source_dim and visual_token_target_dim, "
                "e.g. 3840 and 4096, to enable trainable projection."
            )

        if source_dim != self._visual_token_source_dim or target_dim != self._visual_token_target_dim:
            raise ValueError(
                f"GT visual token dim {source_dim} and prompt feature dim {target_dim} do not match configured "
                f"projection {self._visual_token_source_dim}->{self._visual_token_target_dim}."
            )
        return self._visual_token_projection(tokens)

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
