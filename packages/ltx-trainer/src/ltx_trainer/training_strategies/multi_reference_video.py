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
from ltx_core.multicond.cfg_sampler import CFGModeBatch, sample_cfg_modes
from ltx_core.multicond.rope_mask_builder import build_multiref_sequence
from ltx_core.multicond.visual_tokens import Visual3DResampler
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
        default=2048,
        description="Maximum query tokens reserved by the visual 3D resampler.",
        ge=1,
    )

    visual_resampler_num_heads: int = Field(default=16, ge=1)
    visual_resampler_depth: int = Field(default=1, ge=1)
    visual_resampler_ffn_multiplier: float = Field(default=4.0, gt=0.0)
    visual_resampler_dropout: float = Field(default=0.0, ge=0.0)
    visual_resampler_zero_init_output: bool = Field(default=True)
    visual_resampler_gate_init: float = Field(default=0.0)

    visual_connector_enabled: bool = Field(
        default=True,
        description="Use a trainable visual connector initialized from embeddings_processor.video_connector.",
    )

    visual_gate_init: float = Field(
        default=0.0,
        description="Initial scalar gate for visual_context before concatenating to DiT context.",
    )

    cfg_dropout_enabled: bool = Field(
        default=False,
        description=(
            "Enable per-sample condition dropout for factorized CFG training. "
            "Modes are mutually exclusive: full, drop_text, drop_ref, drop_all/null."
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

    cfg_drop_ref_p: float = Field(
        default=0.1,
        description=(
            "Probability of dropping image/visual conditions. This masks reference latent tokens, zeros "
            "GT/predicted visual condition tokens, and swaps VLM context to text-only conditions when "
            "cfg_text_conditions_dir is available."
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
        self._visual_connector: nn.Module | None = None
        self._visual_gate: ScalarParameterModule | None = None
        self._last_visual_context_shape: list[int] | None = None

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

        self._visual_resampler = Visual3DResampler(
            dim=int(connector_dim),
            max_query_tokens=self.config.visual_context_max_tokens,
            num_heads=self.config.visual_resampler_num_heads,
            depth=self.config.visual_resampler_depth,
            ffn_multiplier=self.config.visual_resampler_ffn_multiplier,
            dropout=self.config.visual_resampler_dropout,
            zero_init_output=self.config.visual_resampler_zero_init_output,
            gate_init=self.config.visual_resampler_gate_init,
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
        if "visual_connector" in modules:
            self._visual_connector = modules["visual_connector"]
        if "visual_gate" in modules:
            self._visual_gate = modules["visual_gate"]

    def load_extra_checkpoint_state_dict(self, state_dict: dict[str, Tensor]) -> None:
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
        return self._apply_cfg_preconnector_context_switch(batch, conditions)

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

    def _cfg_probabilities(self) -> dict[str, float]:
        return {
            "full": self.config.cfg_full_p,
            "drop_text": self.config.cfg_drop_text_p,
            "drop_ref": self.config.cfg_drop_ref_p,
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
                    drop_ref=existing.drop_ref.to(device=device),
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
        if not self.config.cfg_dropout_enabled:
            return conditions

        video_feature_key = "video_prompt_embeds" if "video_prompt_embeds" in conditions else "prompt_embeds"
        video_features = conditions[video_feature_key]
        modes = self._get_or_sample_cfg_modes(batch, video_features.shape[0], video_features.device)
        if modes is None:
            return conditions

        out = dict(conditions)

        if torch.any(modes.drop_ref) and "cfg_text_conditions" in batch:
            out = self._select_condition_rows(
                primary=out,
                alternate=batch["cfg_text_conditions"],
                use_alternate=modes.drop_ref,
            )

        if torch.any(modes.drop_text) and "cfg_ref_only_conditions" in batch:
            out = self._select_condition_rows(
                primary=out,
                alternate=batch["cfg_ref_only_conditions"],
                use_alternate=modes.drop_text,
            )

        # drop_text/drop_all text zeroing happens after the frozen connector so
        # connector registers/FFN do not synthesize a nonzero null pattern.
        return out

    def _apply_cfg_reference_dropout(
        self,
        batch: dict[str, Any],
        ref_valid_mask: Tensor,
        *,
        device: torch.device,
    ) -> Tensor:
        if not self.config.cfg_dropout_enabled:
            return ref_valid_mask
        modes = self._get_or_sample_cfg_modes(batch, ref_valid_mask.shape[0], device)
        if modes is None:
            return ref_valid_mask
        drop_reference = modes.drop_ref | modes.drop_all
        if not torch.any(drop_reference):
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
        """Zero visual condition tokens for drop_ref and drop_all/null.

        The historical method name is kept for compatibility; this is not a
        planner-only dropout branch.
        """

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
        return modes.drop_ref | modes.drop_all

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

    def _cfg_drop_ref_mask(
        self,
        batch: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor | None:
        modes = self._get_or_sample_cfg_modes(batch, batch_size, device)
        if modes is None:
            return None
        return modes.drop_ref | modes.drop_all

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
            "cfg_drop_ref_p": self.config.cfg_drop_ref_p,
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
            "visual_context_spatial_grid": self.config.visual_context_spatial_grid,
            "visual_context_frame_stride": self.config.visual_context_frame_stride,
            "visual_context_token_count": self._last_visual_context_shape[1] if self._last_visual_context_shape else None,
            "visual_resampler_depth": self.config.visual_resampler_depth,
            "visual_resampler_num_heads": self.config.visual_resampler_num_heads,
            "visual_connector_enabled": self.config.visual_connector_enabled,
            "visual_gate_init": self.config.visual_gate_init,
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
