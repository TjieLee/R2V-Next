"""Joint flow in frozen Gemma semantic space and native video latent space."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from functools import partial
from itertools import pairwise
from typing import Any, Literal

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn

from ltx_core.model.transformer.modality import Modality, SemanticVideoPrediction
from ltx_core.multicond.gemma3_attention import build_gemma3_attention_masks, resolve_gemma3_sliding_window
from ltx_core.multicond.semantic_tokens import (
    EVIDENCE_GRID_SIZE,
    EVIDENCE_TOKENS_PER_FRAME,
    build_semantic_vlm_teacher_attention_mask,
)
from ltx_core.types import VideoLatentShape
from ltx_trainer.online_data.anchor_geometry import normalized_anchor_timestamps
from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_inference.semantic_guidance import (
    SemanticGuidanceConfig,
    SemanticGuidanceStateBundle,
    build_stg_perturbation,
    combine_guided_denoised,
    denoised_to_velocity,
    rescale_guided_denoised_branches,
    validate_stg_blocks,
    velocity_to_denoised,
)
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)
from ltx_trainer.training_strategies.semantic_flow import (
    ENTITY_GLOBAL,
    MAX_REFERENCE_ENTITIES,
    PHASE2_CONDITION_MODES,
    TYPE_REFERENCE,
    TYPE_SEMANTIC,
    TYPE_TARGET,
    SemanticFlowStrategy,
    SemanticInferenceState,
    _attention_allows_any_key,
    _attention_key_spans_equal,
    build_compact_valid_token_mask,
)
from ltx_trainer.training_strategies.semantic_flow_bridge import (
    configure_phase2_bridge_trainability,
    resolve_phase2_bridge_modules,
)

DEFAULT_SEMANTIC_VLM_CONDITION_PROBABILITIES = {
    "til_111": 0.50,
    "til_110": 0.10,
    "til_101": 0.10,
    "til_011": 0.10,
    "til_100": 0.05,
    "til_010": 0.05,
    "til_001": 0.05,
    "til_000": 0.05,
}


class SemanticVLMFlowConfig(TrainingStrategyConfigBase):
    """Configuration for frozen-VLM semantic and native-video joint flow."""

    name: Literal["semantic_vlm_flow"] = "semantic_vlm_flow"
    reference_latents_dir: str = "reference_latents"
    conditions_dir: str = "conditions"
    max_ref_images_per_sample: int = Field(default=4, ge=1, le=MAX_REFERENCE_ENTITIES)
    reference_rope_mode: Literal["negative_adjacent_shifted_hw"] = "negative_adjacent_shifted_hw"
    required_fsdp_world_size: Literal[8] = 8

    semantic_anchor_count: int = Field(default=8, ge=1)
    semantic_grid_size: int = Field(default=16, ge=1)
    semantic_tokens_per_frame: int = Field(default=256, ge=1)
    vlm_prefix_max_length: int = Field(default=2560, ge=128)
    vlm_teacher_max_length: int = Field(default=6656, ge=2560)

    video_flow_weight: float = Field(default=1.0, ge=0.0)
    semantic_flow_weight: float = Field(default=1.0, ge=0.0)
    condition_probabilities: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_SEMANTIC_VLM_CONDITION_PROBABILITIES)
    )

    @model_validator(mode="after")
    def _validate_semantic_vlm_contract(self) -> "SemanticVLMFlowConfig":
        fixed_values = {
            "semantic_anchor_count": (self.semantic_anchor_count, 8),
            "semantic_grid_size": (self.semantic_grid_size, EVIDENCE_GRID_SIZE),
            "semantic_tokens_per_frame": (
                self.semantic_tokens_per_frame,
                EVIDENCE_TOKENS_PER_FRAME,
            ),
        }
        invalid = {name: value for name, (value, expected) in fixed_values.items() if value != expected}
        if invalid:
            raise ValueError(f"semantic_vlm_joint_flow_v1 fixed geometry mismatch: {invalid}")
        expected_keys = set(PHASE2_CONDITION_MODES)
        actual_keys = set(self.condition_probabilities)
        if actual_keys != expected_keys:
            raise ValueError(
                "condition_probabilities must contain exactly the T/I/L 8-way modes; "
                f"missing={sorted(expected_keys - actual_keys)}, unexpected={sorted(actual_keys - expected_keys)}"
            )
        if any(value < 0.0 for value in self.condition_probabilities.values()):
            raise ValueError("condition_probabilities must be non-negative")
        probability_sum = sum(self.condition_probabilities.values())
        if abs(probability_sum - 1.0) > 1.0e-6:
            raise ValueError(f"condition_probabilities must sum to 1.0, got {probability_sum}")
        maximum_teacher_length = self.vlm_prefix_max_length + (self.semantic_anchor_count * EVIDENCE_TOKENS_PER_FRAME)
        if self.vlm_teacher_max_length < maximum_teacher_length:
            raise ValueError(
                "vlm_teacher_max_length cannot hold the R2V evidence suffix: "
                f"required={maximum_teacher_length}, configured={self.vlm_teacher_max_length}"
            )
        return self

    def get_data_sources(self) -> dict[str, str]:
        return {
            "latents": "latents",
            self.conditions_dir: "conditions",
            self.reference_latents_dir: "reference_latents",
        }


class SemanticVLMFlowStrategy(SemanticFlowStrategy):
    """Train one DiT over frozen-Gemma semantic and native-video flow states."""

    config: SemanticVLMFlowConfig

    def __init__(self, config: SemanticVLMFlowConfig) -> None:
        TrainingStrategy.__init__(self, config)
        self._text_encoder: nn.Module | None = None
        self._embeddings_processor: nn.Module | None = None
        self._transformer: nn.Module | None = None
        self._gemma_dim: int | None = None
        self._video_dim: int | None = None
        self._dit_hidden_dim: int | None = None
        self._last_training_metrics: dict[str, Tensor] = {}
        self._last_bridge_metrics: dict[str, Tensor] = {}
        self._condition_counts: Counter[str] = Counter()

    def requires_text_encoder(self) -> bool:
        return True

    def train_text_encoder(self) -> bool:
        return False

    def train_embeddings_processor(self) -> bool:
        return True

    def get_trainable_modules(self) -> dict[str, nn.Module]:
        return {}

    def set_trainable_modules(self, modules: dict[str, nn.Module]) -> None:
        if modules:
            raise RuntimeError(
                "semantic_vlm_flow keeps all semantic flow modules inside the transformer"
            )

    def semantic_tokens_per_frame(self) -> int:
        return self.config.semantic_tokens_per_frame

    def semantic_frame_count_for_task(self, *, task: str, pixel_frame_count: int) -> int:
        del pixel_frame_count
        if task == IMAGE_TASK:
            return 1
        if task == VIDEO_TASK:
            return self.config.semantic_anchor_count
        raise ValueError(f"Unsupported semantic VLM flow task: {task!r}")

    def attach_models(
        self,
        *,
        transformer: nn.Module,
        embeddings_processor: nn.Module,
        text_encoder: nn.Module | None = None,
    ) -> None:
        if text_encoder is None:
            raise ValueError("semantic_vlm_flow requires the frozen Gemma multimodal text encoder")
        self._text_encoder = text_encoder
        self._text_encoder.requires_grad_(False).eval()
        self._embeddings_processor = embeddings_processor
        self._transformer = transformer

        language_model = self._get_language_model()
        language_model.requires_grad_(False).eval()
        input_embeddings = self._get_input_embeddings(language_model)
        self._gemma_dim = int(input_embeddings.weight.shape[-1])
        patchify_projection = getattr(transformer, "patchify_proj", None)
        if patchify_projection is None:
            raise ValueError("semantic_vlm_flow requires transformer.patchify_proj")
        self._video_dim = int(patchify_projection.in_features)
        self._dit_hidden_dim = int(transformer.inner_dim)
        if self._video_dim != int(transformer.proj_out.out_features):
            raise ValueError("transformer video input and output token widths differ")

        enable_semantic_flow = getattr(transformer, "enable_semantic_vlm_flow", None)
        if not callable(enable_semantic_flow):
            raise ValueError("transformer does not expose enable_semantic_vlm_flow")
        enable_semantic_flow(
            semantic_dim=self._gemma_dim,
            num_reference_slots=self.config.max_ref_images_per_sample,
            semantic_token_type_id=TYPE_SEMANTIC,
            reference_token_type_id=TYPE_REFERENCE,
        )

    def set_transformer(self, transformer: nn.Module) -> None:
        """Receive the accelerator-prepared transformer wrapper."""
        self._transformer = transformer

    def configure_embeddings_processor_trainability(self, embeddings_processor: nn.Module) -> None:
        configure_phase2_bridge_trainability(embeddings_processor)

    def get_embeddings_processor_trainable_modules(self, embeddings_processor: nn.Module) -> dict[str, nn.Module]:
        return resolve_phase2_bridge_modules(embeddings_processor)

    def set_embeddings_processor_trainable_modules(
        self,
        embeddings_processor: nn.Module,
        modules: dict[str, nn.Module],
    ) -> None:
        projection_names = [name for name in modules if name.startswith("feature_extractor.")]
        if len(projection_names) != 1 or "video_connector" not in modules:
            raise RuntimeError(f"Prepared semantic VLM flow bridge modules are incomplete: {sorted(modules)}")
        projection_name = projection_names[0].removeprefix("feature_extractor.")
        setattr(
            embeddings_processor.feature_extractor,
            projection_name,
            modules[projection_names[0]],
        )
        embeddings_processor.video_connector = modules["video_connector"]
        self._embeddings_processor = embeddings_processor

    def enforce_frozen_module_eval(self) -> None:
        if self._text_encoder is not None:
            self._text_encoder.eval()
        if self._embeddings_processor is None:
            return
        self._embeddings_processor.eval()
        for module in resolve_phase2_bridge_modules(self._embeddings_processor).values():
            module.train()
        audio_connector = getattr(self._embeddings_processor, "audio_connector", None)
        if isinstance(audio_connector, nn.Module):
            audio_connector.eval()

    def prepare_conditions(self, batch: dict[str, Any], conditions: dict[str, Tensor]) -> dict[str, Tensor]:
        del batch
        if self._embeddings_processor is None:
            raise RuntimeError("semantic VLM flow embedding processor is not attached")
        hidden_states = conditions.get("frozen_vlm_hidden_states")
        if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
            raise RuntimeError("semantic VLM flow conditions require frozen_vlm_hidden_states")
        if any(not isinstance(value, Tensor) for value in hidden_states):
            raise TypeError("frozen VLM hidden states must be tensors")
        if any(value.requires_grad or torch.is_inference(value) for value in hidden_states):
            raise RuntimeError("frozen VLM hidden states must be detached normal tensors")
        attention_mask = conditions["prompt_attention_mask"]
        video_features, audio_features = self._embeddings_processor.feature_extractor(
            tuple(hidden_states),
            attention_mask,
            "right",
        )
        prepared = {key: value for key, value in conditions.items() if key != "frozen_vlm_hidden_states"}
        prepared["video_prompt_embeds"] = video_features
        if audio_features is not None:
            prepared["audio_prompt_embeds"] = audio_features
        self._last_bridge_metrics = {
            "train/bridge_input_rms": torch.stack([value.detach().float().pow(2).mean() for value in hidden_states])
            .mean()
            .sqrt(),
            "train/bridge_feature_output_rms": video_features.detach().float().pow(2).mean().sqrt(),
        }
        return prepared

    def postprocess_conditions_after_connector(
        self,
        batch: dict[str, Any],
        conditions: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        del batch
        self._last_bridge_metrics["train/bridge_output_rms"] = (
            conditions["video_prompt_embeds"].detach().float().pow(2).mean().sqrt()
        )
        return conditions

    def build_semantic_teacher_outputs(self, teacher_inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Contextualize all native 16x16 image tokens under frozen no-grad Gemma."""
        prefix_embeddings = teacher_inputs["prefix_inputs_embeds"].detach()
        prefix_attention_mask = teacher_inputs["prefix_attention_mask"].to(dtype=torch.bool)
        prefix_image_token_mask = teacher_inputs.get(
            "prefix_image_token_mask",
            teacher_inputs.get("prefix_reference_segment_mask"),
        )
        evidence = teacher_inputs["evidence_tokens"].detach()
        if evidence.ndim != 4 or evidence.shape[-2] != EVIDENCE_TOKENS_PER_FRAME:
            raise ValueError(f"semantic VLM evidence must be [B,F,256,D], got {tuple(evidence.shape)}")
        batch_size, frame_count, _tokens, gemma_dim = evidence.shape
        if gemma_dim != prefix_embeddings.shape[-1]:
            raise ValueError("prefix and evidence Gemma dimensions differ")
        suffix = evidence.reshape(batch_size, frame_count * EVIDENCE_TOKENS_PER_FRAME, gemma_dim)
        inputs_embeds = torch.cat([prefix_embeddings, suffix], dim=1)
        if inputs_embeds.shape[1] > self.config.vlm_teacher_max_length:
            raise ValueError(
                f"Gemma teacher length {inputs_embeds.shape[1]} exceeds {self.config.vlm_teacher_max_length}"
            )
        allowed = build_semantic_vlm_teacher_attention_mask(
            prefix_attention_mask,
            frame_count=frame_count,
            image_token_mask=prefix_image_token_mask,
        )
        suffix_valid = torch.ones(batch_size, suffix.shape[1], device=inputs_embeds.device, dtype=torch.bool)
        valid_token_mask = torch.cat([prefix_attention_mask, suffix_valid], dim=1)
        if prefix_image_token_mask is None:
            prefix_image_token_mask = torch.zeros_like(prefix_attention_mask)
        image_token_mask = torch.cat(
            [
                prefix_image_token_mask.to(device=inputs_embeds.device, dtype=torch.bool),
                suffix_valid,
            ],
            dim=1,
        )
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        language_model = self._get_language_model()
        language_model.eval()
        gemma_masks = build_gemma3_attention_masks(
            valid_token_mask=valid_token_mask,
            image_token_mask=image_token_mask,
            custom_visibility=allowed,
            sliding_window=resolve_gemma3_sliding_window(language_model),
            dtype=inputs_embeds.dtype,
        )
        with torch.no_grad():
            outputs = language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=gemma_masks.as_mapping(),
                position_ids=position_ids,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
        prefix_length = prefix_embeddings.shape[1]
        teacher_hidden = outputs.last_hidden_state[:, prefix_length:].reshape(
            batch_size,
            frame_count,
            EVIDENCE_TOKENS_PER_FRAME,
            gemma_dim,
        )
        if teacher_hidden.requires_grad:
            raise RuntimeError("frozen Gemma teacher hidden unexpectedly requires gradients")
        return {"teacher_hidden": teacher_hidden.detach()}

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        teacher = self.build_semantic_teacher_outputs(batch["semantic_teacher_inputs"])
        semantic_clean_grid = teacher["teacher_hidden"].detach()
        semantic_clean = semantic_clean_grid.flatten(1, 2)
        latents = batch["latents"]
        target_latents = latents["latents"]
        target_tokens = self._video_patchifier.patchify(target_latents)
        batch_size = target_tokens.shape[0]
        device = target_tokens.device
        sigma = timestep_sampler.sample_for(target_tokens)
        sigma_expanded = sigma.view(batch_size, 1, 1)
        target_noise = torch.randn_like(target_tokens)
        semantic_noise = torch.randn_like(semantic_clean)
        noisy_target = (1.0 - sigma_expanded) * target_tokens + sigma_expanded * target_noise
        noisy_semantic = (1.0 - sigma_expanded) * semantic_clean + sigma_expanded * semantic_noise
        video_flow_target = target_noise - target_tokens
        semantic_flow_target = semantic_noise - semantic_clean

        target_fps = latents.get(
            "fps",
            torch.full((batch_size,), float(DEFAULT_FPS), device=device),
        ).flatten()
        target_positions = self._get_video_positions(
            num_frames=int(latents["num_frames"][0].item()),
            height=int(latents["height"][0].item()),
            width=int(latents["width"][0].item()),
            batch_size=batch_size,
            fps=target_fps,
            device=device,
        )
        if target_positions.shape[2] != target_tokens.shape[1]:
            raise RuntimeError("target position/token count mismatch")
        normalized_timestamps = batch["semantic_teacher_inputs"]["normalized_timestamps"]
        expected_frames = self.semantic_frame_count_for_task(
            task=str(batch["task"][0]),
            pixel_frame_count=int(batch["latents"]["num_frames"][0].item()),
        )
        if semantic_clean_grid.shape[1:3] != (expected_frames, self.config.semantic_tokens_per_frame):
            raise RuntimeError(
                "semantic VLM teacher geometry mismatch: "
                f"actual={tuple(semantic_clean_grid.shape[1:3])}, "
                f"expected={(expected_frames, self.config.semantic_tokens_per_frame)}"
            )
        semantic_positions, _semantic_bounds = self._semantic_positions(
            target_positions,
            normalized_timestamps,
        )
        ref_tokens, ref_positions, ref_valid, ref_entities = self._reference_sequence(
            batch["reference_latents"],
            target_latents=target_latents,
            target_positions=target_positions,
        )
        ref_length = ref_tokens.shape[1]
        semantic_length = noisy_semantic.shape[1]
        target_length = target_tokens.shape[1]
        valid_tokens = torch.cat(
            [
                ref_valid,
                torch.ones(batch_size, semantic_length + target_length, device=device, dtype=torch.bool),
            ],
            dim=1,
        )
        attention_mask = build_compact_valid_token_mask(valid_tokens)
        token_type_ids = torch.cat(
            [
                torch.full((batch_size, ref_length), TYPE_REFERENCE, device=device, dtype=torch.long),
                torch.full((batch_size, semantic_length), TYPE_SEMANTIC, device=device, dtype=torch.long),
                torch.full((batch_size, target_length), TYPE_TARGET, device=device, dtype=torch.long),
            ],
            dim=1,
        )
        entity_ids = torch.cat(
            [
                ref_entities,
                torch.full(
                    (batch_size, semantic_length + target_length),
                    ENTITY_GLOBAL,
                    device=device,
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )
        timesteps = torch.cat(
            [
                torch.zeros(batch_size, ref_length, device=device, dtype=sigma.dtype),
                sigma[:, None].expand(batch_size, semantic_length + target_length),
            ],
            dim=1,
        )
        conditions = batch["conditions"]
        modality = Modality(
            enabled=True,
            latent=noisy_target,
            sigma=sigma,
            timesteps=timesteps,
            positions=torch.cat([ref_positions, semantic_positions, target_positions], dim=2),
            context=conditions["video_prompt_embeds"],
            context_mask=conditions["prompt_attention_mask"],
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            reference_latent=ref_tokens,
            semantic_latent=noisy_semantic,
        )

        condition_mode = str(batch["condition_mode"])
        if isinstance(batch["condition_mode"], (list, tuple)):
            condition_mode = str(batch["condition_mode"][0])
        if condition_mode not in PHASE2_CONDITION_MODES:
            raise RuntimeError(f"Invalid semantic VLM flow condition mode: {condition_mode!r}")
        self._condition_counts[condition_mode] += batch_size
        self._last_training_metrics = {
            **self._last_bridge_metrics,
            "train/anchor_frame_count": torch.tensor(float(expected_frames), device=device),
            "train/semantic_token_count": torch.tensor(float(semantic_length), device=device),
            "train/video_clean_rms": target_tokens.detach().float().pow(2).mean().sqrt(),
            "train/semantic_clean_rms": semantic_clean.float().pow(2).mean().sqrt(),
            "train/video_velocity_target_rms": video_flow_target.detach().float().pow(2).mean().sqrt(),
            "train/semantic_velocity_target_rms": semantic_flow_target.float().pow(2).mean().sqrt(),
            **{
                f"train/condition_count_{name}": torch.tensor(
                    float(self._condition_counts[name]),
                    device=device,
                )
                for name in PHASE2_CONDITION_MODES
            },
        }
        return ModelInputs(
            video=modality,
            audio=None,
            video_targets=video_flow_target,
            audio_targets=None,
            video_loss_mask=torch.ones(batch_size, target_length, device=device, dtype=torch.bool),
            audio_loss_mask=None,
            semantic_targets=semantic_flow_target,
            semantic_loss_mask=torch.ones(batch_size, semantic_length, device=device, dtype=torch.bool),
            sequence_offsets={
                "reference_end": ref_length,
                "semantic_end": ref_length + semantic_length,
                "target_end": ref_length + semantic_length + target_length,
            },
        )

    def compute_loss(
        self,
        video_pred: Tensor | SemanticVideoPrediction,
        _audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        if not isinstance(video_pred, SemanticVideoPrediction):
            raise TypeError("semantic_vlm_flow requires separated semantic/video predictions")
        video_loss = self._masked_token_mse(
            video_pred.video,
            inputs.video_targets,
            inputs.video_loss_mask,
        )
        semantic_loss = self._masked_token_mse(
            video_pred.semantic,
            inputs.semantic_targets,
            inputs.semantic_loss_mask,
        )
        weighted_video_loss = self.config.video_flow_weight * video_loss
        weighted_semantic_loss = self.config.semantic_flow_weight * semantic_loss
        total = weighted_video_loss + weighted_semantic_loss
        self._last_training_metrics.update(
            {
                "train/loss_video_flow": video_loss.detach().mean(),
                "train/loss_semantic_flow": semantic_loss.detach().mean(),
                "train/loss_video_flow_weighted": weighted_video_loss.detach().mean(),
                "train/loss_semantic_flow_weighted": weighted_semantic_loss.detach().mean(),
            }
        )
        return total

    def get_last_training_metrics(self) -> dict[str, Tensor]:
        return dict(self._last_training_metrics)

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "semantic_vlm_joint_flow_v1",
            "training_regime": "single_stage_full",
            "condition_factorization": "T_I_L_8way_v1",
            "gemma_dim": self._gemma_dim,
            "video_dim": self._video_dim,
            "semantic_anchor_count": self.config.semantic_anchor_count,
            "semantic_tokens_per_frame": self.config.semantic_tokens_per_frame,
            "required_fsdp_world_size": self.config.required_fsdp_world_size,
            "semantic_state_space": "frozen_gemma_last_hidden_state",
            "semantic_noise_space": "frozen_gemma_hidden",
            "semantic_flow_target": "epsilon_minus_teacher_hidden",
            "semantic_input_projection": "linear_gemma_to_dit_after_noising",
            "semantic_output_projection": "linear_dit_to_gemma_velocity",
            "semantic_teacher_gradient": "frozen_no_grad",
            "flow_loss_reduction": "independent_branch_mean_then_weighted_sum",
            "video_flow_weight": self.config.video_flow_weight,
            "semantic_flow_weight": self.config.semantic_flow_weight,
            "token_sequence": ["reference", "semantic", "target"],
            "reference_rope_layout_version": 3,
            "reference_rope_mode": self.config.reference_rope_mode,
            "reference_rope_temporal_slots": "shared_negative_adjacent",
            "reference_rope_spatial_shift": "height_width_adjacent",
            "semantic_rope_mode": "target_interpolated_16x16",
            "reference_type_embedding": "enabled",
            "reference_slot_embedding": "enabled",
            "semantic_type_embedding": "enabled",
        }

    def load_extra_checkpoint_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        checkpoint_metadata: dict[str, str] | None = None,
    ) -> None:
        architecture = (checkpoint_metadata or {}).get("architecture")
        if architecture is not None and architecture != "semantic_vlm_joint_flow_v1":
            raise RuntimeError(
                "Unsupported semantic VLM flow checkpoint architecture: "
                f"{architecture!r}"
            )
        del state_dict

    @staticmethod
    def _semantic_positions(
        target_positions: Tensor,
        normalized_timestamps: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Map every 16x16 semantic token to its target-space cell and anchor time."""
        batch_size, frame_count = normalized_timestamps.shape
        device, dtype = target_positions.device, target_positions.dtype
        timestamps = normalized_timestamps.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        t_min = target_positions[:, 0, :, 0].amin(dim=1)
        t_max = target_positions[:, 0, :, 1].amax(dim=1)
        h_min = target_positions[:, 1, :, 0].amin(dim=1)
        h_max = target_positions[:, 1, :, 1].amax(dim=1)
        w_min = target_positions[:, 2, :, 0].amin(dim=1)
        w_max = target_positions[:, 2, :, 1].amax(dim=1)
        anchor_time = t_min[:, None] + timestamps * (t_max - t_min)[:, None]

        grid = torch.arange(EVIDENCE_GRID_SIZE, device=device, dtype=dtype)
        row = grid[:, None].expand(-1, EVIDENCE_GRID_SIZE).reshape(-1)
        column = grid[None, :].expand(EVIDENCE_GRID_SIZE, -1).reshape(-1)
        h0_unit = row / EVIDENCE_GRID_SIZE
        h1_unit = (row + 1) / EVIDENCE_GRID_SIZE
        w0_unit = column / EVIDENCE_GRID_SIZE
        w1_unit = (column + 1) / EVIDENCE_GRID_SIZE
        h0 = h_min[:, None] + (h_max - h_min)[:, None] * h0_unit[None]
        h1 = h_min[:, None] + (h_max - h_min)[:, None] * h1_unit[None]
        w0 = w_min[:, None] + (w_max - w_min)[:, None] * w0_unit[None]
        w1 = w_min[:, None] + (w_max - w_min)[:, None] * w1_unit[None]
        spatial = torch.stack([torch.stack([h0, h1], dim=-1), torch.stack([w0, w1], dim=-1)], dim=1)
        spatial = spatial[:, None].expand(-1, frame_count, -1, -1, -1)
        temporal = torch.stack([anchor_time, anchor_time], dim=-1)[:, :, None, :]
        temporal = temporal.expand(-1, -1, EVIDENCE_TOKENS_PER_FRAME, -1).unsqueeze(3)
        spatial = spatial.permute(0, 1, 3, 2, 4)
        positions = torch.cat([temporal, spatial], dim=3)
        positions = positions.permute(0, 3, 1, 2, 4).reshape(batch_size, 3, -1, 2)

        bounds_per_frame = torch.stack(
            [
                timestamps[:, :, None].expand(-1, -1, EVIDENCE_TOKENS_PER_FRAME),
                timestamps[:, :, None].expand(-1, -1, EVIDENCE_TOKENS_PER_FRAME),
                h0_unit[None, None].expand(batch_size, frame_count, -1),
                h1_unit[None, None].expand(batch_size, frame_count, -1),
                w0_unit[None, None].expand(batch_size, frame_count, -1),
                w1_unit[None, None].expand(batch_size, frame_count, -1),
            ],
            dim=-1,
        )
        return positions, bounds_per_frame.reshape(batch_size, -1, 6)

    def prepare_inference_state(
        self,
        *,
        conditions: dict[str, Tensor],
        reference_latents: dict[str, Tensor],
        target_shape: VideoLatentShape,
        semantic_frame_count: int,
        pixel_frame_count: int,
        fps: float,
        seed: int,
        semantic_noise: Tensor | None = None,
        target_noise: Tensor | None = None,
    ) -> SemanticInferenceState:
        """Initialize strict-no-GT separated semantic and video states."""
        if semantic_frame_count < 1 or pixel_frame_count < semantic_frame_count:
            raise ValueError("semantic frame count must be positive and not exceed pixel frames")
        fps_value = float(fps)
        if not math.isfinite(fps_value) or fps_value <= 0:
            raise ValueError(f"Inference fps must be finite and positive, got {fps}")
        if self._gemma_dim is None or self._video_dim is None:
            raise RuntimeError("semantic VLM flow modules are not initialized")
        reference_tensor = reference_latents["latents"]
        device, dtype = reference_tensor.device, reference_tensor.dtype
        batch_size = int(reference_tensor.shape[0])
        if target_shape.batch != batch_size:
            raise ValueError("target and reference batch sizes differ")
        target_template = torch.zeros(target_shape.to_torch_shape(), device=device, dtype=dtype)
        target_template_tokens = self._video_patchifier.patchify(target_template)
        if target_template_tokens.shape[-1] != self._video_dim:
            raise ValueError(
                "target token width differs from pretrained patchify projection: "
                f"target={target_template_tokens.shape[-1]}, expected={self._video_dim}"
            )
        semantic_shape = (
            batch_size,
            semantic_frame_count * self.semantic_tokens_per_frame(),
            self._gemma_dim,
        )
        if (semantic_noise is None) != (target_noise is None):
            raise ValueError("semantic_noise and target_noise must be supplied together")
        if semantic_noise is None:
            generator = torch.Generator(device=device).manual_seed(int(seed))
            target_noise = torch.randn(
                target_template_tokens.shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            semantic_noise = torch.randn(
                semantic_shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            assert target_noise is not None
            self._validate_inference_noise(
                semantic_noise,
                name="semantic_noise",
                expected_shape=semantic_shape,
                device=device,
                dtype=dtype,
            )
            self._validate_inference_noise(
                target_noise,
                name="target_noise",
                expected_shape=tuple(target_template_tokens.shape),
                device=device,
                dtype=dtype,
            )

        target_positions = self._get_video_positions(
            num_frames=target_shape.frames,
            height=target_shape.height,
            width=target_shape.width,
            batch_size=batch_size,
            fps=fps_value,
            device=device,
        )
        timestamps = normalized_anchor_timestamps(
            frame_count=pixel_frame_count,
            anchor_count=semantic_frame_count,
            device=device,
            dtype=target_positions.dtype,
        ).unsqueeze(0).expand(batch_size, -1)
        semantic_positions, _bounds = self._semantic_positions(target_positions, timestamps)
        ref_tokens, ref_positions, ref_valid, ref_entities = self._reference_sequence(
            reference_latents,
            target_latents=target_template,
            target_positions=target_positions,
        )
        ref_length = ref_tokens.shape[1]
        semantic_length = semantic_noise.shape[1]
        target_length = target_noise.shape[1]
        valid = torch.cat(
            [
                ref_valid,
                torch.ones(
                    batch_size,
                    semantic_length + target_length,
                    device=device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
        token_type_ids = torch.cat(
            [
                torch.full((batch_size, ref_length), TYPE_REFERENCE, device=device, dtype=torch.long),
                torch.full((batch_size, semantic_length), TYPE_SEMANTIC, device=device, dtype=torch.long),
                torch.full((batch_size, target_length), TYPE_TARGET, device=device, dtype=torch.long),
            ],
            dim=1,
        )
        entity_ids = torch.cat(
            [
                ref_entities,
                torch.full(
                    (batch_size, semantic_length + target_length),
                    ENTITY_GLOBAL,
                    device=device,
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )
        sigma = torch.ones(batch_size, device=device, dtype=torch.float32)
        timesteps = torch.cat(
            [
                torch.zeros(batch_size, ref_length, device=device, dtype=torch.float32),
                sigma[:, None].expand(batch_size, semantic_length + target_length),
            ],
            dim=1,
        )
        modality = Modality(
            enabled=True,
            latent=target_noise,
            reference_latent=ref_tokens,
            semantic_latent=semantic_noise,
            sigma=sigma,
            timesteps=timesteps,
            positions=torch.cat([ref_positions, semantic_positions, target_positions], dim=2),
            context=conditions["video_prompt_embeds"],
            context_mask=conditions["prompt_attention_mask"],
            attention_mask=build_compact_valid_token_mask(valid),
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
        )
        return SemanticInferenceState(
            modality=modality,
            target_shape=target_shape,
            sequence_offsets={
                "reference_end": ref_length,
                "semantic_end": ref_length + semantic_length,
                "target_end": ref_length + semantic_length + target_length,
            },
            semantic_frame_count=semantic_frame_count,
        )

    @staticmethod
    def _require_separated_prediction(value: Any) -> SemanticVideoPrediction:
        if not isinstance(value, SemanticVideoPrediction):
            raise TypeError("transformer did not return separated semantic/video velocities")
        return value

    def _predict_guidance_denoised(
        self,
        *,
        transformer: nn.Module,
        branch: SemanticInferenceState,
        semantic_state: Tensor,
        video_state: Tensor,
        sigma: Tensor,
        reference_end: int,
        target_end: int,
        perturbations: Any = None,
    ) -> SemanticVideoPrediction:
        timesteps = torch.zeros_like(branch.modality.timesteps)
        timesteps[:, reference_end:target_end] = sigma[:, None]
        modality = replace(
            branch.modality,
            latent=video_state,
            semantic_latent=semantic_state,
            sigma=sigma,
            timesteps=timesteps,
        )
        raw, _ = transformer(video=modality, audio=None, perturbations=perturbations)
        velocity = self._require_separated_prediction(raw)
        return SemanticVideoPrediction(
            semantic=velocity_to_denoised(semantic_state, velocity.semantic, sigma),
            video=velocity_to_denoised(video_state, velocity.video, sigma),
        )

    @staticmethod
    def _combine_guidance_prediction(
        name: Literal["semantic", "video"],
        *,
        positive: SemanticVideoPrediction,
        negative: SemanticVideoPrediction | None,
        no_reference: SemanticVideoPrediction | None,
        no_latent_reference: SemanticVideoPrediction | None,
        empty_reference: SemanticVideoPrediction | None,
        empty_no_reference: SemanticVideoPrediction | None,
        stg: SemanticVideoPrediction | None,
        guidance: SemanticGuidanceConfig,
    ) -> Tensor:
        return combine_guided_denoised(
            positive=getattr(positive, name),
            negative=getattr(negative, name) if negative is not None else None,
            no_reference=getattr(no_reference, name) if no_reference is not None else None,
            no_latent_reference=(
                getattr(no_latent_reference, name) if no_latent_reference is not None else None
            ),
            empty_reference=getattr(empty_reference, name) if empty_reference is not None else None,
            empty_no_reference=(
                getattr(empty_no_reference, name) if empty_no_reference is not None else None
            ),
            stg=getattr(stg, name) if stg is not None else None,
            config=guidance,
        )

    @torch.inference_mode()
    def denoise_joint(
        self,
        *,
        transformer: nn.Module,
        state: SemanticInferenceState,
        num_inference_steps: int,
    ) -> tuple[Tensor, Tensor]:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        semantic_state = state.modality.semantic_latent
        if semantic_state is None:
            raise ValueError("semantic VLM inference state is missing semantic noise")
        video_state = state.modality.latent
        ref_end = state.sequence_offsets["reference_end"]
        target_end = state.sequence_offsets["target_end"]
        schedule = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=video_state.device)
        for sigma_value, next_sigma in pairwise(schedule):
            sigma = torch.full(
                (video_state.shape[0],),
                float(sigma_value.item()),
                device=video_state.device,
                dtype=torch.float32,
            )
            timesteps = torch.zeros_like(state.modality.timesteps)
            timesteps[:, ref_end:target_end] = sigma[:, None]
            modality = replace(
                state.modality,
                latent=video_state,
                semantic_latent=semantic_state,
                sigma=sigma,
                timesteps=timesteps,
            )
            prediction, _ = transformer(video=modality, audio=None, perturbations=None)
            velocity = self._require_separated_prediction(prediction)
            semantic_delta = (next_sigma - sigma_value).to(
                device=semantic_state.device,
                dtype=semantic_state.dtype,
            )
            video_delta = semantic_delta.to(device=video_state.device, dtype=video_state.dtype)
            semantic_state = semantic_state + semantic_delta * velocity.semantic
            video_state = video_state + video_delta * velocity.video
        return semantic_state, self._video_patchifier.unpatchify(video_state, state.target_shape)

    @torch.inference_mode()
    def denoise_joint_guided(  # noqa: PLR0915
        self,
        *,
        transformer: nn.Module,
        states: SemanticGuidanceStateBundle,
        guidance: SemanticGuidanceConfig,
        num_inference_steps: int,
    ) -> tuple[Tensor, Tensor]:
        if (
            not guidance.uses_factorized_til_guidance
            and guidance.guidance_scale == 1.0
            and guidance.ref_guidance_scale == 0.0
            and guidance.stg_scale == 0.0
            and guidance.guidance_rescale == 0.0
        ):
            return self.denoise_joint(
                transformer=transformer,
                state=states.positive,
                num_inference_steps=num_inference_steps,
            )
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        self.validate_guidance_state_bundle(states, guidance)
        positive = states.positive
        semantic_state = positive.modality.semantic_latent
        if semantic_state is None:
            raise ValueError("positive guidance state is missing semantic noise")
        video_state = positive.modality.latent
        ref_end = positive.sequence_offsets["reference_end"]
        target_end = positive.sequence_offsets["target_end"]
        perturbations = None
        if guidance.need_stg:
            blocks = getattr(transformer, "transformer_blocks", None)
            if blocks is None:
                raise ValueError("STG requires transformer.transformer_blocks")
            validate_stg_blocks(guidance.stg_blocks, transformer_block_count=len(blocks))
            perturbations = build_stg_perturbation(
                guidance.stg_blocks,
                batch_size=video_state.shape[0],
            )

        schedule = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=video_state.device)
        for sigma_value, next_sigma in pairwise(schedule):
            sigma = torch.full(
                (video_state.shape[0],),
                float(sigma_value.item()),
                device=video_state.device,
                dtype=torch.float32,
            )

            predict = partial(
                self._predict_guidance_denoised,
                transformer=transformer,
                semantic_state=semantic_state,
                video_state=video_state,
                sigma=sigma,
                reference_end=ref_end,
                target_end=target_end,
            )

            positive_denoised = predict(branch=positive)
            negative = predict(branch=states.negative) if guidance.need_negative else None
            no_reference = None
            no_latent_reference = None
            empty_reference = None
            empty_no_reference = None
            if guidance.uses_factorized_til_guidance:
                no_reference = predict(branch=states.no_reference)
                no_latent_reference = predict(branch=states.no_latent_reference)
            elif guidance.uses_q_reference_comparison and guidance.need_reference:
                no_reference = predict(branch=states.no_reference)
            elif guidance.uses_ql_reference_comparison and guidance.need_reference:
                no_latent_reference = predict(branch=states.no_latent_reference)
            elif guidance.need_control_pair:
                empty_reference = predict(branch=states.empty_reference)
                empty_no_reference = predict(branch=states.empty_no_reference)
            stg = predict(branch=positive, perturbations=perturbations) if guidance.need_stg else None

            guided_semantic, guided_video, _factor = rescale_guided_denoised_branches(
                positive_semantic=positive_denoised.semantic,
                guided_semantic=self._combine_guidance_prediction(
                    "semantic",
                    positive=positive_denoised,
                    negative=negative,
                    no_reference=no_reference,
                    no_latent_reference=no_latent_reference,
                    empty_reference=empty_reference,
                    empty_no_reference=empty_no_reference,
                    stg=stg,
                    guidance=guidance,
                ),
                positive_video=positive_denoised.video,
                guided_video=self._combine_guidance_prediction(
                    "video",
                    positive=positive_denoised,
                    negative=negative,
                    no_reference=no_reference,
                    no_latent_reference=no_latent_reference,
                    empty_reference=empty_reference,
                    empty_no_reference=empty_no_reference,
                    stg=stg,
                    guidance=guidance,
                ),
                guidance_rescale=guidance.guidance_rescale,
            )
            semantic_velocity = denoised_to_velocity(semantic_state, guided_semantic, sigma)
            video_velocity = denoised_to_velocity(video_state, guided_video, sigma)
            semantic_delta = (next_sigma - sigma_value).to(
                device=semantic_state.device,
                dtype=semantic_state.dtype,
            )
            video_delta = semantic_delta.to(device=video_state.device, dtype=video_state.dtype)
            semantic_state = semantic_state + semantic_delta * semantic_velocity
            video_state = video_state + video_delta * video_velocity
        return semantic_state, self._video_patchifier.unpatchify(video_state, positive.target_shape)

    @staticmethod
    def validate_guidance_state_bundle(
        states: SemanticGuidanceStateBundle,
        guidance: SemanticGuidanceConfig,
    ) -> None:
        required = [states.positive]
        if guidance.need_negative:
            if states.negative is None:
                raise ValueError("CFG requires a negative inference state")
            required.append(states.negative)
        if guidance.uses_factorized_til_guidance:
            if states.no_reference is None or states.no_latent_reference is None:
                raise ValueError("Factorized guidance requires T and I states")
            required.extend((states.no_reference, states.no_latent_reference))
        elif guidance.uses_q_reference_comparison and guidance.need_reference:
            if states.no_reference is None:
                raise ValueError("Reference guidance requires a Q state")
            required.append(states.no_reference)
        elif guidance.uses_ql_reference_comparison and guidance.need_reference:
            if states.no_latent_reference is None:
                raise ValueError("Latent-reference guidance requires a QL state")
            required.append(states.no_latent_reference)
        elif guidance.need_control_pair:
            if states.empty_reference is None or states.empty_no_reference is None:
                raise ValueError("Debiased guidance requires R and U states")
            required.extend((states.empty_reference, states.empty_no_reference))

        positive = states.positive
        offsets = positive.sequence_offsets
        ref_end = offsets["reference_end"]
        positive_semantic = positive.modality.semantic_latent
        positive_reference = positive.modality.reference_latent
        if positive_semantic is None or positive_reference is None:
            raise ValueError("positive state is not a separated semantic VLM flow state")
        for branch in required[1:]:
            if branch.sequence_offsets != offsets or branch.target_shape != positive.target_shape:
                raise ValueError("Guidance branches have different geometry")
            if branch.modality.semantic_latent is None or not torch.equal(
                branch.modality.semantic_latent,
                positive_semantic,
            ):
                raise ValueError("Guidance branches must share semantic initial noise")
            if not torch.equal(branch.modality.latent, positive.modality.latent):
                raise ValueError("Guidance branches must share video initial noise")
            for name in ("positions", "token_type_ids", "timesteps"):
                if not torch.equal(getattr(branch.modality, name), getattr(positive.modality, name)):
                    raise ValueError(f"Guidance branches differ in {name}")
            if not torch.equal(
                branch.modality.entity_ids[:, ref_end:],
                positive.modality.entity_ids[:, ref_end:],
            ):
                raise ValueError("Guidance branches differ in generated entity IDs")
            if not _attention_key_spans_equal(
                branch.modality.attention_mask,
                positive.modality.attention_mask,
                key_start=ref_end,
                key_end=offsets["target_end"],
                query_start=ref_end,
            ):
                raise ValueError("Guidance branches differ in generated attention layout")

        def require_inactive_reference(label: str, branch: Any) -> None:
            reference = branch.modality.reference_latent
            if reference is None or torch.count_nonzero(reference).item():
                raise ValueError(f"{label} reference tokens must be inactive")
            if _attention_allows_any_key(
                branch.modality.attention_mask,
                key_start=0,
                key_end=ref_end,
                query_start=ref_end,
            ):
                raise ValueError(f"{label} generated tokens must not attend to references")

        shared_reference_modes = {
            "debiased_ref",
            "standard_negative_latent_ref",
            "factorized_til_guidance",
        }
        if (
            states.negative is not None
            and guidance.guidance_mode not in shared_reference_modes
            and not torch.equal(states.negative.modality.reference_latent, positive_reference)
        ):
            raise ValueError("P and N must share reference latents")
        if guidance.uses_drop_all_negative and guidance.need_negative:
            require_inactive_reference("N0", states.negative)
        if guidance.uses_factorized_til_guidance:
            require_inactive_reference("T", states.no_reference)
            require_inactive_reference("I", states.no_latent_reference)
            if not torch.equal(states.no_latent_reference.modality.context, positive.modality.context):
                raise ValueError("P and I must share VLM context")
        if guidance.need_reference and guidance.uses_q_reference_comparison:
            require_inactive_reference("Q", states.no_reference)
        if guidance.need_reference and guidance.uses_ql_reference_comparison:
            require_inactive_reference("QL", states.no_latent_reference)
            if not torch.equal(states.no_latent_reference.modality.context, positive.modality.context):
                raise ValueError("P and QL must share VLM context")
        if guidance.guidance_mode == "debiased_ref":
            if states.negative is not None:
                require_inactive_reference("N", states.negative)
            if states.empty_no_reference is not None:
                require_inactive_reference("U", states.empty_no_reference)
            if guidance.need_control_pair:
                if not torch.equal(
                    states.empty_reference.modality.reference_latent,
                    positive_reference,
                ):
                    raise ValueError("P and R must share reference latents")
