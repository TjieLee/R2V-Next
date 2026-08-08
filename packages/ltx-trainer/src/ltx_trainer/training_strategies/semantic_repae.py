"""Single-stage semantic/video flow with frozen Gemma REPA-E supervision."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn

from ltx_core.model.transformer.modality import Modality
from ltx_core.multicond.gemma3_attention import build_gemma3_attention_masks, resolve_gemma3_sliding_window
from ltx_core.multicond.semantic_tokens import (
    EVIDENCE_TOKENS_PER_FRAME,
    REPAE_SEMANTIC_GRID_SIZE,
    REPAE_SEMANTIC_TOKENS_PER_FRAME,
    SemanticInputProjection,
    SemanticRepaProjector,
    build_semantic_repae_teacher_attention_mask,
    semantic_projection_smooth_l1_loss,
    semantic_repa_loss,
)
from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
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
    build_compact_valid_token_mask,
)
from ltx_trainer.training_strategies.semantic_flow_bridge import (
    configure_phase2_bridge_trainability,
    resolve_phase2_bridge_modules,
)

DEFAULT_REPAE_CONDITION_PROBABILITIES = {
    "til_111": 0.50,
    "til_110": 0.10,
    "til_101": 0.10,
    "til_011": 0.10,
    "til_100": 0.05,
    "til_010": 0.05,
    "til_001": 0.05,
    "til_000": 0.05,
}


def scale_semantic_dit_input_gradient(value: Tensor, scale: float) -> Tensor:
    """Preserve ``value`` exactly while scaling its upstream autograd Jacobian."""
    detached = value.detach()
    return detached + scale * (value - detached)


class SemanticRepaEConfig(TrainingStrategyConfigBase):
    """Configuration for the independent single-stage semantic REPA-E route."""

    name: Literal["semantic_repae"] = "semantic_repae"
    reference_latents_dir: str = "reference_latents"
    conditions_dir: str = "conditions"
    max_ref_images_per_sample: int = Field(default=4, ge=1, le=MAX_REFERENCE_ENTITIES)
    reference_rope_mode: Literal[
        "negative_adjacent_shifted_hw",
        "negative_adjacent_aligned_hw",
    ] = "negative_adjacent_aligned_hw"
    required_fsdp_world_size: Literal[8] = 8

    semantic_anchor_count: int = Field(default=8, ge=1)
    semantic_grid_size: int = Field(default=16, ge=1)
    semantic_tokens_per_frame: int = Field(default=256, ge=1)
    semantic_hidden_dim: int = Field(default=512, ge=1)
    repa_hidden_dim: int = Field(default=1024, ge=1)
    semantic_repa_block: int = Field(default=16, ge=1)
    vlm_prefix_max_length: int = Field(default=2560, ge=128)
    vlm_teacher_max_length: int = Field(default=6656, ge=2560)

    video_flow_weight: float = Field(default=1.0, ge=0.0)
    semantic_flow_weight: float = Field(default=1.0, ge=0.0)
    semantic_projection_repa_weight: float = Field(default=1.0, ge=0.0)
    semantic_dit_repa_weight: float = Field(default=0.5, ge=0.0)
    semantic_dit_input_gradient_scale: float = Field(default=0.1, ge=0.0, le=1.0)
    condition_probabilities: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_REPAE_CONDITION_PROBABILITIES)
    )

    @model_validator(mode="after")
    def _validate_repae_contract(self) -> "SemanticRepaEConfig":
        fixed_values = {
            "semantic_anchor_count": (self.semantic_anchor_count, 8),
            "semantic_grid_size": (self.semantic_grid_size, REPAE_SEMANTIC_GRID_SIZE),
            "semantic_tokens_per_frame": (
                self.semantic_tokens_per_frame,
                REPAE_SEMANTIC_TOKENS_PER_FRAME,
            ),
        }
        invalid = {name: value for name, (value, expected) in fixed_values.items() if value != expected}
        if invalid:
            raise ValueError(f"semantic_repae_v1 fixed geometry mismatch: {invalid}")
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


class SemanticRepaEStrategy(SemanticFlowStrategy):
    """Train full DiT, T/I/L bridge, semantic projection, and REPA heads from step zero."""

    config: SemanticRepaEConfig

    def __init__(self, config: SemanticRepaEConfig) -> None:
        TrainingStrategy.__init__(self, config)
        self._text_encoder: nn.Module | None = None
        self._embeddings_processor: nn.Module | None = None
        self._transformer: nn.Module | None = None
        self._semantic_input_projection: SemanticInputProjection | None = None
        self._semantic_repa_projector: SemanticRepaProjector | None = None
        self._dit_repa_projector: SemanticRepaProjector | None = None
        self._semantic_dim: int | None = None
        self._gemma_dim: int | None = None
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

    def semantic_tokens_per_frame(self) -> int:
        return self.config.semantic_tokens_per_frame

    def semantic_frame_count_for_task(self, *, task: str, pixel_frame_count: int) -> int:
        del pixel_frame_count
        if task == IMAGE_TASK:
            return 1
        if task == VIDEO_TASK:
            return self.config.semantic_anchor_count
        raise ValueError(f"Unsupported semantic REPA-E task: {task!r}")

    def attach_models(
        self,
        *,
        transformer: nn.Module,
        embeddings_processor: nn.Module,
        text_encoder: nn.Module | None = None,
    ) -> None:
        if text_encoder is None:
            raise ValueError("semantic_repae requires the frozen Gemma multimodal text encoder")
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
            raise ValueError("semantic_repae requires transformer.patchify_proj")
        self._semantic_dim = int(patchify_projection.in_features)
        self._dit_hidden_dim = int(getattr(transformer, "inner_dim"))
        if self._semantic_dim != int(transformer.proj_out.out_features):
            raise ValueError("semantic dimension must match transformer video token width")
        if self.config.semantic_repa_block > len(transformer.transformer_blocks):
            raise ValueError(
                f"semantic_repa_block={self.config.semantic_repa_block} exceeds "
                f"transformer block count {len(transformer.transformer_blocks)}"
            )

        parameter = next(input_embeddings.parameters())
        device, dtype = parameter.device, parameter.dtype
        self._semantic_input_projection = SemanticInputProjection(
            self._gemma_dim,
            self._semantic_dim,
            hidden_dim=self.config.semantic_hidden_dim,
        ).to(device=device, dtype=dtype)
        self._semantic_repa_projector = SemanticRepaProjector(
            self._semantic_dim,
            self._gemma_dim,
            hidden_dim=self.config.repa_hidden_dim,
        ).to(device=device, dtype=dtype)
        self._dit_repa_projector = SemanticRepaProjector(
            self._dit_hidden_dim,
            self._gemma_dim,
            hidden_dim=self.config.repa_hidden_dim,
        ).to(device=device, dtype=dtype)
        enable_repae = getattr(transformer, "enable_semantic_repae_conditioning", None)
        if not callable(enable_repae):
            raise ValueError("transformer does not expose enable_semantic_repae_conditioning")
        enable_repae(
            semantic_dim=self._semantic_dim,
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
            raise RuntimeError(f"Prepared semantic REPA-E bridge modules are incomplete: {sorted(modules)}")
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

    def get_trainable_modules(self) -> dict[str, nn.Module]:
        modules = {
            "semantic_input_projection": self._semantic_input_projection,
            "semantic_repa_projector": self._semantic_repa_projector,
            "dit_repa_projector": self._dit_repa_projector,
        }
        return {name: module for name, module in modules.items() if module is not None}

    def set_trainable_modules(self, modules: dict[str, nn.Module]) -> None:
        if "semantic_input_projection" in modules:
            self._semantic_input_projection = modules["semantic_input_projection"]  # type: ignore[assignment]
        if "semantic_repa_projector" in modules:
            self._semantic_repa_projector = modules["semantic_repa_projector"]  # type: ignore[assignment]
        if "dit_repa_projector" in modules:
            self._dit_repa_projector = modules["dit_repa_projector"]  # type: ignore[assignment]

    def prepare_conditions(self, batch: dict[str, Any], conditions: dict[str, Tensor]) -> dict[str, Tensor]:
        del batch
        if self._embeddings_processor is None:
            raise RuntimeError("semantic REPA-E embedding processor is not attached")
        hidden_states = conditions.get("frozen_vlm_hidden_states")
        if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
            raise RuntimeError("semantic REPA-E conditions require frozen_vlm_hidden_states")
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
        semantic_input, semantic_projector, _dit_projector = self._require_repae_modules()
        prefix_embeddings = teacher_inputs["prefix_inputs_embeds"].detach()
        prefix_attention_mask = teacher_inputs["prefix_attention_mask"].to(dtype=torch.bool)
        prefix_image_token_mask = teacher_inputs.get(
            "prefix_image_token_mask",
            teacher_inputs.get("prefix_reference_segment_mask"),
        )
        evidence = teacher_inputs["evidence_tokens"].detach()
        if evidence.ndim != 4 or evidence.shape[-2] != EVIDENCE_TOKENS_PER_FRAME:
            raise ValueError(f"semantic REPA-E evidence must be [B,F,256,D], got {tuple(evidence.shape)}")
        batch_size, frame_count, _tokens, gemma_dim = evidence.shape
        if gemma_dim != prefix_embeddings.shape[-1]:
            raise ValueError("prefix and evidence Gemma dimensions differ")
        suffix = evidence.reshape(batch_size, frame_count * EVIDENCE_TOKENS_PER_FRAME, gemma_dim)
        inputs_embeds = torch.cat([prefix_embeddings, suffix], dim=1)
        if inputs_embeds.shape[1] > self.config.vlm_teacher_max_length:
            raise ValueError(
                f"Gemma teacher length {inputs_embeds.shape[1]} exceeds {self.config.vlm_teacher_max_length}"
            )
        allowed = build_semantic_repae_teacher_attention_mask(
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
        semantic_clean = semantic_input(teacher_hidden)
        semantic_projection_repa_prediction = semantic_projector(semantic_clean)
        return {
            "teacher_hidden": teacher_hidden,
            "semantic_clean": semantic_clean,
            "semantic_projection_repa_prediction": semantic_projection_repa_prediction,
            "semantic_repa_target": teacher_hidden.detach(),
        }

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        teacher = self.build_semantic_teacher_outputs(batch["semantic_teacher_inputs"])
        semantic_clean_grid = teacher["semantic_clean"]
        semantic_clean = semantic_clean_grid.flatten(1, 2)
        semantic_clean_detached = semantic_clean.detach()
        semantic_clean_for_flow = scale_semantic_dit_input_gradient(
            semantic_clean,
            self.config.semantic_dit_input_gradient_scale,
        )
        latents = batch["latents"]
        target_latents = latents["latents"]
        target_tokens = self._video_patchifier.patchify(target_latents)
        batch_size = target_tokens.shape[0]
        device = target_tokens.device
        sigma = timestep_sampler.sample_for(target_tokens)
        sigma_expanded = sigma.view(batch_size, 1, 1)
        target_noise = torch.randn_like(target_tokens)
        semantic_noise = torch.randn_like(semantic_clean_for_flow)
        noisy_target = (1.0 - sigma_expanded) * target_tokens + sigma_expanded * target_noise
        noisy_semantic = (1.0 - sigma_expanded) * semantic_clean_for_flow + sigma_expanded * semantic_noise
        video_flow_target = target_noise - target_tokens
        semantic_flow_target = semantic_noise - semantic_clean_detached

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
                "semantic REPA-E teacher geometry mismatch: "
                f"actual={tuple(semantic_clean_grid.shape[1:3])}, "
                f"expected={(expected_frames, self.config.semantic_tokens_per_frame)}"
            )
        semantic_positions, semantic_bounds = self._semantic_positions(
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
        sequence = torch.cat([ref_tokens, noisy_semantic, noisy_target], dim=1)
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
        position_bounds = torch.cat(
            [
                torch.zeros(batch_size, ref_length, 6, device=device, dtype=semantic_bounds.dtype),
                semantic_bounds,
                torch.zeros(batch_size, target_length, 6, device=device, dtype=semantic_bounds.dtype),
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
            latent=sequence,
            sigma=sigma,
            timesteps=timesteps,
            positions=torch.cat([ref_positions, semantic_positions, target_positions], dim=2),
            context=conditions["video_prompt_embeds"],
            context_mask=conditions["prompt_attention_mask"],
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            semantic_position_bounds=position_bounds,
        )
        self._configure_dit_capture(reference_end=ref_length, semantic_end=ref_length + semantic_length)

        condition_mode = str(batch["condition_mode"])
        if isinstance(batch["condition_mode"], (list, tuple)):
            condition_mode = str(batch["condition_mode"][0])
        if condition_mode not in PHASE2_CONDITION_MODES:
            raise RuntimeError(f"Invalid semantic REPA-E condition mode: {condition_mode!r}")
        self._condition_counts[condition_mode] += batch_size
        self._last_training_metrics = {
            **self._last_bridge_metrics,
            "train/anchor_frame_count": torch.tensor(float(expected_frames), device=device),
            "train/semantic_token_count": torch.tensor(float(semantic_length), device=device),
            "train/semantic_latent_rms": semantic_clean.detach().float().pow(2).mean().sqrt(),
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
            semantic_projection_repa_prediction=teacher["semantic_projection_repa_prediction"].flatten(1, 2),
            semantic_repa_target=teacher["semantic_repa_target"].flatten(1, 2),
            sequence_offsets={
                "reference_end": ref_length,
                "semantic_end": ref_length + semantic_length,
                "target_end": ref_length + semantic_length + target_length,
            },
        )

    def compute_loss(
        self,
        video_pred: Tensor,
        _audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        if inputs.sequence_offsets is None:
            raise ValueError("semantic_repae ModelInputs are missing sequence offsets")
        reference_end = inputs.sequence_offsets["reference_end"]
        semantic_end = inputs.sequence_offsets["semantic_end"]
        target_end = inputs.sequence_offsets["target_end"]
        video_loss = self._masked_token_mse(
            video_pred[:, semantic_end:target_end],
            inputs.video_targets,
            inputs.video_loss_mask,
        )
        semantic_loss = self._masked_token_mse(
            video_pred[:, reference_end:semantic_end],
            inputs.semantic_targets,
            inputs.semantic_loss_mask,
        )
        if inputs.semantic_projection_repa_prediction is None or inputs.semantic_repa_target is None:
            raise ValueError("semantic_repae ModelInputs are missing REPA targets")
        projection_cosine_loss = semantic_repa_loss(
            inputs.semantic_projection_repa_prediction,
            inputs.semantic_repa_target,
        )
        projection_smooth_l1_loss = semantic_projection_smooth_l1_loss(
            inputs.semantic_projection_repa_prediction,
            inputs.semantic_repa_target,
        )
        projection_repa_loss = projection_cosine_loss + 0.1 * projection_smooth_l1_loss
        _semantic_input, _semantic_projector, dit_projector = self._require_repae_modules()
        captured = self._consume_dit_capture()
        if captured.shape[1] != semantic_end - reference_end:
            raise RuntimeError("captured DiT semantic span has the wrong length")
        dit_repa_loss = semantic_repa_loss(
            dit_projector(captured),
            inputs.semantic_repa_target,
        )
        weighted_video_loss = self.config.video_flow_weight * video_loss
        weighted_semantic_loss = self.config.semantic_flow_weight * semantic_loss
        weighted_projection_repa_loss = self.config.semantic_projection_repa_weight * projection_repa_loss
        weighted_dit_repa_loss = self.config.semantic_dit_repa_weight * dit_repa_loss
        total = weighted_video_loss + weighted_semantic_loss + weighted_projection_repa_loss + weighted_dit_repa_loss
        self._last_training_metrics.update(
            {
                "train/loss_video_flow": video_loss.detach().mean(),
                "train/loss_semantic_flow": semantic_loss.detach().mean(),
                "train/loss_semantic_projection_repa": projection_repa_loss.detach().mean(),
                "train/loss_semantic_projection_cosine": (projection_cosine_loss.detach().mean()),
                "train/loss_semantic_projection_smooth_l1": (projection_smooth_l1_loss.detach().mean()),
                "train/loss_semantic_dit_repa": dit_repa_loss.detach().mean(),
                "train/semantic_dit_input_gradient_scale": torch.tensor(
                    self.config.semantic_dit_input_gradient_scale,
                    device=video_pred.device,
                ),
                "train/loss_video_flow_weighted": weighted_video_loss.detach().mean(),
                "train/loss_semantic_flow_weighted": weighted_semantic_loss.detach().mean(),
                "train/loss_semantic_projection_repa_weighted": (weighted_projection_repa_loss.detach().mean()),
                "train/loss_semantic_dit_repa_weighted": (weighted_dit_repa_loss.detach().mean()),
            }
        )
        return total

    def get_last_training_metrics(self) -> dict[str, Tensor]:
        return dict(self._last_training_metrics)

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "semantic_repae_v1",
            "training_regime": "single_stage_full",
            "condition_factorization": "T_I_L_8way_v1",
            "semantic_dim": self._semantic_dim,
            "gemma_dim": self._gemma_dim,
            "semantic_anchor_count": self.config.semantic_anchor_count,
            "semantic_tokens_per_frame": self.config.semantic_tokens_per_frame,
            "required_fsdp_world_size": self.config.required_fsdp_world_size,
            "semantic_repa_block": self.config.semantic_repa_block,
            "semantic_teacher_gradient": "frozen_no_grad",
            "semantic_dit_input_gradient_scale": self.config.semantic_dit_input_gradient_scale,
            "semantic_flow_gradient_boundary": ("scaled_semantic_input_gradient_target_detached"),
            "semantic_flow_target_gradient": "detached",
            "projection_alignment_loss": "cosine_plus_0p1_smooth_l1_beta_1",
            "dit_repa_loss": "normalized_cosine_distance",
            "token_sequence": ["reference", "semantic", "target"],
            "reference_rope_layout_version": (
                3 if self.config.reference_rope_mode == "negative_adjacent_shifted_hw" else 4
            ),
            "reference_rope_mode": self.config.reference_rope_mode,
            "reference_rope_temporal_slots": "shared_negative_adjacent",
            "reference_rope_spatial_shift": (
                "height_width_adjacent"
                if self.config.reference_rope_mode == "negative_adjacent_shifted_hw"
                else "target_aligned"
            ),
            "semantic_rope_mode": "target_interpolated_16x16",
        }

    def load_extra_checkpoint_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        checkpoint_metadata: dict[str, str] | None = None,
    ) -> None:
        architecture = (checkpoint_metadata or {}).get("architecture")
        if architecture not in {None, "semantic_repae_v1"}:
            raise RuntimeError(f"Unsupported semantic REPA-E checkpoint architecture: {architecture!r}")
        if architecture == "semantic_repae_v1":
            required_prefixes = [f"training_strategy.{name}." for name in self.get_trainable_modules()]
            missing = [prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in state_dict)]
            if missing:
                raise RuntimeError(f"Semantic REPA-E checkpoint is missing strategy modules: {missing}")
        TrainingStrategy.load_extra_checkpoint_state_dict(
            self,
            state_dict,
            checkpoint_metadata=checkpoint_metadata,
        )

    def _require_repae_modules(
        self,
    ) -> tuple[SemanticInputProjection, SemanticRepaProjector, SemanticRepaProjector]:
        if (
            self._semantic_input_projection is None
            or self._semantic_repa_projector is None
            or self._dit_repa_projector is None
        ):
            raise RuntimeError("semantic REPA-E modules are not initialized; attach_models must run first")
        return (
            self._semantic_input_projection,
            self._semantic_repa_projector,
            self._dit_repa_projector,
        )

    def _transformer_core(self) -> nn.Module:
        if self._transformer is None:
            raise RuntimeError("semantic REPA-E transformer is not attached")
        transformer = self._transformer
        seen: set[int] = set()
        while id(transformer) not in seen:
            seen.add(id(transformer))
            if callable(getattr(transformer, "configure_semantic_repae_capture", None)):
                return transformer
            wrapped = getattr(transformer, "module", None)
            if wrapped is None:
                wrapped = getattr(transformer, "_fsdp_wrapped_module", None)
            if wrapped is None:
                wrapped = getattr(transformer, "_orig_mod", None)
            if not isinstance(wrapped, nn.Module):
                break
            transformer = wrapped
        raise RuntimeError("prepared transformer does not expose semantic REPA-E capture APIs")

    def _configure_dit_capture(self, *, reference_end: int, semantic_end: int) -> None:
        self._transformer_core().configure_semantic_repae_capture(
            block_index=self.config.semantic_repa_block - 1,
            semantic_start=reference_end,
            semantic_end=semantic_end,
        )

    def _consume_dit_capture(self) -> Tensor:
        return self._transformer_core().consume_semantic_repae_capture()

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

        grid = torch.arange(REPAE_SEMANTIC_GRID_SIZE, device=device, dtype=dtype)
        row = grid[:, None].expand(-1, REPAE_SEMANTIC_GRID_SIZE).reshape(-1)
        column = grid[None, :].expand(REPAE_SEMANTIC_GRID_SIZE, -1).reshape(-1)
        h0_unit = row / REPAE_SEMANTIC_GRID_SIZE
        h1_unit = (row + 1) / REPAE_SEMANTIC_GRID_SIZE
        w0_unit = column / REPAE_SEMANTIC_GRID_SIZE
        w1_unit = (column + 1) / REPAE_SEMANTIC_GRID_SIZE
        h0 = h_min[:, None] + (h_max - h_min)[:, None] * h0_unit[None]
        h1 = h_min[:, None] + (h_max - h_min)[:, None] * h1_unit[None]
        w0 = w_min[:, None] + (w_max - w_min)[:, None] * w0_unit[None]
        w1 = w_min[:, None] + (w_max - w_min)[:, None] * w1_unit[None]
        spatial = torch.stack([torch.stack([h0, h1], dim=-1), torch.stack([w0, w1], dim=-1)], dim=1)
        spatial = spatial[:, None].expand(-1, frame_count, -1, -1, -1)
        temporal = torch.stack([anchor_time, anchor_time], dim=-1)[:, :, None, :]
        temporal = temporal.expand(-1, -1, REPAE_SEMANTIC_TOKENS_PER_FRAME, -1).unsqueeze(3)
        spatial = spatial.permute(0, 1, 3, 2, 4)
        positions = torch.cat([temporal, spatial], dim=3)
        positions = positions.permute(0, 3, 1, 2, 4).reshape(batch_size, 3, -1, 2)

        bounds_per_frame = torch.stack(
            [
                timestamps[:, :, None].expand(-1, -1, REPAE_SEMANTIC_TOKENS_PER_FRAME),
                timestamps[:, :, None].expand(-1, -1, REPAE_SEMANTIC_TOKENS_PER_FRAME),
                h0_unit[None, None].expand(batch_size, frame_count, -1),
                h1_unit[None, None].expand(batch_size, frame_count, -1),
                w0_unit[None, None].expand(batch_size, frame_count, -1),
                w1_unit[None, None].expand(batch_size, frame_count, -1),
            ],
            dim=-1,
        )
        return positions, bounds_per_frame.reshape(batch_size, -1, 6)
