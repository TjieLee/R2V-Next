"""Reference-aware semantic/video joint flow matching strategy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ltx_core.multicond.gemma3_attention import build_gemma3_attention_masks, resolve_gemma3_sliding_window
from ltx_core.model.transformer.modality import Modality
from ltx_core.multicond.semantic_tokens import (
    EVIDENCE_TOKENS_PER_FRAME,
    SEMANTIC_GRID_SIZE,
    SEMANTIC_TOKENS_PER_FRAME,
    SemanticEncoder,
    SemanticQueryInitializer,
    SemanticReconstructionDecoder,
    build_semantic_teacher_attention_mask,
    gather_local_evidence,
    sample_semantic_keep_mask_with_stats,
    semantic_reconstruction_loss,
)
from ltx_core.types import VideoLatentShape
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)

TYPE_REFERENCE = 0
TYPE_SEMANTIC = 1
TYPE_TARGET = 2

ENTITY_GLOBAL = 0
ENTITY_REF_0 = 1
MAX_REFERENCE_ENTITIES = 4
REQUIRED_SEMANTIC_CHECKPOINT_MODULES = (
    "semantic_query",
    "semantic_encoder",
    "semantic_reconstruction_decoder",
)


@dataclass(frozen=True)
class SemanticInferenceState:
    """All state needed to integrate reference/semantic/video tokens jointly."""

    modality: Modality
    target_shape: VideoLatentShape
    sequence_offsets: dict[str, int]
    semantic_frame_count: int


class SemanticFlowConfig(TrainingStrategyConfigBase):
    """Configuration for frozen Gemma prefix plus joint semantic/video flow."""

    name: Literal["semantic_flow"] = "semantic_flow"
    reference_latents_dir: str = "reference_latents"
    conditions_dir: str = "conditions"
    max_ref_images_per_sample: int = Field(default=4, ge=1, le=MAX_REFERENCE_ENTITIES)

    anchor_frame_ratio: float = Field(default=0.10, gt=0.0, le=1.0)
    vlm_prefix_max_length: int = Field(default=2560, ge=128)
    vlm_teacher_max_length: int = Field(default=6656, ge=320)

    semantic_hidden_dim: int = Field(default=512, ge=1)
    semantic_position_gate_init: float = Field(default=0.0, ge=0.0, le=0.01)
    semantic_maximum_drop_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    semantic_minimum_tokens_per_frame: int = Field(default=48, ge=1, le=SEMANTIC_TOKENS_PER_FRAME)

    video_flow_weight: float = Field(default=1.0, ge=0.0)
    semantic_flow_weight: float = Field(default=1.0, ge=0.0)
    semantic_reconstruction_weight: float = Field(default=1.0, ge=0.0)

    condition_full_p: float = Field(default=0.70, ge=0.0)
    condition_drop_text_p: float = Field(default=0.10, ge=0.0)
    condition_drop_reference_all_p: float = Field(default=0.15, ge=0.0)
    condition_drop_all_p: float = Field(default=0.05, ge=0.0)

    @model_validator(mode="after")
    def _validate_semantic_architecture(self) -> "SemanticFlowConfig":
        if self.vlm_teacher_max_length < self.vlm_prefix_max_length + 320:
            raise ValueError("vlm_teacher_max_length must leave room for at least one evidence/query frame")
        probability_sum = (
            self.condition_full_p
            + self.condition_drop_text_p
            + self.condition_drop_reference_all_p
            + self.condition_drop_all_p
        )
        if abs(probability_sum - 1.0) > 1.0e-6:
            raise ValueError(f"condition dropout probabilities must sum to 1.0, got {probability_sum}")
        return self

    def get_data_sources(self) -> dict[str, str]:
        return {
            "latents": "latents",
            self.conditions_dir: "conditions",
            self.reference_latents_dir: "reference_latents",
        }


class SemanticFlowStrategy(TrainingStrategy):
    """Generate semantic and target video latents in one DiT self-attention stream."""

    config: SemanticFlowConfig

    def __init__(self, config: SemanticFlowConfig) -> None:
        super().__init__(config)
        self._text_encoder: nn.Module | None = None
        self._query_initializer: SemanticQueryInitializer | None = None
        self._semantic_encoder: SemanticEncoder | None = None
        self._reconstruction_decoder: SemanticReconstructionDecoder | None = None
        self._semantic_dim: int | None = None
        self._gemma_dim: int | None = None
        self._last_training_metrics: dict[str, Tensor] = {}

    def requires_text_encoder(self) -> bool:
        return True

    def train_text_encoder(self) -> bool:
        return False

    def train_embeddings_processor(self) -> bool:
        return False

    def attach_models(
        self,
        *,
        transformer: nn.Module,
        embeddings_processor: nn.Module,
        text_encoder: nn.Module | None = None,
    ) -> None:
        del embeddings_processor
        if text_encoder is None:
            raise ValueError("semantic_flow requires the frozen Gemma multimodal text encoder")
        self._text_encoder = text_encoder
        self._text_encoder.requires_grad_(False).eval()

        language_model = self._get_language_model()
        self._enable_frozen_teacher_gradient_checkpointing(language_model)
        input_embeddings = self._get_input_embeddings(language_model)
        self._gemma_dim = int(input_embeddings.weight.shape[-1])
        patchify_projection = getattr(transformer, "patchify_proj", None)
        if patchify_projection is None:
            raise ValueError("semantic_flow requires transformer.patchify_proj")
        self._semantic_dim = int(patchify_projection.in_features)
        if self._semantic_dim != int(transformer.proj_out.out_features):
            raise ValueError("semantic dimension must match transformer video input/output token width")

        parameter = next(input_embeddings.parameters())
        device, dtype = parameter.device, parameter.dtype
        self._query_initializer = SemanticQueryInitializer(
            self._gemma_dim,
            position_gate_init=self.config.semantic_position_gate_init,
        ).to(device=device, dtype=dtype)
        self._semantic_encoder = SemanticEncoder(
            self._gemma_dim,
            self._semantic_dim,
            hidden_dim=self.config.semantic_hidden_dim,
        ).to(device=device, dtype=dtype)
        self._reconstruction_decoder = SemanticReconstructionDecoder(
            self._semantic_dim,
            self._gemma_dim,
            hidden_dim=self.config.semantic_hidden_dim,
        ).to(device=device, dtype=dtype)

        enable_semantic_flow = getattr(transformer, "enable_semantic_flow_conditioning", None)
        if not callable(enable_semantic_flow):
            raise ValueError("transformer does not expose enable_semantic_flow_conditioning")
        enable_semantic_flow(
            semantic_dim=self._semantic_dim,
            num_token_types=3,
            num_entities=1 + MAX_REFERENCE_ENTITIES,
            semantic_token_type_id=TYPE_SEMANTIC,
        )

    def get_trainable_modules(self) -> dict[str, nn.Module]:
        modules = {
            "semantic_query": self._query_initializer,
            "semantic_encoder": self._semantic_encoder,
            "semantic_reconstruction_decoder": self._reconstruction_decoder,
        }
        return {name: module for name, module in modules.items() if module is not None}

    def set_trainable_modules(self, modules: dict[str, nn.Module]) -> None:
        if "semantic_query" in modules:
            self._query_initializer = modules["semantic_query"]  # type: ignore[assignment]
        if "semantic_encoder" in modules:
            self._semantic_encoder = modules["semantic_encoder"]  # type: ignore[assignment]
        if "semantic_reconstruction_decoder" in modules:
            self._reconstruction_decoder = modules["semantic_reconstruction_decoder"]  # type: ignore[assignment]

    def load_extra_checkpoint_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Load semantic-flow adapters only from a complete, shape-compatible checkpoint."""
        modules = self.get_trainable_modules()
        missing_modules = [
            name
            for name in REQUIRED_SEMANTIC_CHECKPOINT_MODULES
            if name not in modules
        ]
        if missing_modules:
            raise RuntimeError(
                f"semantic_flow checkpoint modules are not initialized: {missing_modules}"
            )

        errors: list[str] = []
        module_states: dict[str, dict[str, Tensor]] = {}
        for name in REQUIRED_SEMANTIC_CHECKPOINT_MODULES:
            prefix = f"training_strategy.{name}."
            module_state = {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if not module_state:
                errors.append(f"{name}: missing prefix {prefix}")
                continue

            expected_state = modules[name].state_dict()
            missing_keys = sorted(set(expected_state) - set(module_state))
            unexpected_keys = sorted(set(module_state) - set(expected_state))
            shape_errors = [
                f"{key}: expected {tuple(expected_state[key].shape)}, got {tuple(value.shape)}"
                for key, value in module_state.items()
                if key in expected_state and tuple(value.shape) != tuple(expected_state[key].shape)
            ]
            if missing_keys:
                errors.append(f"{name}: missing keys {missing_keys}")
            if unexpected_keys:
                errors.append(f"{name}: unexpected keys {unexpected_keys}")
            if shape_errors:
                errors.append(f"{name}: shape mismatch {shape_errors}")
            module_states[name] = module_state

        if errors:
            raise RuntimeError("semantic_flow checkpoint is incomplete: " + "; ".join(errors))

        for name, module_state in module_states.items():
            modules[name].load_state_dict(module_state, strict=True)

    def build_semantic_teacher_outputs(self, teacher_inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run frozen Gemma with the prefix plus native evidence/local-query suffix."""
        query_initializer, semantic_encoder, reconstruction_decoder = self._require_semantic_modules()
        prefix_embeddings = teacher_inputs["prefix_inputs_embeds"].detach()
        prefix_attention_mask = teacher_inputs["prefix_attention_mask"].to(dtype=torch.bool)
        prefix_image_token_mask = teacher_inputs.get(
            "prefix_image_token_mask",
            teacher_inputs.get("prefix_reference_region_mask"),
        )
        evidence = teacher_inputs["evidence_tokens"].detach()
        normalized_timestamps = teacher_inputs["normalized_timestamps"]
        if evidence.shape[-2] != EVIDENCE_TOKENS_PER_FRAME:
            raise ValueError(f"semantic evidence must contain 256 native tokens per frame, got {evidence.shape[-2]}")
        if evidence.shape[-1] != prefix_embeddings.shape[-1]:
            raise ValueError("prefix and evidence Gemma dimensions differ")

        queries = query_initializer(evidence, normalized_timestamps)
        batch_size, frame_count = evidence.shape[:2]
        suffix = torch.cat([evidence, queries], dim=2).reshape(batch_size, -1, evidence.shape[-1])
        inputs_embeds = torch.cat([prefix_embeddings, suffix], dim=1)
        if inputs_embeds.shape[1] > self.config.vlm_teacher_max_length:
            raise ValueError(
                "Gemma teacher length exceeded: "
                f"prefix={prefix_embeddings.shape[1]}, frames={frame_count}, total={inputs_embeds.shape[1]}, "
                f"maximum={self.config.vlm_teacher_max_length}"
            )

        allowed = build_semantic_teacher_attention_mask(
            prefix_attention_mask,
            frame_count=frame_count,
            image_token_mask=prefix_image_token_mask,
        )
        suffix_valid = torch.ones(batch_size, suffix.shape[1], device=prefix_attention_mask.device, dtype=torch.bool)
        valid_token_mask = torch.cat([prefix_attention_mask, suffix_valid], dim=1)
        if prefix_image_token_mask is None:
            prefix_image_token_mask = torch.zeros_like(prefix_attention_mask)
        prefix_image_token_mask = prefix_image_token_mask.to(device=prefix_attention_mask.device, dtype=torch.bool)
        image_token_mask = torch.cat([prefix_image_token_mask, torch.zeros_like(suffix_valid)], dim=1)
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
        outputs = language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=gemma_masks.as_mapping(),
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        final_hidden = outputs.hidden_states[-1]
        prefix_length = prefix_embeddings.shape[1]
        suffix_hidden = final_hidden[:, prefix_length:].reshape(batch_size, frame_count, 320, -1)
        evidence_hidden = suffix_hidden[:, :, :EVIDENCE_TOKENS_PER_FRAME]
        query_hidden = suffix_hidden[:, :, EVIDENCE_TOKENS_PER_FRAME:]
        semantic_clean = semantic_encoder(query_hidden)
        reconstruction_prediction = reconstruction_decoder(semantic_clean)
        reconstruction_target = gather_local_evidence(evidence_hidden).detach()
        return {
            "prefix_hidden": final_hidden[:, :prefix_length],
            "query_hidden": query_hidden,
            "evidence_hidden": evidence_hidden,
            "semantic_clean": semantic_clean,
            "reconstruction_prediction": reconstruction_prediction,
            "reconstruction_target": reconstruction_target,
        }

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        teacher = self.build_semantic_teacher_outputs(batch["semantic_teacher_inputs"])
        semantic_clean = teacher["semantic_clean"]
        reconstruction_prediction = teacher["reconstruction_prediction"]
        reconstruction_target = teacher["reconstruction_target"]

        latents = batch["latents"]
        target_latents = latents["latents"]
        target_tokens = self._video_patchifier.patchify(target_latents)
        batch_size = target_tokens.shape[0]
        device = target_tokens.device
        sigma = timestep_sampler.sample_for(target_tokens)
        sigma_expanded = sigma.view(batch_size, 1, 1)

        target_noise = torch.randn_like(target_tokens)
        noisy_target = (1.0 - sigma_expanded) * target_tokens + sigma_expanded * target_noise
        video_flow_target = target_noise - target_tokens
        target_timesteps = sigma.flatten()[:, None].expand(batch_size, target_tokens.shape[1])
        target_positions = self._get_video_positions(
            num_frames=int(latents["num_frames"][0].item()),
            height=int(latents["height"][0].item()),
            width=int(latents["width"][0].item()),
            batch_size=batch_size,
            fps=latents.get("fps", torch.full((batch_size,), float(DEFAULT_FPS), device=device)).flatten(),
            device=device,
        )

        keep_sample = sample_semantic_keep_mask_with_stats(
            semantic_clean,
            maximum_drop_rate=self.config.semantic_maximum_drop_rate,
            minimum_tokens_per_frame=self.config.semantic_minimum_tokens_per_frame,
        )
        keep_mask = keep_sample.keep_mask
        semantic_noise = torch.randn_like(semantic_clean)
        semantic_noisy_all = (1.0 - sigma[:, None, None, None]) * semantic_clean + (
            sigma[:, None, None, None] * semantic_noise
        )
        semantic_flow_target_all = semantic_noise - semantic_clean.detach()
        normalized_timestamps = batch["semantic_teacher_inputs"]["normalized_timestamps"]
        semantic_positions, semantic_bounds = self._semantic_positions(
            target_positions,
            normalized_timestamps,
        )
        semantic_tokens, semantic_targets, semantic_valid, semantic_positions, semantic_bounds = (
            self._pack_kept_semantic_tokens(
                semantic_noisy_all,
                semantic_flow_target_all,
                keep_mask,
                semantic_positions,
                semantic_bounds,
            )
        )
        prefix_attention_mask = batch["semantic_teacher_inputs"]["prefix_attention_mask"].to(device=device)
        query_initializer, semantic_encoder, _reconstruction_decoder = self._require_semantic_modules()
        semantic_token_count_before_dropout = torch.full(
            (batch_size,),
            semantic_clean.shape[1] * semantic_clean.shape[2],
            device=device,
            dtype=torch.float32,
        )
        semantic_token_count_kept = semantic_valid.sum(dim=1).to(dtype=torch.float32)
        self._last_training_metrics = {
            "train/semantic_requested_drop_rate": keep_sample.requested_drop_rate.to(device=device).detach().mean(),
            "train/semantic_drop_count_per_frame": keep_sample.drop_count_per_frame.to(
                device=device, dtype=torch.float32
            )
            .detach()
            .mean(),
            "train/semantic_token_count_before_dropout": semantic_token_count_before_dropout.detach().mean(),
            "train/semantic_token_count_kept": semantic_token_count_kept.detach().mean(),
            "train/semantic_keep_ratio": (
                semantic_token_count_kept / semantic_token_count_before_dropout.clamp_min(1.0)
            ).detach().mean(),
            "train/prefix_token_count": prefix_attention_mask.to(dtype=torch.float32).sum(dim=1).detach().mean(),
            "train/anchor_frame_count": torch.tensor(
                float(normalized_timestamps.shape[1]),
                device=device,
                dtype=torch.float32,
            ),
            "train/semantic_latent_rms": semantic_clean.detach().float().pow(2).mean().sqrt(),
            "train/query_position_gate": self._module_scalar(query_initializer, "position_gate", device),
            "train/semantic_global_scale": self._module_scalar(semantic_encoder, "global_scale", device),
        }

        ref_tokens, ref_positions, ref_valid, ref_entities = self._reference_sequence(
            batch["reference_latents"],
            target_latents=target_latents,
        )
        ref_length = ref_tokens.shape[1]
        semantic_length = semantic_tokens.shape[1]
        target_length = target_tokens.shape[1]
        sequence = torch.cat([ref_tokens, semantic_tokens, noisy_target], dim=1)
        timesteps = torch.cat(
            [
                torch.zeros(batch_size, ref_length, device=device, dtype=target_timesteps.dtype),
                sigma.flatten()[:, None].expand(batch_size, semantic_length),
                target_timesteps,
            ],
            dim=1,
        )
        positions = torch.cat([ref_positions, semantic_positions, target_positions], dim=2)
        valid_tokens = torch.cat(
            [
                ref_valid,
                semantic_valid,
                torch.ones(batch_size, target_length, device=device, dtype=torch.bool),
            ],
            dim=1,
        )
        attention_mask = valid_tokens[:, :, None] & valid_tokens[:, None, :]
        invalid = ~valid_tokens
        if invalid.any():
            diagonal = torch.eye(sequence.shape[1], device=device, dtype=torch.bool).unsqueeze(0)
            attention_mask |= diagonal & invalid[:, :, None]

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
        conditions = batch["conditions"]
        modality = Modality(
            enabled=True,
            latent=sequence,
            sigma=sigma,
            timesteps=timesteps,
            positions=positions,
            context=conditions["video_prompt_embeds"],
            context_mask=conditions["prompt_attention_mask"],
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            semantic_position_bounds=position_bounds,
        )
        return ModelInputs(
            video=modality,
            audio=None,
            video_targets=video_flow_target,
            audio_targets=None,
            video_loss_mask=torch.ones(batch_size, target_length, device=device, dtype=torch.bool),
            audio_loss_mask=None,
            semantic_targets=semantic_targets,
            semantic_loss_mask=semantic_valid,
            semantic_reconstruction_prediction=reconstruction_prediction,
            semantic_reconstruction_target=reconstruction_target,
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
            raise ValueError("semantic_flow ModelInputs are missing sequence offsets")
        reference_end = inputs.sequence_offsets["reference_end"]
        semantic_end = inputs.sequence_offsets["semantic_end"]
        target_end = inputs.sequence_offsets["target_end"]
        semantic_pred = video_pred[:, reference_end:semantic_end]
        target_pred = video_pred[:, semantic_end:target_end]

        video_loss = self._masked_token_mse(target_pred, inputs.video_targets, inputs.video_loss_mask)
        semantic_loss = self._masked_token_mse(
            semantic_pred,
            inputs.semantic_targets,
            inputs.semantic_loss_mask,
        )
        reconstruction_loss = semantic_reconstruction_loss(
            inputs.semantic_reconstruction_prediction,
            inputs.semantic_reconstruction_target,
        )
        total = (
            self.config.video_flow_weight * video_loss
            + self.config.semantic_flow_weight * semantic_loss
            + self.config.semantic_reconstruction_weight * reconstruction_loss
        )
        self._last_training_metrics = {
            **self._last_training_metrics,
            "train/loss_video_flow": video_loss.detach().mean(),
            "train/loss_semantic_flow": semantic_loss.detach().mean(),
            "train/loss_semantic_reconstruction": reconstruction_loss.detach().mean(),
        }
        return total

    def get_last_training_metrics(self) -> dict[str, Tensor]:
        return dict(self._last_training_metrics)

    def prepare_inference_state(
        self,
        *,
        conditions: dict[str, Tensor],
        reference_latents: dict[str, Tensor],
        target_shape: VideoLatentShape,
        semantic_frame_count: int,
        seed: int,
    ) -> SemanticInferenceState:
        """Initialize strict-no-GT joint state from references, context, and noise only."""
        if semantic_frame_count < 1:
            raise ValueError("semantic_frame_count must be positive")
        if self._semantic_dim is None:
            raise RuntimeError("semantic modules are not initialized; attach_models must run first")
        reference_latent_tensor = reference_latents["latents"]
        device, dtype = reference_latent_tensor.device, reference_latent_tensor.dtype
        batch_size = int(reference_latent_tensor.shape[0])
        if target_shape.batch != batch_size:
            raise ValueError(
                f"target batch {target_shape.batch} does not match references {batch_size}"
            )
        target_template = torch.zeros(target_shape.to_torch_shape(), device=device, dtype=dtype)
        target_template_tokens = self._video_patchifier.patchify(target_template)
        generator = torch.Generator(device=device).manual_seed(int(seed))
        target_noise = torch.randn(
            target_template_tokens.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        semantic_noise = torch.randn(
            (batch_size, semantic_frame_count * SEMANTIC_TOKENS_PER_FRAME, self._semantic_dim),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        target_positions = self._get_video_positions(
            num_frames=target_shape.frames,
            height=target_shape.height,
            width=target_shape.width,
            batch_size=batch_size,
            fps=float(DEFAULT_FPS),
            device=device,
        )
        normalized_timestamps = torch.linspace(
            0.0,
            1.0,
            semantic_frame_count,
            device=device,
            dtype=target_positions.dtype,
        ).unsqueeze(0).expand(batch_size, -1)
        semantic_positions, semantic_bounds = self._semantic_positions(
            target_positions,
            normalized_timestamps,
        )
        ref_tokens, ref_positions, ref_valid, ref_entities = self._reference_sequence(
            reference_latents,
            target_latents=target_template,
        )
        ref_length = ref_tokens.shape[1]
        semantic_length = semantic_noise.shape[1]
        target_length = target_noise.shape[1]
        sequence = torch.cat([ref_tokens, semantic_noise, target_noise], dim=1)
        valid = torch.cat(
            [
                ref_valid,
                torch.ones(batch_size, semantic_length + target_length, device=device, dtype=torch.bool),
            ],
            dim=1,
        )
        attention_mask = valid[:, :, None] & valid[:, None, :]
        invalid = ~valid
        if invalid.any():
            diagonal = torch.eye(sequence.shape[1], device=device, dtype=torch.bool).unsqueeze(0)
            attention_mask |= diagonal & invalid[:, :, None]
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

    @torch.inference_mode()
    def denoise_joint(
        self,
        *,
        transformer: nn.Module,
        state: SemanticInferenceState,
        num_inference_steps: int,
    ) -> tuple[Tensor, Tensor]:
        """Euler-integrate semantic and video flow together while references stay clean."""
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        offsets = state.sequence_offsets
        ref_end = offsets["reference_end"]
        semantic_end = offsets["semantic_end"]
        target_end = offsets["target_end"]
        latent = state.modality.latent
        schedule = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=latent.device)
        for sigma_value, next_sigma in zip(schedule[:-1], schedule[1:], strict=True):
            sigma = torch.full(
                (latent.shape[0],),
                float(sigma_value.item()),
                device=latent.device,
                dtype=torch.float32,
            )
            timesteps = torch.zeros_like(state.modality.timesteps)
            timesteps[:, ref_end:target_end] = sigma[:, None]
            step_modality = replace(state.modality, latent=latent, sigma=sigma, timesteps=timesteps)
            velocity, _ = transformer(video=step_modality, audio=None, perturbations=None)
            if velocity is None:
                raise RuntimeError("transformer returned no video velocity")
            velocity = velocity.clone()
            velocity[:, :ref_end] = 0
            delta = (next_sigma - sigma_value).to(device=latent.device, dtype=latent.dtype)
            updated = latent[:, ref_end:target_end] + delta * velocity[:, ref_end:target_end]
            latent = torch.cat([latent[:, :ref_end], updated], dim=1)
        semantic = latent[:, ref_end:semantic_end]
        target = latent[:, semantic_end:target_end]
        return semantic, self._video_patchifier.unpatchify(target, state.target_shape)

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "semantic_flow_v1",
            "semantic_dim": self._semantic_dim,
            "gemma_dim": self._gemma_dim,
            "token_sequence": ["reference", "semantic", "target"],
        }

    def _require_semantic_modules(
        self,
    ) -> tuple[SemanticQueryInitializer, SemanticEncoder, SemanticReconstructionDecoder]:
        if self._query_initializer is None or self._semantic_encoder is None or self._reconstruction_decoder is None:
            raise RuntimeError("semantic modules are not initialized; attach_models must run first")
        return self._query_initializer, self._semantic_encoder, self._reconstruction_decoder

    def _get_language_model(self) -> nn.Module:
        if self._text_encoder is None:
            raise RuntimeError("semantic_flow text encoder is not attached")
        text_encoder = getattr(self._text_encoder, "module", self._text_encoder)
        language_model = getattr(text_encoder.model.model, "language_model", None)
        if language_model is None:
            raise RuntimeError("Gemma text encoder does not expose model.model.language_model")
        return language_model

    def _enable_frozen_teacher_gradient_checkpointing(self, language_model: nn.Module) -> str:
        """Enable non-reentrant checkpointing while keeping the Gemma teacher frozen."""
        core = getattr(language_model, "module", language_model)
        config = getattr(core, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False

        enable_checkpointing = getattr(core, "gradient_checkpointing_enable", None)
        if callable(enable_checkpointing):
            try:
                enable_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
                return "native_non_reentrant"
            except TypeError:
                pass

        wrapped_layers = self._install_non_reentrant_decoder_checkpointing(core)
        if wrapped_layers > 0:
            return "manual_non_reentrant"
        raise RuntimeError(
            "Gemma language model does not support non-reentrant gradient checkpointing "
            "and no decoder layers were found for the manual fallback"
        )

    @staticmethod
    def _install_non_reentrant_decoder_checkpointing(language_model: nn.Module) -> int:
        layers = None
        for path in ("model.layers", "layers", "decoder.layers"):
            candidate: Any = language_model
            for part in path.split("."):
                candidate = getattr(candidate, part, None)
                if candidate is None:
                    break
            if isinstance(candidate, (nn.ModuleList, list, tuple)):
                layers = candidate
                break
        if layers is None:
            return 0

        wrapped = 0
        for layer in layers:
            if getattr(layer, "_semantic_flow_non_reentrant_checkpoint", False):
                continue
            original_forward = layer.forward

            def checkpointed_forward(*args: Any, _original_forward=original_forward, **kwargs: Any) -> Any:
                return checkpoint(_original_forward, *args, use_reentrant=False, **kwargs)

            layer.forward = checkpointed_forward  # type: ignore[method-assign]
            layer._semantic_flow_non_reentrant_checkpoint = True  # type: ignore[attr-defined]
            wrapped += 1
        return wrapped

    @staticmethod
    def _get_input_embeddings(language_model: nn.Module) -> nn.Module:
        core = getattr(language_model, "module", language_model)
        embeddings = core.get_input_embeddings()
        if embeddings is None:
            raise RuntimeError("Gemma language model has no input embeddings")
        return embeddings

    @staticmethod
    def _masked_token_mse(prediction: Tensor, target: Tensor | None, mask: Tensor | None) -> Tensor:
        if target is None or mask is None:
            raise ValueError("masked token MSE requires target and mask")
        if prediction.shape != target.shape:
            raise ValueError(f"prediction/target shape mismatch: {prediction.shape} != {target.shape}")
        token_loss = (prediction - target).pow(2).mean(dim=-1)
        weights = mask.to(dtype=token_loss.dtype)
        return (token_loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _module_scalar(module: nn.Module, name: str, device: torch.device) -> Tensor:
        for candidate in (module, getattr(module, "module", None), getattr(module, "_fsdp_wrapped_module", None)):
            if candidate is None:
                continue
            value = getattr(candidate, name, None)
            if isinstance(value, Tensor):
                return value.detach().float().mean()
        for parameter_name, parameter in module.named_parameters():
            if parameter_name.endswith(name):
                return parameter.detach().float().mean()
        return torch.tensor(float("nan"), device=device, dtype=torch.float32)

    def _reference_sequence(
        self,
        ref_data: dict[str, Tensor],
        *,
        target_latents: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ref_latents = ref_data["latents"]
        if ref_latents.ndim == 5:
            ref_latents = ref_latents.unsqueeze(3)
        if ref_latents.ndim != 6:
            raise ValueError(f"reference latents must be [B,R,C,F,H,W], got {tuple(ref_latents.shape)}")
        ref_latents = ref_latents[:, : self.config.max_ref_images_per_sample]
        batch_size, reference_count, channels, frames, height, width = ref_latents.shape
        ref_valid = ref_data.get("ref_valid_mask")
        if ref_valid is None:
            ref_valid = torch.ones(batch_size, reference_count, device=ref_latents.device, dtype=torch.bool)
        else:
            ref_valid = ref_valid[:, :reference_count].to(device=ref_latents.device, dtype=torch.bool)
        drop_reference = ref_data.get("drop_reference_all_mask")
        if drop_reference is not None:
            ref_valid &= ~drop_reference.to(device=ref_latents.device, dtype=torch.bool).flatten()[:, None]

        flat = ref_latents.reshape(batch_size * reference_count, channels, frames, height, width)
        ref_tokens = self._video_patchifier.patchify(flat)
        tokens_per_reference = ref_tokens.shape[1]
        ref_tokens = ref_tokens.reshape(batch_size, reference_count * tokens_per_reference, -1)
        positions = self._get_video_positions(
            num_frames=frames,
            height=height,
            width=width,
            batch_size=batch_size * reference_count,
            fps=1.0,
            device=ref_latents.device,
        )
        positions = positions.reshape(batch_size, reference_count, 3, tokens_per_reference, 2)
        target_height, target_width = target_latents.shape[-2:]
        positions[:, :, 1] *= float(target_height) / float(height)
        positions[:, :, 2] *= float(target_width) / float(width)
        positions = positions.permute(0, 2, 1, 3, 4).reshape(batch_size, 3, -1, 2)

        token_valid = ref_valid[:, :, None].expand(-1, -1, tokens_per_reference).reshape(batch_size, -1)
        ref_tokens = ref_tokens * token_valid.unsqueeze(-1).to(dtype=ref_tokens.dtype)
        entity = torch.arange(reference_count, device=ref_latents.device, dtype=torch.long) + ENTITY_REF_0
        entity = entity[None, :, None].expand(batch_size, -1, tokens_per_reference).reshape(batch_size, -1)
        entity = torch.where(token_valid, entity, torch.full_like(entity, ENTITY_GLOBAL))
        return ref_tokens, positions, token_valid, entity

    @staticmethod
    def _semantic_positions(
        target_positions: Tensor,
        normalized_timestamps: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, frame_count = normalized_timestamps.shape
        device = target_positions.device
        dtype = target_positions.dtype
        timestamps = normalized_timestamps.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        positions = torch.empty(
            batch_size,
            frame_count,
            SEMANTIC_TOKENS_PER_FRAME,
            3,
            2,
            device=device,
            dtype=dtype,
        )
        bounds = torch.empty(
            batch_size,
            frame_count,
            SEMANTIC_TOKENS_PER_FRAME,
            6,
            device=device,
            dtype=dtype,
        )
        for batch_index in range(batch_size):
            t_min = target_positions[batch_index, 0, :, 0].min()
            t_max = target_positions[batch_index, 0, :, 1].max()
            h_min = target_positions[batch_index, 1, :, 0].min()
            h_max = target_positions[batch_index, 1, :, 1].max()
            w_min = target_positions[batch_index, 2, :, 0].min()
            w_max = target_positions[batch_index, 2, :, 1].max()
            temporal_interval = (t_max - t_min).clamp_min(1.0) / max(1, frame_count)
            for frame_index in range(frame_count):
                normalized_t = timestamps[batch_index, frame_index]
                t_start = t_min + normalized_t * (t_max - t_min)
                t_start = torch.minimum(t_start, t_max - temporal_interval)
                t_start = torch.maximum(t_start, t_min)
                t_end = torch.minimum(t_max, t_start + temporal_interval)
                for row in range(SEMANTIC_GRID_SIZE):
                    for column in range(SEMANTIC_GRID_SIZE):
                        query_index = row * SEMANTIC_GRID_SIZE + column
                        h0 = h_min + (h_max - h_min) * row / SEMANTIC_GRID_SIZE
                        h1 = h_min + (h_max - h_min) * (row + 1) / SEMANTIC_GRID_SIZE
                        w0 = w_min + (w_max - w_min) * column / SEMANTIC_GRID_SIZE
                        w1 = w_min + (w_max - w_min) * (column + 1) / SEMANTIC_GRID_SIZE
                        positions[batch_index, frame_index, query_index] = torch.stack(
                            [torch.stack([t_start, t_end]), torch.stack([h0, h1]), torch.stack([w0, w1])]
                        )
                        bounds[batch_index, frame_index, query_index] = torch.stack(
                            [
                                normalized_t,
                                normalized_t,
                                torch.tensor(row / SEMANTIC_GRID_SIZE, device=device, dtype=dtype),
                                torch.tensor((row + 1) / SEMANTIC_GRID_SIZE, device=device, dtype=dtype),
                                torch.tensor(column / SEMANTIC_GRID_SIZE, device=device, dtype=dtype),
                                torch.tensor((column + 1) / SEMANTIC_GRID_SIZE, device=device, dtype=dtype),
                            ]
                        )
        positions = positions.permute(0, 3, 1, 2, 4).reshape(batch_size, 3, -1, 2)
        bounds = bounds.reshape(batch_size, -1, 6)
        return positions, bounds

    @staticmethod
    def _pack_kept_semantic_tokens(
        noisy: Tensor,
        targets: Tensor,
        keep_mask: Tensor,
        positions: Tensor,
        bounds: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, _frames, _queries, dim = noisy.shape
        flat_noisy = noisy.reshape(batch_size, -1, dim)
        flat_targets = targets.reshape(batch_size, -1, dim)
        flat_keep = keep_mask.reshape(batch_size, -1)
        lengths = flat_keep.sum(dim=1)
        maximum = int(lengths.max().item())
        packed_noisy = noisy.new_zeros(batch_size, maximum, dim)
        packed_targets = targets.new_zeros(batch_size, maximum, dim)
        packed_valid = torch.zeros(batch_size, maximum, device=noisy.device, dtype=torch.bool)
        packed_positions = positions.new_zeros(batch_size, 3, maximum, 2)
        packed_bounds = bounds.new_zeros(batch_size, maximum, 6)
        for batch_index in range(batch_size):
            selected = torch.nonzero(flat_keep[batch_index], as_tuple=False).flatten()
            count = selected.numel()
            packed_noisy[batch_index, :count] = flat_noisy[batch_index, selected]
            packed_targets[batch_index, :count] = flat_targets[batch_index, selected]
            packed_valid[batch_index, :count] = True
            packed_positions[batch_index, :, :count] = positions[batch_index, :, selected]
            packed_bounds[batch_index, :count] = bounds[batch_index, selected]
        return packed_noisy, packed_targets, packed_valid, packed_positions, packed_bounds
