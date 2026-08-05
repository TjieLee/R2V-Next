"""Reference-aware semantic/video joint flow matching strategy."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
import torch.utils.checkpoint
from pydantic import Field, model_validator
from torch import Tensor, nn

from ltx_core.model.transformer.modality import Modality
from ltx_core.multicond.gemma3_attention import build_gemma3_attention_masks, resolve_gemma3_sliding_window
from ltx_core.multicond.semantic_tokens import (
    EVIDENCE_TOKENS_PER_FRAME,
    SEMANTIC_GRID_SIZE,
    SEMANTIC_TOKENS_PER_FRAME,
    SemanticAlignmentHead,
    SemanticEncoder,
    SemanticQueryInitializer,
    SemanticReconstructionDecoder,
    build_semantic_teacher_attention_mask,
    gather_local_evidence,
    sample_semantic_keep_mask_with_stats,
    semantic_alignment_loss,
    semantic_reconstruction_loss,
)
from ltx_core.types import VideoLatentShape
from ltx_trainer import logger
from ltx_trainer.online_data.anchor_geometry import normalized_anchor_timestamps
from ltx_trainer.online_inference.semantic_guidance import (
    SemanticGuidanceConfig,
    SemanticGuidanceStateBundle,
    build_stg_perturbation,
    combine_guided_denoised,
    denoised_to_velocity,
    rescale_guided_denoised,
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
from ltx_trainer.training_strategies.semantic_flow_bridge import (
    configure_phase2_bridge_trainability,
    resolve_phase2_bridge_modules,
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
    "semantic_alignment_head",
)
SINGLE_VALUE_CHECKPOINT_PARAMETERS = {
    ("semantic_query", "position_gate"),
}
PHASE2_CONDITION_MODES = (
    "til_111",
    "til_110",
    "til_101",
    "til_011",
    "til_100",
    "til_010",
    "til_001",
    "til_000",
)
DEFAULT_PHASE2_CONDITION_PROBABILITIES = {
    "til_111": 0.50,
    "til_110": 0.10,
    "til_101": 0.10,
    "til_011": 0.10,
    "til_100": 0.05,
    "til_010": 0.05,
    "til_001": 0.05,
    "til_000": 0.05,
}


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
    training_phase: Literal["phase1", "phase2"] = "phase1"
    parent_checkpoint_step: int = Field(default=14000, ge=0)
    phase2_condition_probabilities: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_PHASE2_CONDITION_PROBABILITIES)
    )
    reference_latents_dir: str = "reference_latents"
    conditions_dir: str = "conditions"
    max_ref_images_per_sample: int = Field(default=4, ge=1, le=MAX_REFERENCE_ENTITIES)
    reference_rope_mode: Literal[
        "native_overlap",
        "appended_time_shifted_width",
    ] = "appended_time_shifted_width"

    anchor_frame_ratio: float = Field(default=0.10, gt=0.0, le=1.0)
    vlm_prefix_max_length: int = Field(default=2560, ge=128)
    vlm_teacher_max_length: int = Field(default=6656, ge=320)

    semantic_hidden_dim: int = Field(default=512, ge=1)
    semantic_alignment_hidden_dim: int = Field(default=1024, ge=1)
    semantic_position_gate_init: float = Field(default=0.0, ge=0.0, le=0.01)
    semantic_maximum_drop_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    semantic_minimum_tokens_per_frame: int = Field(default=48, ge=1, le=SEMANTIC_TOKENS_PER_FRAME)

    video_flow_weight: float = Field(default=1.0, ge=0.0)
    semantic_flow_weight: float = Field(default=1.0, ge=0.0)
    semantic_reconstruction_weight: float = Field(default=1.0, ge=0.0)
    semantic_alignment_weight: float = Field(default=1.0, ge=0.0)

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
        if self.training_phase == "phase1":
            return self
        phase2_keys = set(self.phase2_condition_probabilities)
        expected_phase2_keys = set(PHASE2_CONDITION_MODES)
        if phase2_keys != expected_phase2_keys:
            raise ValueError(
                "phase2_condition_probabilities must contain exactly "
                f"{list(PHASE2_CONDITION_MODES)}; missing={sorted(expected_phase2_keys - phase2_keys)}, "
                f"unexpected={sorted(phase2_keys - expected_phase2_keys)}"
            )
        invalid_phase2 = {
            name: probability
            for name, probability in self.phase2_condition_probabilities.items()
            if probability < 0.0
        }
        if invalid_phase2:
            raise ValueError(f"Phase 2 condition probabilities must be non-negative: {invalid_phase2}")
        phase2_sum = sum(self.phase2_condition_probabilities.values())
        if abs(phase2_sum - 1.0) > 1.0e-6:
            raise ValueError(f"Phase 2 condition probabilities must sum to 1.0, got {phase2_sum}")
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
        self._embeddings_processor: nn.Module | None = None
        self._query_initializer: SemanticQueryInitializer | None = None
        self._semantic_encoder: SemanticEncoder | None = None
        self._reconstruction_decoder: SemanticReconstructionDecoder | None = None
        self._semantic_alignment_head: SemanticAlignmentHead | None = None
        self._semantic_dim: int | None = None
        self._gemma_dim: int | None = None
        self._last_training_metrics: dict[str, Tensor] = {}
        self._last_bridge_metrics: dict[str, Tensor] = {}
        self._geometry_logged_tasks: set[str] = set()
        self.checkpoint_loaded_as_warm_start = False
        self.teacher_checkpointed_layer_count = 0
        self.teacher_checkpoint_forward_calls = 0
        self._phase2_condition_counts: Counter[str] = Counter()
        self.phase2_initialization_source: str | None = None
        self.phase2_parent_checkpoint_sha256: str | None = None
        self.phase2_loaded_bridge_key_count = 0

    def requires_text_encoder(self) -> bool:
        return True

    def train_text_encoder(self) -> bool:
        return False

    def train_embeddings_processor(self) -> bool:
        return self.config.training_phase == "phase2"

    def attach_models(
        self,
        *,
        transformer: nn.Module,
        embeddings_processor: nn.Module,
        text_encoder: nn.Module | None = None,
    ) -> None:
        self._embeddings_processor = embeddings_processor
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
        self._semantic_alignment_head = SemanticAlignmentHead(
            self._semantic_dim,
            self._gemma_dim,
            hidden_dim=self.config.semantic_alignment_hidden_dim,
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

    def configure_embeddings_processor_trainability(
        self,
        embeddings_processor: nn.Module,
    ) -> None:
        if self.config.training_phase != "phase2":
            raise RuntimeError("Phase 1 must not configure trainable embedding-processor modules")
        configure_phase2_bridge_trainability(embeddings_processor)

    def get_embeddings_processor_trainable_modules(
        self,
        embeddings_processor: nn.Module,
    ) -> dict[str, nn.Module]:
        if self.config.training_phase != "phase2":
            return {}
        return resolve_phase2_bridge_modules(embeddings_processor)

    def set_embeddings_processor_trainable_modules(
        self,
        embeddings_processor: nn.Module,
        modules: dict[str, nn.Module],
    ) -> None:
        if self.config.training_phase != "phase2":
            if modules:
                raise RuntimeError("Phase 1 received unexpected trainable bridge modules")
            return
        projection_names = [
            name
            for name in modules
            if name.startswith("feature_extractor.")
        ]
        if len(projection_names) != 1 or "video_connector" not in modules:
            raise RuntimeError(
                "Prepared Phase 2 bridge modules are incomplete: "
                f"{sorted(modules)}"
            )
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
        if self.config.training_phase == "phase2":
            for module in resolve_phase2_bridge_modules(
                self._embeddings_processor
            ).values():
                module.train()
        audio_connector = getattr(self._embeddings_processor, "audio_connector", None)
        if isinstance(audio_connector, nn.Module):
            audio_connector.eval()

    def get_trainable_modules(self) -> dict[str, nn.Module]:
        modules = {
            "semantic_query": self._query_initializer,
            "semantic_encoder": self._semantic_encoder,
            "semantic_reconstruction_decoder": self._reconstruction_decoder,
            "semantic_alignment_head": self._semantic_alignment_head,
        }
        return {name: module for name, module in modules.items() if module is not None}

    def set_trainable_modules(self, modules: dict[str, nn.Module]) -> None:
        if "semantic_query" in modules:
            self._query_initializer = modules["semantic_query"]  # type: ignore[assignment]
        if "semantic_encoder" in modules:
            self._semantic_encoder = modules["semantic_encoder"]  # type: ignore[assignment]
        if "semantic_reconstruction_decoder" in modules:
            self._reconstruction_decoder = modules["semantic_reconstruction_decoder"]  # type: ignore[assignment]
        if "semantic_alignment_head" in modules:
            self._semantic_alignment_head = modules["semantic_alignment_head"]  # type: ignore[assignment]

    def prepare_conditions(
        self,
        batch: dict[str, Any],
        conditions: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        del batch
        if self.config.training_phase == "phase1":
            return conditions
        if self._embeddings_processor is None:
            raise RuntimeError("Phase 2 embedding processor is not attached")
        hidden_states = conditions.get("frozen_vlm_hidden_states")
        if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
            raise RuntimeError("Phase 2 conditions require frozen_vlm_hidden_states")
        if any(not isinstance(value, Tensor) for value in hidden_states):
            raise TypeError("Phase 2 frozen VLM hidden states must be tensors")
        if any(value.requires_grad or torch.is_inference(value) for value in hidden_states):
            raise RuntimeError("Phase 2 frozen VLM hidden states must be detached normal tensors")
        attention_mask = conditions["prompt_attention_mask"]
        feature_extractor = self._embeddings_processor.feature_extractor
        video_features, audio_features = feature_extractor(
            tuple(hidden_states),
            attention_mask,
            "right",
        )
        prepared = {
            key: value
            for key, value in conditions.items()
            if key != "frozen_vlm_hidden_states"
        }
        prepared["video_prompt_embeds"] = video_features
        if audio_features is not None:
            prepared["audio_prompt_embeds"] = audio_features
        bridge_input = torch.stack(
            [value.detach().float().pow(2).mean() for value in hidden_states]
        ).mean().sqrt()
        self._last_bridge_metrics = {
            "train/bridge_input_rms": bridge_input,
            "train/bridge_feature_output_rms": video_features.detach().float().pow(2).mean().sqrt(),
        }
        return prepared

    def postprocess_conditions_after_connector(
        self,
        batch: dict[str, Any],
        conditions: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        del batch
        if self.config.training_phase == "phase2":
            self._last_bridge_metrics["train/bridge_output_rms"] = (
                conditions["video_prompt_embeds"].detach().float().pow(2).mean().sqrt()
            )
        return conditions

    def load_extra_checkpoint_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        checkpoint_metadata: dict[str, str] | None = None,
    ) -> None:
        """Strictly load v2 modules or explicitly warm-migrate a v1 checkpoint."""
        modules = self.get_trainable_modules()
        metadata = checkpoint_metadata or {}
        architecture = metadata.get("architecture")
        if architecture is None:
            has_alignment = any(
                key.startswith("training_strategy.semantic_alignment_head.")
                for key in state_dict
            )
            has_legacy_scale = "training_strategy.semantic_encoder.global_scale" in state_dict
            architecture = "semantic_flow_v1" if has_legacy_scale and not has_alignment else "semantic_flow_v2"
        if architecture not in {"semantic_flow_v1", "semantic_flow_v2"}:
            raise RuntimeError(f"Unsupported semantic-flow checkpoint architecture: {architecture!r}")
        self.checkpoint_loaded_as_warm_start = architecture == "semantic_flow_v1"
        if self.config.training_phase == "phase2":
            checkpoint_phase = metadata.get("training_phase", "phase1")
            if checkpoint_phase == "phase2":
                if metadata.get("condition_factorization") != "T_I_L_8way_v1":
                    raise RuntimeError("Phase 2 checkpoint has incompatible condition factorization")
                self.phase2_initialization_source = "phase2_resume"
                self.phase2_parent_checkpoint_sha256 = metadata.get("parent_checkpoint_sha256")
            elif checkpoint_phase == "phase1":
                parent_step = metadata.get("global_step")
                if parent_step is None or int(parent_step) != self.config.parent_checkpoint_step:
                    raise RuntimeError(
                        "Phase 2 warm start requires the configured Phase 1 parent step: "
                        f"expected={self.config.parent_checkpoint_step}, actual={parent_step!r}"
                    )
                self.phase2_initialization_source = "phase1_parent"
                self.checkpoint_loaded_as_warm_start = True
            else:
                raise RuntimeError(f"Unsupported semantic-flow training_phase metadata: {checkpoint_phase!r}")

        required_modules = (
            REQUIRED_SEMANTIC_CHECKPOINT_MODULES[:-1]
            if architecture == "semantic_flow_v1"
            else REQUIRED_SEMANTIC_CHECKPOINT_MODULES
        )
        missing_modules = [
            name
            for name in required_modules
            if name not in modules
        ]
        if missing_modules:
            raise RuntimeError(
                f"semantic_flow checkpoint modules are not initialized: {missing_modules}"
            )

        errors: list[str] = []
        module_states: dict[str, dict[str, Tensor]] = {}
        for name in required_modules:
            prefix = f"training_strategy.{name}."
            module_state = {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if not module_state:
                errors.append(f"{name}: missing prefix {prefix}")
                continue

            if architecture == "semantic_flow_v1" and name == "semantic_encoder":
                module_state.pop("global_scale", None)
            expected_state = modules[name].state_dict()
            for key, value in module_state.items():
                expected = expected_state.get(key)
                if (
                    (name, key) in SINGLE_VALUE_CHECKPOINT_PARAMETERS
                    and expected is not None
                    and value.numel() == 1
                    and expected.numel() == 1
                    and tuple(value.shape) != tuple(expected.shape)
                ):
                    module_state[key] = value.reshape(expected.shape)
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
        if architecture == "semantic_flow_v1":
            logger.warning(
                "Warm-migrating semantic_flow_v1 checkpoint to semantic_flow_v2: "
                "discarded semantic_encoder.global_scale, left semantic_alignment_head "
                "at its new initialization, and changed the representation gradient contract. "
                "Use a new output directory; this is not an exact resume."
            )

    def build_semantic_teacher_outputs(self, teacher_inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run frozen Gemma with the prefix plus native evidence/local-query suffix."""
        query_initializer, semantic_encoder, reconstruction_decoder, alignment_head = (
            self._require_semantic_modules()
        )
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
        frame_image_mask = torch.cat(
            [
                torch.ones(
                    EVIDENCE_TOKENS_PER_FRAME,
                    dtype=torch.bool,
                    device=prefix_attention_mask.device,
                ),
                torch.zeros(
                    SEMANTIC_TOKENS_PER_FRAME,
                    dtype=torch.bool,
                    device=prefix_attention_mask.device,
                ),
            ],
            dim=0,
        )
        suffix_image_token_mask = frame_image_mask.repeat(frame_count).unsqueeze(0).expand(batch_size, -1)
        image_token_mask = torch.cat([prefix_image_token_mask, suffix_image_token_mask], dim=1)
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
        alignment_prediction = alignment_head(semantic_clean)
        alignment_target = reconstruction_target.mean(dim=-2).detach()
        return {
            "prefix_hidden": final_hidden[:, :prefix_length],
            "query_hidden": query_hidden,
            "evidence_hidden": evidence_hidden,
            "semantic_clean": semantic_clean,
            "reconstruction_prediction": reconstruction_prediction,
            "reconstruction_target": reconstruction_target,
            "alignment_prediction": alignment_prediction,
            "alignment_target": alignment_target,
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
        alignment_prediction = teacher["alignment_prediction"]
        alignment_target = teacher["alignment_target"]

        latents = batch["latents"]
        target_latents = latents["latents"]
        target_tokens = self._video_patchifier.patchify(target_latents)
        batch_size = target_tokens.shape[0]
        device = target_tokens.device
        for key in ("num_frames", "height", "width"):
            values = latents[key].flatten()
            if values.numel() != batch_size:
                raise RuntimeError(
                    f"Latent geometry metadata for {key} has {values.numel()} values, expected {batch_size}"
                )
            if not torch.equal(values, values[:1].expand_as(values)):
                raise RuntimeError(f"Mixed latent geometry inside one batch for {key}: {values.tolist()}")
        sigma = timestep_sampler.sample_for(target_tokens)
        sigma_expanded = sigma.view(batch_size, 1, 1)

        target_noise = torch.randn_like(target_tokens)
        noisy_target = (1.0 - sigma_expanded) * target_tokens + sigma_expanded * target_noise
        video_flow_target = target_noise - target_tokens
        target_timesteps = sigma.flatten()[:, None].expand(batch_size, target_tokens.shape[1])
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
            raise RuntimeError(
                "Target position/token count mismatch: "
                f"positions={target_positions.shape[2]}, "
                f"tokens={target_tokens.shape[1]}, "
                f"latent_shape={tuple(target_latents.shape)}, "
                f"metadata_frames={latents['num_frames'].tolist()}, "
                f"metadata_height={latents['height'].tolist()}, "
                f"metadata_width={latents['width'].tolist()}"
            )

        semantic_for_dit = semantic_clean.detach()
        keep_sample = sample_semantic_keep_mask_with_stats(
            semantic_for_dit,
            maximum_drop_rate=self.config.semantic_maximum_drop_rate,
            minimum_tokens_per_frame=self.config.semantic_minimum_tokens_per_frame,
        )
        keep_mask = keep_sample.keep_mask
        semantic_noise = torch.randn_like(semantic_for_dit)
        semantic_noisy_all = (1.0 - sigma[:, None, None, None]) * semantic_for_dit + (
            sigma[:, None, None, None] * semantic_noise
        )
        semantic_flow_target_all = semantic_noise - semantic_for_dit
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
        query_initializer, _semantic_encoder, _reconstruction_decoder, _alignment_head = (
            self._require_semantic_modules()
        )
        semantic_token_count_before_dropout = torch.full(
            (batch_size,),
            semantic_clean.shape[1] * semantic_clean.shape[2],
            device=device,
            dtype=torch.float32,
        )
        semantic_token_count_kept = semantic_valid.sum(dim=1).to(dtype=torch.float32)
        self._last_training_metrics = {
            **self._last_bridge_metrics,
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
        }
        if self.config.training_phase == "phase2":
            raw_condition_mode = batch.get("condition_mode")
            condition_mode = (
                str(raw_condition_mode[0])
                if isinstance(raw_condition_mode, (list, tuple))
                else str(raw_condition_mode)
            )
            if condition_mode not in PHASE2_CONDITION_MODES:
                raise RuntimeError(f"Invalid Phase 2 condition mode in encoded batch: {condition_mode!r}")
            mode_index = PHASE2_CONDITION_MODES.index(condition_mode)
            t_active, i_active, l_active = (float(value) for value in condition_mode.removeprefix("til_"))
            self._phase2_condition_counts[condition_mode] += batch_size
            self._last_training_metrics.update(
                {
                    "train/condition_t": torch.tensor(t_active, device=device),
                    "train/condition_i": torch.tensor(i_active, device=device),
                    "train/condition_l": torch.tensor(l_active, device=device),
                    "train/condition_mode_id": torch.tensor(float(mode_index), device=device),
                    **{
                        f"train/condition_count_{name}": torch.tensor(
                            float(self._phase2_condition_counts[name]),
                            device=device,
                        )
                        for name in PHASE2_CONDITION_MODES
                    },
                }
            )

        ref_tokens, ref_positions, ref_valid, ref_entities = self._reference_sequence(
            batch["reference_latents"],
            target_latents=target_latents,
            target_positions=target_positions,
        )
        ref_length = ref_tokens.shape[1]
        semantic_length = semantic_tokens.shape[1]
        target_length = target_tokens.shape[1]
        reference_data = batch["reference_latents"]
        reference_capacity = int(reference_data["latents"].shape[1])
        ref_valid_mask = reference_data.get("ref_valid_mask")
        if ref_valid_mask is None:
            valid_reference_counts = torch.full(
                (batch_size,),
                reference_capacity,
                device=device,
                dtype=torch.long,
            )
        else:
            valid_reference_counts = ref_valid_mask.to(
                device=device,
                dtype=torch.bool,
            ).sum(dim=1)
        raw_task = batch.get("task", ["unknown"])
        task = str(raw_task[0]) if isinstance(raw_task, (list, tuple)) else str(raw_task)
        if task not in self._geometry_logged_tasks:
            logger.info(
                "semantic-flow geometry: "
                f"task={task} "
                f"target_latent={list(target_latents.shape)} "
                f"target_tokens={target_length} "
                f"target_positions={target_positions.shape[2]} "
                f"fps={float(target_fps[0].item())} "
                f"valid_references={valid_reference_counts.tolist()} "
                f"reference_capacity={reference_capacity} "
                f"reference_tokens={ref_length} "
                f"reference_positions={ref_positions.shape[2]} "
                f"reference_rope_mode={self.config.reference_rope_mode} "
                f"semantic_frames={semantic_clean.shape[1]} "
                f"semantic_tokens={int(semantic_valid.sum().item())}/"
                f"{semantic_clean.shape[1] * semantic_clean.shape[2]}"
            )
            self._geometry_logged_tasks.add(task)
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
            semantic_alignment_prediction=alignment_prediction,
            semantic_alignment_target=alignment_target,
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
        alignment_loss = semantic_alignment_loss(
            inputs.semantic_alignment_prediction,
            inputs.semantic_alignment_target,
        )
        total = (
            self.config.video_flow_weight * video_loss
            + self.config.semantic_flow_weight * semantic_loss
            + self.config.semantic_reconstruction_weight * reconstruction_loss
            + self.config.semantic_alignment_weight * alignment_loss
        )
        self._last_training_metrics = {
            **self._last_training_metrics,
            "train/loss_video_flow": video_loss.detach().mean(),
            "train/loss_semantic_flow": semantic_loss.detach().mean(),
            "train/loss_semantic_reconstruction": reconstruction_loss.detach().mean(),
            "train/loss_semantic_alignment": alignment_loss.detach().mean(),
        }
        if self.config.training_phase == "phase2":
            active_mode = max(
                PHASE2_CONDITION_MODES,
                key=lambda name: self._phase2_condition_counts[name],
            )
            raw_condition_mode = self._last_training_metrics.get("train/condition_mode_id")
            if raw_condition_mode is not None:
                active_mode = PHASE2_CONDITION_MODES[int(raw_condition_mode.item())]
            self._last_training_metrics.update(
                {
                    f"train/loss_video_flow_{active_mode}": video_loss.detach().mean(),
                    f"train/loss_semantic_flow_{active_mode}": semantic_loss.detach().mean(),
                    f"train/loss_semantic_reconstruction_{active_mode}": reconstruction_loss.detach().mean(),
                    f"train/loss_semantic_alignment_{active_mode}": alignment_loss.detach().mean(),
                }
            )
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
        pixel_frame_count: int,
        fps: float,
        seed: int,
        semantic_noise: Tensor | None = None,
        target_noise: Tensor | None = None,
    ) -> SemanticInferenceState:
        """Initialize strict-no-GT joint state from references, context, and noise only."""
        if semantic_frame_count < 1:
            raise ValueError("semantic_frame_count must be positive")
        if pixel_frame_count < 1:
            raise ValueError(f"pixel_frame_count must be positive, got {pixel_frame_count}")
        if semantic_frame_count > pixel_frame_count:
            raise ValueError(
                "semantic_frame_count cannot exceed pixel_frame_count: "
                f"semantic_frame_count={semantic_frame_count}, "
                f"pixel_frame_count={pixel_frame_count}"
            )
        fps_value = float(fps)
        if not math.isfinite(fps_value) or fps_value <= 0:
            raise ValueError(f"Inference fps must be finite and positive, got {fps}")
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
        if (semantic_noise is None) != (target_noise is None):
            raise ValueError("semantic_noise and target_noise must be supplied together")
        semantic_shape = (
            batch_size,
            semantic_frame_count * SEMANTIC_TOKENS_PER_FRAME,
            self._semantic_dim,
        )
        if semantic_noise is None or target_noise is None:
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
        normalized_timestamps = normalized_anchor_timestamps(
            frame_count=pixel_frame_count,
            anchor_count=semantic_frame_count,
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
            target_positions=target_positions,
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

    @staticmethod
    def _validate_inference_noise(
        value: Tensor,
        *,
        name: str,
        expected_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} does not match expected {expected_shape}"
            )
        if value.device != device:
            raise ValueError(f"{name} device {value.device} does not match {device}")
        if value.dtype != dtype:
            raise ValueError(f"{name} dtype {value.dtype} does not match {dtype}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")

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

    @torch.inference_mode()
    def denoise_joint_guided(  # noqa: PLR0915
        self,
        *,
        transformer: nn.Module,
        states: SemanticGuidanceStateBundle,
        guidance: SemanticGuidanceConfig,
        num_inference_steps: int,
    ) -> tuple[Tensor, Tensor]:
        """Guide one canonical semantic/video trajectory with cached condition branches."""
        if (
            guidance.guidance_scale == 1.0
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
        offsets = positive.sequence_offsets
        ref_end = offsets["reference_end"]
        semantic_end = offsets["semantic_end"]
        target_end = offsets["target_end"]
        semantic_length = semantic_end - ref_end
        latent = positive.modality.latent
        perturbations = None
        if guidance.need_stg:
            blocks = getattr(transformer, "transformer_blocks", None)
            if blocks is None:
                raise ValueError("STG requires transformer.transformer_blocks")
            validate_stg_blocks(
                guidance.stg_blocks,
                transformer_block_count=len(blocks),
            )
            perturbations = build_stg_perturbation(
                guidance.stg_blocks,
                batch_size=latent.shape[0],
            )

        schedule = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=latent.device)
        for sigma_value, next_sigma in zip(schedule[:-1], schedule[1:], strict=True):
            sigma_scalar = float(sigma_value.item())
            if not math.isfinite(sigma_scalar) or sigma_scalar <= 0.0:
                raise RuntimeError(f"Guided denoising requires positive sigma, got {sigma_scalar}")
            sigma = torch.full(
                (latent.shape[0],),
                sigma_scalar,
                device=latent.device,
                dtype=torch.float32,
            )
            current_generated = latent[:, ref_end:target_end]

            def predict(
                branch: SemanticInferenceState,
                *,
                branch_perturbations: Any = None,
            ) -> Tensor:
                branch_latent = torch.cat(
                    [branch.modality.latent[:, :ref_end], current_generated],
                    dim=1,
                )
                timesteps = torch.zeros_like(branch.modality.timesteps)
                timesteps[:, ref_end:target_end] = sigma[:, None]
                modality = replace(
                    branch.modality,
                    latent=branch_latent,
                    sigma=sigma,
                    timesteps=timesteps,
                )
                velocity, _ = transformer(
                    video=modality,
                    audio=None,
                    perturbations=branch_perturbations,
                )
                if velocity is None:
                    raise RuntimeError("transformer returned no video velocity")
                return velocity_to_denoised(
                    current_generated,
                    velocity[:, ref_end:target_end],
                    sigma,
                )

            denoised_positive = predict(positive)
            denoised_negative = predict(states.negative) if guidance.need_negative else None
            denoised_no_reference = None
            denoised_no_latent_reference = None
            denoised_empty_reference = None
            denoised_empty_no_reference = None
            if guidance.uses_q_reference_comparison:
                if guidance.need_reference:
                    denoised_no_reference = predict(states.no_reference)
            elif guidance.uses_ql_reference_comparison:
                if guidance.need_reference:
                    denoised_no_latent_reference = predict(
                        states.no_latent_reference
                    )
            elif guidance.need_control_pair:
                denoised_empty_reference = predict(states.empty_reference)
                denoised_empty_no_reference = predict(states.empty_no_reference)
            denoised_stg = (
                predict(positive, branch_perturbations=perturbations)
                if guidance.need_stg
                else None
            )
            guided = combine_guided_denoised(
                positive=denoised_positive,
                negative=denoised_negative,
                no_reference=denoised_no_reference,
                no_latent_reference=denoised_no_latent_reference,
                empty_reference=denoised_empty_reference,
                empty_no_reference=denoised_empty_no_reference,
                stg=denoised_stg,
                config=guidance,
            )
            guided, _factor = rescale_guided_denoised(
                positive_generated=denoised_positive,
                guided_generated=guided,
                semantic_token_count=semantic_length,
                guidance_rescale=guidance.guidance_rescale,
            )
            velocity = denoised_to_velocity(current_generated, guided, sigma)
            delta = (next_sigma - sigma_value).to(device=latent.device, dtype=latent.dtype)
            updated = current_generated + delta * velocity
            latent = torch.cat([latent[:, :ref_end], updated], dim=1)

        semantic = latent[:, ref_end:semantic_end]
        target = latent[:, semantic_end:target_end]
        return semantic, self._video_patchifier.unpatchify(target, positive.target_shape)

    @staticmethod
    def validate_guidance_state_bundle(  # noqa: PLR0912, PLR0915
        states: SemanticGuidanceStateBundle,
        guidance: SemanticGuidanceConfig,
    ) -> None:
        required = [states.positive]
        if guidance.need_negative:
            if states.negative is None:
                raise ValueError("CFG requires a negative inference state")
            required.append(states.negative)
        if guidance.uses_q_reference_comparison:
            if guidance.need_reference:
                if states.no_reference is None:
                    raise ValueError("Reference guidance requires a Q inference state")
                required.append(states.no_reference)
        elif guidance.uses_ql_reference_comparison:
            if guidance.need_reference:
                if states.no_latent_reference is None:
                    raise ValueError(
                        "Latent reference guidance requires a QL inference state"
                    )
                required.append(states.no_latent_reference)
        elif guidance.need_control_pair:
            if states.empty_reference is None or states.empty_no_reference is None:
                raise ValueError("Debiased reference guidance requires R and U inference states")
            required.extend((states.empty_reference, states.empty_no_reference))
        positive = states.positive
        offsets = positive.sequence_offsets
        ref_end = offsets["reference_end"]
        generated = positive.modality.latent[:, ref_end:]
        for branch in required[1:]:
            if branch.sequence_offsets != offsets:
                raise ValueError("Guidance branches have different sequence offsets")
            if branch.target_shape != positive.target_shape:
                raise ValueError("Guidance branches have different target shapes")
            if not torch.equal(branch.modality.latent[:, ref_end:], generated):
                raise ValueError("Guidance branches must share bitwise-identical generated noise")
            for name in ("positions", "token_type_ids", "semantic_position_bounds", "timesteps"):
                if not torch.equal(
                    getattr(branch.modality, name),
                    getattr(positive.modality, name),
                ):
                    raise ValueError(f"Guidance branches differ in {name}")
            if not torch.equal(
                branch.modality.entity_ids[:, ref_end:],
                positive.modality.entity_ids[:, ref_end:],
            ):
                raise ValueError("Guidance branches differ in generated-span entity_ids")
            if not torch.equal(
                branch.modality.attention_mask[:, ref_end:, ref_end:],
                positive.modality.attention_mask[:, ref_end:, ref_end:],
            ):
                raise ValueError("Guidance branches differ in generated-span attention layout")
        if (
            guidance.guidance_mode
            not in {"debiased_ref", "standard_negative_latent_ref"}
            and states.negative is not None
            and not torch.equal(
                states.negative.modality.latent[:, :ref_end],
                positive.modality.latent[:, :ref_end],
            )
        ):
            raise ValueError("P and N must share reference latents")
        if (
            guidance.guidance_mode
            not in {"debiased_ref", "standard_negative_latent_ref"}
            and states.negative is not None
        ):
            for name in ("attention_mask", "entity_ids"):
                if not torch.equal(
                    getattr(states.negative.modality, name),
                    getattr(positive.modality, name),
                ):
                    raise ValueError(f"P and N must share {name}")
        if guidance.uses_standard_drop_all_negative and guidance.need_negative:
            assert states.negative is not None
            negative = states.negative.modality
            if torch.count_nonzero(negative.latent[:, :ref_end]).item():
                raise ValueError("N0 reference tokens must be zero")
            if negative.attention_mask[:, ref_end:, :ref_end].any():
                raise ValueError(
                    "N0 generated tokens must not attend to reference tokens"
                )
        if guidance.need_reference and guidance.uses_q_reference_comparison:
            assert states.no_reference is not None
            if torch.count_nonzero(states.no_reference.modality.latent[:, :ref_end]).item():
                raise ValueError("Q reference tokens must be zero")
            attention = states.no_reference.modality.attention_mask
            if attention[:, ref_end:, :ref_end].any():
                raise ValueError("Q generated tokens must not attend to reference tokens")
        if guidance.need_reference and guidance.uses_ql_reference_comparison:
            assert states.no_latent_reference is not None
            no_latent_reference = states.no_latent_reference.modality
            if not torch.equal(
                no_latent_reference.context,
                positive.modality.context,
            ) or not torch.equal(
                no_latent_reference.context_mask,
                positive.modality.context_mask,
            ):
                raise ValueError("P and QL must share the same VLM condition")
            if torch.count_nonzero(
                no_latent_reference.latent[:, :ref_end]
            ).item():
                raise ValueError("QL reference tokens must be zero")
            if no_latent_reference.attention_mask[:, ref_end:, :ref_end].any():
                raise ValueError(
                    "QL generated tokens must not attend to reference tokens"
                )
        if guidance.guidance_mode == "debiased_ref":
            no_reference_states = [
                ("N", states.negative),
                ("U", states.empty_no_reference),
            ]
            for name, branch in no_reference_states:
                if branch is None:
                    continue
                if torch.count_nonzero(branch.modality.latent[:, :ref_end]).item():
                    raise ValueError(f"{name} reference tokens must be zero")
                if branch.modality.attention_mask[:, ref_end:, :ref_end].any():
                    raise ValueError(
                        f"{name} generated tokens must not attend to reference tokens"
                    )
            if guidance.need_control_pair:
                assert states.empty_reference is not None
                if not torch.equal(
                    states.empty_reference.modality.latent[:, :ref_end],
                    positive.modality.latent[:, :ref_end],
                ):
                    raise ValueError("P and R must share reference latents")
                for name in ("attention_mask", "entity_ids"):
                    if not torch.equal(
                        getattr(states.empty_reference.modality, name),
                        getattr(positive.modality, name),
                    ):
                        raise ValueError(f"P and R must share {name}")

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        appended = self.config.reference_rope_mode == "appended_time_shifted_width"
        metadata = {
            "architecture": "semantic_flow_v2",
            "semantic_dim": self._semantic_dim,
            "gemma_dim": self._gemma_dim,
            "semantic_encoder_dit_gradient": "detached",
            "semantic_alignment_target": "pooled_contextual_local_2x2",
            "semantic_alignment_head": "tokenwise_mlp",
            "semantic_velocity_head_init": "zero",
            "token_sequence": ["reference", "semantic", "target"],
            "reference_rope_layout_version": 2,
            "reference_rope_mode": self.config.reference_rope_mode,
            "reference_rope_temporal_slots": "fixed_after_target" if appended else "native_overlap",
            "reference_rope_spatial_shift": "width_adjacent" if appended else "native_overlap",
            "semantic_rope_mode": "target_interpolated_8x8",
        }
        if self.config.training_phase == "phase2":
            metadata.update(
                {
                    "training_phase": "phase2",
                    "parent_checkpoint_step": self.config.parent_checkpoint_step,
                    "parent_checkpoint_sha256": self.phase2_parent_checkpoint_sha256,
                    "train_dit": True,
                    "train_semantic_modules": True,
                    "train_conditioning_bridge": True,
                    "freeze_gemma": True,
                    "freeze_vision_tower": True,
                    "freeze_multimodal_projector": True,
                    "condition_factorization": "T_I_L_8way_v1",
                }
            )
        return metadata

    def _require_semantic_modules(
        self,
    ) -> tuple[
        SemanticQueryInitializer,
        SemanticEncoder,
        SemanticReconstructionDecoder,
        SemanticAlignmentHead,
    ]:
        if (
            self._query_initializer is None
            or self._semantic_encoder is None
            or self._reconstruction_decoder is None
            or self._semantic_alignment_head is None
        ):
            raise RuntimeError("semantic modules are not initialized; attach_models must run first")
        return (
            self._query_initializer,
            self._semantic_encoder,
            self._reconstruction_decoder,
            self._semantic_alignment_head,
        )

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

        wrapped_layers = self._install_non_reentrant_decoder_checkpointing(core)
        if wrapped_layers <= 0:
            raise RuntimeError(
                "No Gemma decoder layers were found for frozen-teacher "
                "non-reentrant checkpointing"
            )

        self.teacher_checkpointed_layer_count = wrapped_layers
        logger.info("frozen teacher checkpoint mode = manual_non_reentrant")
        logger.info("wrapped Gemma decoder layers = %d", wrapped_layers)
        return "manual_non_reentrant"

    def _install_non_reentrant_decoder_checkpointing(self, language_model: nn.Module) -> int:
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

        checkpointed_layers = 0
        for layer in layers:
            if getattr(layer, "_semantic_flow_non_reentrant_checkpoint", False):
                checkpointed_layers += 1
                continue
            original_forward = layer.forward

            def checkpointed_forward(*args: Any, _original_forward=original_forward, **kwargs: Any) -> Any:
                self.teacher_checkpoint_forward_calls += 1
                return torch.utils.checkpoint.checkpoint(
                    _original_forward,
                    *args,
                    use_reentrant=False,
                    **kwargs,
                )

            layer.forward = checkpointed_forward  # type: ignore[method-assign]
            layer._semantic_flow_original_forward = original_forward  # type: ignore[attr-defined]
            layer._semantic_flow_non_reentrant_checkpoint = True  # type: ignore[attr-defined]
            checkpointed_layers += 1
        return checkpointed_layers

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
        target_positions: Tensor,
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
        if positions.shape[2] != tokens_per_reference:
            raise RuntimeError(
                "Reference position/token count mismatch: "
                f"positions={positions.shape[2]}, tokens={tokens_per_reference}"
            )
        positions = positions.reshape(batch_size, reference_count, 3, tokens_per_reference, 2)
        if positions.shape[3] != tokens_per_reference:
            raise RuntimeError(
                "Reference position/token count mismatch after reshape: "
                f"positions={positions.shape[3]}, tokens={tokens_per_reference}"
            )
        target_height, target_width = target_latents.shape[-2:]
        positions[:, :, 1] *= float(target_height) / float(height)
        positions[:, :, 2] *= float(target_width) / float(width)
        if target_positions.shape[0] != batch_size or target_positions.ndim != 4 or target_positions.shape[1] != 3:
            raise ValueError(
                "target_positions must be [B,3,N,2] with the same batch as references, "
                f"got {tuple(target_positions.shape)}"
            )
        if target_positions.device != positions.device:
            raise ValueError("target_positions and reference positions must be on the same device")
        if self.config.reference_rope_mode == "appended_time_shifted_width":
            target_t_start = target_positions[:, 0, :, 0]
            target_t_end = target_positions[:, 0, :, 1]
            video_end = target_t_end.amax(dim=1)
            last_start = target_t_start.amax(dim=1)
            last_interval_width = video_end - last_start
            if not torch.isfinite(last_interval_width).all() or (last_interval_width <= 0).any():
                raise RuntimeError("Target final temporal interval must be finite and positive")
            delta_t = last_interval_width.clamp_min(1.0e-6)
            slot_ids = torch.arange(
                reference_count,
                device=positions.device,
                dtype=positions.dtype,
            )
            slot_start = video_end[:, None].to(dtype=positions.dtype) + slot_ids[None, :] * delta_t[
                :, None
            ].to(dtype=positions.dtype)
            slot_end = slot_start + delta_t[:, None].to(dtype=positions.dtype)
            positions[:, :, 0, :, 0] = slot_start[:, :, None]
            positions[:, :, 0, :, 1] = slot_end[:, :, None]

            target_w_max = target_positions[:, 2, :, 1].amax(dim=1).to(dtype=positions.dtype)
            ref_w_min = positions[:, :, 2, :, 0].amin(dim=2)
            w_shift = target_w_max[:, None] - ref_w_min
            positions[:, :, 2] = positions[:, :, 2] + w_shift[:, :, None, None]
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
