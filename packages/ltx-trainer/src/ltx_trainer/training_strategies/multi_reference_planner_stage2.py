"""Stage 2 Baton-style VLM planner training for multi-reference video generation.

Stage 1 teaches the DiT to consume text/thinking tokens plus frozen
SigLIP/projector visual tokens. Stage 2 keeps that DiT interface unchanged but
replaces the GT visual tokens with fixed-count VLM planner placeholder outputs.
The placeholder count must match the GT SigLIP/projector token count exactly;
an MSE loss aligns the VLM-predicted visual tokens to the GT visual tokens.
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import torch
import torch.nn.functional as F
from pydantic import Field, model_validator
from torch import Tensor, nn

from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.multicond.visual_tokens import (
    VisualPlannerTokens,
    extract_projected_visual_tokens,
    scatter_visual_tokens_into_embeddings,
)
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.multi_reference_video import (
    MultiReferenceVideoConfig,
    MultiReferenceVideoStrategy,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlannerInferenceConditionParts:
    """Explicit Stage 2 inference outputs for reusing one planner visual context."""

    connected_text_conditions: dict[str, Tensor]
    visual_context: Tensor
    visual_mask: Tensor
    final_conditions: dict[str, Tensor]
    diagnostics: dict[str, Any]


def configure_stage3_transformer_trainability(transformer: nn.Module) -> list[str]:
    """Freeze the base DiT and enable only the existing Stage 1 LoRA weights."""
    transformer.requires_grad_(False)
    trainable_names: list[str] = []
    for name, parameter in transformer.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(True)
            trainable_names.append(name)

    if not trainable_names:
        raise RuntimeError("Stage 3 requires trainable Stage 1 DiT LoRA parameters")
    non_lora_trainable = [
        name
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if non_lora_trainable:
        raise RuntimeError(f"Stage 3 must not train base DiT parameters: {non_lora_trainable[:20]}")
    return trainable_names


class MultiReferencePlannerStage2Config(MultiReferenceVideoConfig):
    """Stage 2 VLM planner token training."""

    name: Literal["multi_reference_planner_stage2"] = "multi_reference_planner_stage2"
    training_phase: Literal["stage2", "stage3"] = "stage2"
    train_stage1_dit_lora: bool = False

    visual_branch_enabled: bool = True
    visual_context_mode: Literal["qformer_512", "full_tokens_3d_sa"] = "full_tokens_3d_sa"
    visual_context_expected_tokens: int | None = 2048
    visual_context_max_tokens: int = 2048
    visual_connector_enabled: bool = False
    visual_token_source_dim: int | None = 3840
    visual_token_target_dim: int | None = 4096
    cfg_full_p: float = Field(default=0.65, ge=0.0)
    cfg_drop_text_p: float = Field(default=0.10, ge=0.0)
    cfg_drop_siglip_p: float = Field(default=0.10, ge=0.0)
    cfg_drop_ref_latents_p: float = Field(default=0.10, ge=0.0)
    cfg_drop_all_p: float | None = Field(default=0.05, ge=0.0)
    cfg_drop_ref_p: float = Field(default=0.0, ge=0.0)
    cfg_drop_planner_p: float = Field(default=0.0, ge=0.0)

    use_online_vlm: bool = Field(
        default=True,
        description="Run Gemma/VLM online during training. Offline mode expects precomputed predicted visual tokens.",
    )

    planner_vlm_inputs_dir: str = Field(
        default="planner_vlm_inputs",
        description="Directory containing tokenized Gemma/VLM inputs with fixed planner placeholder masks.",
    )

    planner_conditions_dir: str = Field(
        default="planner_conditions",
        description="Offline directory containing predicted visual tokens and masks.",
    )

    vlm_placeholder_mask_key: str = Field(
        default="planner_placeholder_mask",
        description="Bool mask selecting the fixed learnable planner placeholder positions in the VLM sequence.",
    )

    vlm_output_mask_key: str = Field(
        default="planner_output_mask",
        description=(
            "Bool [B,planner_token_count] mask selecting valid planner outputs. "
            "Missing legacy caches fall back to all placeholder outputs being valid."
        ),
    )

    vlm_hidden_layer: int = Field(
        default=-1,
        description="Gemma language-model hidden-state layer used as predicted visual tokens.",
    )

    vlm_lm_labels_key: str = Field(
        default="ntp_labels",
        description="NTP labels key. New caches use ntp_labels; labels remains a fallback.",
    )

    vlm_lm_loss_weight: float = Field(
        default=0.0,
        description="Deprecated alias for ntp_loss_weight.",
        ge=0.0,
    )

    train_gemma_backbone: bool = Field(
        default=False,
        description=(
            "Train the Gemma language-model backbone during planner training. Keep false for lightweight planner "
            "training that only optimizes planner/register/connector parameters."
        ),
    )

    train_vlm_language_model: bool | None = Field(
        default=None,
        description=(
            "Deprecated alias for train_gemma_backbone. If set, it overrides train_gemma_backbone for backward "
            "compatibility with older configs."
        ),
    )

    freeze_vlm_vision_tower: bool = Field(
        default=True,
        description="Freeze Gemma/SigLIP vision tower parameters.",
    )

    freeze_vlm_multi_modal_projector: bool = Field(
        default=True,
        description="Freeze Gemma multi-modal image projection parameters.",
    )

    planner_token_count: int = Field(
        default=2048,
        description=(
            "Fixed target visual planner token count. Must equal the token count saved in "
            "gt_siglip_tokens/ for every sample."
        ),
        ge=1,
    )

    planner_cross_attention_heads: int = Field(
        default=16,
        description="Number of heads in the zero-init planner cross-attention bridge.",
        ge=1,
    )

    planner_cross_attention_dropout: float = Field(
        default=0.0,
        description="Dropout used inside the planner cross-attention bridge.",
        ge=0.0,
        le=1.0,
    )

    planner_zero_init_cross_attention: bool = Field(
        default=False,
        description="Zero-initialize the planner cross-attention output projection.",
    )

    planner_ffn_multiplier: float = Field(
        default=4.0,
        description="Hidden-size multiplier for the planner FFN after cross-attention.",
        gt=0.0,
    )

    planner_ffn_dropout: float = Field(
        default=0.0,
        description="Dropout used inside the planner FFN.",
        ge=0.0,
        le=1.0,
    )

    planner_zero_init_ffn: bool = Field(
        default=False,
        description="Zero-initialize the planner FFN output projection.",
    )

    planner_slot_encoding: bool = Field(
        default=True,
        description="Add trainable slot/type encodings to repeated query registers and planner hidden states.",
    )

    planner_slot_init_std: float = Field(
        default=1e-4,
        description=(
            "Small normal-init std for per-slot planner query/KV encodings. "
            "Set to 0.0 to recover exact zero-initialized slot encodings."
        ),
        ge=0.0,
    )

    planner_slot_init_seed: int | None = Field(
        default=0,
        description=(
            "Optional deterministic seed for planner slot encoding initialization. "
            "Set to null to use the current torch RNG."
        ),
    )

    planner_source_dim: int | None = Field(
        default=3840,
        description="Hidden size of the selected VLM layer. None defaults to visual_token_source_dim when set.",
        ge=1,
    )

    planner_output_dim: int | None = Field(
        default=3840,
        description=(
            "Output dimension of the planner/Q-former. Defaults to visual_token_source_dim. "
            "For SigLIP-space alignment this should be 3840."
        ),
        ge=1,
    )

    planner_query_init_std: float = Field(
        default=1e-4,
        description="Normal init std for learned planner query tokens in planner_output_dim space.",
        ge=0.0,
    )

    planner_use_content_residual: bool = True
    planner_use_3d_rope: bool = True
    planner_rope_use_middle_positions: bool = True
    planner_residual_init_gain: float = Field(default=0.1, gt=0.0)
    planner_query_chunk_size: int | None = Field(default=256, ge=1)

    use_connector_register_queries: bool = Field(
        default=False,
        description=(
            "Legacy option. False uses new learned planner query tokens in planner_output_dim space. "
            "Do not enable for SigLIP-space alignment when connector dim is 4096 and planner_output_dim is 3840."
        ),
    )

    predicted_visual_token_key: str = Field(
        default="predicted_visual_tokens",
        description="Offline predicted visual token key for use_online_vlm=false. Shape per sample: [K,D].",
    )

    predicted_visual_token_mask_key: str = Field(
        default="predicted_visual_token_mask",
        description="Optional offline bool mask key. Shape per sample: [K].",
    )

    planner_mse_weight: float = Field(
        default=1.0,
        description="Deprecated alias for siglip_loss_weight.",
        ge=0.0,
    )

    siglip_loss_weight: float | None = Field(default=None, ge=0.0)

    ntp_loss_weight: float = Field(default=0.1, ge=0.0)

    ntp_logits_chunk_size: int = Field(default=128, ge=1)

    flow_loss_weight: float = Field(
        default=1.0,
        description="Weight for the Stage 1 target flow-matching loss.",
        ge=0.0,
    )

    freeze_transformer: bool = Field(
        default=True,
        description=(
            "Freeze the LTX base DiT. Stage 2 also freezes the Stage 1 LoRA; Stage 3 can train only that LoRA."
        ),
    )

    train_text_connector: bool = Field(
        default=True,
        description="Also optimize the LTX video embedding connector during Stage 2.",
    )

    gemma_gradient_checkpointing: bool = True
    gemma_gradient_checkpointing_use_reentrant: bool = False

    @model_validator(mode="after")
    def _validate_stage2_architecture(self) -> "MultiReferencePlannerStage2Config":
        if self.visual_context_mode != "full_tokens_3d_sa":
            raise ValueError("Stage 2 planner requires visual_context_mode='full_tokens_3d_sa'")
        if not self.visual_branch_enabled:
            raise ValueError("Stage 2 planner requires visual_branch_enabled=true")
        if self.visual_connector_enabled:
            raise ValueError("Stage 2 full-token planner requires visual_connector_enabled=false")
        if not self.freeze_vlm_vision_tower or not self.freeze_vlm_multi_modal_projector:
            raise ValueError("Stage 2 requires the Gemma vision tower and multimodal projector to remain frozen")
        if self.train_gemma_backbone or self.train_vlm_language_model is True:
            raise ValueError("Stage 2 trains Gemma LoRA only; the Gemma base language model must remain frozen")
        if self.training_phase == "stage2" and self.train_stage1_dit_lora:
            raise ValueError("Stage 2 requires train_stage1_dit_lora=false")
        if self.training_phase == "stage3" and not self.train_stage1_dit_lora:
            raise ValueError("Stage 3 requires train_stage1_dit_lora=true")
        if not self.freeze_transformer:
            raise ValueError("Stage 2/3 requires freeze_transformer=true so the LTX base DiT stays frozen")
        if not self.train_text_connector:
            raise ValueError("Stage 2 requires train_text_connector=true")
        if self.gemma_gradient_checkpointing and self.gemma_gradient_checkpointing_use_reentrant:
            raise ValueError("Stage 2 DDP requires non-reentrant Gemma gradient checkpointing")
        if self.cfg_drop_ref_p != 0.0:
            raise ValueError("Stage 2 does not use cfg_drop_ref_p; set cfg_drop_ref_p=0")
        if self.cfg_dropout_enabled:
            drop_all = self.cfg_drop_all_p if self.cfg_drop_all_p is not None else self.cfg_drop_planner_p
            total = (
                self.cfg_full_p
                + self.cfg_drop_text_p
                + self.cfg_drop_siglip_p
                + self.cfg_drop_ref_latents_p
                + drop_all
            )
            if abs(total - 1.0) > 1.0e-6:
                raise ValueError(f"Stage 2 CFG probabilities must sum to 1.0, got {total}")
        return self

    def get_data_sources(self) -> dict[str, str]:
        data_sources = super().get_data_sources()
        if (
            self.cfg_dropout_enabled
            and self.cfg_drop_ref_latents_p > 0
            and self.cfg_text_conditions_dir is not None
            and self.cfg_text_conditions_dir != self.conditions_dir
        ):
            data_sources[self.cfg_text_conditions_dir] = "cfg_text_conditions"
        if self.use_online_vlm:
            data_sources[self.planner_vlm_inputs_dir] = "planner_vlm_inputs"
        else:
            data_sources[self.planner_conditions_dir] = "planner_conditions"
        return data_sources


class MultiReferencePlannerStage2Strategy(MultiReferenceVideoStrategy):
    """Predict fixed visual condition tokens with Gemma/VLM and align to GT SigLIP tokens."""

    config: MultiReferencePlannerStage2Config

    def __init__(self, config: MultiReferencePlannerStage2Config):
        super().__init__(config)
        self.planner_tokens: VisualPlannerTokens | None = None
        self.text_encoder: nn.Module | None = None
        self._inference_embeddings_processor: nn.Module | None = None
        self._planner_query_registers: Tensor | None = None
        self._last_planner_mse_loss: Tensor | None = None
        self._last_vlm_lm_loss: Tensor | None = None
        self._last_flow_loss: Tensor | None = None
        self._last_siglip_loss: Tensor | None = None
        self._last_ntp_loss: Tensor | None = None
        self._last_siglip_cosine: Tensor | None = None
        self._last_predicted_token_std: Tensor | None = None
        self._last_gt_token_std: Tensor | None = None
        self._last_predicted_token_norm: Tensor | None = None
        self._last_gt_token_norm: Tensor | None = None

    def attach_models(
        self,
        *,
        transformer: nn.Module,
        embeddings_processor: nn.Module,
        text_encoder: nn.Module | None = None,
    ) -> None:
        super().attach_models(
            transformer=transformer,
            embeddings_processor=embeddings_processor,
            text_encoder=text_encoder,
        )
        self._inference_embeddings_processor = embeddings_processor
        video_connector = embeddings_processor.video_connector
        connector_param = next(video_connector.parameters(), None)
        connector_device = connector_param.device if connector_param is not None else torch.device("cpu")
        connector_dtype = connector_param.dtype if connector_param is not None else torch.float32
        planner_source_dim = self._resolved_planner_source_dim()
        planner_output_dim = self._resolved_planner_output_dim(planner_source_dim=planner_source_dim)

        base_tokens = getattr(video_connector, "learnable_registers", None)
        use_learned_query_tokens = not self.config.use_connector_register_queries
        if self.config.use_connector_register_queries:
            connector_dim = getattr(video_connector, "inner_dim", None)
            if base_tokens is None:
                if connector_dim is None:
                    raise ValueError(
                        "Cannot initialize connector-register planner queries: video connector has no inner_dim"
                    )
                base_tokens = torch.zeros(1, connector_dim, device=connector_device, dtype=connector_dtype)
            elif connector_dim is None:
                connector_dim = base_tokens.shape[-1]
            if connector_dim != planner_output_dim:
                raise ValueError(
                    "Cannot use connector register queries: "
                    f"connector dim {connector_dim} != planner_output_dim {planner_output_dim}. "
                    "Set use_connector_register_queries=false."
                )
            self._planner_query_registers = base_tokens
        else:
            self._planner_query_registers = None

        self.planner_tokens = VisualPlannerTokens(
            token_count=self.config.planner_token_count,
            dim=planner_output_dim,
            source_dim=planner_source_dim,
            num_heads=self.config.planner_cross_attention_heads,
            dropout=self.config.planner_cross_attention_dropout,
            zero_init_output=self.config.planner_zero_init_cross_attention,
            use_slot_encoding=self.config.planner_slot_encoding,
            slot_init_std=self.config.planner_slot_init_std,
            slot_init_seed=self.config.planner_slot_init_seed,
            ffn_multiplier=self.config.planner_ffn_multiplier,
            ffn_dropout=self.config.planner_ffn_dropout,
            zero_init_ffn=self.config.planner_zero_init_ffn,
            use_learned_query_tokens=use_learned_query_tokens,
            query_init_std=self.config.planner_query_init_std,
            positional_embedding_theta=getattr(transformer, "positional_embedding_theta", 10000.0),
            positional_embedding_max_pos=getattr(
                transformer,
                "positional_embedding_max_pos",
                [20, 2048, 2048],
            ),
            rope_type=getattr(transformer, "rope_type", LTXRopeType.SPLIT),
            use_content_residual=self.config.planner_use_content_residual,
            use_3d_rope=self.config.planner_use_3d_rope,
            rope_use_middle_positions=self.config.planner_rope_use_middle_positions,
            residual_init_gain=self.config.planner_residual_init_gain,
            query_chunk_size=self.config.planner_query_chunk_size,
        ).to(device=connector_device, dtype=connector_dtype)

        if self.config.visual_context_mode != "full_tokens_3d_sa":
            raise ValueError("Stage 2 planner requires visual_context_mode='full_tokens_3d_sa'")
        if self._visual_full_encoder is None:
            raise RuntimeError("Stage 2 planner requires the Stage 1 Visual3DTokenEncoder")
        if self._visual_resampler is not None or self._visual_connector is not None or self._visual_gate is not None:
            raise RuntimeError("Stage 2 full-token planner must not create a visual resampler, connector, or gate")

        self.text_encoder = text_encoder
        if self.config.use_online_vlm:
            if self.text_encoder is None:
                raise ValueError("Stage 2 online VLM training requires a loaded Gemma text_encoder.")
            self._configure_vlm_trainable_parameters(self.text_encoder)

    def train_transformer(self) -> bool:
        return self.config.training_phase == "stage3" and self.config.train_stage1_dit_lora

    def configure_transformer_trainability(self, transformer: nn.Module) -> list[str]:
        if self.config.training_phase == "stage3":
            return configure_stage3_transformer_trainability(transformer)
        transformer.requires_grad_(False)
        return []

    def train_embeddings_processor(self) -> bool:
        return self.config.train_text_connector

    def requires_text_encoder(self) -> bool:
        return self.config.use_online_vlm

    def requires_text_encoder_lora(self) -> bool:
        return self.config.use_online_vlm

    def train_text_encoder(self) -> bool:
        if not self.config.use_online_vlm or self.text_encoder is None:
            return False
        return any(parameter.requires_grad for parameter in self.text_encoder.parameters())

    def get_trainable_modules(self) -> dict[str, nn.Module]:
        modules = super().get_trainable_modules()
        if self.planner_tokens is not None:
            modules["planner_tokens"] = self.planner_tokens
        return modules

    def set_trainable_modules(self, modules: dict[str, nn.Module]) -> None:
        super().set_trainable_modules(modules)
        if "planner_tokens" in modules:
            self.planner_tokens = modules["planner_tokens"]

    def set_text_encoder(self, text_encoder: nn.Module) -> None:
        self.text_encoder = text_encoder

    def enforce_frozen_module_eval(self) -> None:
        self._keep_frozen_vlm_modules_in_eval()

    def get_text_encoder_trainable_module(self) -> nn.Module:
        return self._get_language_model()

    def set_text_encoder_trainable_module(self, language_model: nn.Module) -> None:
        gemma_model = self._unwrap_text_encoder().model.model
        gemma_model.language_model = language_model

    def get_text_encoder_checkpoint_state_dict(self, accelerator: Any) -> dict[str, Tensor]:
        language_model = self._get_language_model()
        unwrapped = accelerator.unwrap_model(language_model, keep_torch_compile=False)
        trainable_names = {name for name, parameter in unwrapped.named_parameters() if parameter.requires_grad}
        full_state = accelerator.get_state_dict(language_model)
        prefix = "model.model.language_model."
        return {f"{prefix}{key}": value for key, value in full_state.items() if key in trainable_names}

    def prepare_conditions(self, batch: dict[str, Any], conditions: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.planner_tokens is None:
            raise RuntimeError("Planner tokens were not initialized. Did attach_models() run?")

        self._last_planner_mse_loss = None
        self._last_vlm_lm_loss = None
        self._last_flow_loss = None
        self._last_siglip_loss = None
        self._last_ntp_loss = None

        conditions = self._apply_cfg_preconnector_context_switch(batch, conditions)
        video_feature_key = "video_prompt_embeds" if "video_prompt_embeds" in conditions else "prompt_embeds"
        video_features = conditions[video_feature_key]
        gt_tokens, gt_mask = self._load_raw_condition_visual_tokens(
            batch["gt_visual_tokens"],
            device=video_features.device,
            dtype=video_features.dtype,
        )
        self._assert_token_count("GT visual tokens", gt_tokens, gt_mask)
        planner_output_dim = self._resolved_planner_output_dim(planner_source_dim=self._resolved_planner_source_dim())
        self._assert_visual_token_dim("GT visual tokens", gt_tokens, planner_output_dim)
        token_positions = self._build_visual_token_positions(
            batch["gt_visual_tokens"],
            batch["latents"],
            token_count=gt_tokens.shape[1],
            device=gt_tokens.device,
            dtype=torch.float32,
        )

        if self.config.use_online_vlm:
            predicted_tokens, predicted_mask = self._run_online_vlm(
                batch,
                batch["planner_vlm_inputs"],
                gt_tokens.device,
                token_positions=token_positions,
            )
        else:
            predicted_tokens, predicted_mask = self._load_offline_predicted_tokens(
                batch["planner_conditions"],
                gt_tokens,
            )

        self._assert_token_count("Predicted visual tokens", predicted_tokens, predicted_mask)
        predicted_tokens = predicted_tokens.to(device=gt_tokens.device, dtype=gt_tokens.dtype)
        predicted_mask = predicted_mask.to(device=gt_tokens.device, dtype=torch.bool) & gt_mask
        self._assert_visual_token_dim("Predicted visual tokens", predicted_tokens, planner_output_dim)
        if predicted_tokens.shape[-1] != gt_tokens.shape[-1]:
            raise ValueError(
                f"Predicted raw visual dim {predicted_tokens.shape[-1]} must match raw GT dim {gt_tokens.shape[-1]} "
                "for Stage 2 planner MSE."
            )
        exclude_siglip_loss = self._cfg_drop_ref_latents_mask(
            batch,
            batch_size=predicted_tokens.shape[0],
            device=predicted_tokens.device,
        )

        if self._siglip_loss_weight() > 0:
            mse_mask = predicted_mask
            if exclude_siglip_loss is not None and torch.any(exclude_siglip_loss):
                mse_mask = mse_mask & ~exclude_siglip_loss[:, None]
            self._last_planner_mse_loss = self._compute_visual_alignment_loss(
                predicted_tokens=predicted_tokens,
                gt_tokens=gt_tokens,
                mask=mse_mask,
            )
            self._last_siglip_loss = self._last_planner_mse_loss

        cosine = F.cosine_similarity(predicted_tokens, gt_tokens, dim=-1)
        mask_float = predicted_mask.to(dtype=cosine.dtype)
        mask_denominator = mask_float.sum(dim=1).clamp(min=1.0)
        self._last_siglip_cosine = cosine.mul(mask_float).sum(dim=1) / mask_denominator
        self._last_predicted_token_std = self._masked_visual_stat(predicted_tokens, predicted_mask, "std")
        self._last_gt_token_std = self._masked_visual_stat(gt_tokens, predicted_mask, "std")
        self._last_predicted_token_norm = self._masked_visual_stat(predicted_tokens, predicted_mask, "norm")
        self._last_gt_token_norm = self._masked_visual_stat(gt_tokens, predicted_mask, "norm")

        batch["_planner_predicted_raw_tokens"] = predicted_tokens
        batch["_planner_predicted_mask"] = predicted_mask
        batch["_planner_visual_positions"] = token_positions
        return self._pad_conditions_to_connector_multiple(conditions)

    def prepare_inference_conditions(
        self,
        *,
        conditions: dict[str, Tensor],
        planner_vlm_inputs: dict[str, Any],
        latents_metadata: dict[str, Any],
        visual_position_metadata: dict[str, Any],
        drop_reference_images: bool = False,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """Build Stage 2 planner conditions without GT visual tokens or training-only losses/dropout."""
        parts = MultiReferencePlannerStage2Strategy.prepare_inference_condition_parts(
            self,
            conditions=conditions,
            planner_vlm_inputs=planner_vlm_inputs,
            latents_metadata=latents_metadata,
            visual_position_metadata=visual_position_metadata,
            drop_reference_images=drop_reference_images,
        )
        return parts.final_conditions, parts.diagnostics

    def prepare_inference_condition_parts(
        self,
        *,
        conditions: dict[str, Tensor],
        planner_vlm_inputs: dict[str, Any],
        latents_metadata: dict[str, Any],
        visual_position_metadata: dict[str, Any],
        drop_reference_images: bool = False,
    ) -> PlannerInferenceConditionParts:
        """Build and expose text/planner condition parts from exactly one VLM planner forward."""
        if not self.config.use_online_vlm:
            raise RuntimeError("Stage 2 planner inference requires use_online_vlm=true")
        if self.planner_tokens is None or self.text_encoder is None:
            raise RuntimeError("Stage 2 planner inference models are not attached")
        if self._inference_embeddings_processor is None:
            raise RuntimeError("Stage 2 planner inference requires an attached embeddings processor")
        if self._visual_full_encoder is None:
            raise RuntimeError("Stage 2 planner inference requires visual_full_encoder")
        visual_token_key = getattr(self.config, "visual_token_key", "visual_tokens")
        if visual_token_key in visual_position_metadata:
            raise ValueError(
                "Stage 2 planner inference accepts visual position metadata only; "
                f"remove raw {visual_token_key!r} before building conditions"
            )

        conditions = dict(conditions)
        feature_key = "video_prompt_embeds" if "video_prompt_embeds" in conditions else "prompt_embeds"
        if feature_key not in conditions:
            raise ValueError("Stage 2 inference conditions contain no video/prompt embeddings")
        original_shape = list(conditions[feature_key].shape)
        token_positions = self._build_visual_token_positions(
            visual_position_metadata,
            latents_metadata,
            token_count=self.config.planner_token_count,
            device=conditions[feature_key].device,
            dtype=torch.float32,
        )
        predicted_tokens, predicted_mask = self._run_online_vlm_inference(
            planner_vlm_inputs,
            device=conditions[feature_key].device,
            token_positions=token_positions,
            drop_reference_images=drop_reference_images,
        )
        self._assert_token_count("Predicted visual tokens", predicted_tokens, predicted_mask)
        planner_output_dim = self._resolved_planner_output_dim(
            planner_source_dim=self._resolved_planner_source_dim()
        )
        self._assert_visual_token_dim("Predicted visual tokens", predicted_tokens, planner_output_dim)

        conditions = self._pad_conditions_to_connector_multiple(conditions)
        pre_connector_shape = list(conditions[feature_key].shape)
        video_features = conditions[feature_key]
        audio_features = conditions.get("audio_prompt_embeds")
        additive_mask = convert_to_additive_mask(conditions["prompt_attention_mask"], video_features.dtype)
        video_embeds, audio_embeds, attention_mask = self._inference_embeddings_processor.create_embeddings(
            video_features,
            audio_features,
            additive_mask,
        )
        connected_conditions = dict(conditions)
        connected_conditions["video_prompt_embeds"] = video_embeds
        if audio_embeds is not None:
            connected_conditions["audio_prompt_embeds"] = audio_embeds
        connected_conditions["prompt_attention_mask"] = attention_mask
        post_connector_text_shape = list(video_embeds.shape)

        projected_tokens = self._project_visual_tokens(predicted_tokens, target_dim=video_embeds.shape[-1])
        self._validate_full_visual_token_layout(
            visual_position_metadata,
            token_count=projected_tokens.shape[1],
        )
        visual_context, visual_mask = self._visual_full_encoder(
            tokens=projected_tokens,
            token_positions=token_positions,
            token_mask=predicted_mask,
        )
        final_conditions = self._append_postconnector_visual_context(
            connected_conditions,
            visual_context,
            visual_mask,
        )
        diagnostics = {
            "predicted_visual_tokens": predicted_tokens,
            "predicted_visual_token_mask": predicted_mask,
            "token_positions": token_positions,
            "planner_raw_shape": list(predicted_tokens.shape),
            "planner_projected_shape": list(projected_tokens.shape),
            "visual_context_shape": list(visual_context.shape),
            "original_condition_shape": original_shape,
            "pre_connector_condition_shape": pre_connector_shape,
            "post_connector_text_condition_shape": post_connector_text_shape,
            "final_condition_shape": list(final_conditions["video_prompt_embeds"].shape),
            "drop_reference_images": drop_reference_images,
        }
        return PlannerInferenceConditionParts(
            connected_text_conditions=connected_conditions,
            visual_context=visual_context,
            visual_mask=visual_mask,
            final_conditions=final_conditions,
            diagnostics=diagnostics,
        )

    def append_shared_inference_visual_context(
        self,
        connected_text_conditions: dict[str, Tensor | None],
        *,
        visual_context: Tensor,
        visual_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Append an already-computed planner context to connected text without another VLM forward."""
        conditions = dict(connected_text_conditions)
        feature_key = "video_prompt_embeds" if "video_prompt_embeds" in conditions else "prompt_embeds"
        text_context = conditions.get(feature_key)
        if not isinstance(text_context, Tensor):
            raise ValueError("Shared planner inference conditions contain no text context tensor")
        if text_context.shape[0] != visual_context.shape[0] or text_context.shape[-1] != visual_context.shape[-1]:
            raise ValueError(
                "Shared planner text/visual context shape mismatch: "
                f"text={list(text_context.shape)}, visual={list(visual_context.shape)}"
            )
        if conditions.get("prompt_attention_mask") is None:
            conditions["prompt_attention_mask"] = torch.ones(
                text_context.shape[:2],
                dtype=torch.long,
                device=text_context.device,
            )
        return self._append_postconnector_visual_context(
            conditions,
            visual_context,
            visual_mask,
        )

    @staticmethod
    def _prepare_inference_planner_data(
        planner_data: dict[str, Any],
        *,
        drop_reference_images: bool,
        device: torch.device,
    ) -> dict[str, Any]:
        if not drop_reference_images:
            return planner_data

        planner_forward_data = dict(planner_data)
        planner_forward_data.pop("pixel_values", None)
        image_counts = planner_data.get("num_ref_images")
        if isinstance(image_counts, Tensor):
            planner_forward_data["num_ref_images"] = torch.zeros_like(image_counts)
        else:
            input_ids = planner_data.get("input_ids")
            batch_size = input_ids.shape[0] if isinstance(input_ids, Tensor) and input_ids.ndim > 1 else 1
            planner_forward_data["num_ref_images"] = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=device,
            )
        return planner_forward_data

    def _run_online_vlm_inference(
        self,
        planner_data: dict[str, Any],
        *,
        device: torch.device,
        token_positions: Tensor,
        drop_reference_images: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if self.text_encoder is None or self.planner_tokens is None:
            raise RuntimeError("Online VLM inference requires text_encoder and planner_tokens")
        planner_forward_data = self._prepare_inference_planner_data(
            planner_data,
            drop_reference_images=drop_reference_images,
            device=device,
        )

        forward_inputs = self._build_vlm_forward_inputs(planner_forward_data, device)
        placeholder_mask = planner_forward_data[self.config.vlm_placeholder_mask_key].to(
            device=device,
            dtype=torch.bool,
        )
        self._assert_placeholder_mask(placeholder_mask)
        dropped_image_token_mask = None
        drop_ref_mask = None
        if drop_reference_images:
            drop_ref_mask = torch.ones(
                forward_inputs["input_ids"].shape[0],
                dtype=torch.bool,
                device=device,
            )
            dropped_image_token_mask, _ = self._build_vlm_dropout_masks(
                planner_data=planner_forward_data,
                forward_inputs=forward_inputs,
                drop_ref_mask=drop_ref_mask,
                drop_text_mask=None,
            )
            forward_inputs = self._apply_vlm_condition_dropout(
                forward_inputs=forward_inputs,
                dropped_image_token_mask=dropped_image_token_mask,
                dropped_text_token_mask=None,
            )
        inputs_embeds = self._build_vlm_inputs_embeds(
            forward_inputs,
            planner_forward_data,
            placeholder_mask,
            dropped_image_token_mask=dropped_image_token_mask,
            drop_ref_mask=drop_ref_mask,
        )
        planner_hidden_source, _ = self._forward_language_model_for_planner(
            inputs_embeds=inputs_embeds,
            attention_mask=forward_inputs["attention_mask"],
            position_ids=forward_inputs.get("position_ids"),
            cache_position=forward_inputs.get("cache_position"),
            force_eval=True,
        )
        selected_hidden, selected_mask = self._select_masked_hidden(
            planner_hidden_source,
            placeholder_mask,
        )
        selected_mask = self._resolve_planner_output_mask(planner_forward_data, selected_mask)
        predicted_tokens = self.planner_tokens(
            planner_hidden=selected_hidden,
            query_registers=self._planner_query_registers,
            planner_mask=selected_mask,
            token_positions=token_positions,
        )
        return predicted_tokens, selected_mask

    def postprocess_conditions_after_connector(
        self,
        batch: dict[str, Any],
        conditions: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        conditions = self._apply_cfg_postconnector_text_dropout(batch, conditions)
        predicted_tokens = batch.get("_planner_predicted_raw_tokens")
        predicted_mask = batch.get("_planner_predicted_mask")
        token_positions = batch.get("_planner_visual_positions")
        if not isinstance(predicted_tokens, Tensor) or not isinstance(predicted_mask, Tensor):
            raise RuntimeError("Stage 2 planner raw tokens were not prepared before the text connector")
        if not isinstance(token_positions, Tensor):
            raise RuntimeError("Stage 2 planner 3D positions were not prepared")
        if self._visual_full_encoder is None:
            raise RuntimeError("Stage 2 planner requires loaded Stage 1 visual_full_encoder weights")

        context_key = "video_prompt_embeds" if "video_prompt_embeds" in conditions else "prompt_embeds"
        target_dim = conditions[context_key].shape[-1]
        projected_tokens = self._project_visual_tokens(predicted_tokens, target_dim=target_dim)
        self._validate_full_visual_token_layout(
            batch["gt_visual_tokens"],
            token_count=projected_tokens.shape[1],
        )
        visual_context, visual_mask = self._visual_full_encoder(
            tokens=projected_tokens,
            token_positions=token_positions.to(device=projected_tokens.device),
            token_mask=predicted_mask,
        )
        visual_context, visual_mask = self._apply_cfg_visual_dropout_after_connector(
            batch,
            visual_context,
            visual_mask,
        )
        return self._append_postconnector_visual_context(conditions, visual_context, visual_mask)

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        flow_loss = super().compute_loss(video_pred, audio_pred, inputs)
        self._last_flow_loss = flow_loss
        siglip_loss = self._loss_or_zeros(self._last_siglip_loss, flow_loss)
        ntp_loss = self._loss_or_zeros(self._last_ntp_loss, flow_loss)
        return self._combine_losses(flow_loss, siglip_loss, ntp_loss)

    def _combine_losses(self, flow_loss: Tensor, siglip_loss: Tensor, ntp_loss: Tensor) -> Tensor:
        return (
            flow_loss * self.config.flow_loss_weight
            + siglip_loss * self._siglip_loss_weight()
            + ntp_loss * self._ntp_loss_weight()
        )

    def _apply_cfg_preconnector_context_switch(
        self,
        batch: dict[str, Any],
        conditions: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        if not self.config.cfg_dropout_enabled:
            return conditions
        context_key = "video_prompt_embeds" if "video_prompt_embeds" in conditions else "prompt_embeds"
        batch_size = conditions[context_key].shape[0]
        drop_ref = self._cfg_drop_ref_latents_mask(
            batch,
            batch_size=batch_size,
            device=conditions[context_key].device,
        )
        if drop_ref is None or not torch.any(drop_ref):
            return conditions
        text_only = batch.get("cfg_text_conditions")
        if text_only is None:
            raise ValueError(
                "Stage 2 synchronized reference dropout requires cfg_text_conditions. "
                "Set cfg_text_conditions_dir to precomputed text-only conditions."
            )
        return self._select_condition_rows(primary=conditions, alternate=text_only, use_alternate=drop_ref)

    @staticmethod
    def _loss_or_zeros(loss: Tensor | None, reference: Tensor) -> Tensor:
        if loss is None:
            return torch.zeros_like(reference)
        return loss.to(device=reference.device, dtype=reference.dtype)

    def _siglip_loss_weight(self) -> float:
        if self.config.siglip_loss_weight is not None:
            return self.config.siglip_loss_weight
        return self.config.planner_mse_weight

    def _ntp_loss_weight(self) -> float:
        explicit_fields = getattr(self.config, "model_fields_set", set())
        if "ntp_loss_weight" in explicit_fields:
            return self.config.ntp_loss_weight
        if "vlm_lm_loss_weight" in explicit_fields:
            return self.config.vlm_lm_loss_weight
        return self.config.ntp_loss_weight

    def get_last_training_metrics(self) -> dict[str, Tensor]:
        metrics: dict[str, Tensor | None] = {
            "train/loss_flow": self._last_flow_loss,
            "train/loss_siglip": self._last_siglip_loss,
            "train/loss_ntp": self._last_ntp_loss,
            "train/siglip_cosine": self._last_siglip_cosine,
            "train/predicted_token_std": self._last_predicted_token_std,
            "train/gt_token_std": self._last_gt_token_std,
            "train/predicted_token_norm": self._last_predicted_token_norm,
            "train/gt_token_norm": self._last_gt_token_norm,
        }
        return {name: value.detach().mean() for name, value in metrics.items() if value is not None}

    def load_extra_checkpoint_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        legacy_prefixes = (
            "training_strategy.visual_resampler.",
            "training_strategy.visual_connector.",
            "training_strategy.visual_gate.",
        )
        if any(key.startswith(legacy_prefixes) for key in state_dict):
            raise ValueError("Stage 2 full-token planner cannot load legacy visual resampler/connector/gate weights")
        stage1_modules = MultiReferenceVideoStrategy.get_trainable_modules(self)
        required = {"visual_token_projection", "visual_full_encoder"}
        for name, module in stage1_modules.items():
            prefix = f"training_strategy.{name}."
            module_state = {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if module_state:
                module.load_state_dict(module_state, strict=True)
            elif name in required:
                raise ValueError(f"Stage 2 requires checkpoint keys under {prefix}*")

        if self.planner_tokens is not None:
            prefix = "training_strategy.planner_tokens."
            planner_state = {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if planner_state:
                self.planner_tokens.load_state_dict(planner_state, strict=True)
            elif any(key.startswith("text_encoder.") for key in state_dict):
                raise ValueError("Stage 2 checkpoint contains Gemma weights but no training_strategy.planner_tokens.*")

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        metadata = super().get_checkpoint_metadata()
        metadata.update(
            {
                "conditioning": "multi_reference_planner_stage2",
                "stage": 3 if self.config.training_phase == "stage3" else 2,
                "training_phase": self.config.training_phase,
                "train_stage1_dit_lora": self.config.train_stage1_dit_lora,
                "planner_architecture": "content_residual_3d_rope_cross_attention",
                "planner_token_count": self.config.planner_token_count,
                "planner_cross_attention_heads": self.config.planner_cross_attention_heads,
                "planner_zero_init_cross_attention": self.config.planner_zero_init_cross_attention,
                "planner_ffn_multiplier": self.config.planner_ffn_multiplier,
                "planner_zero_init_ffn": self.config.planner_zero_init_ffn,
                "planner_slot_encoding": self.config.planner_slot_encoding,
                "planner_slot_init_std": self.config.planner_slot_init_std,
                "planner_slot_init_seed": self.config.planner_slot_init_seed,
                "planner_output_dim": self._resolved_planner_output_dim(
                    planner_source_dim=self._resolved_planner_source_dim()
                ),
                "planner_query_init_std": self.config.planner_query_init_std,
                "use_connector_register_queries": self.config.use_connector_register_queries,
                "planner_mse_weight": self.config.planner_mse_weight,
                "siglip_loss_weight": self._siglip_loss_weight(),
                "ntp_loss_weight": self._ntp_loss_weight(),
                "flow_loss_weight": self.config.flow_loss_weight,
                "freeze_transformer": self.config.freeze_transformer,
                "train_text_connector": self.config.train_text_connector,
                "use_online_vlm": self.config.use_online_vlm,
                "train_gemma_backbone": self._train_gemma_backbone(),
                "freeze_vlm_vision_tower": self.config.freeze_vlm_vision_tower,
                "freeze_vlm_multi_modal_projector": self.config.freeze_vlm_multi_modal_projector,
                "cfg_drop_all_p": self._cfg_drop_all_probability(),
                "cfg_drop_planner_p_is_legacy_drop_all_alias": True,
                "visual_context_mode": self.config.visual_context_mode,
                "losses": ["ntp", "flow_matching", "siglip_mse"],
            }
        )
        return metadata

    def validate_initial_checkpoint_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        if self.config.training_phase == "stage3":
            self.validate_checkpoint_state_dict(state_dict)

    @staticmethod
    def validate_checkpoint_state_dict(state_dict: dict[str, Tensor]) -> None:
        required_prefixes = (
            "diffusion_model.",
            "training_strategy.planner_tokens.",
            "training_strategy.visual_token_projection.",
            "training_strategy.visual_full_encoder.",
            "embeddings_processor.video_connector.",
            "text_encoder.model.model.language_model.",
        )
        missing = [prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in state_dict)]
        if missing:
            raise RuntimeError(f"Stage 2 checkpoint is missing required components: {missing}")

        forbidden_prefixes = (
            "text_encoder.model.model.vision_tower.",
            "text_encoder.model.model.multi_modal_projector.",
        )
        forbidden = [key for key in state_dict if key.startswith(forbidden_prefixes)]
        if forbidden:
            raise RuntimeError(f"Stage 2 checkpoint contains frozen Gemma vision/projector weights: {forbidden[:5]}")
        non_lora_text = [
            key
            for key in state_dict
            if key.startswith("text_encoder.model.model.language_model.") and "lora_" not in key
        ]
        non_lora_dit = [
            key for key in state_dict if key.startswith("diffusion_model.") and "lora_" not in key
        ]
        if non_lora_text or non_lora_dit:
            raise RuntimeError(
                "Stage 2 checkpoint contains frozen base weights: "
                f"text={non_lora_text[:5]}, dit={non_lora_dit[:5]}"
            )
        non_finite = [
            key
            for key, value in state_dict.items()
            if isinstance(value, Tensor) and value.is_floating_point() and not torch.isfinite(value).all()
        ]
        if non_finite:
            raise RuntimeError(f"Planner checkpoint contains non-finite tensors: {non_finite[:20]}")

    def _run_online_vlm(
        self,
        batch: dict[str, Any],
        planner_data: dict[str, Any],
        device: torch.device,
        *,
        token_positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.text_encoder is None or self.planner_tokens is None:
            raise RuntimeError("Online VLM mode requires text_encoder and planner_tokens.")

        self._keep_frozen_vlm_modules_in_eval()
        forward_inputs = self._build_vlm_forward_inputs(planner_data, device)
        placeholder_mask = planner_data[self.config.vlm_placeholder_mask_key].to(device=device, dtype=torch.bool)
        self._assert_placeholder_mask(placeholder_mask)
        drop_ref_mask = self._cfg_drop_ref_latents_mask(
            batch,
            batch_size=forward_inputs["input_ids"].shape[0],
            device=device,
        )
        drop_text_mask = self._cfg_drop_text_mask(
            batch,
            batch_size=forward_inputs["input_ids"].shape[0],
            device=device,
        )
        dropped_image_token_mask, dropped_text_token_mask = self._build_vlm_dropout_masks(
            planner_data=planner_data,
            forward_inputs=forward_inputs,
            drop_ref_mask=drop_ref_mask,
            drop_text_mask=drop_text_mask,
        )
        forward_inputs = self._apply_vlm_condition_dropout(
            forward_inputs=forward_inputs,
            dropped_image_token_mask=dropped_image_token_mask,
            dropped_text_token_mask=dropped_text_token_mask,
        )

        inputs_embeds = self._build_vlm_inputs_embeds(
            forward_inputs,
            planner_data,
            placeholder_mask,
            dropped_image_token_mask=dropped_image_token_mask,
            dropped_text_token_mask=dropped_text_token_mask,
            drop_ref_mask=drop_ref_mask,
        )
        planner_hidden_source, final_hidden = self._forward_language_model_for_planner(
            inputs_embeds=inputs_embeds,
            attention_mask=forward_inputs["attention_mask"],
            position_ids=forward_inputs.get("position_ids"),
            cache_position=forward_inputs.get("cache_position"),
        )
        selected_hidden, selected_mask = self._select_masked_hidden(
            planner_hidden_source,
            placeholder_mask,
        )
        selected_mask = self._resolve_planner_output_mask(planner_data, selected_mask)
        predicted_tokens = self.planner_tokens(
            planner_hidden=selected_hidden,
            query_registers=self._planner_query_registers,
            planner_mask=selected_mask,
            token_positions=token_positions,
        )

        labels = planner_data.get(self.config.vlm_lm_labels_key)
        if labels is None:
            labels = planner_data.get("ntp_labels", planner_data.get("labels"))
        if labels is None and self._ntp_loss_weight() > 0:
            raise ValueError(
                "NTP loss is enabled but planner_vlm_inputs contain no ntp_labels/labels. "
                "Regenerate them with precompute_planner_vlm_inputs.py."
            )
        if labels is not None and self._ntp_loss_weight() > 0:
            labels = labels.to(device=device, dtype=torch.long).clone()
            if drop_text_mask is not None and torch.any(drop_text_mask):
                labels[drop_text_mask] = -100
            self._last_vlm_lm_loss = self._compute_lm_loss(final_hidden, labels)
            self._last_ntp_loss = self._last_vlm_lm_loss

        return predicted_tokens, selected_mask

    def _forward_language_model_for_planner(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor | None = None,
        cache_position: Tensor | None = None,
        force_eval: bool = False,
    ) -> tuple[Tensor, Tensor]:
        language_model = self._get_language_model()
        enable_grad = self.train_text_encoder() and not force_eval
        language_model.train(enable_grad)
        self._keep_frozen_vlm_modules_in_eval()
        need_intermediate_hidden = self.config.vlm_hidden_layer != -1
        lm_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "output_hidden_states": need_intermediate_hidden,
            "return_dict": True,
        }
        if position_ids is not None:
            lm_inputs["position_ids"] = position_ids
        if cache_position is not None:
            lm_inputs["cache_position"] = cache_position

        with torch.set_grad_enabled(enable_grad):
            outputs = language_model(**lm_inputs)
        if need_intermediate_hidden:
            hidden_states = self._extract_hidden_states(outputs)
            planner_hidden_source = hidden_states[self.config.vlm_hidden_layer]
            final_hidden = hidden_states[-1]
        else:
            final_hidden = getattr(outputs, "last_hidden_state", None)
            if final_hidden is None:
                raise ValueError(
                    "Gemma language_model output has no last_hidden_state while vlm_hidden_layer=-1"
                )
            planner_hidden_source = final_hidden
        return planner_hidden_source, final_hidden

    def _build_vlm_inputs_embeds(
        self,
        forward_inputs: dict[str, Tensor],
        planner_data: dict[str, Any],
        placeholder_mask: Tensor,
        *,
        dropped_image_token_mask: Tensor | None = None,
        dropped_text_token_mask: Tensor | None = None,
        drop_ref_mask: Tensor | None = None,
    ) -> Tensor:
        input_ids = forward_inputs["input_ids"]
        language_model = self._get_language_model()
        embed_tokens = self._get_input_embeddings(language_model)
        inputs_embeds = embed_tokens(input_ids)

        pixel_values = forward_inputs.get("pixel_values")
        if pixel_values is not None:
            image_counts = planner_data.get("num_ref_images")
            if isinstance(image_counts, Tensor):
                image_counts = image_counts.to(device=pixel_values.device, dtype=torch.long)
                if dropped_image_token_mask is not None and drop_ref_mask is not None:
                    image_counts = image_counts.masked_fill(drop_ref_mask.to(device=image_counts.device), 0)
            scatter_exclusion_mask = placeholder_mask
            if dropped_image_token_mask is not None:
                scatter_exclusion_mask = scatter_exclusion_mask | dropped_image_token_mask
            visual_batch = extract_projected_visual_tokens(
                self._unwrap_text_encoder().model,
                pixel_values,
                image_counts=image_counts,
            )
            inputs_embeds = scatter_visual_tokens_into_embeddings(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                visual_tokens=visual_batch.tokens,
                visual_mask=visual_batch.mask,
                image_token_index=GEMMA3_CONFIG_FOR_LTX.image_token_index,
                planner_placeholder_mask=scatter_exclusion_mask,
            )

        if dropped_image_token_mask is not None:
            inputs_embeds = inputs_embeds.masked_fill(dropped_image_token_mask.unsqueeze(-1), 0)
        if dropped_text_token_mask is not None:
            inputs_embeds = inputs_embeds.masked_fill(dropped_text_token_mask.unsqueeze(-1), 0)
        return inputs_embeds

    def _apply_vlm_condition_dropout(
        self,
        *,
        forward_inputs: dict[str, Tensor],
        dropped_image_token_mask: Tensor | None,
        dropped_text_token_mask: Tensor | None,
    ) -> dict[str, Tensor]:
        if dropped_image_token_mask is None and dropped_text_token_mask is None:
            return forward_inputs
        forward_inputs = dict(forward_inputs)
        attention_mask = forward_inputs["attention_mask"].to(device=forward_inputs["input_ids"].device)
        if dropped_image_token_mask is not None:
            attention_mask = attention_mask.masked_fill(dropped_image_token_mask, 0)
        if dropped_text_token_mask is not None:
            attention_mask = attention_mask.masked_fill(dropped_text_token_mask, 0)
        forward_inputs["attention_mask"] = attention_mask
        return forward_inputs

    def _apply_vlm_reference_dropout(
        self,
        *,
        forward_inputs: dict[str, Tensor],
        planner_data: dict[str, Any],
        drop_ref_mask: Tensor | None,
    ) -> dict[str, Tensor]:
        dropped_image_token_mask, _ = self._build_vlm_dropout_masks(
            planner_data=planner_data,
            forward_inputs=forward_inputs,
            drop_ref_mask=drop_ref_mask,
            drop_text_mask=None,
        )
        return self._apply_vlm_condition_dropout(
            forward_inputs=forward_inputs,
            dropped_image_token_mask=dropped_image_token_mask,
            dropped_text_token_mask=None,
        )

    def _build_vlm_dropout_masks(
        self,
        *,
        planner_data: dict[str, Any],
        forward_inputs: dict[str, Tensor],
        drop_ref_mask: Tensor | None,
        drop_text_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        device = forward_inputs["input_ids"].device
        dropped_image_token_mask = self._get_dropped_vlm_image_token_mask(
            planner_data=planner_data,
            forward_inputs=forward_inputs,
            drop_ref_mask=drop_ref_mask,
            device=device,
        )
        dropped_text_token_mask = self._get_dropped_vlm_text_token_mask(
            planner_data=planner_data,
            forward_inputs=forward_inputs,
            drop_text_mask=drop_text_mask,
            device=device,
        )
        return dropped_image_token_mask, dropped_text_token_mask

    @staticmethod
    def _get_dropped_vlm_image_token_mask(
        *,
        planner_data: dict[str, Any],
        forward_inputs: dict[str, Tensor],
        drop_ref_mask: Tensor | None,
        device: torch.device,
    ) -> Tensor | None:
        if drop_ref_mask is None or not torch.any(drop_ref_mask):
            return None
        ref_image_region_mask = MultiReferencePlannerStage2Strategy._get_vlm_ref_image_region_mask(
            planner_data=planner_data,
            forward_inputs=forward_inputs,
            device=device,
        )
        return ref_image_region_mask & drop_ref_mask.to(device=device, dtype=torch.bool)[:, None]

    @staticmethod
    def _get_dropped_vlm_text_token_mask(
        *,
        planner_data: dict[str, Any],
        forward_inputs: dict[str, Tensor],
        drop_text_mask: Tensor | None,
        device: torch.device,
    ) -> Tensor | None:
        if drop_text_mask is None or not torch.any(drop_text_mask):
            return None
        text_token_mask = planner_data.get("text_token_mask")
        if text_token_mask is None:
            text_token_mask = MultiReferencePlannerStage2Strategy._infer_vlm_text_token_mask(
                planner_data=planner_data,
                forward_inputs=forward_inputs,
                device=device,
            )
        else:
            text_token_mask = text_token_mask.to(device=device, dtype=torch.bool)
        return text_token_mask & drop_text_mask.to(device=device, dtype=torch.bool)[:, None]

    @staticmethod
    def _infer_vlm_text_token_mask(
        *,
        planner_data: dict[str, Any],
        forward_inputs: dict[str, Tensor],
        device: torch.device,
    ) -> Tensor:
        active_mask = forward_inputs["attention_mask"].to(device=device, dtype=torch.bool)
        planner_region_mask = MultiReferencePlannerStage2Strategy._get_vlm_planner_region_mask(
            planner_data=planner_data,
            forward_inputs=forward_inputs,
            device=device,
        )
        ref_image_region_mask = MultiReferencePlannerStage2Strategy._get_vlm_ref_image_region_mask(
            planner_data=planner_data,
            forward_inputs=forward_inputs,
            device=device,
        )

        return active_mask & ~ref_image_region_mask & ~planner_region_mask

    @staticmethod
    def _get_vlm_planner_region_mask(
        *,
        planner_data: dict[str, Any],
        forward_inputs: dict[str, Tensor],
        device: torch.device,
    ) -> Tensor:
        input_ids = forward_inputs["input_ids"].to(device=device)
        planner_region_mask = planner_data.get("planner_region_mask")
        if planner_region_mask is None:
            placeholder_mask = planner_data.get("planner_placeholder_mask")
            boundary_mask = planner_data.get("planner_boundary_mask")
            planner_region_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
            if placeholder_mask is not None:
                planner_region_mask = planner_region_mask | placeholder_mask.to(device=device, dtype=torch.bool)
            if boundary_mask is not None:
                planner_region_mask = planner_region_mask | boundary_mask.to(device=device, dtype=torch.bool)
        else:
            planner_region_mask = planner_region_mask.to(device=device, dtype=torch.bool)
        return planner_region_mask

    @staticmethod
    def _infer_vlm_ref_image_region_mask(
        *,
        planner_data: dict[str, Any],
        forward_inputs: dict[str, Tensor],
        device: torch.device,
    ) -> Tensor:
        input_ids = forward_inputs["input_ids"].to(device=device)
        active_mask = forward_inputs["attention_mask"].to(device=device, dtype=torch.bool)
        planner_region_mask = MultiReferencePlannerStage2Strategy._get_vlm_planner_region_mask(
            planner_data=planner_data,
            forward_inputs=forward_inputs,
            device=device,
        )
        return (
            (input_ids == GEMMA3_CONFIG_FOR_LTX.image_token_index)
            | (input_ids == GEMMA3_CONFIG_FOR_LTX.boi_token_index)
            | (input_ids == GEMMA3_CONFIG_FOR_LTX.eoi_token_index)
        ) & ~planner_region_mask & active_mask

    @staticmethod
    def _get_vlm_ref_image_region_mask(
        *,
        planner_data: dict[str, Any],
        forward_inputs: dict[str, Tensor],
        device: torch.device,
    ) -> Tensor:
        ref_image_region_mask = planner_data.get("ref_image_region_mask")
        if ref_image_region_mask is None:
            ref_image_region_mask = planner_data.get("gt_image_token_mask")
        if ref_image_region_mask is None:
            return MultiReferencePlannerStage2Strategy._infer_vlm_ref_image_region_mask(
                planner_data=planner_data,
                forward_inputs=forward_inputs,
                device=device,
            )
        return ref_image_region_mask.to(device=device, dtype=torch.bool)

    def _build_vlm_forward_inputs(self, planner_data: dict[str, Any], device: torch.device) -> dict[str, Tensor]:
        allowed_keys = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "position_ids",
            "cache_position",
        }
        inputs: dict[str, Tensor] = {}
        vlm_dtype = next(self.text_encoder.parameters()).dtype

        for key in allowed_keys:
            value = planner_data.get(key)
            if not isinstance(value, Tensor):
                continue
            if key in {"input_ids", "attention_mask", "position_ids", "cache_position"}:
                inputs[key] = value.to(device=device, dtype=torch.long)
            else:
                inputs[key] = value.to(device=device, dtype=vlm_dtype)

        if "input_ids" not in inputs:
            raise ValueError("planner_vlm_inputs must contain input_ids for online VLM mode.")
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long, device=device)
        return inputs

    def _load_offline_predicted_tokens(
        self,
        planner_data: dict[str, Any],
        gt_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        tokens = planner_data[self.config.predicted_visual_token_key].to(device=gt_tokens.device, dtype=gt_tokens.dtype)
        if tokens.ndim != 3:
            raise ValueError(f"Offline predicted visual tokens must be [B,K,D], got {tuple(tokens.shape)}")
        expected_dim = self._resolved_planner_output_dim(planner_source_dim=self._resolved_planner_source_dim())
        if tokens.shape[-1] != expected_dim:
            raise ValueError(
                "Offline predicted visual tokens must be raw SigLIP-space tokens "
                f"with dim={expected_dim}; got dim={tokens.shape[-1]}. "
                "Regenerate planner_conditions or set compatible planner_output_dim."
            )
        mask = planner_data.get(self.config.predicted_visual_token_mask_key)
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            mask = mask.to(device=tokens.device, dtype=torch.bool)
        return tokens, mask

    def _resolved_planner_source_dim(self) -> int:
        dim = self.config.planner_source_dim or self.config.visual_token_source_dim
        if dim is None:
            raise ValueError("Set planner_source_dim or visual_token_source_dim for Stage 2 planner training.")
        return dim

    def _resolved_planner_output_dim(self, *, planner_source_dim: int | None = None) -> int:
        dim = self.config.planner_output_dim or self.config.visual_token_source_dim or planner_source_dim
        if dim is None:
            raise ValueError("Set planner_output_dim or visual_token_source_dim for Stage 2 planner training.")
        return dim

    @staticmethod
    def _assert_visual_token_dim(name: str, tokens: Tensor, expected_dim: int) -> None:
        if tokens.shape[-1] != expected_dim:
            raise ValueError(f"{name} dim {tokens.shape[-1]} must equal planner_output_dim={expected_dim}.")

    def _configure_vlm_trainable_parameters(self, text_encoder: nn.Module) -> None:
        text_encoder.requires_grad_(False)
        gemma_model = self._unwrap_text_encoder().model.model

        language_model = getattr(gemma_model, "language_model", None)
        if language_model is not None:
            if self._train_gemma_backbone():
                language_model.requires_grad_(True)
            else:
                for name, parameter in language_model.named_parameters():
                    parameter.requires_grad_("lora_" in name)
            if hasattr(language_model, "config"):
                language_model.config.use_cache = False
            if self.config.gemma_gradient_checkpointing:
                self._enable_gemma_gradient_checkpointing(language_model)
                self.assert_gemma_non_reentrant_checkpointing()
                logger.info("Gemma gradient checkpointing: enabled")
                logger.info("Gemma checkpoint use_reentrant: false")

        lm_head = getattr(self._unwrap_text_encoder().model, "lm_head", None)
        if lm_head is not None:
            lm_head.requires_grad_(False)

        vision_tower = getattr(gemma_model, "vision_tower", None)
        if vision_tower is not None:
            vision_tower.requires_grad_(not self.config.freeze_vlm_vision_tower)

        projector = getattr(gemma_model, "multi_modal_projector", None)
        if projector is not None:
            projector.requires_grad_(not self.config.freeze_vlm_multi_modal_projector)

        self._keep_frozen_vlm_modules_in_eval()

    @staticmethod
    def _enable_gemma_gradient_checkpointing(language_model: nn.Module) -> None:
        language_model = getattr(language_model, "module", language_model)
        checkpoint_model = language_model
        if hasattr(language_model, "get_base_model"):
            base_model = language_model.get_base_model()
            if callable(getattr(base_model, "gradient_checkpointing_enable", None)):
                checkpoint_model = base_model

        enable = getattr(checkpoint_model, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise RuntimeError("Gemma model does not support gradient checkpointing")

        try:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError as exc:
            raise RuntimeError(
                "The installed Transformers version does not support non-reentrant gradient checkpointing. "
                "Do not fall back to reentrant checkpointing under DDP."
            ) from exc

        enable_inputs = getattr(checkpoint_model, "enable_input_require_grads", None)
        if callable(enable_inputs):
            enable_inputs()

    def assert_gemma_non_reentrant_checkpointing(self) -> None:
        language_model = self._get_language_model()
        language_model = getattr(language_model, "module", language_model)
        checkpoint_model = (
            language_model.get_base_model() if hasattr(language_model, "get_base_model") else language_model
        )
        if not getattr(checkpoint_model, "is_gradient_checkpointing", False):
            raise RuntimeError("Gemma gradient checkpointing is not enabled")

        checkpoint_func = getattr(checkpoint_model, "_gradient_checkpointing_func", None)
        if isinstance(checkpoint_func, partial) and checkpoint_func.keywords.get("use_reentrant") is not False:
            raise RuntimeError("Gemma gradient checkpointing must use use_reentrant=False")

    def _keep_frozen_vlm_modules_in_eval(self) -> None:
        if self.text_encoder is None:
            return
        gemma_model = self._unwrap_text_encoder().model.model
        vision_tower = getattr(gemma_model, "vision_tower", None)
        projector = getattr(gemma_model, "multi_modal_projector", None)
        if vision_tower is not None and self.config.freeze_vlm_vision_tower:
            vision_tower.eval()
        if projector is not None and self.config.freeze_vlm_multi_modal_projector:
            projector.eval()

    def _train_gemma_backbone(self) -> bool:
        if self.config.train_vlm_language_model is not None:
            return self.config.train_vlm_language_model
        return self.config.train_gemma_backbone

    def _assert_token_count(self, name: str, tokens: Tensor, mask: Tensor) -> None:
        if tokens.ndim != 3:
            raise ValueError(f"{name} must be [B,K,D], got {tuple(tokens.shape)}")
        if tokens.shape[1] != self.config.planner_token_count:
            raise ValueError(
                f"{name} count {tokens.shape[1]} does not match planner_token_count="
                f"{self.config.planner_token_count}. Recompute GT/VLM inputs or fix the config."
            )
        if mask.shape != tokens.shape[:2]:
            raise ValueError(f"{name} mask must be [B,K], got {tuple(mask.shape)}")
        counts = mask.sum(dim=1)
        if torch.any(counts <= 0) or torch.any(counts > self.config.planner_token_count):
            raise ValueError(
                f"{name} valid counts {counts.tolist()} must be in [1,{self.config.planner_token_count}]."
            )

    def _resolve_planner_output_mask(self, planner_data: dict[str, Any], fallback_mask: Tensor) -> Tensor:
        output_mask = planner_data.get(self.config.vlm_output_mask_key)
        if output_mask is None:
            return fallback_mask
        if not isinstance(output_mask, Tensor):
            raise ValueError(f"{self.config.vlm_output_mask_key} must be a tensor")
        output_mask = output_mask.to(device=fallback_mask.device, dtype=torch.bool)
        if output_mask.ndim == 1:
            output_mask = output_mask.unsqueeze(0)
        if output_mask.shape != fallback_mask.shape:
            raise ValueError(
                f"{self.config.vlm_output_mask_key} must have shape {tuple(fallback_mask.shape)}, "
                f"got {tuple(output_mask.shape)}"
            )
        if torch.any(output_mask & ~fallback_mask):
            raise ValueError("planner_output_mask cannot select positions outside planner placeholders")
        if torch.any(output_mask.sum(dim=1) <= 0):
            raise ValueError("planner_output_mask must keep at least one output per sample")
        return output_mask

    @staticmethod
    def _masked_visual_stat(tokens: Tensor, mask: Tensor, statistic: str) -> Tensor:
        values: list[Tensor] = []
        for sample, sample_mask in zip(tokens.float(), mask, strict=True):
            valid = sample[sample_mask]
            if valid.numel() == 0:
                values.append(sample.new_zeros(()))
            elif statistic == "std":
                values.append(valid.std(unbiased=False))
            elif statistic == "norm":
                values.append(valid.norm(dim=-1).mean())
            else:
                raise ValueError(f"Unsupported visual statistic {statistic!r}")
        return torch.stack(values)

    def _assert_placeholder_mask(self, placeholder_mask: Tensor) -> None:
        counts = placeholder_mask.sum(dim=1)
        if not torch.all(counts == self.config.planner_token_count):
            raise ValueError(
                "planner_placeholder_mask counts must equal planner_token_count. "
                f"Got {counts.tolist()} vs {self.config.planner_token_count}."
            )

    def _compute_visual_alignment_loss(
        self,
        *,
        predicted_tokens: Tensor,
        gt_tokens: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if predicted_tokens.shape != gt_tokens.shape:
            raise ValueError(
                f"Predicted visual tokens {tuple(predicted_tokens.shape)} must match GT {tuple(gt_tokens.shape)}"
            )
        loss = F.mse_loss(predicted_tokens, gt_tokens, reduction="none")
        token_loss = loss.mean(dim=-1)
        loss_mask = mask.to(dtype=token_loss.dtype)
        return token_loss.mul(loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)

    def _compute_lm_loss(self, final_hidden: Tensor, labels: Tensor) -> Tensor:
        lm_head = getattr(self._unwrap_text_encoder().model, "lm_head", None)
        if lm_head is None:
            raise ValueError("NTP loss requires text_encoder.model.lm_head")
        lm_head_parameter = next(
            (parameter for parameter in lm_head.parameters() if parameter.is_floating_point()),
            None,
        )
        if lm_head_parameter is None:
            lm_head_device = final_hidden.device
            lm_head_dtype = final_hidden.dtype
        else:
            lm_head_device = lm_head_parameter.device
            lm_head_dtype = lm_head_parameter.dtype
        shift_hidden = final_hidden[:, :-1]
        shift_labels = labels[:, 1:].contiguous()
        losses = []
        for sample_hidden, sample_labels in zip(shift_hidden, shift_labels, strict=True):
            valid = sample_labels != -100
            selected_hidden = sample_hidden[valid]
            selected_labels = sample_labels[valid]
            if selected_labels.numel() == 0:
                losses.append(final_hidden.new_zeros((), dtype=torch.float32))
                continue
            loss_sum = final_hidden.new_zeros((), dtype=torch.float32)
            count = 0
            for start in range(0, selected_labels.numel(), self.config.ntp_logits_chunk_size):
                chunk_hidden = selected_hidden[start : start + self.config.ntp_logits_chunk_size]
                chunk_labels = selected_labels[start : start + self.config.ntp_logits_chunk_size]
                chunk_hidden_for_head = chunk_hidden.to(
                    device=lm_head_device,
                    dtype=lm_head_dtype,
                )
                chunk_labels_for_loss = chunk_labels.to(
                    device=lm_head_device,
                    dtype=torch.long,
                )
                logits = lm_head(chunk_hidden_for_head)
                chunk_loss = F.cross_entropy(
                    logits.float(),
                    chunk_labels_for_loss,
                    reduction="sum",
                )
                loss_sum = loss_sum + chunk_loss.to(device=loss_sum.device)
                count += int(chunk_labels.numel())
            losses.append(loss_sum / max(count, 1))
        return torch.stack(losses)

    def _get_language_model(self) -> nn.Module:
        gemma_model = self._unwrap_text_encoder().model.model
        language_model = getattr(gemma_model, "language_model", None)
        if language_model is None:
            raise ValueError("Gemma model does not expose language_model")
        return language_model

    def _unwrap_text_encoder(self) -> nn.Module:
        if self.text_encoder is None:
            raise RuntimeError("Stage 2 online planner has no text encoder")
        return getattr(self.text_encoder, "module", self.text_encoder)

    @staticmethod
    def _get_input_embeddings(language_model: nn.Module) -> nn.Module:
        language_model = getattr(language_model, "module", language_model)
        if hasattr(language_model, "get_input_embeddings"):
            embeddings = language_model.get_input_embeddings()
            if embeddings is not None:
                return embeddings
        embeddings = getattr(language_model, "embed_tokens", None)
        if embeddings is None:
            raise ValueError("Gemma language_model does not expose input embeddings")
        return embeddings

    @staticmethod
    def _extract_hidden_states(outputs: Any) -> tuple[Tensor, ...]:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None:
            return hidden_states
        raise ValueError("Gemma/VLM output did not include hidden_states.")

    @staticmethod
    def _select_masked_hidden(hidden: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        selected = []
        lengths = mask.sum(dim=1).tolist()
        max_len = max(max(lengths), 1)
        for sample_hidden, sample_mask in zip(hidden, mask, strict=True):
            sample_selected = sample_hidden[sample_mask]
            if sample_selected.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - sample_selected.shape[0],
                    sample_hidden.shape[-1],
                    dtype=sample_hidden.dtype,
                    device=sample_hidden.device,
                )
                sample_selected = torch.cat([sample_selected, pad], dim=0)
            selected.append(sample_selected[:max_len])
        planner_hidden = torch.stack(selected, dim=0)
        planner_mask = torch.arange(max_len, device=hidden.device).unsqueeze(0) < torch.tensor(
            lengths,
            device=hidden.device,
        ).unsqueeze(1)
        return planner_hidden, planner_mask
