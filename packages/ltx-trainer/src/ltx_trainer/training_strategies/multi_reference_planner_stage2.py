"""Stage 2 Baton-style VLM planner training for multi-reference video generation.

Stage 1 teaches the DiT to consume text/thinking tokens plus frozen
SigLIP/projector visual tokens. Stage 2 keeps that DiT interface unchanged but
replaces the GT visual tokens with fixed-count VLM planner placeholder outputs.
The placeholder count must match the GT SigLIP/projector token count exactly;
an MSE loss aligns the VLM-predicted visual tokens to the GT visual tokens.
"""

from typing import Any, Literal

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import Tensor, nn

from ltx_core.multicond.visual_tokens import (
    VisualPlannerTokens,
    extract_projected_visual_tokens,
    scatter_visual_tokens_into_embeddings,
)
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.multi_reference_video import (
    MultiReferenceVideoConfig,
    MultiReferenceVideoStrategy,
)


class MultiReferencePlannerStage2Config(MultiReferenceVideoConfig):
    """Stage 2 VLM planner token training."""

    name: Literal["multi_reference_planner_stage2"] = "multi_reference_planner_stage2"

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

    vlm_hidden_layer: int = Field(
        default=-1,
        description="Gemma language-model hidden-state layer used as predicted visual tokens.",
    )

    vlm_lm_labels_key: str = Field(
        default="labels",
        description="Optional labels key for next-token/language-model loss. Use -100 for ignored tokens.",
    )

    vlm_lm_loss_weight: float = Field(
        default=0.0,
        description="Weight for optional Gemma language-model loss.",
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
        default=1024,
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
        default=True,
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
        default=True,
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
        default=None,
        description="Hidden size of the selected VLM layer. None assumes it equals the connector input dimension.",
        ge=1,
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
        description="Weight for MSE(predicted visual tokens, GT SigLIP/projector visual tokens).",
        ge=0.0,
    )

    flow_loss_weight: float = Field(
        default=1.0,
        description="Weight for the Stage 1 target flow-matching loss.",
        ge=0.0,
    )

    freeze_transformer: bool = Field(
        default=True,
        description="Freeze the Stage 1 transformer/LoRA weights while training the VLM planner.",
    )

    train_text_connector: bool = Field(
        default=True,
        description="Also optimize the LTX video embedding connector during Stage 2.",
    )

    def get_data_sources(self) -> dict[str, str]:
        data_sources = super().get_data_sources()
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
        self._planner_query_registers: Tensor | None = None
        self._last_planner_mse_loss: Tensor | None = None
        self._last_vlm_lm_loss: Tensor | None = None

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
        video_connector = embeddings_processor.video_connector
        base_tokens = getattr(video_connector, "learnable_registers", None)
        dim = getattr(video_connector, "inner_dim", None)

        if base_tokens is None:
            if dim is None:
                raise ValueError("Cannot initialize planner tokens: video connector has no inner_dim")
            base_tokens = torch.zeros(1, dim, device=next(video_connector.parameters()).device)
        elif dim is None:
            dim = base_tokens.shape[-1]
        self._planner_query_registers = base_tokens

        self.planner_tokens = VisualPlannerTokens(
            token_count=self.config.planner_token_count,
            dim=dim,
            source_dim=self.config.planner_source_dim,
            num_heads=self.config.planner_cross_attention_heads,
            dropout=self.config.planner_cross_attention_dropout,
            zero_init_output=self.config.planner_zero_init_cross_attention,
            use_slot_encoding=self.config.planner_slot_encoding,
            slot_init_std=self.config.planner_slot_init_std,
            slot_init_seed=self.config.planner_slot_init_seed,
            ffn_multiplier=self.config.planner_ffn_multiplier,
            ffn_dropout=self.config.planner_ffn_dropout,
            zero_init_ffn=self.config.planner_zero_init_ffn,
        ).to(device=base_tokens.device)

        self.text_encoder = text_encoder
        if self.config.use_online_vlm:
            if self.text_encoder is None:
                raise ValueError("Stage 2 online VLM training requires a loaded Gemma text_encoder.")
            if self.config.vlm_lm_loss_weight > 0 and not self._train_gemma_backbone():
                raise ValueError("vlm_lm_loss_weight > 0 requires train_gemma_backbone: true.")
            self._configure_vlm_trainable_parameters(self.text_encoder)

    def train_transformer(self) -> bool:
        return not self.config.freeze_transformer

    def train_embeddings_processor(self) -> bool:
        return self.config.train_text_connector

    def requires_text_encoder(self) -> bool:
        return self.config.use_online_vlm

    def train_text_encoder(self) -> bool:
        return self.config.use_online_vlm and (
            self._train_gemma_backbone()
            or not self.config.freeze_vlm_vision_tower
            or not self.config.freeze_vlm_multi_modal_projector
        )

    def get_trainable_modules(self) -> dict[str, nn.Module]:
        modules = super().get_trainable_modules()
        if self.planner_tokens is not None:
            modules["planner_tokens"] = self.planner_tokens
        return modules

    def set_trainable_modules(self, modules: dict[str, nn.Module]) -> None:
        super().set_trainable_modules(modules)
        if "planner_tokens" in modules:
            self.planner_tokens = modules["planner_tokens"]

    def prepare_conditions(self, batch: dict[str, Any], conditions: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.planner_tokens is None:
            raise RuntimeError("Planner tokens were not initialized. Did attach_models() run?")

        self._last_planner_mse_loss = None
        self._last_vlm_lm_loss = None

        conditions = self._apply_cfg_context_dropout(batch, conditions)
        gt_tokens, gt_mask = self._load_condition_visual_tokens(batch["gt_visual_tokens"], conditions)
        self._assert_token_count("GT visual tokens", gt_tokens, gt_mask)

        if self.config.use_online_vlm:
            predicted_tokens, predicted_mask = self._run_online_vlm(batch, batch["planner_vlm_inputs"], gt_tokens.device)
        else:
            predicted_tokens, predicted_mask = self._load_offline_predicted_tokens(batch["planner_conditions"], gt_tokens)

        self._assert_token_count("Predicted visual tokens", predicted_tokens, predicted_mask)
        predicted_tokens = predicted_tokens.to(device=gt_tokens.device, dtype=gt_tokens.dtype)
        predicted_mask = predicted_mask.to(device=gt_tokens.device, dtype=torch.bool) & gt_mask
        drop_visual_mask = self._cfg_drop_visual_mask(
            batch,
            batch_size=predicted_tokens.shape[0],
            device=predicted_tokens.device,
        )

        if self.config.planner_mse_weight > 0:
            mse_mask = predicted_mask
            if drop_visual_mask is not None and torch.any(drop_visual_mask):
                mse_mask = mse_mask & ~drop_visual_mask[:, None]
            self._last_planner_mse_loss = self._compute_visual_alignment_loss(
                predicted_tokens=predicted_tokens,
                gt_tokens=gt_tokens,
                mask=mse_mask,
            )

        predicted_tokens, predicted_mask = self._apply_cfg_planner_dropout(batch, predicted_tokens, predicted_mask)
        return self._append_visual_tokens_to_conditions(conditions, predicted_tokens, predicted_mask)

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        flow_loss = super().compute_loss(video_pred, audio_pred, inputs) * self.config.flow_loss_weight
        loss = flow_loss
        if self._last_planner_mse_loss is not None:
            loss = loss + self._last_planner_mse_loss.to(device=flow_loss.device, dtype=flow_loss.dtype) * (
                self.config.planner_mse_weight
            )
        if self._last_vlm_lm_loss is not None and self.config.vlm_lm_loss_weight > 0:
            loss = loss + self._last_vlm_lm_loss.to(device=flow_loss.device, dtype=flow_loss.dtype) * (
                self.config.vlm_lm_loss_weight
            )
        return loss

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        metadata = super().get_checkpoint_metadata()
        metadata.update(
            {
                "conditioning": "multi_reference_planner_stage2",
                "planner_token_count": self.config.planner_token_count,
                "planner_cross_attention_heads": self.config.planner_cross_attention_heads,
                "planner_zero_init_cross_attention": self.config.planner_zero_init_cross_attention,
                "planner_ffn_multiplier": self.config.planner_ffn_multiplier,
                "planner_zero_init_ffn": self.config.planner_zero_init_ffn,
                "planner_slot_encoding": self.config.planner_slot_encoding,
                "planner_slot_init_std": self.config.planner_slot_init_std,
                "planner_slot_init_seed": self.config.planner_slot_init_seed,
                "planner_mse_weight": self.config.planner_mse_weight,
                "flow_loss_weight": self.config.flow_loss_weight,
                "freeze_transformer": self.config.freeze_transformer,
                "train_text_connector": self.config.train_text_connector,
                "use_online_vlm": self.config.use_online_vlm,
                "train_gemma_backbone": self._train_gemma_backbone(),
                "freeze_vlm_vision_tower": self.config.freeze_vlm_vision_tower,
                "freeze_vlm_multi_modal_projector": self.config.freeze_vlm_multi_modal_projector,
                "cfg_drop_all_p": self._cfg_drop_all_probability(),
                "cfg_drop_planner_p_is_legacy_drop_all_alias": True,
            }
        )
        return metadata

    def _run_online_vlm(
        self,
        batch: dict[str, Any],
        planner_data: dict[str, Any],
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        if self.text_encoder is None or self.planner_tokens is None or self._planner_query_registers is None:
            raise RuntimeError("Online VLM mode requires text_encoder, planner_tokens and query registers.")

        forward_inputs = self._build_vlm_forward_inputs(planner_data, device)
        placeholder_mask = planner_data[self.config.vlm_placeholder_mask_key].to(device=device, dtype=torch.bool)
        self._assert_placeholder_mask(placeholder_mask)
        drop_ref_mask = self._cfg_drop_ref_mask(
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
        language_model = self._get_language_model()
        lm_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": forward_inputs["attention_mask"],
            "output_hidden_states": True,
            "return_dict": True,
        }
        for key in ("position_ids", "cache_position"):
            if key in forward_inputs:
                lm_inputs[key] = forward_inputs[key]

        with torch.set_grad_enabled(self.train_text_encoder()):
            outputs = language_model(**lm_inputs)
        hidden_states = self._extract_hidden_states(outputs)
        selected_hidden, selected_mask = self._select_masked_hidden(
            hidden_states[self.config.vlm_hidden_layer],
            placeholder_mask,
        )
        predicted_tokens = self.planner_tokens(
            planner_hidden=selected_hidden,
            query_registers=self._planner_query_registers,
            planner_mask=selected_mask,
        )

        labels = planner_data.get(self.config.vlm_lm_labels_key)
        if labels is not None and self.config.vlm_lm_loss_weight > 0:
            self._last_vlm_lm_loss = self._compute_lm_loss(hidden_states[-1], labels.to(device=device, dtype=torch.long))

        return predicted_tokens, selected_mask

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
                self.text_encoder.model,
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
        mask = planner_data.get(self.config.predicted_visual_token_mask_key)
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            mask = mask.to(device=tokens.device, dtype=torch.bool)
        return tokens, mask

    def _configure_vlm_trainable_parameters(self, text_encoder: nn.Module) -> None:
        text_encoder.requires_grad_(False)
        gemma_model = text_encoder.model.model

        language_model = getattr(gemma_model, "language_model", None)
        if language_model is not None and self._train_gemma_backbone():
            language_model.requires_grad_(True)

        lm_head = getattr(text_encoder.model, "lm_head", None)
        if lm_head is not None and self._train_gemma_backbone() and self.config.vlm_lm_loss_weight > 0:
            lm_head.requires_grad_(True)

        vision_tower = getattr(gemma_model, "vision_tower", None)
        if vision_tower is not None:
            vision_tower.requires_grad_(not self.config.freeze_vlm_vision_tower)

        projector = getattr(gemma_model, "multi_modal_projector", None)
        if projector is not None:
            projector.requires_grad_(not self.config.freeze_vlm_multi_modal_projector)

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
        if not torch.all(counts == self.config.planner_token_count):
            raise ValueError(
                f"{name} valid counts {counts.tolist()} must all equal planner_token_count="
                f"{self.config.planner_token_count}."
            )

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
        loss_mask = mask.unsqueeze(-1).to(dtype=loss.dtype)
        return loss.mul(loss_mask).sum(dim=[1, 2]) / loss_mask.sum(dim=[1, 2]).clamp(min=1.0)

    def _compute_lm_loss(self, final_hidden: Tensor, labels: Tensor) -> Tensor:
        lm_head = getattr(self.text_encoder.model, "lm_head", None)
        if lm_head is None:
            raise ValueError("vlm_lm_loss_weight > 0 requires text_encoder.model.lm_head")
        logits = lm_head(final_hidden)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def _get_language_model(self) -> nn.Module:
        gemma_model = self.text_encoder.model.model
        language_model = getattr(gemma_model, "language_model", None)
        if language_model is None:
            raise ValueError("Gemma model does not expose language_model")
        return language_model

    @staticmethod
    def _get_input_embeddings(language_model: nn.Module) -> nn.Module:
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
