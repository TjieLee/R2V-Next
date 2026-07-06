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

    train_vlm_language_model: bool = Field(
        default=True,
        description="Train Gemma language-model parameters during Stage 2.",
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
            "Fixed learnable visual planner token count. Must equal the token count saved in "
            "gt_siglip_tokens/ for every sample."
        ),
        ge=1,
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

        self.planner_tokens = VisualPlannerTokens(
            base_tokens=base_tokens.detach().float(),
            token_count=self.config.planner_token_count,
            dim=dim,
            source_dim=self.config.planner_source_dim,
        ).to(device=base_tokens.device)

        self.text_encoder = text_encoder
        if self.config.use_online_vlm:
            if self.text_encoder is None:
                raise ValueError("Stage 2 online VLM training requires a loaded Gemma text_encoder.")
            self._configure_vlm_trainable_parameters(self.text_encoder)

    def train_transformer(self) -> bool:
        return not self.config.freeze_transformer

    def train_embeddings_processor(self) -> bool:
        return self.config.train_text_connector

    def requires_text_encoder(self) -> bool:
        return self.config.use_online_vlm

    def train_text_encoder(self) -> bool:
        return self.config.use_online_vlm and (
            self.config.train_vlm_language_model
            or not self.config.freeze_vlm_vision_tower
            or not self.config.freeze_vlm_multi_modal_projector
        )

    def get_trainable_modules(self) -> dict[str, nn.Module]:
        if self.planner_tokens is None:
            return {}
        return {"planner_tokens": self.planner_tokens}

    def set_trainable_modules(self, modules: dict[str, nn.Module]) -> None:
        if "planner_tokens" in modules:
            self.planner_tokens = modules["planner_tokens"]

    def prepare_conditions(self, batch: dict[str, Any], conditions: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.planner_tokens is None:
            raise RuntimeError("Planner tokens were not initialized. Did attach_models() run?")

        self._last_planner_mse_loss = None
        self._last_vlm_lm_loss = None

        gt_tokens, gt_mask = self._load_condition_visual_tokens(batch["gt_visual_tokens"], conditions)
        self._assert_token_count("GT visual tokens", gt_tokens, gt_mask)

        if self.config.use_online_vlm:
            predicted_tokens, predicted_mask = self._run_online_vlm(batch["planner_vlm_inputs"], gt_tokens.device)
        else:
            predicted_tokens, predicted_mask = self._load_offline_predicted_tokens(batch["planner_conditions"], gt_tokens)

        self._assert_token_count("Predicted visual tokens", predicted_tokens, predicted_mask)
        predicted_tokens = predicted_tokens.to(device=gt_tokens.device, dtype=gt_tokens.dtype)
        predicted_mask = predicted_mask.to(device=gt_tokens.device, dtype=torch.bool) & gt_mask

        if self.config.planner_mse_weight > 0:
            self._last_planner_mse_loss = self._compute_visual_alignment_loss(
                predicted_tokens=predicted_tokens,
                gt_tokens=gt_tokens,
                mask=predicted_mask,
            )

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
                "planner_mse_weight": self.config.planner_mse_weight,
                "flow_loss_weight": self.config.flow_loss_weight,
                "freeze_transformer": self.config.freeze_transformer,
                "train_text_connector": self.config.train_text_connector,
                "use_online_vlm": self.config.use_online_vlm,
                "train_vlm_language_model": self.config.train_vlm_language_model,
                "freeze_vlm_vision_tower": self.config.freeze_vlm_vision_tower,
                "freeze_vlm_multi_modal_projector": self.config.freeze_vlm_multi_modal_projector,
            }
        )
        return metadata

    def _run_online_vlm(self, planner_data: dict[str, Any], device: torch.device) -> tuple[Tensor, Tensor]:
        if self.text_encoder is None or self.planner_tokens is None:
            raise RuntimeError("Online VLM mode requires text_encoder and planner_tokens.")

        forward_inputs = self._build_vlm_forward_inputs(planner_data, device)
        placeholder_mask = planner_data[self.config.vlm_placeholder_mask_key].to(device=device, dtype=torch.bool)
        self._assert_placeholder_mask(placeholder_mask)

        inputs_embeds = self._build_vlm_inputs_embeds(forward_inputs, planner_data, placeholder_mask)
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

        outputs = language_model(**lm_inputs)
        hidden_states = self._extract_hidden_states(outputs)
        selected_hidden, selected_mask = self._select_masked_hidden(
            hidden_states[self.config.vlm_hidden_layer],
            placeholder_mask,
        )
        predicted_tokens = self.planner_tokens.project_hidden(selected_hidden)

        labels = planner_data.get(self.config.vlm_lm_labels_key)
        if labels is not None and self.config.vlm_lm_loss_weight > 0:
            self._last_vlm_lm_loss = self._compute_lm_loss(hidden_states[-1], labels.to(device=device, dtype=torch.long))

        return predicted_tokens, selected_mask

    def _build_vlm_inputs_embeds(
        self,
        forward_inputs: dict[str, Tensor],
        planner_data: dict[str, Any],
        placeholder_mask: Tensor,
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
                planner_placeholder_mask=placeholder_mask,
            )

        planner_inputs = self.planner_tokens.input_embeddings(
            batch_size=input_ids.shape[0],
            device=input_ids.device,
            dtype=inputs_embeds.dtype,
        )
        out = inputs_embeds.clone()
        for batch_index in range(out.shape[0]):
            out[batch_index, placeholder_mask[batch_index]] = planner_inputs[batch_index]
        return out

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
        if language_model is not None and self.config.train_vlm_language_model:
            language_model.requires_grad_(True)

        lm_head = getattr(text_encoder.model, "lm_head", None)
        if lm_head is not None and self.config.train_vlm_language_model and self.config.vlm_lm_loss_weight > 0:
            lm_head.requires_grad_(True)

        vision_tower = getattr(gemma_model, "vision_tower", None)
        if vision_tower is not None:
            vision_tower.requires_grad_(not self.config.freeze_vlm_vision_tower)

        projector = getattr(gemma_model, "multi_modal_projector", None)
        if projector is not None:
            projector.requires_grad_(not self.config.freeze_vlm_multi_modal_projector)

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
