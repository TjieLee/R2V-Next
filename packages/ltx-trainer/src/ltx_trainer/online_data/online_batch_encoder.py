"""Online raw-media encoding for the semantic-flow training strategy."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from transformers import AutoImageProcessor, AutoTokenizer, Gemma3Processor

from ltx_core.multicond.semantic_tokens import (
    EVIDENCE_TOKENS_PER_FRAME,
    build_multimodal_prefix_attention_mask,
    round_up_prefix_length,
)
from ltx_core.multicond.visual_tokens import (
    extract_projected_visual_tokens,
    module_compute_device_dtype,
    scatter_visual_tokens_into_embeddings,
)
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_core.utils import find_matching_file
from ltx_trainer.config import OnlineEncodingConfig
from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_data.transforms import (
    augment_reference_image,
    augment_target_frames,
    augmentation_seed,
)


class OnlineSampleEncodeError(RuntimeError):
    """A sample-specific error for which retrying another manifest row is valid."""

    def __init__(self, message: str, *, reason: str = "online_sample_encode_error") -> None:
        self.reason = reason
        self.manifest_index: int | None = None
        self.sample_key: str | None = None
        self.task: str | None = None
        self.reference_path: str | None = None
        self.reference_paths: list[str] | None = None
        super().__init__(f"{reason}: {message}")

    def attach_sample_context(self, raw_batch: dict[str, Any]) -> None:
        def first(value: Any) -> Any:
            if isinstance(value, Tensor):
                return value.flatten()[0].item() if value.numel() else None
            if isinstance(value, (list, tuple)):
                return value[0] if value else None
            return value

        manifest_index = first(raw_batch.get("manifest_index"))
        self.manifest_index = int(manifest_index) if manifest_index is not None else None
        sample_key = first(raw_batch.get("sample_key"))
        self.sample_key = str(sample_key) if sample_key is not None else None
        task = first(raw_batch.get("task"))
        self.task = str(task) if task is not None else None
        reference_paths = first(raw_batch.get("reference_paths"))
        if isinstance(reference_paths, (list, tuple)):
            self.reference_paths = [str(path) for path in reference_paths]
            if len(self.reference_paths) == 1:
                self.reference_path = self.reference_paths[0]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error_type": type(self).__name__,
            "reason": self.reason,
            "message": str(self),
        }
        for key in ("manifest_index", "sample_key", "task", "reference_path", "reference_paths"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


def _to_pil(image: Tensor) -> Image.Image:
    if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected uint8 [H,W,3] image, got {tuple(image.shape)} {image.dtype}")
    return Image.fromarray(image.cpu().numpy(), mode="RGB")


def _build_messages(
    system_prompt: str,
    user_prompt: str,
    num_images: int,
    *,
    task: str,
) -> list[dict[str, Any]]:
    """Serialize system, references, then user prompt for both training and inference."""
    if not user_prompt.strip():
        user_text = ""
    elif task == IMAGE_TASK:
        user_text = f"Image editing instruction: {user_prompt}"
    elif task == VIDEO_TASK:
        user_text = f"User Raw Input Prompt: {user_prompt}."
    else:
        raise ValueError(f"Unsupported online task {task!r}")
    content: list[dict[str, str]] = []
    if num_images:
        content.append({"type": "text", "text": "Reference images:"})
        for index in range(num_images):
            content.extend(
                [
                    {"type": "text", "text": f"Reference image {index + 1}:"},
                    {"type": "image"},
                ]
            )
    if user_text:
        content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


class OnlineBatchEncoder:
    """Encode one homogeneous local microbatch without feature caches."""

    def __init__(
        self,
        *,
        config: OnlineEncodingConfig,
        model_path: str,
        text_encoder_path: str,
        vae_encoder: nn.Module,
        text_encoder: nn.Module,
        embeddings_processor: nn.Module,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model_path = model_path
        self.text_encoder_path = text_encoder_path
        self.vae_encoder = vae_encoder
        self.text_encoder = text_encoder
        self.embeddings_processor = embeddings_processor
        self.device = device
        self.dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[config.encoder_dtype]
        self.last_dtype_diagnostics: dict[str, str] = {}

        tokenizer_root = str(find_matching_file(text_encoder_path, "tokenizer.model").parent)
        processor_root = str(find_matching_file(text_encoder_path, "preprocessor_config.json").parent)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_root,
            local_files_only=True,
            model_max_length=config.vlm_teacher_max_length,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            processor_root,
            local_files_only=True,
            use_fast=False,
        )
        self.processor = Gemma3Processor(image_processor=self.image_processor, tokenizer=self.tokenizer)
        prompt_root = (
            Path(__file__).resolve().parents[4]
            / "ltx-core"
            / "src"
            / "ltx_core"
            / "text_encoders"
            / "gemma"
            / "encoders"
            / "prompts"
        )
        self.system_prompts = {
            IMAGE_TASK: (prompt_root / "gemma_i2i_semantic_system_prompt.txt").read_text(encoding="utf-8"),
            VIDEO_TASK: (prompt_root / "gemma_r2v_semantic_system_prompt.txt").read_text(encoding="utf-8"),
        }
        self.vae_encoder.requires_grad_(False).eval()
        self._keep_frozen_modules_eval()

    def encode_for_strategy(
        self,
        raw_batch: dict[str, Any],
        *,
        strategy: Any,
        optimizer_step: int = 0,
        microstep: int = 0,
        global_seed: int = 0,
    ) -> dict[str, Any]:
        strategy_config = strategy.config
        self.last_dtype_diagnostics = {}
        augmentation_started = time.perf_counter()
        raw_batch = self._augment_training_batch(
            raw_batch,
            optimizer_step=optimizer_step,
            microstep=microstep,
            global_seed=global_seed,
        )
        target_pixels = raw_batch["target_pixels"]
        if target_pixels.shape[0] != 1:
            raise ValueError("Online semantic-flow encoding currently requires local microbatch size 1")
        task = str(raw_batch["task"][0])
        expected_frames = 1 if task == IMAGE_TASK else 121
        if target_pixels.shape[1] != expected_frames:
            raise ValueError(f"Task {task} expected {expected_frames} target frames, got {target_pixels.shape[1]}")

        metrics = {
            "data_decode_ms": float(raw_batch["data_decode_ms"].float().mean().item()),
            "augmentation_ms": (time.perf_counter() - augmentation_started) * 1000.0,
            "vae_encode_ms": 0.0,
            "gemma_prefix_ms": 0.0,
            "gemma_evidence_ms": 0.0,
        }
        reference_pixels_vae, reference_images_vlm = self._resolve_reference_inputs(raw_batch)
        condition_mode = self._sample_condition_mode(
            sample_key=str(raw_batch["sample_key"][0]),
            optimizer_step=optimizer_step,
            microstep=microstep,
            global_seed=global_seed,
            strategy_config=strategy_config,
        )
        drop_reference = condition_mode in {"drop_reference_all", "drop_all"}
        drop_text = condition_mode in {"drop_text", "drop_all"}
        if drop_reference:
            reference_pixels_vae = [[]]
            reference_images_vlm = [[]]
        references = self._reference_images(reference_images_vlm[0])
        caption = "" if drop_text else str(raw_batch["caption"][0])

        vae_started = time.perf_counter()
        latents = self._encode_target_latents(target_pixels, raw_batch["target_fps"])
        reference_latents = self._encode_reference_latents(
            reference_pixels_vae,
            fallback_height=target_pixels.shape[2],
            fallback_width=target_pixels.shape[3],
        )
        metrics["vae_encode_ms"] = (time.perf_counter() - vae_started) * 1000.0

        prefix_started = time.perf_counter()
        conditions, prefix_inputs = self._encode_prefix(
            caption=caption,
            reference_images=references,
            task=task,
            sample_key=str(raw_batch["sample_key"][0]),
        )
        metrics["gemma_prefix_ms"] = (time.perf_counter() - prefix_started) * 1000.0

        evidence_started = time.perf_counter()
        if condition_mode == "drop_all":
            conditions = {
                key: torch.zeros_like(value) if isinstance(value, Tensor) else value
                for key, value in conditions.items()
            }
        evidence = self._encode_gt_evidence(raw_batch, task=task)
        metrics["gemma_evidence_ms"] = (time.perf_counter() - evidence_started) * 1000.0
        semantic_teacher_inputs = {**prefix_inputs, **evidence}

        return {
            "sample_key": raw_batch["sample_key"],
            "sample_plan_sha256": raw_batch["sample_plan_sha256"],
            "task": raw_batch["task"],
            "target_modality": raw_batch["target_modality"],
            "manifest_index": raw_batch["manifest_index"],
            "target_source_frame_indices": raw_batch["target_source_frame_indices"],
            "semantic_anchor_target_indices": raw_batch["semantic_anchor_target_indices"],
            "semantic_anchor_source_indices": raw_batch["semantic_anchor_source_indices"],
            "latents": latents,
            "reference_latents": reference_latents,
            "conditions": conditions,
            "semantic_teacher_inputs": semantic_teacher_inputs,
            "_online_metrics": metrics,
            "condition_mode": condition_mode,
        }

    def encode_inference_conditions_from_references(
        self,
        *,
        task: str,
        caption: str,
        reference_pixels_vae: list[Tensor],
        reference_images_vlm: list[Tensor],
        width: int,
        height: int,
        num_frames: int,
        fps: float,
    ) -> dict[str, Any]:
        """Encode only text/reference conditions; inference never creates teacher inputs."""
        if task not in {IMAGE_TASK, VIDEO_TASK}:
            raise ValueError(f"Unsupported online inference task {task!r}")
        expected_geometry = (
            self.config.width,
            self.config.height,
            self.config.image_num_frames if task == IMAGE_TASK else self.config.video_num_frames,
            self.config.image_fps if task == IMAGE_TASK else self.config.video_fps,
        )
        if (int(width), int(height), int(num_frames), float(fps)) != expected_geometry:
            raise ValueError(f"Task {task} requires geometry {expected_geometry}")
        if not caption.strip():
            raise ValueError("Online inference caption must not be empty")
        if len(reference_pixels_vae) != len(reference_images_vlm):
            raise ValueError("Reference VAE/VLM counts differ")
        if not 1 <= len(reference_pixels_vae) <= int(self.config.max_ref_images or 4):
            raise ValueError("Online inference requires 1..4 reference images")

        references = self._reference_images(reference_images_vlm)
        reference_latents = self._encode_reference_latents(
            [reference_pixels_vae],
            fallback_height=height,
            fallback_width=width,
        )
        conditions, _prefix_inputs = self._encode_prefix(
            caption=caption,
            reference_images=references,
            task=task,
            sample_key="inference",
        )
        result = {
            "task": task,
            "reference_latents": reference_latents,
            "conditions": conditions,
            "reference_metadata": {
                "reference_count": len(references),
                "reference_order": list(range(len(references))),
            },
        }
        forbidden = {"target_pixels", "latents", "semantic_teacher_inputs", "evidence_tokens"}
        leaked = sorted(forbidden & result.keys())
        if leaked:
            raise RuntimeError(f"Strict-no-GT inference encoder emitted forbidden keys: {leaked}")
        return result

    def _encode_target_latents(self, target_pixels: Tensor, fps: Tensor) -> dict[str, Tensor]:
        video = target_pixels.to(device=self.device, dtype=self.dtype, non_blocking=True)
        video = video.permute(0, 4, 1, 2, 3).div_(127.5).sub_(1.0)
        with torch.inference_mode(), self._frozen_encode_autocast():
            encoded = self.vae_encoder(video)
        return {
            "latents": encoded,
            "num_frames": torch.tensor([target_pixels.shape[1]], device=self.device, dtype=torch.long),
            "height": torch.tensor([target_pixels.shape[2]], device=self.device, dtype=torch.long),
            "width": torch.tensor([target_pixels.shape[3]], device=self.device, dtype=torch.long),
            "fps": fps.to(device=self.device, dtype=torch.float32),
        }

    def _encode_reference_latents(
        self,
        reference_batches: list[list[Tensor]],
        *,
        fallback_height: int,
        fallback_width: int,
    ) -> dict[str, Tensor]:
        batch_size = len(reference_batches)
        max_refs = int(self.config.max_ref_images or 4)
        valid_mask = torch.zeros(batch_size, max_refs, device=self.device, dtype=torch.bool)
        template = next(
            (reference for references in reference_batches for reference in references),
            torch.zeros(fallback_height, fallback_width, 3, dtype=torch.uint8),
        )
        flat: list[Tensor] = []
        for batch_index, references in enumerate(reference_batches):
            references = references[:max_refs]
            valid_mask[batch_index, : len(references)] = True
            flat.extend(references)
            flat.extend(torch.zeros_like(template) for _ in range(max_refs - len(references)))
        pixels = torch.stack(flat).to(device=self.device, dtype=self.dtype, non_blocking=True)
        pixels = pixels.permute(0, 3, 1, 2).unsqueeze(2).div_(127.5).sub_(1.0)
        with torch.inference_mode(), self._frozen_encode_autocast():
            encoded = self.vae_encoder(pixels)
        encoded = encoded.reshape(batch_size, max_refs, *encoded.shape[1:])
        encoded *= valid_mask[:, :, None, None, None, None].to(dtype=encoded.dtype)
        return {
            "latents": encoded,
            "ref_valid_mask": valid_mask,
            "fps": torch.ones(batch_size, device=self.device, dtype=torch.float32),
        }

    def _encode_prefix(
        self,
        *,
        caption: str,
        reference_images: list[Image.Image],
        task: str,
        sample_key: str,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        processed, reference_region_mask = self._process_multimodal_prefix(
            caption=caption,
            reference_images=reference_images,
            task=task,
            sample_key=sample_key,
        )
        input_ids = processed["input_ids"].to(device=self.device, dtype=torch.long)
        attention_mask = processed["attention_mask"].to(device=self.device, dtype=torch.long)
        language_model = self._get_language_model()
        core_language_model = getattr(language_model, "module", language_model)
        inputs_embeds = core_language_model.get_input_embeddings()(input_ids)
        pixel_values = processed.get("pixel_values")
        if pixel_values is not None:
            with torch.inference_mode(), self._frozen_encode_autocast():
                visual = extract_projected_visual_tokens(
                    self._unwrap_text_encoder().model,
                    pixel_values.to(device=self.device, dtype=self.dtype),
                    image_counts=torch.tensor([len(reference_images)], device=self.device, dtype=torch.long),
                    dtype_diagnostics=self.last_dtype_diagnostics,
                )
            inputs_embeds = scatter_visual_tokens_into_embeddings(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                visual_tokens=visual.tokens,
                visual_mask=visual.mask,
                image_token_index=GEMMA3_CONFIG_FOR_LTX.image_token_index,
            )

        feature_extractor = getattr(self.embeddings_processor, "feature_extractor", None)
        if feature_extractor is None:
            raise RuntimeError("Online prefix encoding requires embeddings_processor.feature_extractor")
        feature_extractor.requires_grad_(False).eval()
        language_model.eval()
        prefix_reference_mask = reference_region_mask.to(device=self.device).unsqueeze(0)
        prefix_visibility = build_multimodal_prefix_attention_mask(
            attention_mask,
            reference_region_mask=prefix_reference_mask,
        )
        finfo = torch.finfo(inputs_embeds.dtype)
        attention_bias = torch.zeros_like(prefix_visibility, dtype=inputs_embeds.dtype)
        attention_bias.masked_fill_(~prefix_visibility, finfo.min)
        attention_bias = attention_bias.unsqueeze(1)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
        with torch.inference_mode(), self._frozen_encode_autocast():
            outputs = language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_bias,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_states = self._align_hidden_states(outputs.hidden_states, feature_extractor)
            video_features, audio_features = feature_extractor(hidden_states, attention_mask, "right")
        conditions = {
            "video_prompt_embeds": video_features,
            "prompt_attention_mask": attention_mask,
        }
        if audio_features is not None:
            conditions["audio_prompt_embeds"] = audio_features
        teacher_prefix = {
            "prefix_inputs_embeds": inputs_embeds.detach(),
            "prefix_attention_mask": attention_mask,
            "prefix_reference_region_mask": prefix_reference_mask,
        }
        return conditions, teacher_prefix

    def _encode_gt_evidence(self, raw_batch: dict[str, Any], *, task: str) -> dict[str, Tensor]:
        target_pixels = raw_batch["target_pixels"]
        indices = raw_batch["semantic_anchor_target_indices"][0].to(dtype=torch.long).tolist()
        if task == IMAGE_TASK and indices != [0]:
            raise ValueError(f"I2I semantic anchor indices must be [0], got {indices}")
        selected = target_pixels[0, indices]
        processed = self.image_processor(images=[_to_pil(frame) for frame in selected], return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=self.device, dtype=self.dtype)
        self._keep_frozen_modules_eval()
        with torch.inference_mode(), self._frozen_encode_autocast():
            visual = extract_projected_visual_tokens(
                self._unwrap_text_encoder().model,
                pixel_values,
                image_counts=torch.tensor([len(indices)], device=self.device, dtype=torch.long),
                dtype_diagnostics=self.last_dtype_diagnostics,
            )
        if visual.tokens.shape[1] != len(indices) * EVIDENCE_TOKENS_PER_FRAME:
            raise ValueError(
                "Gemma native visual evidence shape mismatch: "
                f"frames={len(indices)}, tokens={visual.tokens.shape[1]}"
            )
        evidence = visual.tokens.reshape(1, len(indices), EVIDENCE_TOKENS_PER_FRAME, -1).detach()
        denominator = max(1, target_pixels.shape[1] - 1)
        normalized = torch.tensor(indices, device=self.device, dtype=torch.float32).unsqueeze(0) / denominator
        return {
            "evidence_tokens": evidence,
            "normalized_timestamps": normalized,
        }

    def _process_multimodal_prefix(
        self,
        *,
        caption: str,
        reference_images: list[Image.Image],
        task: str,
        sample_key: str,
    ) -> tuple[dict[str, Tensor], Tensor]:
        text = self.tokenizer.apply_chat_template(
            _build_messages(self.system_prompts[task], caption, len(reference_images), task=task),
            tokenize=False,
            add_generation_prompt=True,
        )
        kwargs: dict[str, Any] = {"input_data_format": "channels_last"} if reference_images else {}
        try:
            processed = self.processor(
                text=text,
                images=reference_images or None,
                return_tensors="pt",
                padding=False,
                truncation=False,
                **kwargs,
            )
        except (ValueError, TypeError) as exc:
            raise OnlineSampleEncodeError(
                f"Gemma reference processor failed for sample={sample_key}: {type(exc).__name__}: {exc}",
                reason="gemma_reference_processor_failure",
            ) from exc
        input_ids = processed["input_ids"][0].to(dtype=torch.long)
        attention_mask = processed["attention_mask"][0].to(dtype=torch.long)
        reference_region_mask = self._validate_reference_regions(
            input_ids,
            num_reference_images=len(reference_images),
        )
        input_ids, attention_mask, reference_region_mask = self._truncate_prefix(
            input_ids,
            attention_mask,
            reference_region_mask,
            sample_key=sample_key,
        )
        padded_length = round_up_prefix_length(
            input_ids.numel(),
            maximum=self.config.vlm_prefix_max_length,
        )
        pad_length = padded_length - input_ids.numel()
        if pad_length:
            pad_id = int(self.tokenizer.pad_token_id or 0)
            input_ids = torch.cat([input_ids, torch.full((pad_length,), pad_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
            reference_region_mask = torch.cat(
                [reference_region_mask, torch.zeros(pad_length, dtype=torch.bool)]
            )
        result = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }
        if isinstance(processed.get("pixel_values"), Tensor):
            result["pixel_values"] = processed["pixel_values"]
        return result, reference_region_mask

    def _truncate_prefix(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        reference_region_mask: Tensor,
        *,
        sample_key: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        maximum = self.config.vlm_prefix_max_length
        if input_ids.numel() <= maximum:
            return input_ids, attention_mask, reference_region_mask
        protected = reference_region_mask.clone()
        for token_id in set(getattr(self.tokenizer, "all_special_ids", [])) | {
            GEMMA3_CONFIG_FOR_LTX.boi_token_index,
            GEMMA3_CONFIG_FOR_LTX.eoi_token_index,
            GEMMA3_CONFIG_FOR_LTX.image_token_index,
        }:
            protected |= input_ids == int(token_id)
        removable = torch.nonzero(~protected, as_tuple=False).flatten()
        overflow = input_ids.numel() - maximum
        if removable.numel() < overflow:
            raise OnlineSampleEncodeError(
                "VLM prefix exceeds 2560 after preserving complete reference images: "
                f"sample={sample_key}, actual={input_ids.numel()}, removable={removable.numel()}",
                reason="vlm_prefix_too_long",
            )
        first_reference = (
            int(torch.nonzero(reference_region_mask, as_tuple=False).flatten()[0])
            if reference_region_mask.any()
            else input_ids.numel()
        )
        system_candidates = removable[removable < first_reference]
        user_candidates = removable[removable >= first_reference].flip(0)
        removal_order = torch.cat([system_candidates.flip(0), user_candidates])
        remove = removal_order[:overflow]
        keep = torch.ones(input_ids.numel(), dtype=torch.bool)
        keep[remove] = False
        truncated_ids = input_ids[keep]
        truncated_attention = attention_mask[keep]
        truncated_reference = reference_region_mask[keep]
        self._validate_reference_regions(
            truncated_ids,
            num_reference_images=self._reference_count(reference_region_mask),
        )
        return truncated_ids, truncated_attention, truncated_reference

    def _validate_reference_regions(self, input_ids: Tensor, *, num_reference_images: int) -> Tensor:
        image_token = GEMMA3_CONFIG_FOR_LTX.image_token_index
        boi_token = GEMMA3_CONFIG_FOR_LTX.boi_token_index
        eoi_token = GEMMA3_CONFIG_FOR_LTX.eoi_token_index
        expected_per_image = int(getattr(self.processor, "image_seq_length", None) or EVIDENCE_TOKENS_PER_FRAME)
        if int((input_ids == image_token).sum().item()) != num_reference_images * expected_per_image:
            raise ValueError("Gemma reference image-token count mismatch")
        starts = torch.nonzero(input_ids == boi_token, as_tuple=False).flatten().tolist()
        ends = torch.nonzero(input_ids == eoi_token, as_tuple=False).flatten().tolist()
        if len(starts) != num_reference_images or len(ends) != num_reference_images:
            raise ValueError("Gemma reference boundary count mismatch")
        region_mask = torch.zeros(input_ids.numel(), dtype=torch.bool)
        for start, end in zip(starts, ends, strict=True):
            if end <= start:
                raise ValueError("Gemma reference boundaries are malformed")
            if int((input_ids[start + 1 : end] == image_token).sum().item()) != expected_per_image:
                raise ValueError("Gemma reference image region was truncated")
            region_mask[start : end + 1] = True
        return region_mask

    @staticmethod
    def _reference_count(reference_region_mask: Tensor) -> int:
        if not reference_region_mask.any():
            return 0
        padded = torch.cat(
            [torch.zeros(1, dtype=torch.bool), reference_region_mask, torch.zeros(1, dtype=torch.bool)]
        )
        return int((padded[1:] & ~padded[:-1]).sum().item())

    def _align_hidden_states(self, hidden_states: Any, module: nn.Module) -> Any:
        compute = module_compute_device_dtype(module)
        if compute is None:
            return hidden_states

        def align(value: Any) -> Any:
            if isinstance(value, Tensor) and value.is_floating_point():
                return value.to(device=compute[0], dtype=compute[1])
            return value

        if isinstance(hidden_states, tuple):
            return tuple(align(value) for value in hidden_states)
        if isinstance(hidden_states, list):
            return [align(value) for value in hidden_states]
        return align(hidden_states)

    def _frozen_encode_autocast(self) -> Any:
        if self.device.type == "cuda" and self.dtype in {torch.bfloat16, torch.float16}:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    def _get_language_model(self) -> nn.Module:
        language_model = getattr(self._unwrap_text_encoder().model.model, "language_model", None)
        if language_model is None:
            raise RuntimeError("Gemma text encoder does not expose language_model")
        return language_model

    def _unwrap_text_encoder(self) -> nn.Module:
        return getattr(self.text_encoder, "module", self.text_encoder)

    def _keep_frozen_modules_eval(self) -> None:
        text_encoder = self._unwrap_text_encoder()
        text_encoder.requires_grad_(False).eval()
        model = text_encoder.model.model
        for name in ("vision_tower", "multi_modal_projector", "language_model"):
            module = getattr(model, name, None)
            if module is not None:
                module.requires_grad_(False).eval()

    @staticmethod
    def _reference_images(references: list[Tensor]) -> list[Image.Image]:
        return [_to_pil(reference) for reference in references]

    def _sample_condition_mode(
        self,
        *,
        sample_key: str,
        optimizer_step: int,
        microstep: int,
        global_seed: int,
        strategy_config: Any,
    ) -> str:
        seed = augmentation_seed(
            global_seed=global_seed,
            optimizer_step=optimizer_step,
            microstep=microstep,
            sample_key=sample_key,
            reference_index=-2,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        draw = float(torch.rand((), generator=generator).item())
        probabilities = (
            ("full", strategy_config.condition_full_p),
            ("drop_text", strategy_config.condition_drop_text_p),
            ("drop_reference_all", strategy_config.condition_drop_reference_all_p),
            ("drop_all", strategy_config.condition_drop_all_p),
        )
        cumulative = 0.0
        for name, probability in probabilities:
            cumulative += float(probability)
            if draw < cumulative:
                return name
        return "drop_all"

    def _augment_training_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        optimizer_step: int,
        microstep: int,
        global_seed: int,
    ) -> dict[str, Any]:
        target_pixels = raw_batch["target_pixels"]
        if target_pixels.shape[0] != 1:
            raise ValueError("Online augmentation currently requires local microbatch size 1")
        sample_key = str(raw_batch["sample_key"][0])
        target_seed = augmentation_seed(
            global_seed=global_seed,
            optimizer_step=optimizer_step,
            microstep=microstep,
            sample_key=sample_key,
        )
        augmented_target = augment_target_frames(
            target_pixels[0],
            seed=target_seed,
            config=self.config.augmentation,
            target_height=self.config.height,
            target_width=self.config.width,
            chunk_frames=self.config.cpu_transform_chunk_frames,
        ).unsqueeze(0)
        reference_batches = raw_batch.get("reference_images_vlm", raw_batch.get("reference_pixels"))
        if not isinstance(reference_batches, list) or len(reference_batches) != 1:
            raise ValueError("Online augmentation requires one local reference batch")
        augmented_references = []
        for reference_index, reference in enumerate(reference_batches[0]):
            reference_seed = augmentation_seed(
                global_seed=global_seed,
                optimizer_step=optimizer_step,
                microstep=microstep,
                sample_key=sample_key,
                reference_index=reference_index,
            )
            augmented_references.append(
                augment_reference_image(
                    reference,
                    seed=reference_seed,
                    config=self.config.augmentation,
                    target_height=self.config.height,
                    target_width=self.config.width,
                )
            )
        anchor_indices = raw_batch["semantic_anchor_target_indices"][0].to(dtype=torch.long)
        updated = dict(raw_batch)
        updated["target_pixels"] = augmented_target
        updated["semantic_teacher_pixels"] = augmented_target[0].index_select(0, anchor_indices).unsqueeze(0)
        updated["reference_pixels_vae"] = [augmented_references]
        updated["reference_images_vlm"] = [augmented_references]
        updated["reference_pixels"] = [augmented_references]
        return updated

    def _resolve_reference_inputs(
        self,
        raw_batch: dict[str, Any],
    ) -> tuple[list[list[Tensor]], list[list[Tensor]]]:
        vae_batches = raw_batch.get("reference_pixels_vae", raw_batch.get("reference_pixels"))
        vlm_batches = raw_batch.get("reference_images_vlm", raw_batch.get("reference_pixels"))
        if not isinstance(vae_batches, list) or not isinstance(vlm_batches, list):
            raise ValueError("Online raw batch is missing reference VAE/VLM image lists")
        if len(vae_batches) != len(vlm_batches):
            raise ValueError("Reference VAE and VLM batch sizes differ")
        limit = self.config.max_ref_images
        resolved_vae: list[list[Tensor]] = []
        resolved_vlm: list[list[Tensor]] = []
        for vae_references, vlm_references in zip(vae_batches, vlm_batches, strict=True):
            if len(vae_references) != len(vlm_references):
                raise ValueError("Reference VAE/VLM order cannot align because counts differ")
            resolved_vae.append(list(vae_references if limit is None else vae_references[:limit]))
            resolved_vlm.append(list(vlm_references if limit is None else vlm_references[:limit]))
        return resolved_vae, resolved_vlm
