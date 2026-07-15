"""Per-rank GPU encoding bridge from raw online samples to existing strategy keys."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from transformers import AutoImageProcessor, AutoTokenizer, Gemma3Processor

from ltx_core.multicond.visual_tokens import extract_projected_visual_tokens, scatter_visual_tokens_into_embeddings
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_core.utils import find_matching_file
from ltx_trainer.config import OnlineEncodingConfig
from ltx_trainer.online_data.constants import (
    IMAGE_TASK,
    MAX_VLM_FRAMES,
    VIDEO_TASK,
    VISUAL_TOKEN_CAPACITY,
    VLM_TARGET_INDICES,
)
from ltx_trainer.online_data.visual_token_packing import (
    build_visual_metadata,
    pack_visual_tokens,
    planner_output_mask_from_visual_mask,
)


def _to_pil(image: Tensor) -> Image.Image:
    if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected uint8 [H,W,3] image, got {tuple(image.shape)} {image.dtype}")
    return Image.fromarray(image.cpu().numpy(), mode="RGB")


def _build_messages(system_prompt: str, user_prompt: str, num_images: int) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [{"type": "text", "text": f"User Raw Input Prompt: {user_prompt}."}]
    if num_images:
        content.append({"type": "text", "text": "Reference images:"})
        for index in range(num_images):
            content.extend(
                [
                    {"type": "text", "text": f"Reference image {index + 1}:"},
                    {"type": "image"},
                ]
            )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


class OnlineBatchEncoder:
    """Encode one homogeneous local microbatch without writing tensor caches."""

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

        tokenizer_root = str(find_matching_file(text_encoder_path, "tokenizer.model").parent)
        processor_root = str(find_matching_file(text_encoder_path, "preprocessor_config.json").parent)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_root,
            local_files_only=True,
            model_max_length=config.planner_max_length,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            processor_root,
            local_files_only=True,
            use_fast=False,
        )
        self.processor = Gemma3Processor(image_processor=self.image_processor, tokenizer=self.tokenizer)
        prompt_path = (
            Path(__file__).resolve().parents[4]
            / "ltx-core"
            / "src"
            / "ltx_core"
            / "text_encoders"
            / "gemma"
            / "encoders"
            / "prompts"
            / "gemma_multiref_video_planner_system_prompt.txt"
        )
        self.system_prompt = prompt_path.read_text(encoding="utf-8")
        self.vae_encoder.requires_grad_(False).eval()
        self._keep_frozen_visual_modules_eval()

    def encode_for_strategy(
        self,
        raw_batch: dict[str, Any],
        *,
        strategy: Any,
        training_phase: str,
    ) -> dict[str, Any]:
        del training_phase
        target_pixels = raw_batch["target_pixels"]
        if target_pixels.shape[0] != 1:
            raise ValueError("Online multi-task encoding currently requires local microbatch size 1")
        task = str(raw_batch["task"][0])
        if task not in {IMAGE_TASK, VIDEO_TASK}:
            raise ValueError(f"Unsupported online task {task!r}")
        expected_frames = 1 if task == IMAGE_TASK else 121
        if target_pixels.shape[1] != expected_frames:
            raise ValueError(f"Task {task} expected {expected_frames} target frames, got {target_pixels.shape[1]}")

        metrics = {
            "data_decode_ms": float(raw_batch["data_decode_ms"].float().mean().item()),
            "vae_encode_ms": 0.0,
            "siglip_encode_ms": 0.0,
            "vlm_planner_ms": 0.0,
        }
        self._move_frozen_encoders_for_encode()
        vae_started = time.perf_counter()
        latents = self._encode_target_latents(target_pixels, raw_batch["target_fps"])
        multi_ref_latents = self._encode_reference_latents(raw_batch["reference_pixels"])
        metrics["vae_encode_ms"] = (time.perf_counter() - vae_started) * 1000.0

        siglip_started = time.perf_counter()
        gt_visual_tokens = self._encode_gt_visual_tokens(raw_batch, task=task)
        metrics["siglip_encode_ms"] = (time.perf_counter() - siglip_started) * 1000.0

        condition_started = time.perf_counter()
        caption = str(raw_batch["caption"][0])
        references = self._reference_images(raw_batch["reference_pixels"][0])
        text_conditions = self._encode_condition(caption=caption, reference_images=[])
        vlm_conditions = self._encode_condition(caption=caption, reference_images=references)
        metrics["vlm_planner_ms"] = (time.perf_counter() - condition_started) * 1000.0

        batch: dict[str, Any] = {
            "sample_key": raw_batch["sample_key"],
            "sample_plan_sha256": raw_batch["sample_plan_sha256"],
            "task": raw_batch["task"],
            "target_modality": raw_batch["target_modality"],
            "manifest_index": raw_batch["manifest_index"],
            "target_source_frame_indices": raw_batch["target_source_frame_indices"],
            "vlm_target_frame_indices": raw_batch["vlm_target_frame_indices"],
            "vlm_source_frame_indices": raw_batch["vlm_source_frame_indices"],
            "latents": latents,
            "multi_ref_latents": multi_ref_latents,
            "multi_reference_latents": multi_ref_latents,
            "conditions": vlm_conditions,
            "vlm_conditions": vlm_conditions,
            "text_conditions": text_conditions,
            "cfg_text_conditions": text_conditions,
            "gt_visual_tokens": gt_visual_tokens,
            "gt_siglip_tokens": gt_visual_tokens,
            "_online_metrics": metrics,
        }
        if getattr(strategy.config, "name", "") == "multi_reference_planner_stage2":
            planner_started = time.perf_counter()
            batch["planner_vlm_inputs"] = self._build_planner_vlm_inputs(
                caption=caption,
                reference_images=references,
                planner_output_mask=planner_output_mask_from_visual_mask(
                    gt_visual_tokens["visual_token_mask"]
                ),
            )
            metrics["vlm_planner_ms"] += (time.perf_counter() - planner_started) * 1000.0
        self._offload_frozen_encoders_after_encode()
        return batch

    def _encode_target_latents(self, target_pixels: Tensor, fps: Tensor) -> dict[str, Tensor]:
        video = target_pixels.to(device=self.device, dtype=self.dtype, non_blocking=True)
        video = video.permute(0, 4, 1, 2, 3).div_(127.5).sub_(1.0)
        with torch.inference_mode():
            encoded = self.vae_encoder(video)
        return {
            "latents": encoded,
            "num_frames": torch.full((encoded.shape[0],), encoded.shape[2], device=self.device, dtype=torch.long),
            "height": torch.full((encoded.shape[0],), encoded.shape[3], device=self.device, dtype=torch.long),
            "width": torch.full((encoded.shape[0],), encoded.shape[4], device=self.device, dtype=torch.long),
            "fps": fps.to(device=self.device, dtype=torch.float32),
        }

    def _encode_reference_latents(self, reference_batches: list[list[Tensor]]) -> dict[str, Tensor]:
        max_refs = max(len(references) for references in reference_batches)
        if self.config.max_ref_images is not None:
            max_refs = min(max_refs, self.config.max_ref_images)
        if max_refs <= 0:
            raise ValueError("Every online sample must provide at least one reference image")
        batch_size = len(reference_batches)
        ref_valid_mask = torch.zeros(batch_size, max_refs, dtype=torch.bool, device=self.device)
        flat: list[Tensor] = []
        template = reference_batches[0][0]
        for batch_index, references in enumerate(reference_batches):
            references = references[:max_refs]
            ref_valid_mask[batch_index, : len(references)] = True
            flat.extend(references)
            flat.extend(torch.zeros_like(template) for _ in range(max_refs - len(references)))
        pixels = torch.stack(flat, dim=0).to(device=self.device, dtype=self.dtype, non_blocking=True)
        pixels = pixels.permute(0, 3, 1, 2).unsqueeze(2).div_(127.5).sub_(1.0)
        with torch.inference_mode():
            encoded = self.vae_encoder(pixels)
        encoded = encoded.reshape(batch_size, max_refs, *encoded.shape[1:])
        encoded = encoded * ref_valid_mask[:, :, None, None, None, None].to(dtype=encoded.dtype)
        return {
            "latents": encoded,
            "ref_valid_mask": ref_valid_mask,
            "fps": torch.ones(batch_size, device=self.device, dtype=torch.float32),
        }

    def _encode_gt_visual_tokens(self, raw_batch: dict[str, Any], *, task: str) -> dict[str, Any]:
        target_pixels = raw_batch["target_pixels"]
        if task == IMAGE_TASK:
            selected = target_pixels[0, :1]
            valid_frames = 1
            sampled_indices = torch.zeros(1, MAX_VLM_FRAMES, dtype=torch.long, device=self.device)
        else:
            selected = target_pixels[0, list(VLM_TARGET_INDICES)]
            valid_frames = MAX_VLM_FRAMES
            sampled_indices = torch.tensor([VLM_TARGET_INDICES], dtype=torch.long, device=self.device)
        pil_frames = [_to_pil(frame) for frame in selected]
        processed = self.image_processor(images=pil_frames, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=self.device, dtype=self.dtype)
        self._keep_frozen_visual_modules_eval()
        with torch.inference_mode():
            visual = extract_projected_visual_tokens(
                self._unwrap_text_encoder().model,
                pixel_values,
                image_counts=torch.tensor([valid_frames], device=self.device, dtype=torch.long),
            )
        packed, visual_mask = pack_visual_tokens(visual.tokens, valid_frames=valid_frames)
        if packed.shape[-1] != self.config.raw_visual_dim:
            raise ValueError(
                f"Projected SigLIP dim {packed.shape[-1]} != configured raw_visual_dim={self.config.raw_visual_dim}"
            )
        metadata = build_visual_metadata(
            batch_size=1,
            valid_frames=valid_frames,
            target_num_frames=1 if task == IMAGE_TASK else 121,
            target_fps=1.0 if task == IMAGE_TASK else 24.0,
            sampled_frame_indices=sampled_indices,
            device=self.device,
        )
        return {
            "visual_tokens": packed,
            "visual_token_mask": visual_mask,
            "source_fps": metadata["target_fps"],
            **metadata,
        }

    def _encode_condition(self, *, caption: str, reference_images: list[Image.Image]) -> dict[str, Tensor]:
        text = self.tokenizer.apply_chat_template(
            _build_messages(self.system_prompt, caption, len(reference_images)),
            tokenize=False,
            add_generation_prompt=True,
        )
        processed = self.processor(
            text=text,
            images=reference_images or None,
            return_tensors="pt",
            padding=False,
            max_length=self.config.planner_max_length,
            truncation=True,
        )
        input_ids = processed["input_ids"].to(device=self.device, dtype=torch.long)
        attention_mask = processed["attention_mask"].to(device=self.device, dtype=torch.long)
        pixel_values = processed.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        language_model = self._get_language_model()
        core_language_model = getattr(language_model, "module", language_model)
        embed_tokens = core_language_model.get_input_embeddings()
        adapter_context = (
            core_language_model.disable_adapter()
            if hasattr(core_language_model, "disable_adapter")
            else nullcontext()
        )
        feature_extractor = getattr(self.embeddings_processor, "feature_extractor", None)
        if feature_extractor is None:
            raise RuntimeError("Online condition encoding requires embeddings_processor.feature_extractor")
        feature_extractor.requires_grad_(False).eval()
        language_model_was_training = language_model.training
        language_model.eval()
        try:
            with torch.inference_mode(), adapter_context:
                inputs_embeds = embed_tokens(input_ids)
                if pixel_values is not None:
                    visual = extract_projected_visual_tokens(
                        self._unwrap_text_encoder().model,
                        pixel_values,
                        image_counts=torch.tensor([len(reference_images)], device=self.device, dtype=torch.long),
                    )
                    inputs_embeds = scatter_visual_tokens_into_embeddings(
                        inputs_embeds=inputs_embeds,
                        input_ids=input_ids,
                        visual_tokens=visual.tokens,
                        visual_mask=visual.mask,
                        image_token_index=GEMMA3_CONFIG_FOR_LTX.image_token_index,
                    )
                outputs = language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                video_features, audio_features = feature_extractor(
                    outputs.hidden_states,
                    attention_mask,
                    "right",
                )
        finally:
            language_model.train(language_model_was_training)
        result = {
            "video_prompt_embeds": video_features,
            "prompt_attention_mask": attention_mask,
        }
        if audio_features is not None:
            result["audio_prompt_embeds"] = audio_features
        return result

    def _build_planner_vlm_inputs(
        self,
        *,
        caption: str,
        reference_images: list[Image.Image],
        planner_output_mask: Tensor,
    ) -> dict[str, Tensor]:
        source_max_length = self.config.planner_max_length - VISUAL_TOKEN_CAPACITY - 2
        if source_max_length <= 0:
            raise ValueError("planner_max_length must reserve 2048 placeholders and two boundary tokens")
        text = self.tokenizer.apply_chat_template(
            _build_messages(self.system_prompt, caption, len(reference_images)),
            tokenize=False,
            add_generation_prompt=True,
        )
        processed = self.processor(
            text=text,
            images=reference_images or None,
            return_tensors="pt",
            padding=False,
            max_length=source_max_length,
            truncation=True,
        )
        input_ids = processed["input_ids"][0].to(dtype=torch.long)
        attention_mask = processed["attention_mask"][0].to(dtype=torch.long)
        boundary = torch.tensor(
            [GEMMA3_CONFIG_FOR_LTX.boi_token_index, GEMMA3_CONFIG_FOR_LTX.eoi_token_index],
            dtype=torch.long,
        )
        placeholders = torch.full(
            (VISUAL_TOKEN_CAPACITY,), GEMMA3_CONFIG_FOR_LTX.image_token_index, dtype=torch.long
        )
        source_len = input_ids.shape[0]
        input_ids = torch.cat([input_ids, boundary[:1], placeholders, boundary[1:]], dim=0)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(VISUAL_TOKEN_CAPACITY + 2, dtype=torch.long)], dim=0
        )
        placeholder_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        placeholder_mask[source_len + 1 : source_len + 1 + VISUAL_TOKEN_CAPACITY] = True
        boundary_mask = torch.zeros_like(placeholder_mask)
        boundary_mask[source_len] = True
        boundary_mask[source_len + VISUAL_TOKEN_CAPACITY + 1] = True
        pad_len = self.config.planner_max_length - input_ids.shape[0]
        if pad_len < 0:
            raise ValueError("Planner source input exceeded its reserved max length")
        if pad_len:
            pad_id = int(self.tokenizer.pad_token_id or 0)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            placeholder_mask = torch.cat([placeholder_mask, torch.zeros(pad_len, dtype=torch.bool)])
            boundary_mask = torch.cat([boundary_mask, torch.zeros(pad_len, dtype=torch.bool)])
        planner_region_mask = placeholder_mask | boundary_mask
        active = attention_mask.bool()
        ref_visual_mask = (
            (input_ids == GEMMA3_CONFIG_FOR_LTX.image_token_index) & ~planner_region_mask & active
        )
        ref_region_mask = (
            (
                (input_ids == GEMMA3_CONFIG_FOR_LTX.image_token_index)
                | (input_ids == GEMMA3_CONFIG_FOR_LTX.boi_token_index)
                | (input_ids == GEMMA3_CONFIG_FOR_LTX.eoi_token_index)
            )
            & ~planner_region_mask
            & active
        )
        text_token_mask = active & ~ref_region_mask & ~planner_region_mask
        ntp_labels = input_ids.clone()
        ntp_labels[~text_token_mask] = -100
        result = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "planner_placeholder_mask": placeholder_mask.unsqueeze(0),
            "planner_boundary_mask": boundary_mask.unsqueeze(0),
            "planner_region_mask": planner_region_mask.unsqueeze(0),
            "planner_output_mask": planner_output_mask.to(dtype=torch.bool),
            "ref_visual_token_mask": ref_visual_mask.unsqueeze(0),
            "ref_image_region_mask": ref_region_mask.unsqueeze(0),
            "gt_image_token_mask": ref_region_mask.unsqueeze(0),
            "text_token_mask": text_token_mask.unsqueeze(0),
            "ntp_labels": ntp_labels.unsqueeze(0),
            "ntp_label_mask": (ntp_labels != -100).unsqueeze(0),
            "labels": ntp_labels.unsqueeze(0).clone(),
            "num_ref_images": torch.tensor([len(reference_images)], dtype=torch.long),
            "planner_token_count": torch.tensor([VISUAL_TOKEN_CAPACITY], dtype=torch.long),
            "source_max_length": torch.tensor([source_max_length], dtype=torch.long),
        }
        pixel_values = processed.get("pixel_values")
        if isinstance(pixel_values, Tensor):
            result["pixel_values"] = pixel_values.unsqueeze(0) if pixel_values.ndim == 4 else pixel_values
        return result

    def _get_language_model(self) -> nn.Module:
        text_encoder = self._unwrap_text_encoder()
        language_model = getattr(text_encoder.model.model, "language_model", None)
        if language_model is None:
            raise RuntimeError("Gemma text encoder does not expose language_model")
        return language_model

    def _unwrap_text_encoder(self) -> nn.Module:
        return getattr(self.text_encoder, "module", self.text_encoder)

    def _reference_images(self, references: list[Tensor]) -> list[Image.Image]:
        if self.config.max_ref_images is not None:
            references = references[: self.config.max_ref_images]
        return [_to_pil(reference) for reference in references]

    def _keep_frozen_visual_modules_eval(self) -> None:
        model = self._unwrap_text_encoder().model.model
        for name in ("vision_tower", "multi_modal_projector"):
            module = getattr(model, name, None)
            if module is not None:
                module.requires_grad_(False).eval()

    def _move_frozen_encoders_for_encode(self) -> None:
        if self.config.encoder_device_policy != "resident_cuda":
            self.vae_encoder.to(device=self.device, dtype=self.dtype)

    def _offload_frozen_encoders_after_encode(self) -> None:
        if self.config.encoder_device_policy in {"sequential_cuda", "cpu_offload"}:
            self.vae_encoder.to(device="cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
