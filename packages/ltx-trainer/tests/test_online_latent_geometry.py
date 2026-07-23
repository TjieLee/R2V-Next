from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import ltx_trainer.online_data.online_batch_encoder as online_batch_encoder_module
from ltx_core.multicond.semantic_tokens import EVIDENCE_TOKENS_PER_FRAME, SemanticQueryInitializer
from ltx_trainer.online_data.constants import IMAGE_TASK
from ltx_trainer.online_data.online_batch_encoder import (
    OnlineBatchEncoder,
    _materialize_frozen_tensor,
)
from ltx_trainer.training_strategies.semantic_flow import SemanticFlowConfig, SemanticFlowStrategy


class _FakeVAE(nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            pixels.shape[0],
            128,
            16,
            15,
            26,
            device=pixels.device,
            dtype=pixels.dtype,
        )


class _FakeImageProcessor:
    def __call__(self, *, images: list[object], return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros(len(images), 3, 2, 2)}


class _TinyFakeVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([2.0]))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        mean = pixels.mean(dim=(1, 2, 3, 4), keepdim=True)
        return self.scale * mean.expand(-1, 2, 1, 2, 2)


class _FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.projection = nn.Linear(4, 4)
        self.config = SimpleNamespace(sliding_window=4)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: dict[str, torch.Tensor],
        position_ids: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
        assert output_hidden_states
        assert return_dict
        assert not use_cache
        return SimpleNamespace(hidden_states=(self.projection(inputs_embeds),))


class _FakeFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_projection = nn.Linear(4, 3)
        self.audio_projection = nn.Linear(4, 2)

    def forward(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        padding_side: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del attention_mask
        assert padding_side == "right"
        hidden = hidden_states[-1]
        return self.video_projection(hidden), self.audio_projection(hidden)


def test_materialize_frozen_tensor_converts_inference_tensor_without_changing_value() -> None:
    with torch.inference_mode():
        inference_value = torch.ones(2, 3, dtype=torch.float32)

    assert torch.is_inference(inference_value)
    assert torch.is_inference(inference_value.detach())
    normal_value = _materialize_frozen_tensor(inference_value)

    assert not torch.is_inference(normal_value)
    assert not normal_value.requires_grad
    assert normal_value.device == inference_value.device
    assert normal_value.dtype == inference_value.dtype
    torch.testing.assert_close(normal_value, inference_value)


def test_online_target_metadata_uses_encoded_latent_geometry() -> None:
    encoder = OnlineBatchEncoder.__new__(OnlineBatchEncoder)
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float16
    encoder.vae_encoder = _FakeVAE()
    encoder._frozen_encode_autocast = nullcontext  # type: ignore[method-assign]
    pixels = torch.empty(1, 121, 480, 832, 3, dtype=torch.float16)

    encoded = encoder._encode_target_latents(pixels, torch.tensor([24.0]))

    assert encoded["latents"].shape == (1, 128, 16, 15, 26)
    assert not torch.is_inference(encoded["latents"])
    assert not encoded["latents"].requires_grad
    assert encoded["latents"].device == pixels.device
    assert encoded["latents"].dtype == pixels.dtype
    assert encoded["num_frames"].tolist() == [16]
    assert encoded["height"].tolist() == [15]
    assert encoded["width"].tolist() == [26]
    assert encoded["fps"].tolist() == [24.0]
    assert encoded["num_frames"].item() != pixels.shape[1]
    assert encoded["height"].item() != pixels.shape[2]
    assert encoded["width"].item() != pixels.shape[3]

    head = nn.Conv3d(encoded["latents"].shape[1], 2, kernel_size=1)
    head(encoded["latents"].float()).square().mean().backward()
    assert head.weight.grad is not None
    assert torch.isfinite(head.weight.grad).all()


def test_online_reference_latents_materialize_and_zero_padding_before_backward() -> None:
    encoder = OnlineBatchEncoder.__new__(OnlineBatchEncoder)
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float32
    encoder.config = SimpleNamespace(max_ref_images=4)
    encoder.vae_encoder = _TinyFakeVAE().requires_grad_(False)
    encoder._frozen_encode_autocast = nullcontext  # type: ignore[method-assign]
    white_reference = torch.full((2, 2, 3), 255, dtype=torch.uint8)
    black_reference = torch.zeros(2, 2, 3, dtype=torch.uint8)

    encoded = encoder._encode_reference_latents(
        [[white_reference, black_reference]],
        fallback_height=2,
        fallback_width=2,
    )
    latents = encoded["latents"]

    assert latents.shape == (1, 4, 2, 1, 2, 2)
    assert not torch.is_inference(latents)
    assert not latents.requires_grad
    assert latents.device == encoder.device
    assert latents.dtype == encoder.dtype
    assert encoded["ref_valid_mask"].tolist() == [[True, True, False, False]]
    torch.testing.assert_close(latents[:, 0], torch.full_like(latents[:, 0], 2.0))
    torch.testing.assert_close(latents[:, 1], torch.full_like(latents[:, 1], -2.0))
    assert torch.count_nonzero(latents[:, 2:]) == 0

    head = nn.Conv3d(2, 1, kernel_size=1)
    head(latents.flatten(0, 1)).square().mean().backward()
    assert head.weight.grad is not None
    assert torch.isfinite(head.weight.grad).all()
    assert encoder.vae_encoder.scale.grad is None


def test_online_prompt_features_materialize_before_trainable_connectors() -> None:
    encoder = OnlineBatchEncoder.__new__(OnlineBatchEncoder)
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float32
    encoder.last_dtype_diagnostics = {}
    encoder._frozen_encode_autocast = nullcontext  # type: ignore[method-assign]
    language_model = _FakeLanguageModel().requires_grad_(False)
    feature_extractor = _FakeFeatureExtractor()
    encoder.embeddings_processor = SimpleNamespace(feature_extractor=feature_extractor)
    encoder._get_language_model = lambda: language_model  # type: ignore[method-assign]
    encoder._process_multimodal_prefix = (  # type: ignore[method-assign]
        lambda **_kwargs: (
            {
                "input_ids": torch.tensor([[1, 2, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0]]),
            },
            torch.zeros(3, dtype=torch.bool),
            torch.zeros(3, dtype=torch.bool),
        )
    )

    conditions, _teacher_prefix = encoder._encode_prefix(
        caption="edit the image",
        reference_images=[],
        task=IMAGE_TASK,
        sample_key="prompt-boundary",
    )
    video = conditions["video_prompt_embeds"]
    audio = conditions["audio_prompt_embeds"]

    assert not torch.is_inference(video)
    assert not video.requires_grad
    assert video.shape == (1, 3, 3)
    assert video.device == encoder.device
    assert video.dtype == encoder.dtype
    assert not torch.is_inference(audio)
    assert not audio.requires_grad
    assert audio.shape == (1, 3, 2)
    assert audio.device == encoder.device
    assert audio.dtype == encoder.dtype
    video_head = nn.Linear(video.shape[-1], 1)
    audio_head = nn.Linear(audio.shape[-1], 1)
    loss = video_head(video).square().mean() + audio_head(audio).square().mean()
    loss.backward()
    for head in (video_head, audio_head):
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()
    assert all(parameter.grad is None for parameter in language_model.parameters())
    assert all(parameter.grad is None for parameter in feature_extractor.parameters())


@pytest.mark.parametrize(("frames", "expected_tokens"), [(1, 390), (16, 6240)])
def test_target_and_reference_position_counts_match_latent_tokens(frames: int, expected_tokens: int) -> None:
    strategy = SemanticFlowStrategy(SemanticFlowConfig())
    target_latents = torch.zeros(1, 128, frames, 15, 26)
    target_tokens = strategy._video_patchifier.patchify(target_latents)
    target_positions = strategy._get_video_positions(
        num_frames=frames,
        height=15,
        width=26,
        batch_size=1,
        fps=1.0 if frames == 1 else 24.0,
        device=target_latents.device,
    )
    assert target_tokens.shape[1] == expected_tokens
    assert target_positions.shape[2] == expected_tokens

    references = {
        "latents": torch.zeros(1, 4, 128, 1, 15, 26),
        "ref_valid_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    ref_tokens, ref_positions, _, _ = strategy._reference_sequence(
        references,
        target_latents=target_latents,
        target_positions=target_positions,
    )
    assert ref_tokens.shape[1] == ref_positions.shape[2]


def test_online_encoder_validates_canonical_anchor_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = [0, 11, 22, 33, 44, 55, 65, 76, 87, 98, 109, 120]
    encoder = OnlineBatchEncoder.__new__(OnlineBatchEncoder)
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float32
    encoder.image_processor = _FakeImageProcessor()
    encoder.last_dtype_diagnostics = {}
    encoder._keep_frozen_modules_eval = lambda: None  # type: ignore[method-assign]
    encoder._frozen_encode_autocast = nullcontext  # type: ignore[method-assign]
    frozen_visual_model = nn.Linear(1, 1).requires_grad_(False)
    encoder._unwrap_text_encoder = lambda: SimpleNamespace(model=frozen_visual_model)  # type: ignore[method-assign]
    monkeypatch.setattr(
        online_batch_encoder_module,
        "extract_projected_visual_tokens",
        lambda _model, _pixels, *, image_counts, **_kwargs: SimpleNamespace(
            tokens=torch.randn(1, int(image_counts.item()) * EVIDENCE_TOKENS_PER_FRAME, 4)
        ),
    )
    raw_batch = {
        "target_pixels": torch.zeros(1, 121, 2, 2, 3, dtype=torch.uint8),
        "semantic_anchor_target_indices": torch.tensor([canonical]),
    }

    evidence = encoder._encode_gt_evidence(raw_batch)
    evidence_tokens = evidence["evidence_tokens"]
    expected = torch.tensor(canonical, dtype=torch.float32).unsqueeze(0) / 120
    assert torch.equal(evidence["normalized_timestamps"], expected)
    assert not torch.is_inference(evidence_tokens)
    assert not evidence_tokens.requires_grad
    assert evidence_tokens.shape == (1, len(canonical), EVIDENCE_TOKENS_PER_FRAME, 4)
    assert evidence_tokens.device == encoder.device
    assert evidence_tokens.dtype == encoder.dtype

    query = SemanticQueryInitializer(gemma_dim=4)
    query(evidence_tokens, evidence["normalized_timestamps"]).square().mean().backward()
    assert query.content_norm.weight.grad is not None
    assert torch.isfinite(query.content_norm.weight.grad).all()
    assert all(parameter.grad is None for parameter in frozen_visual_model.parameters())

    invalid = dict(raw_batch)
    invalid["semantic_anchor_target_indices"] = torch.tensor(
        [[0, 11, 22, 33, 44, 55, 66, 76, 87, 98, 109, 120]]
    )
    with pytest.raises(ValueError, match="differ from canonical uniform sampling"):
        encoder._encode_gt_evidence(invalid)
