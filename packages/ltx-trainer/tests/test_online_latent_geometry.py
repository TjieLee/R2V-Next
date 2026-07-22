from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import ltx_trainer.online_data.online_batch_encoder as online_batch_encoder_module
from ltx_core.multicond.semantic_tokens import EVIDENCE_TOKENS_PER_FRAME
from ltx_trainer.online_data.online_batch_encoder import OnlineBatchEncoder
from ltx_trainer.training_strategies.semantic_flow import SemanticFlowConfig, SemanticFlowStrategy


class _FakeVAE(nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
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


def test_online_target_metadata_uses_encoded_latent_geometry() -> None:
    encoder = OnlineBatchEncoder.__new__(OnlineBatchEncoder)
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float16
    encoder.vae_encoder = _FakeVAE()
    encoder._frozen_encode_autocast = nullcontext  # type: ignore[method-assign]
    pixels = torch.empty(1, 121, 480, 832, 3, dtype=torch.float16)

    encoded = encoder._encode_target_latents(pixels, torch.tensor([24.0]))

    assert encoded["latents"].shape == (1, 128, 16, 15, 26)
    assert encoded["num_frames"].tolist() == [16]
    assert encoded["height"].tolist() == [15]
    assert encoded["width"].tolist() == [26]
    assert encoded["fps"].tolist() == [24.0]
    assert encoded["num_frames"].item() != pixels.shape[1]
    assert encoded["height"].item() != pixels.shape[2]
    assert encoded["width"].item() != pixels.shape[3]


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
    encoder._unwrap_text_encoder = lambda: SimpleNamespace(model=nn.Identity())  # type: ignore[method-assign]
    monkeypatch.setattr(
        online_batch_encoder_module,
        "extract_projected_visual_tokens",
        lambda _model, _pixels, *, image_counts, **_kwargs: SimpleNamespace(
            tokens=torch.zeros(1, int(image_counts.item()) * EVIDENCE_TOKENS_PER_FRAME, 4)
        ),
    )
    raw_batch = {
        "target_pixels": torch.zeros(1, 121, 2, 2, 3, dtype=torch.uint8),
        "semantic_anchor_target_indices": torch.tensor([canonical]),
    }

    evidence = encoder._encode_gt_evidence(raw_batch)
    expected = torch.tensor(canonical, dtype=torch.float32).unsqueeze(0) / 120
    assert torch.equal(evidence["normalized_timestamps"], expected)

    invalid = dict(raw_batch)
    invalid["semantic_anchor_target_indices"] = torch.tensor(
        [[0, 11, 22, 33, 44, 55, 66, 76, 87, 98, 109, 120]]
    )
    with pytest.raises(ValueError, match="differ from canonical uniform sampling"):
        encoder._encode_gt_evidence(invalid)
