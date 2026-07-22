from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from torch import nn

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
