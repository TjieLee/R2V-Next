from __future__ import annotations

from types import SimpleNamespace

import torch

from ltx_core.multicond.semantic_tokens import (
    SEMANTIC_TOKENS_PER_FRAME,
    SemanticEncoder,
    SemanticQueryInitializer,
    SemanticReconstructionDecoder,
)
from ltx_core.types import VideoLatentShape
from ltx_trainer.online_data.constants import IMAGE_TASK
from ltx_trainer.online_data.online_batch_encoder import _build_messages
from ltx_trainer.training_strategies.semantic_flow import (
    ENTITY_GLOBAL,
    ENTITY_REF_0,
    SemanticFlowConfig,
    SemanticFlowStrategy,
)


def _target_positions() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[0.0, 1.0], [1.0, 9.0], [9.0, 17.0]],
                [[0.0, 32.0], [32.0, 64.0], [64.0, 96.0]],
                [[0.0, 64.0], [64.0, 128.0], [128.0, 192.0]],
            ]
        ]
    )


def _references(reference_count: int = 4) -> dict[str, torch.Tensor]:
    latents = torch.stack(
        [torch.full((4, 1, 2, 3), float(index + 1)) for index in range(reference_count)],
        dim=0,
    ).unsqueeze(0)
    return {
        "latents": latents,
        "ref_valid_mask": torch.ones(1, reference_count, dtype=torch.bool),
    }


def test_appended_reference_rope_uses_fixed_identity_order_temporal_slots_and_adjacent_width() -> None:
    target_positions = _target_positions()
    target_latents = torch.zeros(1, 4, 3, 4, 6)
    references = _references()
    strategy = SemanticFlowStrategy(SemanticFlowConfig())

    tokens, positions, valid, entities = strategy._reference_sequence(
        references,
        target_latents=target_latents,
        target_positions=target_positions,
    )
    reference_count = references["latents"].shape[1]
    tokens_per_reference = tokens.shape[1] // reference_count
    token_blocks = tokens.reshape(1, reference_count, tokens_per_reference, -1)
    position_blocks = positions.reshape(1, 3, reference_count, tokens_per_reference, 2)
    entity_blocks = entities.reshape(1, reference_count, tokens_per_reference)

    for index in range(reference_count):
        assert torch.equal(token_blocks[0, index], torch.full_like(token_blocks[0, index], index + 1.0))
        assert torch.equal(
            entity_blocks[0, index],
            torch.full_like(entity_blocks[0, index], ENTITY_REF_0 + index),
        )
        expected_start = 17.0 + index * 8.0
        assert torch.equal(position_blocks[0, 0, index, :, 0], torch.full((tokens_per_reference,), expected_start))
        assert torch.equal(
            position_blocks[0, 0, index, :, 1],
            torch.full((tokens_per_reference,), expected_start + 8.0),
        )

    assert valid.all()
    assert torch.all(position_blocks[0, 0, :-1, :, 1].amax(dim=1) <= position_blocks[0, 0, 1:, :, 0].amin(dim=1))
    target_w_max = target_positions[:, 2, :, 1].amax()
    assert torch.allclose(position_blocks[:, 2, :, :, 0].amin(), target_w_max, atol=1.0e-6, rtol=0.0)
    assert position_blocks[:, 2, :, :, 1].amax() > target_w_max

    native_strategy = SemanticFlowStrategy(SemanticFlowConfig(reference_rope_mode="native_overlap"))
    _, native_positions, _, _ = native_strategy._reference_sequence(
        references,
        target_latents=target_latents,
        target_positions=target_positions,
    )
    native_blocks = native_positions.reshape(1, 3, reference_count, tokens_per_reference, 2)
    assert torch.equal(position_blocks[:, 1], native_blocks[:, 1])
    assert not torch.equal(position_blocks[:, 0], native_blocks[:, 0])
    assert not torch.equal(position_blocks[:, 2], native_blocks[:, 2])


def test_semantic_rope_remains_target_interpolated_eight_by_eight() -> None:
    target_positions = _target_positions()
    timestamps = torch.tensor([[0.0, 0.5, 1.0]])
    positions, _ = SemanticFlowStrategy._semantic_positions(target_positions, timestamps)

    assert positions.shape == (1, 3, 3 * SEMANTIC_TOKENS_PER_FRAME, 2)
    for axis in range(3):
        assert positions[:, axis, :, 0].amin() >= target_positions[:, axis, :, 0].amin()
        assert positions[:, axis, :, 1].amax() <= target_positions[:, axis, :, 1].amax()
    assert torch.equal(positions[:, 1, :, 0].amin(), target_positions[:, 1, :, 0].amin())
    assert torch.equal(positions[:, 1, :, 1].amax(), target_positions[:, 1, :, 1].amax())
    assert torch.equal(positions[:, 2, :, 0].amin(), target_positions[:, 2, :, 0].amin())
    assert torch.equal(positions[:, 2, :, 1].amax(), target_positions[:, 2, :, 1].amax())


def test_training_and_inference_reference_positions_are_identical() -> None:
    strategy = SemanticFlowStrategy(SemanticFlowConfig(semantic_minimum_tokens_per_frame=64))
    strategy._semantic_dim = 4
    strategy._gemma_dim = 4
    strategy._query_initializer = SemanticQueryInitializer(4)
    strategy._semantic_encoder = SemanticEncoder(4, 4)
    strategy._reconstruction_decoder = SemanticReconstructionDecoder(4, 4)
    semantic_clean = torch.ones(1, 1, SEMANTIC_TOKENS_PER_FRAME, 4)
    strategy.build_semantic_teacher_outputs = lambda _inputs: {  # type: ignore[method-assign]
        "semantic_clean": semantic_clean,
        "reconstruction_prediction": torch.zeros(1, 1, SEMANTIC_TOKENS_PER_FRAME, 4, 4),
        "reconstruction_target": torch.zeros(1, 1, SEMANTIC_TOKENS_PER_FRAME, 4, 4),
    }
    references = {
        "latents": torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 1, 2, 2),
        "ref_valid_mask": torch.tensor([[True, True]]),
    }
    conditions = {
        "video_prompt_embeds": torch.zeros(1, 2, 4),
        "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    training = strategy.prepare_training_inputs(
        {
            "semantic_teacher_inputs": {
                "prefix_attention_mask": torch.ones(1, 2, dtype=torch.bool),
                "normalized_timestamps": torch.tensor([[0.0]]),
            },
            "latents": {
                "latents": torch.zeros(1, 4, 1, 2, 2),
                "num_frames": torch.tensor([1]),
                "height": torch.tensor([2]),
                "width": torch.tensor([2]),
                "fps": torch.tensor([24.0]),
            },
            "reference_latents": references,
            "conditions": conditions,
        },
        SimpleNamespace(sample_for=lambda tokens: torch.full((tokens.shape[0],), 0.5)),
    )
    inference = strategy.prepare_inference_state(
        conditions=conditions,
        reference_latents=references,
        target_shape=VideoLatentShape(batch=1, channels=4, frames=1, height=2, width=2),
        semantic_frame_count=1,
        seed=7,
    )
    assert training.sequence_offsets is not None
    reference_end = training.sequence_offsets["reference_end"]
    assert torch.equal(
        training.video.positions[:, :, :reference_end],
        inference.modality.positions[:, :, :reference_end],
    )
    assert torch.equal(
        training.video.entity_ids[:, :reference_end],
        inference.modality.entity_ids[:, :reference_end],
    )
    assert torch.count_nonzero(training.video.entity_ids[:, reference_end:] - ENTITY_GLOBAL) == 0


def test_vlm_reference_placeholders_keep_input_order_and_one_based_numbering() -> None:
    messages = _build_messages("system", "Use image 1, then image 2.", 4, task=IMAGE_TASK)
    content = messages[1]["content"]
    labels = [
        item["text"]
        for item in content
        if item["type"] == "text" and item["text"].startswith("Reference image ")
    ]
    assert labels == [f"Reference image {index}:" for index in range(1, 5)]
    assert [item["type"] for item in content].count("image") == 4
