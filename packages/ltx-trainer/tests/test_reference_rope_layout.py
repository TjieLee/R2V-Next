from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ltx_core.multicond.semantic_tokens import (
    SEMANTIC_TOKENS_PER_FRAME,
    SemanticAlignmentHead,
    SemanticEncoder,
    SemanticQueryInitializer,
    SemanticReconstructionDecoder,
)
from ltx_core.types import VideoLatentShape
from ltx_trainer.online_data.anchor_geometry import normalized_anchor_timestamps
from ltx_trainer.online_data.constants import IMAGE_TASK, VIDEO_TASK
from ltx_trainer.online_data.online_batch_encoder import _build_messages
from ltx_trainer.online_inference.checkpoint_runtime import OnlineInferenceRuntime
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.semantic_flow import (
    ENTITY_GLOBAL,
    ENTITY_REF_0,
    SemanticFlowConfig,
    SemanticFlowStrategy,
    SemanticInferenceState,
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


def _prepare_training_and_inference_states(
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    semantic_frame_count: int,
    pixel_frame_count: int,
    fps: float,
    reference_valid_mask: list[bool] | None = None,
) -> tuple[ModelInputs, SemanticInferenceState]:
    feature_dim = 128
    strategy = SemanticFlowStrategy(SemanticFlowConfig(semantic_maximum_drop_rate=0.0))
    strategy._semantic_dim = feature_dim
    strategy._gemma_dim = feature_dim
    strategy._query_initializer = SemanticQueryInitializer(feature_dim)
    strategy._semantic_encoder = SemanticEncoder(feature_dim, feature_dim)
    strategy._reconstruction_decoder = SemanticReconstructionDecoder(feature_dim, feature_dim)
    strategy._semantic_alignment_head = SemanticAlignmentHead(feature_dim, feature_dim)
    semantic_clean = torch.ones(1, semantic_frame_count, SEMANTIC_TOKENS_PER_FRAME, feature_dim)
    strategy.build_semantic_teacher_outputs = lambda _inputs: {  # type: ignore[method-assign]
        "semantic_clean": semantic_clean,
        "reconstruction_prediction": torch.zeros(
            1, semantic_frame_count, SEMANTIC_TOKENS_PER_FRAME, 4, 4
        ),
        "reconstruction_target": torch.zeros(
            1, semantic_frame_count, SEMANTIC_TOKENS_PER_FRAME, 4, 4
        ),
        "alignment_prediction": torch.zeros(
            1, semantic_frame_count, SEMANTIC_TOKENS_PER_FRAME, feature_dim
        ),
        "alignment_target": torch.zeros(
            1, semantic_frame_count, SEMANTIC_TOKENS_PER_FRAME, feature_dim
        ),
    }
    reference_valid_mask = reference_valid_mask or [True, True]
    reference_capacity = len(reference_valid_mask)
    references = {
        "latents": torch.zeros(
            1,
            reference_capacity,
            feature_dim,
            1,
            latent_height,
            latent_width,
        ),
        "ref_valid_mask": torch.tensor([reference_valid_mask]),
    }
    conditions = {
        "video_prompt_embeds": torch.zeros(1, 2, feature_dim),
        "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    target_shape = VideoLatentShape(
        batch=1,
        channels=feature_dim,
        frames=latent_frames,
        height=latent_height,
        width=latent_width,
    )
    task = IMAGE_TASK if latent_frames == 1 else VIDEO_TASK
    training = strategy.prepare_training_inputs(
        {
            "task": [task],
            "semantic_teacher_inputs": {
                "prefix_attention_mask": torch.ones(1, 2, dtype=torch.bool),
                "normalized_timestamps": normalized_anchor_timestamps(
                    frame_count=pixel_frame_count,
                    anchor_count=semantic_frame_count,
                    device=torch.device("cpu"),
                ).unsqueeze(0),
            },
            "latents": {
                "latents": torch.zeros(target_shape.to_torch_shape()),
                "num_frames": torch.tensor([latent_frames]),
                "height": torch.tensor([latent_height]),
                "width": torch.tensor([latent_width]),
                "fps": torch.tensor([fps]),
            },
            "reference_latents": references,
            "conditions": conditions,
        },
        SimpleNamespace(sample_for=lambda tokens: torch.full((tokens.shape[0],), 0.5)),
    )
    inference = strategy.prepare_inference_state(
        conditions=conditions,
        reference_latents=references,
        target_shape=target_shape,
        semantic_frame_count=semantic_frame_count,
        pixel_frame_count=pixel_frame_count,
        fps=fps,
        seed=7,
    )
    assert strategy._geometry_logged_tasks == {task}
    return training, inference


def _assert_position_segments_match(training: ModelInputs, inference: SemanticInferenceState) -> None:
    assert training.sequence_offsets is not None
    assert training.video is not None
    offsets = training.sequence_offsets
    assert offsets == inference.sequence_offsets
    assert torch.equal(training.video.entity_ids, inference.modality.entity_ids)
    assert torch.equal(training.video.token_type_ids, inference.modality.token_type_ids)
    for start, end in (
        (0, offsets["reference_end"]),
        (offsets["reference_end"], offsets["semantic_end"]),
        (offsets["semantic_end"], offsets["target_end"]),
    ):
        assert torch.equal(
            training.video.positions[:, :, start:end],
            inference.modality.positions[:, :, start:end],
        )


def test_i2i_training_and_inference_positions_are_identical_at_one_fps() -> None:
    training, inference = _prepare_training_and_inference_states(
        latent_frames=1,
        latent_height=15,
        latent_width=26,
        semantic_frame_count=1,
        pixel_frame_count=1,
        fps=1.0,
    )
    _assert_position_segments_match(training, inference)
    assert training.video is not None
    assert torch.equal(training.video.entity_ids, inference.modality.entity_ids)
    assert torch.equal(training.video.token_type_ids, inference.modality.token_type_ids)
    assert torch.count_nonzero(training.video.entity_ids - ENTITY_GLOBAL) > 0


def test_r2v_training_and_inference_positions_are_identical_at_24_fps() -> None:
    canonical_timestamps = normalized_anchor_timestamps(
        frame_count=121,
        anchor_count=12,
        device=torch.device("cpu"),
    )
    assert not torch.equal(canonical_timestamps, torch.linspace(0.0, 1.0, 12))
    training, inference = _prepare_training_and_inference_states(
        latent_frames=16,
        latent_height=15,
        latent_width=26,
        semantic_frame_count=12,
        pixel_frame_count=121,
        fps=24.0,
    )
    _assert_position_segments_match(training, inference)


def test_training_geometry_log_reports_valid_references_and_capacity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="ltxv_trainer")
    _prepare_training_and_inference_states(
        latent_frames=1,
        latent_height=15,
        latent_width=26,
        semantic_frame_count=1,
        pixel_frame_count=1,
        fps=1.0,
        reference_valid_mask=[True, False, False, False],
    )
    geometry_logs = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("semantic-flow geometry:")
    ]
    assert len(geometry_logs) == 1
    assert "valid_references=[1]" in geometry_logs[0]
    assert "reference_capacity=4" in geometry_logs[0]
    assert "references=4" not in geometry_logs[0]


def test_online_runtime_records_real_i2i_and_r2v_geometry() -> None:
    strategy = SemanticFlowStrategy(SemanticFlowConfig(max_ref_images_per_sample=1))
    strategy._semantic_dim = 128

    def fake_denoise(**kwargs):
        state = kwargs["state"]
        semantic_length = state.sequence_offsets["semantic_end"] - state.sequence_offsets["reference_end"]
        return (
            torch.zeros(1, semantic_length, 128),
            torch.zeros(state.target_shape.to_torch_shape()),
        )

    strategy.denoise_joint = fake_denoise  # type: ignore[method-assign]
    runtime = object.__new__(OnlineInferenceRuntime)
    runtime.strategy = strategy
    runtime.transformer = torch.nn.Identity()
    runtime.connector_conditions = lambda conditions: conditions  # type: ignore[method-assign]
    encoded = {
        "conditions": {
            "video_prompt_embeds": torch.zeros(1, 2, 128),
            "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
        },
        "reference_latents": {
            "latents": torch.zeros(1, 1, 128, 1, 15, 26),
            "ref_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        },
    }

    for task, frames, fps, expected_shape, expected_tokens in (
        (IMAGE_TASK, 1, 1.0, [1, 128, 1, 15, 26], 390),
        (VIDEO_TASK, 121, 24.0, [1, 128, 16, 15, 26], 6240),
    ):
        runtime.generate_latents(
            {**encoded, "task": task},
            width=832,
            height=480,
            num_frames=frames,
            fps=fps,
            seed=7,
            num_inference_steps=1,
        )
        assert runtime.last_generation_geometry == {
            "fps": fps,
            "target_latent_shape": expected_shape,
            "target_token_count": expected_tokens,
            "target_position_count": expected_tokens,
            "reference_rope_mode": "appended_time_shifted_width",
        }


@pytest.mark.parametrize("fps", [0.0, -1.0, float("inf"), float("nan")])
def test_inference_rejects_non_positive_or_non_finite_fps(fps: float) -> None:
    strategy = SemanticFlowStrategy(SemanticFlowConfig())
    with pytest.raises(ValueError, match="Inference fps must be finite and positive"):
        strategy.prepare_inference_state(
            conditions={},
            reference_latents={},
            target_shape=VideoLatentShape(batch=1, channels=128, frames=1, height=15, width=26),
            semantic_frame_count=1,
            pixel_frame_count=1,
            fps=fps,
            seed=7,
        )


@pytest.mark.parametrize(
    ("semantic_frame_count", "pixel_frame_count", "message"),
    [
        (1, 0, "pixel_frame_count must be positive"),
        (2, 1, "semantic_frame_count cannot exceed pixel_frame_count"),
    ],
)
def test_inference_rejects_invalid_pixel_anchor_geometry(
    semantic_frame_count: int,
    pixel_frame_count: int,
    message: str,
) -> None:
    strategy = SemanticFlowStrategy(SemanticFlowConfig())
    with pytest.raises(ValueError, match=message):
        strategy.prepare_inference_state(
            conditions={},
            reference_latents={},
            target_shape=VideoLatentShape(batch=1, channels=128, frames=1, height=15, width=26),
            semantic_frame_count=semantic_frame_count,
            pixel_frame_count=pixel_frame_count,
            fps=1.0,
            seed=7,
        )


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
