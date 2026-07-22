from __future__ import annotations

import pytest
import torch

from ltx_core.multicond.gemma3_attention import build_gemma3_attention_masks
from ltx_core.multicond.semantic_tokens import (
    EVIDENCE_TOKENS_PER_FRAME,
    SEMANTIC_TOKENS_PER_FRAME,
    build_multimodal_prefix_attention_mask,
    build_semantic_teacher_attention_mask,
)


def _visible(mask: torch.Tensor) -> torch.Tensor:
    return mask[:, 0] == 0


def test_gemma3_full_and_sliding_masks_preserve_image_blocks_and_padding() -> None:
    valid = torch.tensor([[1] * 14 + [0, 0]], dtype=torch.bool)
    image = torch.zeros_like(valid)
    # Segment layout: token 4=BOI, 5..7=image placeholders, 8=EOI.
    image[:, 5:8] = True
    custom = build_multimodal_prefix_attention_mask(valid, image_token_mask=image)
    masks = build_gemma3_attention_masks(
        valid_token_mask=valid,
        image_token_mask=image,
        custom_visibility=custom,
        sliding_window=4,
        dtype=torch.float32,
    )
    full = _visible(masks.full_attention)
    sliding = _visible(masks.sliding_attention)

    assert not torch.equal(full, sliding)
    assert full[0, 10, 0]
    assert not sliding[0, 10, 0]

    assert full[0, 5, 7]
    assert full[0, 7, 5]
    assert sliding[0, 5, 7]
    assert sliding[0, 7, 5]

    assert not full[0, 4, 8]
    assert not full[0, 5, 8]
    assert not sliding[0, 4, 8]
    assert not sliding[0, 5, 8]

    assert not full[0, 14].any()
    assert not full[0, :, 14].any()
    assert not sliding[0, 14].any()
    assert not sliding[0, :, 14].any()


def test_teacher_custom_visibility_keeps_queries_local_and_prefix_blind_to_suffix() -> None:
    prefix = torch.ones(1, 10, dtype=torch.bool)
    image = torch.zeros_like(prefix)
    image[:, 2:5] = True
    custom = build_semantic_teacher_attention_mask(prefix, frame_count=1, image_token_mask=image)
    prefix_length = prefix.shape[1]
    evidence_start = prefix_length
    query_start = prefix_length + EVIDENCE_TOKENS_PER_FRAME
    first_query = query_start

    assert not custom[0, :prefix_length, prefix_length:].any()
    assert custom[0, first_query, :prefix_length].all()
    visible_evidence = torch.nonzero(
        custom[0, first_query, evidence_start : evidence_start + EVIDENCE_TOKENS_PER_FRAME],
        as_tuple=False,
    ).flatten()
    assert visible_evidence.tolist() == [0, 1, 16, 17]
    assert custom[0, first_query, first_query]
    assert not custom[0, first_query, first_query + 1 :].any()

    masks = build_gemma3_attention_masks(
        valid_token_mask=torch.ones(1, custom.shape[1], dtype=torch.bool),
        image_token_mask=torch.cat(
            [
                image,
                torch.zeros(1, EVIDENCE_TOKENS_PER_FRAME + SEMANTIC_TOKENS_PER_FRAME, dtype=torch.bool),
            ],
            dim=1,
        ),
        custom_visibility=custom,
        sliding_window=1024,
        dtype=torch.float32,
    )
    assert torch.equal(
        _visible(masks.full_attention)[:, :prefix_length, :prefix_length],
        _visible(masks.sliding_attention)[:, :prefix_length, :prefix_length],
    )


def test_prefix_and_teacher_use_identical_full_sliding_prefix_masks() -> None:
    prefix = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    image = torch.zeros_like(prefix)
    image[:, 2:5] = True
    prefix_custom = build_multimodal_prefix_attention_mask(prefix, image_token_mask=image)
    teacher_custom = build_semantic_teacher_attention_mask(prefix, frame_count=1, image_token_mask=image)

    prefix_masks = build_gemma3_attention_masks(
        valid_token_mask=prefix,
        image_token_mask=image,
        custom_visibility=prefix_custom,
        sliding_window=4,
        dtype=torch.float32,
    )
    teacher_valid = torch.ones(1, teacher_custom.shape[1], dtype=torch.bool)
    teacher_image = torch.cat(
        [image, torch.zeros(1, EVIDENCE_TOKENS_PER_FRAME + SEMANTIC_TOKENS_PER_FRAME, dtype=torch.bool)],
        dim=1,
    )
    teacher_masks = build_gemma3_attention_masks(
        valid_token_mask=teacher_valid,
        image_token_mask=teacher_image,
        custom_visibility=teacher_custom,
        sliding_window=4,
        dtype=torch.float32,
    )
    prefix_length = prefix.shape[1]
    torch.testing.assert_close(
        prefix_masks.full_attention,
        teacher_masks.full_attention[:, :, :prefix_length, :prefix_length],
    )
    torch.testing.assert_close(
        prefix_masks.sliding_attention,
        teacher_masks.sliding_attention[:, :, :prefix_length, :prefix_length],
    )


def test_small_transformers_gemma3_forward_accepts_full_sliding_mask_mapping() -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(3)
    try:
        config = transformers.Gemma3TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            sliding_window=4,
            layer_types=["full_attention", "sliding_attention"],
            use_cache=False,
        )
        model = transformers.Gemma3TextModel(config).eval()
    except Exception as exc:  # pragma: no cover - depends on installed transformers API
        pytest.skip(f"Installed transformers Gemma3 API is not compatible with this tiny config: {exc}")

    inputs = torch.randn(1, 8, config.hidden_size)
    valid = torch.ones(1, 8, dtype=torch.bool)
    image = torch.zeros_like(valid)
    image[:, 2:4] = True
    custom = build_multimodal_prefix_attention_mask(valid, image_token_mask=image)
    masks = build_gemma3_attention_masks(
        valid_token_mask=valid,
        image_token_mask=image,
        custom_visibility=custom,
        sliding_window=config.sliding_window,
        dtype=inputs.dtype,
    )
    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs,
            attention_mask=masks.as_mapping(),
            position_ids=torch.arange(inputs.shape[1]).unsqueeze(0),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    assert outputs.hidden_states[-1].shape == inputs.shape
