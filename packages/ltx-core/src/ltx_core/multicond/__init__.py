"""Semantic-flow and native Gemma multimodal utilities."""

from ltx_core.multicond.gemma3_attention import (
    Gemma3AttentionMasks,
    build_gemma3_attention_masks,
    build_native_sliding_visibility,
    resolve_gemma3_sliding_window,
)
from ltx_core.multicond.semantic_tokens import (
    EVIDENCE_TOKENS_PER_FRAME,
    SEMANTIC_TOKENS_PER_FRAME,
    SemanticAlignmentHead,
    SemanticEncoder,
    SemanticKeepMaskSample,
    SemanticQueryInitializer,
    SemanticReconstructionDecoder,
    build_multimodal_prefix_attention_mask,
    build_semantic_teacher_attention_mask,
    gather_local_evidence,
    round_up_prefix_length,
    sample_semantic_keep_mask,
    sample_semantic_keep_mask_with_stats,
    semantic_alignment_loss,
    semantic_reconstruction_loss,
)
from ltx_core.multicond.visual_tokens import (
    VisualTokenBatch,
    extract_projected_visual_tokens,
    module_compute_device_dtype,
    scatter_visual_tokens_into_embeddings,
)

__all__ = [
    "EVIDENCE_TOKENS_PER_FRAME",
    "SEMANTIC_TOKENS_PER_FRAME",
    "Gemma3AttentionMasks",
    "SemanticAlignmentHead",
    "SemanticEncoder",
    "SemanticKeepMaskSample",
    "SemanticQueryInitializer",
    "SemanticReconstructionDecoder",
    "VisualTokenBatch",
    "build_gemma3_attention_masks",
    "build_multimodal_prefix_attention_mask",
    "build_native_sliding_visibility",
    "build_semantic_teacher_attention_mask",
    "extract_projected_visual_tokens",
    "gather_local_evidence",
    "module_compute_device_dtype",
    "resolve_gemma3_sliding_window",
    "round_up_prefix_length",
    "sample_semantic_keep_mask",
    "sample_semantic_keep_mask_with_stats",
    "scatter_visual_tokens_into_embeddings",
    "semantic_alignment_loss",
    "semantic_reconstruction_loss",
]
