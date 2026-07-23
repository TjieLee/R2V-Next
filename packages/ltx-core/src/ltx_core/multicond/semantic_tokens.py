"""Semantic teacher primitives for joint semantic/video flow matching."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

EVIDENCE_GRID_SIZE = 16
SEMANTIC_GRID_SIZE = 8
EVIDENCE_TOKENS_PER_FRAME = EVIDENCE_GRID_SIZE**2
SEMANTIC_TOKENS_PER_FRAME = SEMANTIC_GRID_SIZE**2
LOCAL_EVIDENCE_TOKENS = 4


@dataclass(frozen=True)
class SemanticKeepMaskSample:
    keep_mask: Tensor
    requested_drop_rate: Tensor
    drop_count_per_frame: Tensor


class SemanticQueryInitializer(nn.Module):
    """Initialize one 8x8 query grid from local 2x2 native evidence regions."""

    def __init__(self, gemma_dim: int, *, position_gate_init: float = 0.0) -> None:
        super().__init__()
        self.gemma_dim = int(gemma_dim)
        self.content_norm = nn.RMSNorm(self.gemma_dim, elementwise_affine=True)
        self.query_type_embedding = nn.Parameter(torch.zeros(self.gemma_dim))
        self.position_gate = nn.Parameter(torch.tensor([float(position_gate_init)]))
        self.temporal_mlp = nn.Sequential(
            nn.Linear(1, self.gemma_dim),
            nn.SiLU(),
            nn.Linear(self.gemma_dim, self.gemma_dim),
        )
        self.height_embedding = nn.Embedding(SEMANTIC_GRID_SIZE, self.gemma_dim)
        self.width_embedding = nn.Embedding(SEMANTIC_GRID_SIZE, self.gemma_dim)
        self._reset_position_parameters()

    def _reset_position_parameters(self) -> None:
        for module in self.temporal_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=1.0e-4)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.height_embedding.weight, std=1.0e-4)
        nn.init.normal_(self.width_embedding.weight, std=1.0e-4)

    def forward(self, evidence: Tensor, normalized_timestamps: Tensor) -> Tensor:
        """Return query embeddings shaped ``[B, F_anchor, 64, D_gemma]``."""
        if evidence.ndim != 4:
            raise ValueError(f"evidence must be [B,F,256,D], got {tuple(evidence.shape)}")
        batch_size, frame_count, token_count, dim = evidence.shape
        if token_count != EVIDENCE_TOKENS_PER_FRAME or dim != self.gemma_dim:
            raise ValueError(
                "evidence shape mismatch: expected "
                f"[B,F,{EVIDENCE_TOKENS_PER_FRAME},{self.gemma_dim}], got {tuple(evidence.shape)}"
            )
        if normalized_timestamps.shape != (batch_size, frame_count):
            raise ValueError(
                "normalized_timestamps must be [B,F], got "
                f"{tuple(normalized_timestamps.shape)} for evidence {tuple(evidence.shape)}"
            )

        grid = evidence.reshape(batch_size, frame_count, SEMANTIC_GRID_SIZE, 2, SEMANTIC_GRID_SIZE, 2, dim)
        local_content = grid.mean(dim=(3, 5)).reshape(
            batch_size,
            frame_count,
            SEMANTIC_TOKENS_PER_FRAME,
            dim,
        )
        local_content = self.content_norm(local_content)

        time_position = self.temporal_mlp(normalized_timestamps[..., None].to(dtype=evidence.dtype))
        height_position = self.height_embedding.weight.to(dtype=evidence.dtype)
        width_position = self.width_embedding.weight.to(dtype=evidence.dtype)
        spatial_position = (
            height_position[:, None, :] + width_position[None, :, :]
        ).reshape(SEMANTIC_TOKENS_PER_FRAME, dim)
        position = time_position[:, :, None, :] + spatial_position[None, None, :, :]
        return (
            local_content
            + self.query_type_embedding.to(dtype=evidence.dtype)[None, None, None, :]
            + self.position_gate.to(dtype=evidence.dtype) * position
        )


class SemanticEncoder(nn.Module):
    """Map contextualized Gemma query hidden states to clean semantic latents."""

    def __init__(self, gemma_dim: int, semantic_dim: int, *, hidden_dim: int = 512) -> None:
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.network = nn.Sequential(
            nn.RMSNorm(gemma_dim, elementwise_affine=True),
            nn.Linear(gemma_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, semantic_dim),
            nn.RMSNorm(semantic_dim, elementwise_affine=True),
        )
        self.global_scale = nn.Parameter(torch.ones(1))

    def forward(self, query_hidden: Tensor) -> Tensor:
        return self.global_scale.to(dtype=query_hidden.dtype) * self.network(query_hidden)


class SemanticReconstructionDecoder(nn.Module):
    """Reconstruct the four contextualized evidence hiddens local to each query."""

    def __init__(self, semantic_dim: int, gemma_dim: int, *, hidden_dim: int = 512) -> None:
        super().__init__()
        self.input_projection = nn.Linear(semantic_dim, hidden_dim)
        self.local_slot_embedding = nn.Parameter(torch.empty(LOCAL_EVIDENCE_TOKENS, hidden_dim))
        self.output_projection = nn.Linear(hidden_dim, gemma_dim)
        nn.init.normal_(self.local_slot_embedding, std=1.0e-3)

    def forward(self, semantic_latent: Tensor) -> Tensor:
        hidden = self.input_projection(semantic_latent).unsqueeze(-2)
        hidden = hidden + self.local_slot_embedding.to(dtype=hidden.dtype)
        return self.output_projection(F.silu(hidden))


def gather_local_evidence(evidence_hidden: Tensor) -> Tensor:
    """Group a 16x16 evidence grid into ``[B,F,64,4,D]`` local targets."""
    if evidence_hidden.ndim != 4 or evidence_hidden.shape[-2] != EVIDENCE_TOKENS_PER_FRAME:
        raise ValueError(
            f"evidence_hidden must be [B,F,{EVIDENCE_TOKENS_PER_FRAME},D], got {tuple(evidence_hidden.shape)}"
        )
    batch_size, frame_count, _tokens, dim = evidence_hidden.shape
    grid = evidence_hidden.reshape(
        batch_size,
        frame_count,
        SEMANTIC_GRID_SIZE,
        2,
        SEMANTIC_GRID_SIZE,
        2,
        dim,
    )
    return grid.permute(0, 1, 2, 4, 3, 5, 6).reshape(
        batch_size,
        frame_count,
        SEMANTIC_TOKENS_PER_FRAME,
        LOCAL_EVIDENCE_TOKENS,
        dim,
    )


def semantic_reconstruction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Per-sample SmoothL1 plus 0.1 cosine reconstruction loss."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"semantic reconstruction shapes differ: prediction={tuple(prediction.shape)}, target={tuple(target.shape)}"
        )
    target = target.detach()
    smooth_l1 = F.smooth_l1_loss(prediction, target, reduction="none").flatten(1).mean(dim=1)
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1)
    cosine = cosine.flatten(1).mean(dim=1)
    return smooth_l1 + 0.1 * cosine


def build_semantic_teacher_attention_mask(
    prefix_attention_mask: Tensor,
    *,
    frame_count: int,
    image_token_mask: Tensor | None = None,
    reference_region_mask: Tensor | None = None,
) -> Tensor:
    """Build the exact prefix/evidence/local-query teacher visibility mask."""
    if prefix_attention_mask.ndim != 2:
        raise ValueError("prefix_attention_mask must be [B,P]")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    prefix_valid = prefix_attention_mask.to(dtype=torch.bool)
    batch_size, prefix_length = prefix_valid.shape
    frame_span = EVIDENCE_TOKENS_PER_FRAME + SEMANTIC_TOKENS_PER_FRAME
    total_length = prefix_length + frame_count * frame_span
    allowed = torch.zeros(batch_size, total_length, total_length, dtype=torch.bool, device=prefix_valid.device)
    allowed[:, :prefix_length, :prefix_length] = build_multimodal_prefix_attention_mask(
        prefix_attention_mask,
        image_token_mask=image_token_mask,
        reference_region_mask=reference_region_mask,
    )

    for frame_index in range(frame_count):
        frame_start = prefix_length + frame_index * frame_span
        evidence_start = frame_start
        evidence_end = evidence_start + EVIDENCE_TOKENS_PER_FRAME
        query_start = evidence_end
        allowed[:, evidence_start:evidence_end, :prefix_length] = prefix_valid[:, None, :]
        allowed[:, evidence_start:evidence_end, evidence_start:evidence_end] = True

        for query_index in range(SEMANTIC_TOKENS_PER_FRAME):
            query_position = query_start + query_index
            row = query_index // SEMANTIC_GRID_SIZE
            column = query_index % SEMANTIC_GRID_SIZE
            local_indices = (
                (2 * row) * EVIDENCE_GRID_SIZE + (2 * column),
                (2 * row) * EVIDENCE_GRID_SIZE + (2 * column + 1),
                (2 * row + 1) * EVIDENCE_GRID_SIZE + (2 * column),
                (2 * row + 1) * EVIDENCE_GRID_SIZE + (2 * column + 1),
            )
            allowed[:, query_position, :prefix_length] = prefix_valid
            for local_index in local_indices:
                allowed[:, query_position, evidence_start + local_index] = True
            allowed[:, query_position, query_position] = True
    return allowed


def build_multimodal_prefix_attention_mask(
    prefix_attention_mask: Tensor,
    *,
    image_token_mask: Tensor | None = None,
    reference_region_mask: Tensor | None = None,
) -> Tensor:
    """Return [B,P,P] visibility for causal text plus bidirectional image placeholder regions."""
    if prefix_attention_mask.ndim != 2:
        raise ValueError("prefix_attention_mask must be [B,P]")
    prefix_valid = prefix_attention_mask.to(dtype=torch.bool)
    batch_size, prefix_length = prefix_valid.shape
    causal = torch.ones(prefix_length, prefix_length, dtype=torch.bool, device=prefix_valid.device).tril()
    allowed = causal[None] & prefix_valid[:, :, None] & prefix_valid[:, None, :]

    if image_token_mask is None:
        image_token_mask = reference_region_mask
    if image_token_mask is None:
        return allowed
    if image_token_mask.shape != prefix_valid.shape:
        raise ValueError("image_token_mask must match prefix_attention_mask")
    image_token_mask = image_token_mask.to(device=allowed.device, dtype=torch.bool) & prefix_valid
    for batch_index in range(batch_size):
        indices = torch.nonzero(image_token_mask[batch_index], as_tuple=False).flatten().tolist()
        if not indices:
            continue
        run_start = indices[0]
        previous = indices[0]
        for index in indices[1:] + [None]:
            if index is not None and index == previous + 1:
                previous = index
                continue
            allowed[batch_index, run_start : previous + 1, run_start : previous + 1] = True
            if index is not None:
                run_start = previous = index
    return allowed


def sample_semantic_keep_mask(
    semantic_latent: Tensor,
    *,
    maximum_drop_rate: float = 0.25,
    minimum_tokens_per_frame: int = 48,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample exact-count per-frame masks without renumbering tokens."""
    return sample_semantic_keep_mask_with_stats(
        semantic_latent,
        maximum_drop_rate=maximum_drop_rate,
        minimum_tokens_per_frame=minimum_tokens_per_frame,
        generator=generator,
    ).keep_mask


def sample_semantic_keep_mask_with_stats(
    semantic_latent: Tensor,
    *,
    maximum_drop_rate: float = 0.25,
    minimum_tokens_per_frame: int = 48,
    generator: torch.Generator | None = None,
) -> SemanticKeepMaskSample:
    """Sample one exact drop count per sample, with independent spatial positions per frame."""
    if semantic_latent.ndim != 4 or semantic_latent.shape[-2] != SEMANTIC_TOKENS_PER_FRAME:
        raise ValueError("semantic_latent must be [B,F,64,C]")
    if not 0.0 <= maximum_drop_rate <= 1.0:
        raise ValueError("maximum_drop_rate must be in [0,1]")
    minimum_tokens_per_frame = min(max(1, int(minimum_tokens_per_frame)), SEMANTIC_TOKENS_PER_FRAME)
    batch_size, frame_count = semantic_latent.shape[:2]
    device = semantic_latent.device
    max_drop_by_rate = math.floor(float(maximum_drop_rate) * SEMANTIC_TOKENS_PER_FRAME + 1.0e-8)
    max_drop_by_minimum = SEMANTIC_TOKENS_PER_FRAME - minimum_tokens_per_frame
    max_drop_count = max(0, min(max_drop_by_rate, max_drop_by_minimum))
    drop_counts = torch.randint(
        low=0,
        high=max_drop_count + 1,
        size=(batch_size,),
        device=device,
        generator=generator,
        dtype=torch.long,
    )
    keep = torch.zeros(
        batch_size,
        frame_count,
        SEMANTIC_TOKENS_PER_FRAME,
        device=device,
        dtype=torch.bool,
    )
    scores = torch.rand(
        batch_size,
        frame_count,
        SEMANTIC_TOKENS_PER_FRAME,
        device=device,
        generator=generator,
    )
    for batch_index in range(batch_size):
        keep_count = SEMANTIC_TOKENS_PER_FRAME - int(drop_counts[batch_index].item())
        for frame_index in range(frame_count):
            selected = scores[batch_index, frame_index].topk(keep_count).indices
            keep[batch_index, frame_index, selected] = True
    return SemanticKeepMaskSample(
        keep_mask=keep,
        requested_drop_rate=drop_counts.to(dtype=torch.float32) / float(SEMANTIC_TOKENS_PER_FRAME),
        drop_count_per_frame=drop_counts,
    )


def round_up_prefix_length(actual_length: int, *, multiple: int = 128, maximum: int = 2560) -> int:
    if actual_length <= 0:
        raise ValueError("actual_length must be positive")
    if actual_length > maximum:
        raise ValueError(f"prefix length {actual_length} exceeds maximum {maximum}")
    return min(maximum, int(math.ceil(actual_length / multiple) * multiple))
