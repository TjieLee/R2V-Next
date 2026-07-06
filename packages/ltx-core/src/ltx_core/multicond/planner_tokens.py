from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PlannerTokenOutput:
    planner_tokens: Tensor
    planner_valid_mask: Tensor
    placeholder_mask: Tensor | None = None


class SemanticQueryBridge(nn.Module):
    """Map VLM placeholder hidden states to fixed-length planner tokens.

    The bridge follows the stable adapter pattern from the requirements:
    repeated LTX thinking/register tokens, unique query/frame offsets, and a
    zero-initialized residual cross-attention update.
    """

    def __init__(
        self,
        *,
        base_tokens: Tensor,
        target_len: int,
        dim: int,
        num_heads: int,
        source_dim: int | None = None,
        max_frames: int = 64,
    ) -> None:
        super().__init__()
        if base_tokens.ndim != 2:
            raise ValueError(f"base_tokens must be [N, D], got {tuple(base_tokens.shape)}")
        if target_len <= 0:
            raise ValueError("target_len must be positive")

        self.target_len = target_len
        self.dim = dim
        self.source_dim = source_dim or dim

        repeats = (target_len + base_tokens.shape[0] - 1) // base_tokens.shape[0]
        init_tokens = base_tokens.detach().float().repeat(repeats, 1)[:target_len]
        if init_tokens.shape[1] != dim:
            self.base_projection = nn.Linear(init_tokens.shape[1], dim)
        else:
            self.base_projection = nn.Identity()
        self.base_tokens = nn.Parameter(init_tokens, requires_grad=True)
        if self.source_dim != dim:
            self.kv_projection = nn.Linear(self.source_dim, dim)
        else:
            self.kv_projection = nn.Identity()

        self.query_index_embed = nn.Embedding(target_len, dim)
        self.frame_index_embed = nn.Embedding(max_frames, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(dim, dim)
        self.residual_gate = nn.Parameter(torch.zeros(()))

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        planner_hidden: Tensor,
        planner_mask: Tensor | None = None,
        frame_ids: Tensor | None = None,
    ) -> PlannerTokenOutput:
        planner_hidden = self.kv_projection(planner_hidden)
        batch_size = planner_hidden.shape[0]
        device = planner_hidden.device

        queries = self.base_projection(self.base_tokens.to(device=device, dtype=planner_hidden.dtype))
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)

        query_ids = torch.arange(self.target_len, device=device)
        queries = queries + self.query_index_embed(query_ids).to(dtype=queries.dtype).unsqueeze(0)

        if frame_ids is None:
            frame_ids = torch.zeros(self.target_len, dtype=torch.long, device=device)
        frame_ids = frame_ids.to(device=device, dtype=torch.long).clamp_min(0)
        frame_ids = frame_ids[: self.target_len]
        queries = queries + self.frame_index_embed(frame_ids).to(dtype=queries.dtype).unsqueeze(0)

        key_padding_mask = None
        if planner_mask is not None:
            key_padding_mask = ~planner_mask.to(device=device, dtype=torch.bool)

        delta, _ = self.cross_attn(
            self.norm_q(queries),
            self.norm_kv(planner_hidden),
            self.norm_kv(planner_hidden),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        planner_tokens = queries + self.residual_gate.to(dtype=queries.dtype) * self.out_proj(delta)
        planner_valid_mask = torch.ones(batch_size, self.target_len, dtype=torch.bool, device=device)
        return PlannerTokenOutput(planner_tokens=planner_tokens, planner_valid_mask=planner_valid_mask)
