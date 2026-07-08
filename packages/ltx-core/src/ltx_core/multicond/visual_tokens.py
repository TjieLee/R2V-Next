from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class VisualTokenBatch:
    tokens: Tensor
    mask: Tensor


class VisualPlannerTokens(nn.Module):
    """Baton-style visual planner/Q-former for fixed visual placeholder tokens.

    The default Stage 2 path uses internal learned query tokens in raw
    SigLIP/Gemma-projector space, while the legacy path can still repeat external
    connector register tokens as queries. Hidden states at the target
    ``<img_pad>`` positions provide keys/values. The output projection and FFN
    can be zero-initialized so the bridge starts as a small residual adapter over
    the query tokens.
    """

    def __init__(
        self,
        *,
        token_count: int,
        dim: int,
        source_dim: int | None = None,
        num_heads: int = 16,
        dropout: float = 0.0,
        zero_init_output: bool = True,
        use_slot_encoding: bool = True,
        slot_init_std: float = 1e-4,
        slot_init_seed: int | None = 0,
        ffn_multiplier: float = 4.0,
        ffn_dropout: float = 0.0,
        zero_init_ffn: bool = True,
        use_learned_query_tokens: bool = False,
        query_init_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if slot_init_std < 0:
            raise ValueError("slot_init_std must be non-negative")
        if query_init_std < 0:
            raise ValueError("query_init_std must be non-negative")

        self.token_count = token_count
        self.dim = dim
        self.source_dim = source_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.ffn_dropout = ffn_dropout
        self.use_slot_encoding = use_slot_encoding
        self.slot_init_std = slot_init_std
        self.slot_init_seed = slot_init_seed
        self.use_learned_query_tokens = use_learned_query_tokens
        self.query_init_std = query_init_std

        if use_learned_query_tokens:
            self.query_tokens = nn.Parameter(torch.empty(token_count, dim))
            self._init_query_tokens(query_init_std=query_init_std)
        else:
            self.register_parameter("query_tokens", None)

        self.query_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(self.source_dim)
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(self.source_dim, dim)
        self.value_projection = nn.Linear(self.source_dim, dim)
        self.output_projection = nn.Linear(dim, dim)

        if zero_init_output:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

        ffn_hidden_dim = max(1, int(dim * ffn_multiplier))
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_fc1 = nn.Linear(dim, ffn_hidden_dim)
        self.ffn_fc2 = nn.Linear(ffn_hidden_dim, dim)
        if zero_init_ffn:
            nn.init.zeros_(self.ffn_fc2.weight)
            nn.init.zeros_(self.ffn_fc2.bias)

        if use_slot_encoding:
            self.query_slot_encoding = nn.Parameter(torch.zeros(token_count, dim))
            self.kv_slot_encoding = nn.Parameter(torch.zeros(token_count, self.source_dim))
            self._init_slot_encodings(slot_init_std=slot_init_std, slot_init_seed=slot_init_seed)
        else:
            self.register_parameter("query_slot_encoding", None)
            self.register_parameter("kv_slot_encoding", None)

        self.query_type_encoding = nn.Parameter(torch.zeros(1, 1, dim))
        self.kv_type_encoding = nn.Parameter(torch.zeros(1, 1, self.source_dim))

    def _init_query_tokens(self, *, query_init_std: float) -> None:
        with torch.no_grad():
            if query_init_std == 0:
                self.query_tokens.zero_()
            else:
                nn.init.normal_(self.query_tokens, std=query_init_std)

    def _init_slot_encodings(self, *, slot_init_std: float, slot_init_seed: int | None) -> None:
        if slot_init_std == 0:
            return

        randn_kwargs: dict[str, Any] = {}
        if slot_init_seed is not None:
            generator = torch.Generator(device=self.query_slot_encoding.device)
            generator.manual_seed(int(slot_init_seed))
            randn_kwargs["generator"] = generator

        with torch.no_grad():
            query_noise = torch.randn(
                self.query_slot_encoding.shape,
                device=self.query_slot_encoding.device,
                dtype=self.query_slot_encoding.dtype,
                **randn_kwargs,
            )
            kv_noise = torch.randn(
                self.kv_slot_encoding.shape,
                device=self.kv_slot_encoding.device,
                dtype=self.kv_slot_encoding.dtype,
                **randn_kwargs,
            )
            self.query_slot_encoding.copy_(query_noise * slot_init_std)
            self.kv_slot_encoding.copy_(kv_noise * slot_init_std)

    def forward(
        self,
        *,
        planner_hidden: Tensor,
        query_registers: Tensor | None = None,
        planner_mask: Tensor | None = None,
    ) -> Tensor:
        if planner_hidden.ndim != 3:
            raise ValueError(f"planner_hidden must be [B,K,D], got {tuple(planner_hidden.shape)}")
        if planner_hidden.shape[1] != self.token_count:
            raise ValueError(f"planner_hidden token count {planner_hidden.shape[1]} != {self.token_count}")
        batch_size = planner_hidden.shape[0]
        if query_registers is None:
            if self.query_tokens is None:
                raise ValueError("query_registers is required when use_learned_query_tokens=False")
            query = self.query_tokens.to(device=planner_hidden.device, dtype=planner_hidden.dtype)
            query = query.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            if query_registers.ndim != 2:
                raise ValueError(f"query_registers must be [R,D], got {tuple(query_registers.shape)}")
            if query_registers.shape[-1] != self.dim:
                raise ValueError(f"query_registers dim {query_registers.shape[-1]} != planner dim {self.dim}")
            query = self._repeat_query_registers(
                query_registers=query_registers,
                batch_size=batch_size,
                device=planner_hidden.device,
                dtype=planner_hidden.dtype,
            )
        kv = planner_hidden

        if self.query_slot_encoding is not None:
            query = query + self.query_slot_encoding.to(device=query.device, dtype=query.dtype).unsqueeze(0)
            kv = kv + self.kv_slot_encoding.to(device=kv.device, dtype=kv.dtype).unsqueeze(0)
        query = query + self.query_type_encoding.to(device=query.device, dtype=query.dtype)
        kv = kv + self.kv_type_encoding.to(device=kv.device, dtype=kv.dtype)

        q = self.query_projection(self.query_norm(query))
        k = self.key_projection(self.kv_norm(kv))
        v = self.value_projection(self.kv_norm(kv))

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if planner_mask is not None:
            key_mask = planner_mask.to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        attended = torch.matmul(attn, v)
        attended = self._merge_heads(attended)
        x = query + self.output_projection(attended)

        ffn_hidden = self.ffn_fc1(self.ffn_norm(x))
        ffn_hidden = F.gelu(ffn_hidden)
        if self.training and self.ffn_dropout > 0:
            ffn_hidden = F.dropout(ffn_hidden, p=self.ffn_dropout)
        ffn_out = self.ffn_fc2(ffn_hidden)
        if self.training and self.ffn_dropout > 0:
            ffn_out = F.dropout(ffn_out, p=self.ffn_dropout)
        return x + ffn_out

    def _repeat_query_registers(
        self,
        *,
        query_registers: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        repeats = (self.token_count + query_registers.shape[0] - 1) // query_registers.shape[0]
        query = query_registers.to(device=device, dtype=dtype).repeat(repeats, 1)[: self.token_count]
        return query.unsqueeze(0).expand(batch_size, -1, -1)

    def _split_heads(self, value: Tensor) -> Tensor:
        batch_size, seq_len, _dim = value.shape
        return value.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _merge_heads(value: Tensor) -> Tensor:
        batch_size, _heads, seq_len, head_dim = value.shape
        return value.transpose(1, 2).reshape(batch_size, seq_len, -1)


def extract_projected_visual_tokens(
    gemma_causal_lm: nn.Module,
    pixel_values: Tensor,
    *,
    image_counts: Tensor | None = None,
) -> VisualTokenBatch:
    """Run frozen Gemma/SigLIP vision tower + projector and return flattened tokens.

    The helper accepts the tensor layouts produced by ``Gemma3Processor`` both
    before and after DataLoader collation: ``[R,C,H,W]``, ``[B,R,C,H,W]`` or
    ``[B,1,R,C,H,W]``. Returned tokens are grouped per training sample as
    ``[B, R * T, D]`` and padded on the token axis when image counts differ.
    """

    gemma_model = getattr(gemma_causal_lm, "model", gemma_causal_lm)
    vision_tower = getattr(gemma_model, "vision_tower", None)
    projector = getattr(gemma_model, "multi_modal_projector", None)
    if vision_tower is None or projector is None:
        raise ValueError("Gemma model must expose vision_tower and multi_modal_projector to build GT visual tokens.")

    pixel_values, batch_size, images_per_sample = _normalize_pixel_values(pixel_values)
    flat_pixels = pixel_values.reshape(batch_size * images_per_sample, *pixel_values.shape[-3:])

    vision_outputs = vision_tower(pixel_values=flat_pixels)
    vision_hidden = _last_hidden_state(vision_outputs)
    projected = _call_projector(projector, vision_hidden)
    projected = _last_hidden_state(projected)

    if projected.ndim != 3:
        raise ValueError(f"Projected visual tokens must be [B*R,T,D], got {tuple(projected.shape)}")

    tokens_per_image = projected.shape[1]
    projected = projected.reshape(batch_size, images_per_sample * tokens_per_image, projected.shape[-1])

    if image_counts is None:
        mask = torch.ones(projected.shape[:2], dtype=torch.bool, device=projected.device)
        return VisualTokenBatch(tokens=projected, mask=mask)

    image_counts = image_counts.to(device=projected.device, dtype=torch.long).clamp(min=0, max=images_per_sample)
    token_counts = image_counts * tokens_per_image
    mask = torch.arange(projected.shape[1], device=projected.device).unsqueeze(0) < token_counts.unsqueeze(1)
    projected = projected * mask.unsqueeze(-1).to(dtype=projected.dtype)
    return VisualTokenBatch(tokens=projected, mask=mask)


def scatter_visual_tokens_into_embeddings(
    *,
    inputs_embeds: Tensor,
    input_ids: Tensor,
    visual_tokens: Tensor,
    image_token_index: int,
    visual_mask: Tensor | None = None,
    planner_placeholder_mask: Tensor | None = None,
) -> Tensor:
    """Replace real Gemma image-token positions with projected visual tokens."""

    image_mask = input_ids == image_token_index
    if planner_placeholder_mask is not None:
        image_mask = image_mask & ~planner_placeholder_mask.to(device=image_mask.device, dtype=torch.bool)

    if visual_mask is None:
        visual_mask = torch.ones(visual_tokens.shape[:2], dtype=torch.bool, device=visual_tokens.device)
    else:
        visual_mask = visual_mask.to(device=visual_tokens.device, dtype=torch.bool)

    expected = image_mask.sum(dim=1)
    available = visual_mask.sum(dim=1)
    if not torch.all(expected == available):
        raise ValueError(
            "Image token count does not match projected SigLIP token count: "
            f"expected mask counts {expected.tolist()}, valid visual tokens {available.tolist()}"
        )

    out = inputs_embeds.clone()
    for batch_index in range(out.shape[0]):
        valid_tokens = visual_tokens[batch_index, visual_mask[batch_index]].to(dtype=out.dtype)
        out[batch_index, image_mask[batch_index]] = valid_tokens
    return out


def _normalize_pixel_values(pixel_values: Tensor) -> tuple[Tensor, int, int]:
    if pixel_values.ndim == 6 and pixel_values.shape[1] == 1:
        pixel_values = pixel_values.squeeze(1)
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)
    if pixel_values.ndim != 5:
        raise ValueError(
            "pixel_values must be [R,C,H,W], [B,R,C,H,W] or [B,1,R,C,H,W], "
            f"got {tuple(pixel_values.shape)}"
        )
    return pixel_values, pixel_values.shape[0], pixel_values.shape[1]


def _last_hidden_state(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    hidden = getattr(value, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    raise ValueError(f"Cannot extract hidden states from {type(value).__name__}")


def _call_projector(projector: nn.Module, hidden: Tensor) -> Tensor:
    try:
        return projector(hidden)
    except TypeError:
        return projector(image_features=hidden)
