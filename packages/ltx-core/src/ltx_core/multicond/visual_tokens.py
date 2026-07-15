from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ltx_core.model.transformer.rope import (
    LTXRopeType,
    apply_rotary_emb,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)


@dataclass(frozen=True)
class VisualTokenBatch:
    tokens: Tensor
    mask: Tensor


class VisualPlannerTokens(nn.Module):
    """Content-residual 3D-RoPE bridge from VLM placeholders to visual tokens."""

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
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        use_content_residual: bool = True,
        use_3d_rope: bool = False,
        rope_use_middle_positions: bool = True,
        residual_init_gain: float = 0.1,
        query_chunk_size: int | None = None,
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
        if residual_init_gain <= 0:
            raise ValueError("residual_init_gain must be positive")
        if query_chunk_size is not None and query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive when set")

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
        self.use_content_residual = use_content_residual
        self.use_3d_rope = use_3d_rope
        self.rope_use_middle_positions = rope_use_middle_positions
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = list(positional_embedding_max_pos or [20, 2048, 2048])
        self.rope_type = rope_type
        self.query_chunk_size = query_chunk_size
        if use_3d_rope and rope_type == LTXRopeType.SPLIT and self.head_dim % 2 != 0:
            raise ValueError(
                "VisualPlannerTokens requires even head_dim for split RoPE, got "
                f"dim={dim}, num_heads={num_heads}, head_dim={self.head_dim}"
            )

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
        self.content_projection = nn.Linear(self.source_dim, dim, bias=False)

        nn.init.xavier_uniform_(self.query_projection.weight, gain=1.0)
        nn.init.xavier_uniform_(self.key_projection.weight, gain=1.0)
        nn.init.xavier_uniform_(self.value_projection.weight, gain=1.0)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.0 if zero_init_output else residual_init_gain)
        nn.init.zeros_(self.query_projection.bias)
        nn.init.zeros_(self.key_projection.bias)
        nn.init.zeros_(self.value_projection.bias)
        nn.init.zeros_(self.output_projection.bias)
        if self.source_dim == dim:
            nn.init.eye_(self.content_projection.weight)
        else:
            nn.init.xavier_uniform_(self.content_projection.weight, gain=1.0)

        ffn_hidden_dim = max(1, int(dim * ffn_multiplier))
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_fc1 = nn.Linear(dim, ffn_hidden_dim)
        self.ffn_fc2 = nn.Linear(ffn_hidden_dim, dim)
        nn.init.xavier_uniform_(self.ffn_fc1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.ffn_fc2.weight, gain=0.0 if zero_init_ffn else residual_init_gain)
        nn.init.zeros_(self.ffn_fc1.bias)
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
        token_positions: Tensor | None = None,
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

        if self.use_3d_rope:
            if token_positions is None:
                raise ValueError("token_positions is required when use_3d_rope=True")
            if token_positions.shape != (batch_size, 3, self.token_count, 2):
                raise ValueError(
                    "token_positions must be [B,3,K,2], got "
                    f"{tuple(token_positions.shape)} for planner_hidden {tuple(planner_hidden.shape)}"
                )
            freqs_cis = precompute_freqs_cis(
                indices_grid=token_positions.to(device=planner_hidden.device),
                dim=self.dim,
                out_dtype=q.dtype,
                theta=self.positional_embedding_theta,
                max_pos=self.positional_embedding_max_pos,
                use_middle_indices_grid=self.rope_use_middle_positions,
                num_attention_heads=self.num_heads,
                rope_type=self.rope_type,
                freq_grid_generator=generate_freq_grid_pytorch,
            )
            q = apply_rotary_emb(q, freqs_cis, self.rope_type)
            k = apply_rotary_emb(k, freqs_cis, self.rope_type)

        attn_mask = None
        if planner_mask is not None:
            key_mask = planner_mask.to(device=q.device, dtype=torch.bool)
            if key_mask.shape != planner_hidden.shape[:2]:
                raise ValueError(f"planner_mask must be [B,K], got {tuple(key_mask.shape)}")
            if not bool(key_mask.all()):
                no_valid_keys = ~key_mask.any(dim=1)
                if torch.any(no_valid_keys):
                    key_mask = key_mask.clone()
                    key_mask[no_valid_keys, 0] = True
                attn_mask = key_mask[:, None, None, :]

        attended = self._scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attended = self._merge_heads(attended)
        content = self.content_projection(planner_hidden)
        residual = content if self.use_content_residual else query
        x = residual + self.output_projection(attended)

        ffn_hidden = self.ffn_fc1(self.ffn_norm(x))
        ffn_hidden = F.gelu(ffn_hidden)
        if self.training and self.ffn_dropout > 0:
            ffn_hidden = F.dropout(ffn_hidden, p=self.ffn_dropout)
        ffn_out = self.ffn_fc2(ffn_hidden)
        if self.training and self.ffn_dropout > 0:
            ffn_out = F.dropout(ffn_out, p=self.ffn_dropout)
        output = x + ffn_out
        if planner_mask is not None:
            output = output * planner_mask.to(device=output.device, dtype=output.dtype).unsqueeze(-1)
        return output

    def _scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        attn_mask: Tensor | None,
    ) -> Tensor:
        dropout_p = self.dropout if self.training else 0.0
        if self.query_chunk_size is None or q.shape[-2] <= self.query_chunk_size:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        chunks = []
        for start in range(0, q.shape[-2], self.query_chunk_size):
            chunks.append(
                F.scaled_dot_product_attention(
                    q[..., start : start + self.query_chunk_size, :],
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                )
            )
        return torch.cat(chunks, dim=-2)

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


class _Visual3DSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        ffn_multiplier: float,
        dropout: float,
        residual_init_gain: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.norm1 = nn.RMSNorm(dim, eps=1.0e-6, elementwise_affine=True)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.RMSNorm(dim, eps=1.0e-6, elementwise_affine=True)
        hidden_dim = max(1, int(dim * ffn_multiplier))
        self.ffn_fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.ffn_fc2 = nn.Linear(hidden_dim, dim, bias=False)

        nn.init.ones_(self.norm1.weight)
        nn.init.ones_(self.norm2.weight)
        nn.init.xavier_uniform_(self.qkv.weight, gain=1.0)
        nn.init.xavier_uniform_(self.attn_out.weight, gain=residual_init_gain)
        nn.init.xavier_uniform_(self.ffn_fc1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.ffn_fc2.weight, gain=residual_init_gain)

    def forward(
        self,
        x: Tensor,
        *,
        freqs_cis: tuple[Tensor, Tensor],
        token_mask: Tensor,
        rope_type: LTXRopeType,
    ) -> Tensor:
        batch_size, seq_len, _dim = x.shape
        qkv = self.qkv(self.norm1(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = apply_rotary_emb(q, freqs_cis, rope_type)
        k = apply_rotary_emb(k, freqs_cis, rope_type)

        safe_key_mask = token_mask
        no_valid_keys = ~safe_key_mask.any(dim=1)
        if torch.any(no_valid_keys):
            safe_key_mask = safe_key_mask.clone()
            safe_key_mask[no_valid_keys, 0] = True
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=safe_key_mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        x = x + self.attn_out(attended)

        hidden = F.gelu(self.ffn_fc1(self.norm2(x)))
        if self.training and self.dropout > 0.0:
            hidden = F.dropout(hidden, p=self.dropout)
        return x + self.ffn_fc2(hidden)


class Visual3DTokenEncoder(nn.Module):
    """Bias-free 3D-RoPE self-attention encoder for full visual token sequences."""

    def __init__(
        self,
        *,
        dim: int = 4096,
        num_heads: int = 32,
        depth: int = 1,
        ffn_multiplier: float = 2.0,
        dropout: float = 0.0,
        residual_init_gain: float = 0.1,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        use_middle_positions: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        head_dim = dim // num_heads
        if rope_type == LTXRopeType.SPLIT and head_dim % 2 != 0:
            raise ValueError(
                "Visual3DTokenEncoder requires even head_dim for split RoPE, got "
                f"dim={dim}, num_heads={num_heads}, head_dim={head_dim}"
            )
        if depth <= 0:
            raise ValueError("depth must be positive")
        if ffn_multiplier <= 0.0:
            raise ValueError("ffn_multiplier must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if residual_init_gain <= 0.0:
            raise ValueError("residual_init_gain must be positive")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.depth = depth
        self.dropout = dropout
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = list(positional_embedding_max_pos or [20, 2048, 2048])
        self.rope_type = rope_type
        self.use_middle_positions = use_middle_positions

        self.input_norm = nn.RMSNorm(dim, eps=1.0e-6, elementwise_affine=True)
        self.blocks = nn.ModuleList(
            _Visual3DSelfAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_multiplier=ffn_multiplier,
                dropout=dropout,
                residual_init_gain=residual_init_gain,
            )
            for _ in range(depth)
        )
        self.output_norm = nn.RMSNorm(dim, eps=1.0e-6, elementwise_affine=True)
        nn.init.ones_(self.input_norm.weight)
        nn.init.ones_(self.output_norm.weight)

    def forward(
        self,
        *,
        tokens: Tensor,
        token_positions: Tensor,
        token_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,N,D], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.dim:
            raise ValueError(f"tokens dim {tokens.shape[-1]} != encoder dim {self.dim}")
        encoder_compute = module_compute_device_dtype(self.input_norm)
        if encoder_compute is not None:
            tokens = tokens.to(device=encoder_compute[0], dtype=encoder_compute[1])
        if token_positions.shape != (tokens.shape[0], 3, tokens.shape[1], 2):
            raise ValueError(
                "token_positions must be [B,3,N,2], got "
                f"{tuple(token_positions.shape)} for tokens {tuple(tokens.shape)}"
            )
        if token_mask is None:
            encoded_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            if token_mask.shape != tokens.shape[:2]:
                raise ValueError(f"token_mask must be [B,N], got {tuple(token_mask.shape)}")
            encoded_mask = token_mask.to(device=tokens.device, dtype=torch.bool)

        freqs_cis = precompute_freqs_cis(
            indices_grid=token_positions.to(device=tokens.device),
            dim=self.dim,
            out_dtype=tokens.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            use_middle_indices_grid=self.use_middle_positions,
            num_attention_heads=self.num_heads,
            rope_type=self.rope_type,
            freq_grid_generator=generate_freq_grid_pytorch,
        )
        x = self.input_norm(tokens)
        for block in self.blocks:
            x = block(
                x,
                freqs_cis=freqs_cis,
                token_mask=encoded_mask,
                rope_type=self.rope_type,
            )
        x = self.output_norm(x)
        x = x * encoded_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x, encoded_mask



class Visual3DResampler(nn.Module):
    """Position-aware Perceiver/Q-former resampler for target-video SigLIP tokens.

    The module keeps DiT untouched: it compresses raw visual tokens into a
    smaller set of connector-space tokens before they are concatenated to the
    normal post-connector text/VLM context.
    """

    def __init__(
        self,
        *,
        dim: int,
        max_query_tokens: int = 2048,
        num_heads: int = 16,
        depth: int = 1,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.0,
        zero_init_output: bool = True,
        gate_init: float = 0.0,
        query_init_std: float = 1e-4,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if max_query_tokens <= 0:
            raise ValueError("max_query_tokens must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")

        self.dim = dim
        self.max_query_tokens = max_query_tokens
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if rope_type == LTXRopeType.SPLIT and self.head_dim % 2 != 0:
            raise ValueError(
                "Visual3DResampler requires even head_dim for split RoPE, got "
                f"dim={dim}, num_heads={num_heads}, head_dim={self.head_dim}"
            )
        self.depth = depth
        self.dropout = dropout
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos or [20, 2048, 2048]
        self.rope_type = rope_type

        self.query_tokens = nn.Parameter(torch.empty(max_query_tokens, dim))
        nn.init.normal_(self.query_tokens, std=query_init_std)
        self.query_position_proj = nn.Linear(6, dim)
        self.key_position_proj = nn.Linear(6, dim)
        self.query_type_encoding = nn.Parameter(torch.zeros(1, 1, dim))
        self.key_type_encoding = nn.Parameter(torch.zeros(1, 1, dim))

        self.query_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(depth))
        self.key_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(depth))
        self.query_projections = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))
        self.key_projections = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))
        self.value_projections = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))
        self.output_projections = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))
        self.ffn_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(depth))
        ffn_hidden_dim = max(1, int(dim * ffn_multiplier))
        self.ffn_fc1 = nn.ModuleList(nn.Linear(dim, ffn_hidden_dim) for _ in range(depth))
        self.ffn_fc2 = nn.ModuleList(nn.Linear(ffn_hidden_dim, dim) for _ in range(depth))
        self.output_norm = nn.LayerNorm(dim)
        self.residual_gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

        if zero_init_output:
            for projection in self.output_projections:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
            for fc2 in self.ffn_fc2:
                nn.init.zeros_(fc2.weight)
                nn.init.zeros_(fc2.bias)

    def forward(
        self,
        *,
        tokens: Tensor,
        token_positions: Tensor,
        token_mask: Tensor | None,
        query_positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,K,D], got {tuple(tokens.shape)}")
        if token_positions.shape[:3] != (tokens.shape[0], 3, tokens.shape[1]):
            raise ValueError(
                f"token_positions must be [B,3,K,2], got {tuple(token_positions.shape)} for tokens {tuple(tokens.shape)}"
            )
        if query_positions.ndim != 4 or query_positions.shape[0] != tokens.shape[0] or query_positions.shape[1] != 3:
            raise ValueError(f"query_positions must be [B,3,M,2], got {tuple(query_positions.shape)}")
        query_count = query_positions.shape[2]
        if query_count > self.max_query_tokens:
            raise ValueError(f"query token count {query_count} exceeds max_query_tokens={self.max_query_tokens}")
        if tokens.shape[-1] != self.dim:
            raise ValueError(f"tokens dim {tokens.shape[-1]} != resampler dim {self.dim}")
        if token_mask is not None and token_mask.shape != tokens.shape[:2]:
            raise ValueError(f"token_mask must be [B,K], got {tuple(token_mask.shape)}")

        batch_size = tokens.shape[0]
        query = self.query_tokens[:query_count].to(device=tokens.device, dtype=tokens.dtype)
        query = query.unsqueeze(0).expand(batch_size, -1, -1)
        query = query + self.query_type_encoding.to(device=tokens.device, dtype=tokens.dtype)
        query = query + self.query_position_proj(self._flatten_positions(query_positions, dtype=tokens.dtype))

        kv = tokens + self.key_type_encoding.to(device=tokens.device, dtype=tokens.dtype)
        kv = kv + self.key_position_proj(self._flatten_positions(token_positions, dtype=tokens.dtype))
        residual_query = query

        for idx in range(self.depth):
            q = self.query_projections[idx](self.query_norms[idx](query))
            k = self.key_projections[idx](self.key_norms[idx](kv))
            v = self.value_projections[idx](self.key_norms[idx](kv))
            q = self._split_heads(q)
            k = self._split_heads(k)
            v = self._split_heads(v)
            q_pe = self._rope_for_positions(query_positions, dtype=q.dtype)
            k_pe = self._rope_for_positions(token_positions, dtype=k.dtype)
            q = apply_rotary_emb(q, q_pe, self.rope_type)
            k = apply_rotary_emb(k, k_pe, self.rope_type)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if token_mask is not None:
                mask = token_mask.to(device=scores.device, dtype=torch.bool)
                scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout)
            attended = self._merge_heads(torch.matmul(attn, v))
            delta = self.output_projections[idx](attended)
            query = query + self.residual_gate.to(dtype=query.dtype) * delta

            ffn_hidden = self.ffn_fc1[idx](self.ffn_norms[idx](query))
            ffn_hidden = F.gelu(ffn_hidden)
            if self.training and self.dropout > 0:
                ffn_hidden = F.dropout(ffn_hidden, p=self.dropout)
            ffn_delta = self.ffn_fc2[idx](ffn_hidden)
            query = query + self.residual_gate.to(dtype=query.dtype) * ffn_delta

        visual_mask = torch.ones(batch_size, query_count, dtype=torch.bool, device=tokens.device)
        return self.output_norm(query + 0.0 * residual_query), visual_mask

    def _rope_for_positions(self, positions: Tensor, *, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        return precompute_freqs_cis(
            indices_grid=positions[..., 0],
            dim=self.dim,
            out_dtype=dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_heads,
            rope_type=self.rope_type,
            freq_grid_generator=generate_freq_grid_pytorch,
        )

    @staticmethod
    def _flatten_positions(positions: Tensor, *, dtype: torch.dtype) -> Tensor:
        # [B,3,N,2] -> [B,N,6], normalized lightly for stable MLP inputs.
        flat = positions.permute(0, 2, 1, 3).reshape(positions.shape[0], positions.shape[2], 6).to(dtype=dtype)
        scale = flat.detach().abs().amax(dim=1, keepdim=True).clamp(min=1.0)
        return flat / scale

    def _split_heads(self, value: Tensor) -> Tensor:
        batch_size, seq_len, _dim = value.shape
        return value.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _merge_heads(value: Tensor) -> Tensor:
        batch_size, _heads, seq_len, head_dim = value.shape
        return value.transpose(1, 2).reshape(batch_size, seq_len, -1)


def module_compute_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype] | None:
    """Return the first floating parameter/buffer device and dtype, including wrapped modules."""
    candidates: list[nn.Module] = []
    current = module
    seen: set[int] = set()
    while id(current) not in seen:
        candidates.append(current)
        seen.add(id(current))
        wrapped = getattr(current, "module", None)
        if not isinstance(wrapped, nn.Module):
            break
        current = wrapped

    for candidate in candidates:
        for parameter in candidate.parameters():
            if parameter.is_floating_point():
                return parameter.device, parameter.dtype
        for buffer in candidate.buffers():
            if buffer.is_floating_point():
                return buffer.device, buffer.dtype
    return None


def _unwrapped_module_name(module: nn.Module) -> str:
    current = module
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "module", None)
        if not isinstance(wrapped, nn.Module):
            break
        current = wrapped
    return type(current).__name__


def extract_projected_visual_tokens(
    gemma_causal_lm: nn.Module,
    pixel_values: Tensor,
    *,
    image_counts: Tensor | None = None,
    dtype_diagnostics: dict[str, str] | None = None,
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

    vision_compute = module_compute_device_dtype(vision_tower)
    if vision_compute is not None:
        flat_pixels = flat_pixels.to(device=vision_compute[0], dtype=vision_compute[1])
    if dtype_diagnostics is not None:
        dtype_diagnostics["vision_module"] = _unwrapped_module_name(vision_tower)
        dtype_diagnostics["vision_input_dtype"] = str(flat_pixels.dtype)
        dtype_diagnostics["vision_input_device"] = str(flat_pixels.device)
    vision_outputs = vision_tower(pixel_values=flat_pixels)
    vision_hidden = _last_hidden_state(vision_outputs)
    if dtype_diagnostics is not None:
        dtype_diagnostics["vision_output_dtype"] = str(vision_hidden.dtype)
        dtype_diagnostics["vision_output_device"] = str(vision_hidden.device)
    projector_compute = module_compute_device_dtype(projector)
    if projector_compute is not None:
        vision_hidden = vision_hidden.to(device=projector_compute[0], dtype=projector_compute[1])
    if dtype_diagnostics is not None:
        dtype_diagnostics["projector_module"] = _unwrapped_module_name(projector)
        dtype_diagnostics["projector_input_dtype"] = str(vision_hidden.dtype)
        dtype_diagnostics["projector_input_device"] = str(vision_hidden.device)
    projected = _call_projector(projector, vision_hidden)
    projected = _last_hidden_state(projected)
    if dtype_diagnostics is not None:
        dtype_diagnostics["projector_output_dtype"] = str(projected.dtype)
        dtype_diagnostics["projector_output_device"] = str(projected.device)

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
