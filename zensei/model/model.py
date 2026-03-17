"""
Zensei Model — DeepSeek-V3 style MLA + MoE architecture optimized for Japanese.

Implements Multi-head Latent Attention (MLA) with compressed KV cache,
Mixture of Experts (MoE) with shared + routed experts, RMSNorm, SiLU activation,
and Rotary Positional Encoding (RoPE).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelArgs:
    """Configuration for the Zensei / DeepSeek-V3 style model."""

    vocab_size: int = 145408  # expanded from DeepSeek 129K, aligned to 128 for tensor cores
    dim: int = 7168
    inter_dim: int = 18432  # FFN intermediate size for dense layers
    moe_inter_dim: int = 2048  # per-expert FFN intermediate size
    n_layers: int = 61
    n_dense_layers: int = 1  # first N layers use dense FFN instead of MoE
    n_heads: int = 128
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8  # top-k for routing
    route_scale: float = 2.5
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    dtype: str = "bfloat16"
    max_seq_len: int = 4096
    rope_theta: float = 10000.0

    # Derived (computed in __post_init__)
    qk_head_dim: int = field(init=False)
    head_dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.v_head_dim

    @staticmethod
    def from_json(path: Union[str, Path]) -> "ModelArgs":
        """Load model configuration from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return ModelArgs(**{k: v for k, v in data.items() if k in ModelArgs.__dataclass_fields__})

    def to_torch_dtype(self) -> torch.dtype:
        """Convert the string dtype field to a torch.dtype."""
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(self.dtype, torch.bfloat16)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_freqs(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Precompute complex RoPE frequencies of shape (max_seq_len, dim // 2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional encoding to tensor *x*.

    Args:
        x: (..., seq_len, rope_dim) — the rope portion of queries/keys.
        freqs: (seq_len, rope_dim // 2) complex tensor.
    """
    # Reshape to pairs → complex
    *batch, seq_len, d = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(*batch, seq_len, d // 2, 2))
    freqs = freqs[:seq_len].view(1, seq_len, d // 2) if x.dim() == 3 else freqs[:seq_len].view(1, 1, seq_len, d // 2)
    out = torch.view_as_real(x_complex * freqs).flatten(-2)
    return out.type_as(x)


# ---------------------------------------------------------------------------
# Multi-head Latent Attention (MLA)
# ---------------------------------------------------------------------------

class MLAAttention(nn.Module):
    """Multi-head Latent Attention.

    Compresses the KV cache into low-rank projections to reduce memory
    bandwidth during inference while maintaining full attention capacity.

    The key idea: instead of caching full K and V per head, we cache a single
    low-rank vector ``c_kv`` (of size ``kv_lora_rank``) per token, and expand
    it back to K/V on the fly.
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank

        # --- Query path ---
        # Compress hidden → q_lora_rank, then expand to full Q
        self.q_compress = nn.Linear(args.dim, args.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(args.q_lora_rank)
        # Expand to nope + rope portions for all heads
        self.q_expand = nn.Linear(
            args.q_lora_rank,
            args.n_heads * args.qk_head_dim,
            bias=False,
        )

        # --- KV path (compressed) ---
        # Compress hidden → kv_lora_rank  (this is what we cache)
        self.kv_compress = nn.Linear(args.dim, args.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(args.kv_lora_rank)
        # Expand latent to K-nope and V for all heads
        self.kv_expand = nn.Linear(
            args.kv_lora_rank,
            args.n_heads * (args.qk_nope_head_dim + args.v_head_dim),
            bias=False,
        )
        # Separate projection for the RoPE portion of K (small, shared across heads)
        self.k_rope_proj = nn.Linear(args.dim, args.qk_rope_head_dim, bias=False)

        # --- Output ---
        self.o_proj = nn.Linear(args.n_heads * args.v_head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        # --- Q ---
        q = self.q_expand(self.q_norm(self.q_compress(x)))  # (B, T, n_heads * qk_head_dim)
        q = q.view(bsz, seq_len, self.n_heads, self.qk_head_dim)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rope = apply_rope(q_rope, rope_freqs)
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, H, D_qk)

        # --- KV ---
        c_kv = self.kv_norm(self.kv_compress(x))  # (B, T, kv_lora_rank) — compressed cache
        kv = self.kv_expand(c_kv)  # (B, T, n_heads * (nope + v))
        kv = kv.view(bsz, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # K rope portion — shared across heads, then broadcast
        k_rope = self.k_rope_proj(x)  # (B, T, rope_dim)
        k_rope = apply_rope(k_rope.unsqueeze(2), rope_freqs)  # (B, T, 1, rope_dim)
        k_rope = k_rope.expand(-1, -1, self.n_heads, -1)

        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, T, H, D_qk)

        # --- Attention ---
        # Transpose to (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.qk_head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).type_as(q)
        out = torch.matmul(attn, v)  # (B, H, T, D_v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Feed-Forward Networks
# ---------------------------------------------------------------------------

class DenseFFN(nn.Module):
    """Standard SwiGLU FFN used in dense (non-MoE) layers."""

    def __init__(self, dim: int, inter_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter_dim, bias=False)
        self.up_proj = nn.Linear(dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Expert(nn.Module):
    """A single MoE expert — small SwiGLU FFN."""

    def __init__(self, dim: int, inter_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter_dim, bias=False)
        self.up_proj = nn.Linear(dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    """Top-k gating network for routed experts."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.route_scale = args.route_scale
        self.gate = nn.Linear(args.dim, args.n_routed_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch * seq_len, dim)

        Returns:
            weights: (batch * seq_len, n_activated) — normalized gating weights
            indices: (batch * seq_len, n_activated) — expert indices
            aux_loss: scalar — load-balancing auxiliary loss
        """
        logits = self.gate(x) * self.route_scale  # (N, E)
        scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(scores, self.n_activated_experts, dim=-1)
        # Normalize so that selected weights sum to 1
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Load-balancing loss (Switch Transformer style)
        # f_i = fraction of tokens routed to expert i
        # p_i = mean routing probability for expert i
        mask = F.one_hot(topk_indices, self.n_routed_experts).sum(dim=1).float()  # (N, E)
        f = mask.mean(dim=0)  # (E,)
        p = scores.mean(dim=0)  # (E,)
        aux_loss = (f * p).sum() * self.n_routed_experts

        return topk_weights.type_as(x), topk_indices, aux_loss


class MoEFFN(nn.Module):
    """Mixture of Experts feed-forward layer with shared + routed experts."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_activated_experts = args.n_activated_experts
        self.gate = MoEGate(args)

        # Shared expert(s) — always active
        self.shared_experts = nn.ModuleList(
            [Expert(args.dim, args.moe_inter_dim) for _ in range(args.n_shared_experts)]
        )

        # Routed experts — selected via top-k gating
        self.routed_experts = nn.ModuleList(
            [Expert(args.dim, args.moe_inter_dim) for _ in range(args.n_routed_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)

        Returns:
            output: (B, T, D)
            aux_loss: scalar
        """
        bsz, seq_len, dim = x.shape
        flat = x.view(-1, dim)  # (N, D)

        # Shared expert contribution
        shared_out = sum(expert(flat) for expert in self.shared_experts)

        # Gating
        weights, indices, aux_loss = self.gate(flat)  # weights/indices: (N, k)

        # Routed expert contribution (simple loop — production would use grouped GEMM)
        routed_out = torch.zeros_like(flat)
        for i in range(self.n_activated_experts):
            expert_idx = indices[:, i]  # (N,)
            w = weights[:, i].unsqueeze(-1)  # (N, 1)
            for eidx in expert_idx.unique():
                mask = expert_idx == eidx
                if mask.any():
                    routed_out[mask] += w[mask] * self.routed_experts[eidx.item()](flat[mask])

        out = (shared_out + routed_out).view(bsz, seq_len, dim)
        return out, aux_loss


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """A single transformer layer with MLA attention and either dense or MoE FFN."""

    def __init__(self, layer_idx: int, args: ModelArgs) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(args.dim)
        self.attn = MLAAttention(args)
        self.ffn_norm = RMSNorm(args.dim)

        # First n_dense_layers use dense FFN; the rest use MoE
        if layer_idx < args.n_dense_layers:
            self.ffn: nn.Module = DenseFFN(args.dim, args.inter_dim)
            self.is_moe = False
        else:
            self.ffn = MoEFFN(args)
            self.is_moe = True

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Attention with residual
        x = x + self.attn(self.attn_norm(x), rope_freqs, mask)

        # FFN with residual
        ffn_input = self.ffn_norm(x)
        if self.is_moe:
            ffn_out, aux_loss = self.ffn(ffn_input)
        else:
            ffn_out = self.ffn(ffn_input)
            aux_loss = torch.tensor(0.0, device=x.device)
        x = x + ffn_out
        return x, aux_loss


# ---------------------------------------------------------------------------
# Full Transformer Model
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """Zensei Transformer — DeepSeek-V3 architecture with expanded Japanese vocabulary."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.tok_emb = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(i, args) for i in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.dim)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(args.qk_rope_head_dim, args.max_seq_len, args.rope_theta),
            persistent=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, T) integer token ids.
            mask: optional causal mask of shape (1, 1, T, T).

        Returns:
            logits: (B, T, vocab_size)
            total_aux_loss: scalar — accumulated MoE load-balancing loss.
        """
        bsz, seq_len = tokens.shape
        x = self.tok_emb(tokens)

        # Build causal mask if not provided
        if mask is None:
            mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=tokens.device, dtype=x.dtype
            )
            mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        total_aux_loss = torch.tensor(0.0, device=tokens.device)
        for layer in self.layers:
            x, aux_loss = layer(x, self.rope_freqs, mask)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, total_aux_loss

    @staticmethod
    def from_config(path: Union[str, Path]) -> "Transformer":
        """Instantiate a Transformer from a JSON config file."""
        args = ModelArgs.from_json(path)
        return Transformer(args)
