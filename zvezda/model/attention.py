"""Grouped-query attention with optional sliding window."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from zvezda.model.config import ModelConfig, is_sliding_window_layer
from zvezda.model.rope import apply_rope


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.sliding_window = cfg.local_window if is_sliding_window_layer(layer_idx) else None

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos[:seq_len], sin[:seq_len])
        k = apply_rope(k, cos[:seq_len], sin[:seq_len])

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn_mask = self._build_mask(seq_len, x.device, x.dtype, attention_mask)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)

    def _build_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.sliding_window is None and attention_mask is None:
            return None
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
        for i in range(seq_len):
            lower = max(0, i - (self.sliding_window or seq_len) + 1) if self.sliding_window else 0
            mask[i, :lower] = float("-inf")
            mask[i, i + 1 :] = float("-inf")
        if attention_mask is not None:
            mask = mask.unsqueeze(0) + attention_mask
        return mask.unsqueeze(0).unsqueeze(0)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight
