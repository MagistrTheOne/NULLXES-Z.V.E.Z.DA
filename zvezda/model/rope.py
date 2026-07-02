"""Rotary positional embeddings."""

from __future__ import annotations

import torch
from torch import Tensor


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., ::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out
