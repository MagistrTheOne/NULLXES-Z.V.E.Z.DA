"""Deterministic tensor seeds for checkpoint initialization."""

from __future__ import annotations

import hashlib
import struct

import torch


def derive_seed(
    global_seed: int,
    *,
    model_name: str,
    tensor_name: str,
    layer_id: int = -1,
    expert_id: int = -1,
    shard_id: int = 0,
) -> int:
    payload = f"{global_seed}|{model_name}|{tensor_name}|{layer_id}|{expert_id}|{shard_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return struct.unpack("<Q", digest[:8])[0] % (2**63 - 1)


def init_normal_tensor(
    shape: tuple[int, ...],
    *,
    std: float,
    mean: float,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).mul_(std).add_(mean).to(dtype)


def init_ones_tensor(shape: tuple[int, ...], *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.ones(shape, dtype=dtype)
