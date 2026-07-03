"""Minimal distributed setup for smoke training."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_smoke_distributed() -> tuple[int, int]:
    """Initialize NCCL when launched via torchrun; single-GPU works without it."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    if dist.is_initialized():
        return local_rank, dist.get_world_size()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 or os.environ.get("RANK") is not None:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl")

    return local_rank, dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0
