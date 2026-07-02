"""Optimizer configuration for smoke training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    max_grad_norm: float


def build_adamw(params, cfg: OptimizerConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        params,
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )


def mup_lr(hidden_size: int, base_lr: float = 1.0e-4, base_hidden: int = 1024) -> float:
    return base_lr * (base_hidden / hidden_size)
