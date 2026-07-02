"""Multi-token prediction heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from zvezda.model.attention import RMSNorm


class MTPHeads(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, depth: int, eps: float) -> None:
        super().__init__()
        self.depth = depth
        self.projections = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(depth)])
        self.norms = nn.ModuleList([RMSNorm(hidden_size, eps) for _ in range(depth)])
        self.heads = nn.ModuleList([nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(depth)])

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        logits: list[torch.Tensor] = []
        x = hidden_states
        for projection, norm, head in zip(self.projections, self.norms, self.heads, strict=True):
            x = norm(projection(x))
            logits.append(head(x))
        return logits
