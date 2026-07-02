"""MoE feed-forward with normalized sigmoid routing."""

from __future__ import annotations

import torch
import torch.nn as nn

from zvezda.model.ffn import SwiGLUFFN


class MoELayer(nn.Module):
    def __init__(self, hidden_size: int, expert_hidden: int, routed_experts: int, shared_experts: int, routed_top_k: int) -> None:
        super().__init__()
        self.routed_top_k = routed_top_k
        self.router = nn.Linear(hidden_size, routed_experts, bias=False)
        self.register_buffer("load_balance_bias", torch.zeros(routed_experts, dtype=torch.float32), persistent=True)
        self.routed_experts = nn.ModuleList(
            [SwiGLUFFN(hidden_size, expert_hidden) for _ in range(routed_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [SwiGLUFFN(hidden_size, expert_hidden) for _ in range(shared_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, seq_len, hidden = x.shape
        flat = x.reshape(-1, hidden)
        router_logits = self.router(flat).float()
        scores = torch.sigmoid(router_logits) + self.load_balance_bias
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topk_scores, topk_idx = torch.topk(scores, k=self.routed_top_k, dim=-1)

        out = torch.zeros_like(flat)
        for slot in range(self.routed_top_k):
            expert_ids = topk_idx[:, slot]
            weights = topk_scores[:, slot].unsqueeze(-1)
            for expert_id, expert in enumerate(self.routed_experts):
                mask = expert_ids == expert_id
                if mask.any():
                    out[mask] = out[mask] + weights[mask] * expert(flat[mask])

        for expert in self.shared_experts:
            out = out + expert(flat)

        utilization = torch.bincount(topk_idx.reshape(-1), minlength=len(self.routed_experts)).float()
        utilization = utilization / topk_idx.numel()
        metrics = {
            "router_entropy": -(scores * scores.clamp_min(1e-9).log()).sum(dim=-1).mean(),
            "expert_utilization": utilization,
        }
        return out.reshape(batch, seq_len, hidden), metrics
