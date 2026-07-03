"""Z.V.E.Z.D.A decoder-only causal transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from zvezda.model.attention import GroupedQueryAttention, RMSNorm
from zvezda.model.config import ModelConfig, load_model_config
from zvezda.model.ffn import SwiGLUFFN
from zvezda.model.moe import MoELayer
from zvezda.model.mtp import MTPHeads
from zvezda.model.rope import build_rope_cache


@dataclass
class ModelOutput:
    logits: torch.Tensor
    mtp_logits: list[torch.Tensor]
    metrics: dict[str, torch.Tensor]


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rmsnorm_eps)
        self.self_attn = GroupedQueryAttention(cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rmsnorm_eps)
        self.is_moe = layer_idx >= cfg.num_layers - cfg.moe_layers and cfg.moe_layers > 0
        if self.is_moe:
            self.mlp = MoELayer(
                cfg.hidden_size,
                cfg.expert_hidden,
                cfg.routed_experts,
                cfg.shared_experts,
                cfg.routed_top_k,
            )
        else:
            self.mlp = SwiGLUFFN(cfg.hidden_size, cfg.dense_ffn_hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        metrics: dict[str, torch.Tensor] = {}
        h = self.input_layernorm(x)
        h = self.self_attn(h, cos, sin)
        x = x + h
        h = self.post_attention_layernorm(x)
        if self.is_moe:
            h, moe_metrics = self.mlp(h)
            metrics.update(moe_metrics)
        else:
            h = self.mlp(h)
        return x + h, metrics


class ZvezdaTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rmsnorm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.mtp = MTPHeads(cfg.hidden_size, cfg.vocab_size, cfg.mtp_depth, cfg.rmsnorm_eps) if cfg.mtp_enabled else None

    @classmethod
    def from_yaml_config(cls, config: dict) -> "ZvezdaTransformer":
        return cls(load_model_config(config))

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        mtp_weight: float | None = None,
    ) -> ModelOutput | tuple[torch.Tensor, dict[str, float]]:
        if mtp_weight is not None:
            return self.loss(input_ids, mtp_weight=mtp_weight)
        return self._forward_logits(input_ids)

    def _forward_logits(self, input_ids: torch.Tensor) -> ModelOutput:
        batch, seq_len = input_ids.shape
        device = input_ids.device
        x = self.embed_tokens(input_ids)
        cos, sin = build_rope_cache(seq_len, self.cfg.head_dim, self.cfg.rope_theta, device, x.dtype)
        metrics: dict[str, torch.Tensor] = {}
        for layer in self.layers:
            x, layer_metrics = layer(x, cos, sin)
            metrics.update(layer_metrics)
        x = self.norm(x)
        logits = self.lm_head(x)
        mtp_logits = self.mtp(x) if self.mtp is not None else []
        return ModelOutput(logits=logits, mtp_logits=mtp_logits, metrics=metrics)

    def loss(
        self,
        input_ids: torch.Tensor,
        *,
        mtp_weight: float = 0.2,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        out = self._forward_logits(input_ids)
        shift_logits = out.logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]
        lm_loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        total = lm_loss
        log: dict[str, float] = {"lm_loss": float(lm_loss.detach())}
        for depth, mtp_logits in enumerate(out.mtp_logits, start=1):
            offset = depth
            if offset >= input_ids.size(1):
                continue
            pred = mtp_logits[:, :-offset, :].float()
            target = input_ids[:, offset:]
            mtp_loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1))
            total = total + mtp_weight * mtp_loss
            log[f"mtp_{depth}_loss"] = float(mtp_loss.detach())

        log["total_loss"] = float(total.detach())
        if "router_entropy" in out.metrics:
            log["router_entropy"] = float(out.metrics["router_entropy"].detach())
        return total, log
