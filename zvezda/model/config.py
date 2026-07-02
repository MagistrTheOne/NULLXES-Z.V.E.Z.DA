"""Model configuration loaded from Z.V.E.Z.D.A YAML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    dense_ffn_hidden: int
    context_length: int
    local_window: int
    rope_theta: float
    tie_embeddings: bool
    rmsnorm_eps: float
    mtp_enabled: bool
    mtp_depth: int
    moe_layers: int
    routed_experts: int
    shared_experts: int
    expert_hidden: int
    routed_top_k: int

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim


def is_sliding_window_layer(layer_idx: int, period: int = 4, sliding_count: int = 3) -> bool:
    return (layer_idx % period) < sliding_count


def load_model_config(config: dict[str, Any]) -> ModelConfig:
    model = config["model"]
    mtp = model.get("mtp", {})
    moe = model.get("moe", {})
    return ModelConfig(
        name=str(model["name"]),
        vocab_size=int(model["dimensions"]["vocab_size"]),
        hidden_size=int(model["dimensions"]["hidden_size"]),
        num_layers=int(model["dimensions"]["num_layers"]),
        num_attention_heads=int(model["attention"]["num_attention_heads"]),
        num_kv_heads=int(model["attention"]["num_kv_heads"]),
        head_dim=int(model["attention"]["head_dim"]),
        dense_ffn_hidden=int(model["dense_prefix_ffn"]["hidden_size"]),
        context_length=int(model["attention"]["context_length"]),
        local_window=int(model["attention"]["local_window"]),
        rope_theta=float(model["attention"]["position"]["rope_theta"]),
        tie_embeddings=bool(model["dimensions"].get("tie_embeddings", False)),
        rmsnorm_eps=float(model["normalization"]["eps"]),
        mtp_enabled=bool(mtp.get("enabled", False)),
        mtp_depth=int(mtp.get("depth", 0)),
        moe_layers=int(model["dimensions"].get("moe_layers", 0)),
        routed_experts=int(moe.get("routed_experts", 0)),
        shared_experts=int(moe.get("shared_experts", 0)),
        expert_hidden=int(moe.get("expert_hidden_size", 0)),
        routed_top_k=int(moe.get("routed_top_k", 0)),
    )
