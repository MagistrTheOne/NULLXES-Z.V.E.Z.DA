"""Deterministic parameter and memory accounting for Z.V.E.Z.D.A configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GIB = 1024**3


@dataclass(frozen=True)
class Accounting:
    attention: int
    dense_prefix_ffn: int
    moe_layers: int
    active_moe: int
    input_embedding: int
    output_head: int
    mtp: int
    rmsnorm_scales: int
    total: int
    active_training_lm_mtp: int
    active_transformer_body: int
    kv_elements_per_token_per_layer: int
    kv_elements_per_token_all_layers: int
    kv_elements_per_sequence_all_layers: int
    fp8_kv_bytes_per_sequence: int
    bf16_kv_bytes_per_sequence: int

    @property
    def fp8_kv_gib_per_sequence(self) -> float:
        return self.fp8_kv_bytes_per_sequence / GIB

    @property
    def bf16_kv_gib_per_sequence(self) -> float:
        return self.bf16_kv_bytes_per_sequence / GIB


def _require_int(config: dict[str, Any], path: str) -> int:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"missing required config key: {path}")
        value = value[part]
    if not isinstance(value, int):
        raise TypeError(f"{path} must be int, got {type(value).__name__}")
    return value


def _optional_int(config: dict[str, Any], path: str, default: int) -> int:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    if not isinstance(value, int):
        raise TypeError(f"{path} must be int, got {type(value).__name__}")
    return value


def calculate(config: dict[str, Any]) -> Accounting:
    """Calculate architecture counts from one Z.V.E.Z.D.A YAML config dict."""

    model = config.get("model")
    if not isinstance(model, dict):
        raise KeyError("missing required config key: model")

    d_model = _require_int(config, "model.dimensions.hidden_size")
    vocab_size = _require_int(config, "model.dimensions.vocab_size")
    num_layers = _require_int(config, "model.dimensions.num_layers")
    dense_prefix_layers = _require_int(config, "model.dimensions.dense_prefix_layers")
    moe_layers = _require_int(config, "model.dimensions.moe_layers")
    if dense_prefix_layers + moe_layers != num_layers:
        raise ValueError("dense_prefix_layers + moe_layers must equal num_layers")

    num_heads = _require_int(config, "model.attention.num_attention_heads")
    num_kv_heads = _require_int(config, "model.attention.num_kv_heads")
    qk_head_dim = _optional_int(
        config,
        "model.attention.qk_head_dim",
        _require_int(config, "model.attention.head_dim"),
    )
    v_head_dim = _optional_int(config, "model.attention.v_head_dim", qk_head_dim)
    context_length = _require_int(config, "model.attention.context_length")

    dense_ffn = _require_int(config, "model.dense_prefix_ffn.hidden_size")
    routed_experts = _require_int(config, "model.moe.routed_experts")
    shared_experts = _require_int(config, "model.moe.shared_experts")
    expert_ffn = _require_int(config, "model.moe.expert_hidden_size")
    routed_top_k = _require_int(config, "model.moe.routed_top_k")
    mtp_depth = _optional_int(config, "model.mtp.depth", 0)

    q_proj = d_model * num_heads * qk_head_dim
    k_proj = d_model * num_kv_heads * qk_head_dim
    v_proj = d_model * num_kv_heads * v_head_dim
    o_proj = d_model * num_heads * v_head_dim
    per_layer_attention = q_proj + k_proj + v_proj + o_proj
    attention = per_layer_attention * num_layers

    per_dense_ffn = 3 * d_model * dense_ffn
    dense_prefix_ffn = per_dense_ffn * dense_prefix_layers

    per_expert = 3 * d_model * expert_ffn
    router = d_model * routed_experts
    per_moe_layer_total = (routed_experts + shared_experts) * per_expert + router
    moe_total = per_moe_layer_total * moe_layers

    per_moe_layer_active = (routed_top_k + shared_experts) * per_expert + router
    active_moe = per_moe_layer_active * moe_layers

    input_embedding = vocab_size * d_model
    output_head = vocab_size * d_model

    mtp = mtp_depth * (d_model * d_model + d_model)
    rmsnorm_scales = (2 * num_layers * d_model) + d_model

    total = attention + dense_prefix_ffn + moe_total + input_embedding + output_head + mtp + rmsnorm_scales
    active_training = attention + dense_prefix_ffn + active_moe + input_embedding + output_head + mtp + rmsnorm_scales
    active_body = active_training - input_embedding - output_head

    kv_per_token_per_layer = (num_kv_heads * qk_head_dim) + (num_kv_heads * v_head_dim)
    kv_per_token_all_layers = kv_per_token_per_layer * num_layers
    kv_per_sequence = kv_per_token_all_layers * context_length

    return Accounting(
        attention=attention,
        dense_prefix_ffn=dense_prefix_ffn,
        moe_layers=moe_total,
        active_moe=active_moe,
        input_embedding=input_embedding,
        output_head=output_head,
        mtp=mtp,
        rmsnorm_scales=rmsnorm_scales,
        total=total,
        active_training_lm_mtp=active_training,
        active_transformer_body=active_body,
        kv_elements_per_token_per_layer=kv_per_token_per_layer,
        kv_elements_per_token_all_layers=kv_per_token_all_layers,
        kv_elements_per_sequence_all_layers=kv_per_sequence,
        fp8_kv_bytes_per_sequence=kv_per_sequence,
        bf16_kv_bytes_per_sequence=2 * kv_per_sequence,
    )


def expected_breakdown(accounting: Accounting) -> dict[str, int]:
    return {
        "attention": accounting.attention,
        "dense_prefix_ffn": accounting.dense_prefix_ffn,
        "moe_layers": accounting.moe_layers,
        "input_output_embeddings": accounting.input_embedding + accounting.output_head,
        "mtp": accounting.mtp,
        "rmsnorm_scales": accounting.rmsnorm_scales,
        "total": accounting.total,
    }


def expected_active_breakdown(accounting: Accounting) -> dict[str, int]:
    return {
        "attention": accounting.attention,
        "dense_prefix_ffn": accounting.dense_prefix_ffn,
        "active_moe": accounting.active_moe,
        "input_output_embeddings_and_lm_head": accounting.input_embedding + accounting.output_head,
        "mtp": accounting.mtp,
        "rmsnorm_scales": accounting.rmsnorm_scales,
        "total_training_lm_mtp": accounting.active_training_lm_mtp,
        "transformer_body": accounting.active_transformer_body,
    }
