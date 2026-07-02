"""Validation for Z.V.E.Z.D.A architecture configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zvezda.config.accounting import calculate, expected_active_breakdown, expected_breakdown


def _get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _require_equal(errors: list[str], path: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{path}: expected {expected}, got {actual}")


def validate(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    accounting = calculate(config)

    d_model = _get(config, "model.dimensions.hidden_size")
    num_heads = _get(config, "model.attention.num_attention_heads")
    qk_head_dim = _get(config, "model.attention.qk_head_dim", _get(config, "model.attention.head_dim"))
    v_head_dim = _get(config, "model.attention.v_head_dim", qk_head_dim)
    routed_experts = _get(config, "model.moe.routed_experts")
    routed_top_k = _get(config, "model.moe.routed_top_k")

    if num_heads * qk_head_dim != d_model and qk_head_dim == v_head_dim:
        errors.append("attention geometry invalid: num_attention_heads * head_dim must equal hidden_size")
    if num_heads * v_head_dim != d_model and qk_head_dim != v_head_dim:
        # MiMo-style asymmetric QK/V heads intentionally do not make QK equal to hidden size.
        if num_heads * v_head_dim != 2 * d_model:
            errors.append("asymmetric attention geometry must explicitly justify output width")
    if routed_top_k >= routed_experts:
        errors.append("moe.routed_top_k must be smaller than moe.routed_experts")

    param_accounting = _get(config, "model.parameter_accounting", {})
    _require_equal(errors, "model.parameter_accounting.total_parameters", param_accounting.get("total_parameters"), accounting.total)
    _require_equal(
        errors,
        "model.parameter_accounting.active_parameters_per_token_training_lm_mtp",
        param_accounting.get("active_parameters_per_token_training_lm_mtp"),
        accounting.active_training_lm_mtp,
    )
    _require_equal(
        errors,
        "model.parameter_accounting.active_parameters_per_token_transformer_body",
        param_accounting.get("active_parameters_per_token_transformer_body"),
        accounting.active_transformer_body,
    )

    for key, expected in expected_breakdown(accounting).items():
        _require_equal(errors, f"parameter_breakdown.{key}", _get(config, f"parameter_breakdown.{key}"), expected)

    active_breakdown = _get(config, "active_parameter_breakdown")
    if isinstance(active_breakdown, Mapping):
        for key, expected in expected_active_breakdown(accounting).items():
            _require_equal(errors, f"active_parameter_breakdown.{key}", active_breakdown.get(key), expected)

    _require_equal(errors, "kv_cache.elements_per_token_per_layer", _get(config, "kv_cache.elements_per_token_per_layer"), accounting.kv_elements_per_token_per_layer)
    _require_equal(errors, "kv_cache.elements_per_token_all_layers", _get(config, "kv_cache.elements_per_token_all_layers"), accounting.kv_elements_per_token_all_layers)
    _require_equal(errors, "kv_cache.elements_per_sequence_all_layers", _get(config, "kv_cache.elements_per_sequence_all_layers"), accounting.kv_elements_per_sequence_all_layers)
    _require_equal(errors, "kv_cache.fp8_bytes_per_sequence", _get(config, "kv_cache.fp8_bytes_per_sequence"), accounting.fp8_kv_bytes_per_sequence)
    _require_equal(errors, "kv_cache.bf16_bytes_per_sequence", _get(config, "kv_cache.bf16_bytes_per_sequence"), accounting.bf16_kv_bytes_per_sequence)

    return errors
