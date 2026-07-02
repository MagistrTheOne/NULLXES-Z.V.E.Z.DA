"""Tensor inventory for streaming checkpoint initialization."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from zvezda.model.config import ModelConfig
from zvezda.model.transformer import ZvezdaTransformer


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    init_kind: str
    std: float
    layer_id: int = -1
    expert_id: int = -1


def iter_tensor_specs(cfg: ModelConfig) -> list[TensorSpec]:
    with torch.device("meta"):
        model = ZvezdaTransformer(cfg)
    layer_scale = (2 * cfg.num_layers) ** -0.5
    std = 0.02 * layer_scale

    specs: list[TensorSpec] = []
    for name, param in model.named_parameters():
        layer_id = -1
        expert_id = -1
        if ".layers." in name:
            layer_id = int(name.split(".layers.")[1].split(".")[0])
        if "routed_experts." in name:
            expert_id = int(name.split("routed_experts.")[1].split(".")[0])
        elif "shared_experts." in name:
            expert_id = int(name.split("shared_experts.")[1].split(".")[0])

        if name.endswith("load_balance_bias"):
            init_kind = "zeros_fp32"
            specs.append(TensorSpec(name, tuple(param.shape), init_kind, 0.0, layer_id, expert_id))
        elif "norm" in name and name.endswith(".weight"):
            init_kind = "ones"
            specs.append(TensorSpec(name, tuple(param.shape), init_kind, 0.0, layer_id, expert_id))
        elif "router" in name and name.endswith(".weight"):
            init_kind = "normal"
            specs.append(TensorSpec(name, tuple(param.shape), init_kind, 0.001, layer_id, expert_id))
        elif "embed" in name:
            init_kind = "normal"
            specs.append(TensorSpec(name, tuple(param.shape), init_kind, 0.02, layer_id, expert_id))
        else:
            init_kind = "normal"
            specs.append(TensorSpec(name, tuple(param.shape), init_kind, std, layer_id, expert_id))
    return specs
