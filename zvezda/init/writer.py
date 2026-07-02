"""Streaming BF16 checkpoint writer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from zvezda.init.seeds import derive_seed, init_normal_tensor, init_ones_tensor
from zvezda.init.tensor_spec import TensorSpec, iter_tensor_specs
from zvezda.model.config import load_model_config


def _sha256_tensor(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256(tensor.detach().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _materialize_tensor(spec: TensorSpec, *, global_seed: int, model_name: str, shard_id: int, dtype: torch.dtype) -> torch.Tensor:
    seed = derive_seed(
        global_seed,
        model_name=model_name,
        tensor_name=spec.name,
        layer_id=spec.layer_id,
        expert_id=spec.expert_id,
        shard_id=shard_id,
    )
    if spec.init_kind == "ones":
        return init_ones_tensor(spec.shape, dtype=dtype)
    if spec.init_kind == "zeros_fp32":
        return torch.zeros(spec.shape, dtype=torch.float32)
    return init_normal_tensor(spec.shape, std=spec.std, mean=0.0, seed=seed, dtype=dtype)


def write_init_checkpoint(
    config: dict[str, Any],
    output_dir: Path,
    *,
    global_seed: int,
    shard_count: int = 8,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")

    cfg = load_model_config(config)
    specs = iter_tensor_specs(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards: list[list[TensorSpec]] = [[] for _ in range(shard_count)]
    for index, spec in enumerate(specs):
        shards[index % shard_count].append(spec)

    manifest_tensors: list[dict[str, Any]] = []
    for shard_id, shard_specs in enumerate(shards):
        tensors: dict[str, torch.Tensor] = {}
        for spec in shard_specs:
            tensor = _materialize_tensor(spec, global_seed=global_seed, model_name=cfg.name, shard_id=shard_id, dtype=dtype)
            if spec.init_kind == "zeros_fp32":
                tensors[spec.name] = tensor
            else:
                tensors[spec.name] = tensor
            manifest_tensors.append(
                {
                    "tensor_name": spec.name,
                    "tensor_shape": list(spec.shape),
                    "tensor_dtype": str(dtype).replace("torch.", ""),
                    "tensor_init_rule": spec.init_kind,
                    "tensor_sha256": _sha256_tensor(tensor),
                    "shard_id": shard_id,
                    "shard_count": shard_count,
                }
            )
        save_file(tensors, str(output_dir / f"model-{shard_id:05d}-of-{shard_count:05d}.safetensors"))

    manifest = {
        "model_name": cfg.name,
        "global_seed": global_seed,
        "dtype": str(dtype).replace("torch.", ""),
        "shard_count": shard_count,
        "tensor_count": len(specs),
        "optimizer_state": False,
        "tensors": manifest_tensors,
    }
    with (output_dir / "checkpoint_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
