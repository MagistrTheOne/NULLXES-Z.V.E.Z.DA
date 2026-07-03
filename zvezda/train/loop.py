"""Smoke training loop for pod validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader

from zvezda.model.transformer import TransformerBlock, ZvezdaTransformer
from zvezda.train.dataset import CloudTokenStream
from zvezda.train.distributed import is_main_process, setup_smoke_distributed
from zvezda.train.optimizer import OptimizerConfig, build_adamw, mup_lr


@dataclass
class SmokeTrainConfig:
    max_steps: int
    seq_len: int
    log_every: int
    save_every: int
    mtp_weight: float
    optimizer: OptimizerConfig


def _load_sharded_checkpoint(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(checkpoint_dir.glob("model-*-of-*.safetensors")):
        state.update(load_file(str(shard)))
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        raise RuntimeError(f"checkpoint missing tensors: {missing}")
    if unexpected:
        raise RuntimeError(f"checkpoint has unexpected tensors: {unexpected}")


def _wrap_smoke_model(model: ZvezdaTransformer, *, local_rank: int, world_size: int) -> torch.nn.Module:
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda module: isinstance(module, TransformerBlock),
    )
    if world_size < 2:
        return model.to(device=torch.device(f"cuda:{local_rank}"), dtype=torch.bfloat16)

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        use_orig_params=True,
    )


def run_smoke_train(
    model_config: dict[str, Any],
    launch: dict[str, Any],
    *,
    corpus_files,
    storage_config: dict[str, Any] | None,
    checkpoint_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    train_cfg = launch["smoke_train"]
    smoke = SmokeTrainConfig(
        max_steps=int(train_cfg["max_steps"]),
        seq_len=int(train_cfg["seq_len"]),
        log_every=int(train_cfg.get("log_every", 10)),
        save_every=int(train_cfg.get("save_every", 500)),
        mtp_weight=float(train_cfg.get("mtp_weight", 0.2)),
        optimizer=OptimizerConfig(
            lr=float(train_cfg.get("lr", mup_lr(model_config["model"]["dimensions"]["hidden_size"]))),
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        ),
    )

    if not torch.cuda.is_available():
        raise RuntimeError("smoke train requires CUDA")

    local_rank, world_size = setup_smoke_distributed()
    expected_gpus = int(launch.get("pod", {}).get("gpu_count", world_size))
    if world_size != expected_gpus:
        launcher = (
            "python scripts/smoke_train.py"
            if expected_gpus == 1
            else f"torchrun --nproc_per_node={expected_gpus} scripts/smoke_train.py"
        )
        raise RuntimeError(
            f"smoke train expects {expected_gpus} GPU process(es) ({launcher}); got world_size={world_size}"
        )

    device = torch.device(f"cuda:{local_rank}")
    model = ZvezdaTransformer.from_yaml_config(model_config)
    _load_sharded_checkpoint(model, checkpoint_dir)
    model = _wrap_smoke_model(model, local_rank=local_rank, world_size=world_size)

    tokenizer_bundle = Path(launch["paths"]["tokenizer_bundle_dir"])
    dataset = CloudTokenStream(
        corpus_files,
        tokenizer_bundle / "tokenizer.json",
        seq_len=smoke.seq_len,
        storage_config=storage_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda batch: batch[0],
    )
    optimizer = build_adamw(model.parameters(), smoke.optimizer)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_log: list[dict[str, float]] = []
    step = 0
    data_iter = iter(loader)

    model.train()
    while step < smoke.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch.input_ids.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, log = model(input_ids, mtp_weight=smoke.mtp_weight)
        loss.backward()
        if isinstance(model, FSDP):
            model.clip_grad_norm_(smoke.optimizer.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), smoke.optimizer.max_grad_norm)
        optimizer.step()

        if step % smoke.log_every == 0 and is_main_process():
            entry = {"step": float(step), **log}
            metrics_log.append(entry)
            print(json.dumps(entry, sort_keys=True))

        step += 1

    trace_path = output_dir / "smoke_train_trace.json"
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as handle:
            json.dump({"steps": metrics_log, "config": asdict(smoke)}, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return {"trace": str(trace_path), "steps": step}
