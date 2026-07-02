"""Smoke training loop for pod validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from zvezda.model.transformer import ZvezdaTransformer
from zvezda.train.dataset import CloudTokenStream
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

    local_rank = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = ZvezdaTransformer.from_yaml_config(model_config).to(device=device, dtype=torch.bfloat16)
    _load_sharded_checkpoint(model, checkpoint_dir)

    if torch.cuda.device_count() > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if dist.is_initialized():
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    tokenizer_bundle = Path(launch["tokenizer_bundle_dir"])
    dataset = CloudTokenStream(
        corpus_files,
        tokenizer_bundle / "tokenizer.json",
        seq_len=smoke.seq_len,
        storage_config=storage_config,
    )
    loader = DataLoader(dataset, batch_size=1)
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
        raw_model = model.module if hasattr(model, "module") else model
        loss, log = raw_model.loss(input_ids, mtp_weight=smoke.mtp_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), smoke.optimizer.max_grad_norm)
        optimizer.step()

        if step % smoke.log_every == 0:
            entry = {"step": float(step), **log}
            metrics_log.append(entry)
            print(json.dumps(entry, sort_keys=True))

        step += 1

    trace_path = output_dir / "smoke_train_trace.json"
    with trace_path.open("w", encoding="utf-8") as handle:
        json.dump({"steps": metrics_log, "config": asdict(smoke)}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"trace": str(trace_path), "steps": step}
