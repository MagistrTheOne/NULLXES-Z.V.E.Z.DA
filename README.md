# NULLXES-Z.V.E.Z.DA

Continuous Cognitive MoE System, NULLXES Research Division.

Canonical document: `docs/RD-2026-001.md`.

## Current Scope

- Diagnostic dense model: `configs/zvezda-15b-dense.yaml` (14.85B, init + smoke train).
- Diagnostic MoE model: `configs/zvezda-100b-a15b-moe.yaml` (after dense gates).
- Init curriculum (no web crawl): `configs/data/init-curriculum-v1.yaml`.
- **RunPod-only storage:** `configs/data/runpod-storage-v0.yaml`.
- Pod launch manifest: `configs/diagnostic/pod-launch-1xb200.yaml` (1×B200 survival smoke, primary).
- Pod launch manifest (2×H200 FSDP): `configs/diagnostic/pod-launch.yaml`.
- B200 parallelism profile: `configs/parallelism-b200-diagnostic.yaml`.
- H200 parallelism profile: `configs/parallelism-h200-diagnostic.yaml`.

VK Cloud / Yandex Cloud: internal reference only (`configs/data/storage-providers-internal.yaml`), not active.

No mock data, stubs, placeholders, fake metrics, or AWS.

## RunPod Launch

1. Upload corpus to **RunPod network volume** under `/workspace/corpus/`.
2. Fill `configs/tokenizer/corpus-manifest.yaml` with `runpod://` URIs and real `sha256` hashes.
3. On pod: `RUNPOD_VOLUME_ROOT=/workspace` (default).
4. Run phases from `configs/diagnostic/pod-launch-1xb200.yaml` (1×B200) or `configs/diagnostic/pod-launch.yaml` (2×H200 FSDP):

**1×B200 180 GB (primary survival smoke — no torchrun):**

```bash
python scripts/pod_preflight.py --launch configs/diagnostic/pod-launch-1xb200.yaml --probe-storage
python scripts/train_tokenizer.py --config configs/tokenizer/zvezda-bpe-152k-diagnostic.yaml \
  --corpus-manifest configs/tokenizer/corpus-manifest.yaml \
  --storage-config configs/data/runpod-storage-v0.yaml \
  --output-dir /workspace/zvezda/artifacts/tokenizer --probe-storage
python scripts/init_checkpoint.py --config configs/zvezda-15b-dense.yaml \
  --init-policy configs/init/init-policy-v0.yaml \
  --output-dir /workspace/zvezda/artifacts/init/zvezda-15b-dense --global-seed 20260703 --shard-count 4
python scripts/pod_preflight.py --launch configs/diagnostic/pod-launch-1xb200.yaml --probe-storage --require-artifacts
python scripts/smoke_train.py --launch configs/diagnostic/pod-launch-1xb200.yaml
```

**2×H200 141 GB (FSDP via torchrun):**

```bash
python scripts/pod_preflight.py --launch configs/diagnostic/pod-launch.yaml --probe-storage
python scripts/train_tokenizer.py --config configs/tokenizer/zvezda-bpe-152k-diagnostic.yaml \
  --corpus-manifest configs/tokenizer/corpus-manifest.yaml \
  --storage-config configs/data/runpod-storage-v0.yaml \
  --output-dir /workspace/zvezda/artifacts/tokenizer --probe-storage
python scripts/init_checkpoint.py --config configs/zvezda-15b-dense.yaml \
  --init-policy configs/init/init-policy-v0.yaml \
  --output-dir /workspace/zvezda/artifacts/init/zvezda-15b-dense --global-seed 20260703 --shard-count 4
python scripts/pod_preflight.py --launch configs/diagnostic/pod-launch.yaml --probe-storage --require-artifacts
torchrun --nproc_per_node=2 scripts/smoke_train.py --launch configs/diagnostic/pod-launch.yaml
```

### Corpus URI format

```yaml
files:
  - segment: ru_curated
    uri: runpod://corpus/init/ru_curated.jsonl
    license: proprietary
    content_hash:
      algorithm: sha256
      value: "<real-sha256-hex>"
```

Placeholder values and template URIs **fail validation** (stubs = fail).

## Repository Layout

```text
configs/      Model, data, tokenizer, pod launch configs.
docs/         RD specifications and design records.
experiments/  Gate records.
scripts/      Tokenizer, init, smoke train, preflight.
zvezda/       Model, train, init, data, tokenizer packages.
```

## Execution Rule

Any training job above 500 GPU-hours requires explicit operator confirmation with a cost estimate.

Any benchmark not run is recorded as `not measured`.
