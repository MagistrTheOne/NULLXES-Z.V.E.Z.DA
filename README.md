# NULLXES-Z.V.E.Z.DA

Continuous Cognitive MoE System, NULLXES Research Division.

Canonical document: `docs/RD-2026-001.md`.

## Current Scope

This repository currently contains the P0 production-machine specification only:

- Final model parameter envelope: 305,996,722,176 total parameters.
- Active training path: 17,445,384,192 parameters/token including LM head and MTP.
- Production config: `configs/zvezda-300b.yaml`.
- B300 parallelism config: `configs/parallelism-b300.yaml`.

No proxy stack, local execution harness, mock data, fake metrics, or stub model code is included.

## Repository Layout

```text
configs/      Production configuration files.
docs/         RD specifications and design records.
experiments/  Gate records: hypothesis, config, seed, metrics, cost, result.
scripts/      Operational scripts only when fully specified.
zvezda/       Future production package namespace.
```

## Execution Rule

Any training job above 500 GPU-hours requires explicit operator confirmation with a cost estimate.

Any benchmark not run is recorded as `not measured`.
