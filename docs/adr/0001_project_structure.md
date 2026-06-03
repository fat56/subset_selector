# ADR 0001: Long-Running Experiment Structure

## Status

Accepted.

## Context

This project is a research codebase whose first risk is not implementation complexity, but losing track of hypotheses, baselines, runs, metrics, and decisions. The initial research report also makes Stage 1 a hard gate: register/readout embedding distance must correlate with 3DGS quality before selector training is justified.

## Decision

Use a VFM_GS-style structure:

- `configs/experiments` stores reproducible experiment inputs.
- `docs/experiments/<id>` stores proposal, runbook, results, and review.
- `runs/<experiment>/<run_id>` stores local run manifests, copied config snapshots, lightweight metrics, selected indices, and notes.
- `docs/registry` stores global CSV ledgers and metric schema.
- `src/vggt_omega_selector` starts as a thin package and grows only when a stage needs code.

## Consequences

Every iteration has a durable local record. Heavy outputs remain outside git or under ignored artifact folders, but their paths and checksums are recorded in manifests. This keeps the repository light while preserving reproducibility.

