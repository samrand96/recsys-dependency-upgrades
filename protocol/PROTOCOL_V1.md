# Full-study protocol v1

**Approved by the researcher on 13 November 2026.** The `protocol-v1` tag freezes this document, the probe policy and the held-out manifests before any held-out upgrade outcome exists. It extends [PROTOCOL.md](PROTOCOL.md) (protocol v0), which stays in force unchanged except where this document says otherwise.

## Gate G1 decision

On 13 November 2026 the researcher chose **Path B**: an empirical study of how dependency upgrades change recommender rankings, why, and whether the change harms offline quality. Working title: *Silent Ranking Regressions Under Dependency Upgrades: An Empirical Study of Recommender Pipelines*. Evidence at the gate is in [G1_MEMO.md](G1_MEMO.md): no harmful silent case, two attributed silent mechanisms, several visible failures.

Research questions for the full study:
- **RQ1:** What outcomes arise under eligible upgrades (install, runtime, behavioural, quality), with denominators?
- **RQ2:** Which pipeline stages and user activity groups are affected, and by which mechanisms?
- **RQ3 (light):** Which cheap check separates benign list churn from material quality loss? Measured on held-out cases and on synthetic positive controls, reported separately.

## Unchanged from protocol v0

Dataset preparation and split rules, relevance, candidates, seen masks, contracts, metrics, activity groups, loss definition, the tolerance set δ ∈ {0.002, 0.005, 0.010}, verdict rules, interval methods, analysis seed 713, retraining seeds 17, 29, 43, 59, 71, one-thread same-host runs, A/A before A/B, stage hashing and replay diagnosis.

The pilot gives no reason to change the tolerances: development losses lay between −0.0000335 and 0.0000477, far inside the smallest tolerance. Five seeds are kept. A case whose primary retraining interval is `inconclusive` at δ = 0.005 is rerun once with ten additional fixed seeds (83, 97, 101, 113, 127, 131, 149, 157, 163, 173), and both analyses are reported.

## Development and held-out cases

- **Development:** every transition in `cases/systematic_single.yaml` and `cases/systematic_snapshot.yaml` (N1–N3, S1–S4, D1–D2, B1), on every pipeline. These may inform design and are never reported as held-out evidence.
- **Held-out:** every transition in `cases/systematic_single_heldout.yaml` and `cases/systematic_snapshot_heldout.yaml`. They were selected on 13 November 2026 from PyPI release metadata only, before any of them was built or run.

## Selection rules for the held-out collections

**Single-dependency collection.** Window: releases uploaded from 1 January 2024 to 31 October 2026. For NumPy, SciPy, pandas and PyTorch, take every pair of consecutive minor releases on Python 3.12 whose newer side falls in the window and which is not already a development case. Use the latest patch of each minor inside the window. One direct dependency moves; the others are held at versions the resolver accepts on both sides. scikit-learn and numba are excluded because no declared pipeline calls them; if a later pipeline does, their transitions are added under a new protocol version.

**Snapshot collection.** The unpinned requirement set `numpy, scipy, pandas, implicit` resolved with an upload cutoff at 00:00 UTC on the first day of each quarter from 2024-01-01 to 2026-10-01. Each pair of consecutive snapshots is one bundle-level transition. If two consecutive snapshots resolve to identical locks, the transition is recorded as "no change" and kept in the denominator.

Every exclusion (resolve, build or import failure, inapplicable pipeline) is recorded with its reason and stays in the eligibility flow.

## Pipelines and data

P0–P4 as declared in protocol v0 and [pipelines_p3_p4.md](pipelines_p3_p4.md). LensKit and Cornac pipelines, and the MovieLens 20M and Amazon Reviews 2023 datasets, are each declared in their own file under `protocol/` before their first ranking run, using data counts and library documentation only. PyTorch transitions are screened and run in Phase 3 on rented compute; CPU and GPU results are never mixed.

## Detection policy

[probe_policy_v1.md](probe_policy_v1.md), approved as drafted. Synthetic positive controls are reported in separate tables from real cases.

## Reporting

Systematic and issue-guided collections are never pooled. Single-dependency and snapshot collections are tabulated separately. The unit for cross-case statements is the upgrade family and the mechanism. Null, improved and failed outcomes are all reported.
