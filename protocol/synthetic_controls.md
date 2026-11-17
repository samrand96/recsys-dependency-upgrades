# Synthetic positive controls: implementation declaration

Declared on 14 November 2026, before any synthetic control was generated. It fills in the details [probe_policy_v1.md](probe_policy_v1.md) left open for the four frozen faults. It changes no frozen rule: tolerances, verdict rule, panels and baselines are as in protocol v1. Synthetic results are reported in their own tables and are never counted as real upgrade outcomes.

## Base cases

Every held-out ranking case with an accepted inference A/A control and a completed inference A/B comparison: 97 cases across P1, P2, P3, P4 and P6 on 13 held-out transitions. The list is fixed by that rule and can be regenerated from `results/compare/`. Development cases are not used.

Each fault is applied to the **new side's saved inference artifacts** of the base case. The old side's baseline inference run is the reference. Nothing is retrained and no environment is run: the faulted output is written to `results/synthetic/<case>/<fault>/` and evaluated by the frozen evaluator (`orchestrator/evaluation.py`), exactly as a worker output would be.

## Faults

Random choices use `numpy.random.Generator(PCG64(713))`, seeded freshly for each base case and fault, so every case gets the same selection rule.

| ID | Fault | Exact construction |
|---|---|---|
| F1 | Swapped item-id mapping for 1% of items | Choose ⌈1% of the catalogue⌉ item columns (37 of 3,655). Rotate their score columns by one position: item i receives item i+1's scores. Rebuild native top-10 and top-20 from the permuted scores. |
| F2 | Seen-item mask dropped | Rebuild native top-10 and top-20 from the unchanged scores over all catalogue items, ignoring the seen mask. |
| F3 | Scores rounded to 2 decimals before ranking | Round all scores to 2 decimal places (`numpy.round(scores, 2)`) and rebuild the native lists. The rounded scores are also the saved scores. |
| F4 | Reversed sort direction for 1% of users | Choose ⌈1% of evaluation users⌉ (11 of 1,007) and rank their eligible items by ascending score. All other users keep their native lists. |

"Rebuild native lists" uses the pipeline-independent native rule of the reference pipeline (`worker/ranking.native_topk`: `numpy.argpartition`, then `numpy.argsort`), with the seen mask applied unless the fault removes it. The scale of rounding in F3 is deliberately the raw score scale of each model, as declared. F3 is therefore strong for small-scale scores and weak or absent for large-scale ones.

## Labels

A faulted case is a **positive** at tolerance δ only if the full labelled reference (all 1,007 users, paired-user bootstrap, protocol verdict rule) is `harmful` at δ, or the evaluator reports a contract violation. Faults that the reference finds `equivalent` or `inconclusive` are reported as such, not as positives.

## Detectors and measures

As frozen in [probe_policy_v1.md](probe_policy_v1.md):
- **Panels:** uniform, activity, score margin × 100, 300, 1,000 users. An alarm is `harmful` or `contract_failure` on the panel users.
- **Baselines:** exact snapshot (any native top-10 list differs); score tolerance (relative L2 of eligible scores > 1e-6); overlap strict (any top-10 set changes); overlap lenient (mean overlap < 0.99); full labelled check.

Reported per detector and δ:
- recall on synthetic positives;
- alarm rate on the 97 real held-out A/A controls (false alarms);
- alarm rate on the real held-out A/B cases the reference found `equivalent` (benign churn: an alarm here is a review cost, not a harm);
- cost as paired user requests (2 × panel size, or 2 × 1,007 for full checks).

The primary configuration is the activity panel of 300 users at δ = 0.005, named in advance.
