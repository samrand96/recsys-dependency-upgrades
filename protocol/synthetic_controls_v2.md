# Synthetic positive controls: analysis version 2

Declared on 15 November 2026, before any version-2 output was generated. It corrects the implementation of [synthetic_controls.md](synthetic_controls.md) (version 1). Version 1 files in `results/synthetic/` and [SYNTHETIC_RESULTS.md](SYNTHETIC_RESULTS.md) are kept unchanged as the record of the first analysis. The paper reports version 2 and states what changed.

## Why a new version

An evidence review on 15 November found three defects in version 1:

1. **Fault isolation.** Version 1 rebuilt every user's list with the reference ranker (`worker/ranking.native_topk`). Where a pipeline's own native list differs from the reference ranker's (tie order, mainly), users outside the fault's declared scope changed too. For F4, 90 of 97 base cases had changed users outside the 1% of reversed rows, against the declaration ("all other users keep their native lists").
2. **Resume.** Saved records store tolerance keys as strings, and the summary indexes them as floats, so a cached run could not be summarised again.
3. **Cohort.** `base_cases()` is a rule over `results/compare/`. PyTorch and MovieLens 20M cases added later changed what the rule returns (107 instead of 97).

The researcher asked on 15 November for the work to be rechecked, rerun where needed, and finished. This version is that correction. Tolerances, the verdict rule, the panels, the baselines, the primary detector (activity, 300 users, δ = 0.005), the fault definitions and the random selections are unchanged from version 1.

## What changes

**Cohort.** The 97 base cases of version 1, pinned by name in [synthetic_cohort_v1.json](synthetic_cohort_v1.json).

**Isolated faults.** For each fault, user and cutoff (10 and 20), the reference ranker is run twice: on the unchanged new-side inputs (R0) and on the faulted inputs (R1).
- If R1 equals R0, the fault does not change this user's reference ranking, and the user keeps the **saved native list** exactly.
- Otherwise the user's list becomes R1.
- The listed scores are read from the faulted score matrix for every user.

So a list changes only where the fault itself changes the reference ranking. The faulted score matrix (F1 permuted columns, F3 rounded scores) is saved for all users, as in version 1, because those faults act on scores.

The reference ranker depends on the fault:
- **F1 and F2** do not create ties themselves. They use a *tie-faithful* ranker: descending score, with exact ties resolved in the user's saved native order (items of the native list by their native position, then all other items in catalogue order). With unchanged inputs it reproduces every native list that is consistent with the scores, so the reconstruction adds no tie reordering of its own. The number of users for whom it does not reproduce the native top-10 list is recorded per case.
- **F3** creates ties by rounding. How the pipeline-independent ranker of version 1 (`numpy.argpartition`, then `numpy.argsort`) orders them is part of the declared fault, so F3 keeps that ranker for R0 and R1.
- **F4** replaces exactly the reversed rows, ranked in ascending order by the version-1 ranker. Every other row keeps its native list.

One limit remains and is inherent: for a user the fault does touch, the new list comes from a reference ranker, not from the pipeline's own ranker, which cannot be rerun in the orchestrator.

**Amendment, 15 November, before the cohort run.** The first draft of this declaration used the version-1 ranker for all four faults. A single trial case (N4-P3-svd, since deleted from the output folder and rerun) showed, through the F0 control, that this ranker's tie order alone changes 230 of 1,007 users for Surprise SVD, whose clipped predictions tie heavily, and that this would still leak into F1 for the users F1 touches. The tie-faithful ranker for F1 and F2 was added for that reason. No other version-2 result had been produced or seen.

**No-fault control (F0).** The version-1 full rebuild applied with no fault. It measures what reconstruction alone does: how many users' lists change against the native lists, and what the full reference labels it. F0 is reported in its own row and is never counted as a positive or as a benign control.

**Serialisation.** Tolerance keys are written as strings and converted back on load. A test covers the round trip.

## Reported

As in version 1, per detector and δ: recall on synthetic positives, alarms on the 97 real A/A controls, alarms on the real A/B cases the reference finds `equivalent`, and the paired-user request count. New in version 2:
- the footprint of each fault: base cases with changed lists, median and maximum users changed, full-population loss;
- the F0 control's labels and alarms;
- recall per pipeline, because positives from the same pipeline are not independent.

Positives are still synthetic outputs on real held-out results. They are reported apart from the real upgrade outcomes and say nothing about how often real upgrades are harmful.

## Commands

```
uv run --locked python -m silentrank.detect.synthetic_v2            # run or resume, then write results/synthetic_v2/summary.json
uv run --locked python -m silentrank.detect.synthetic_v2 --shard 0/4  # one slice of the cohort, for parallel runs
```
