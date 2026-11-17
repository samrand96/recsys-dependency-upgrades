# Probe and baseline detection policy (protocol v1)

**Status: approved as drafted by the researcher on 13 November 2026 and frozen by the `protocol-v1` tag.** The rules below are binding for every held-out case and must not be retuned after held-out results are seen.

## Basis for the rules

- No development case had a supported harm at any tolerance. Harm recall therefore cannot be estimated from the pilot, and no threshold can be tuned for recall.
- Every accepted inference A/A control was bit-identical (47 of 47). Inference noise is zero, so A/A data cannot calibrate a threshold either.
- The values below are therefore chosen from numerical precision and from the observed development distributions of *benign* change. They were not tuned against outcomes, and must not be retuned after held-out results are seen.

## Development distributions (all development cases, 13 November)

| Comparison | Pairs | With any changed top-10 list | Lowest mean top-10 overlap | Highest share of users changed | Largest absolute score difference | Largest relative L2 score difference | NDCG@10 loss range |
|---|---:|---:|---:|---:|---:|---:|---|
| A/A inference | 47 | 0 | 1.0 | 0 | 0 | 0 | 0 |
| A/B inference | 46 | 5 | 0.997021 | 0.003 | 0 | 0 | 0 |
| A/B retraining (5 seeds) | 52 | 7 | 0.997021 | 0.0848 | 0.416 | 0.00246 | −0.0000335 to 0.0000477 |

Inference changes all had identical scores. Only the ranking of exactly tied scores changed (the NumPy 2.0 partition mechanism). Retraining changes came from that tie mechanism acting at neighbour pruning (P1, P2 BM25) or from the BLAS kernel switch (P2 ALS).

## Frozen rules

**Panel detector (the method under test).**
- Strategies: uniform, proportional by training activity, and baseline score margin, as implemented in `src/silentrank/detect/`. Sizes 100, 300 and 1,000. Selection seed 713.
- Decision: the protocol-v0 verdict rule applied to panel users only, at δ ∈ {0.002, 0.005, 0.010}. `harmful` or `contract_failure` raises an alarm. `equivalent`, `improved` or `inconclusive` raises none, and `inconclusive` is reported as "review required", never as a pass.
- Primary configuration, named in advance: proportional activity panel, 300 users, δ = 0.005. The others are reported as secondary, so no single favourable configuration is picked after the results.

**Simple baselines** (each sees the same artifacts as the panel detector):
- **Exact snapshot:** alarm if the native top-10 lists differ for any user. No threshold.
- **Score tolerance:** alarm if the relative L2 difference of eligible scores exceeds 1e-6. This is about eight times float32 machine epsilon (1.19e-7), set from precision, not from outcomes.
- **Overlap, strict:** alarm if any user's top-10 set changes.
- **Overlap, lenient:** alarm if mean top-10 overlap falls below 0.99. That is below every development value (lowest 0.997), so it is not tuned to them.
- **Full labelled check:** the protocol-v0 verdict on all evaluation users. This is the reference, and also the most expensive baseline.

## Positives for held-out evaluation

With no real harmful case, recall must come from synthetic positive controls, reported separately (BLUEPRINT.md §17). Proposed faults, injected on the new side only:
1. swapped item-id mapping for 1% of items;
2. seen-item mask dropped;
3. scores rounded to 2 decimals before ranking (creates ties);
4. reversed sort direction for 1% of users.

Each is checked against the full labelled reference first. A fault counts as a positive only if that reference says `harmful` or `contract_failure`.
