# MovieLens 20M: second dataset declaration

Declared on 14 November 2026, before any MovieLens 20M ranking run, as [PROTOCOL_V1.md](PROTOCOL_V1.md) requires for new datasets. The `dataset-ml20m-v1` tag freezes this file, `data/prepare_ml20m.py` and `data/ml20m/manifest.json`. All other rules of protocols v0 and v1 apply unchanged. Results are reported separately from MovieLens 1M and never pooled with it.

## Purpose

Test whether the pilot's mechanisms, and their lack of quality impact, hold on a dataset about 20 times larger, with about 3.4 times the catalogue (12,388 vs 3,655 training items).

## Data

[MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) from GroupLens (archive SHA-256 `96f243c3…`, publisher MD5 `cd245b17…`), prepared once with `uv run --locked python data/prepare_ml20m.py`. The archive is downloaded with resume (`curl -C -`) and checked by the script. The data stay local under the GroupLens terms.

- **Split:** the protocol-v0 global chronological rule, unchanged. The 80/10/10 target meets the 1,500-user minimum with 2,834 warm relevant test users. Training has 112,466 users, 12,388 items and 16,000,210 events, with no duplicate pairs.
- **Ratings:** half-star ratings are stored as their integer part. This keeps the relevance rule (rating ≥ 4) exact. It changes explicit rating values, so only pipelines that use relevant events alone run here: P1 (reference item-kNN) and P2 (`implicit`).
- **Evaluation sample:** to keep per-run score matrices manageable, a fixed sample is drawn with `numpy.random.Generator(PCG64(713))`: 2,000 of the 2,834 warm relevant test users, and 2,000 of the 4,894 warm relevant validation users. Test rows are kept for sampled test users only. Validation rows are kept for users in either sample, so every sampled test user keeps its full seen history. The sample was drawn by rule from counts only, before any run.
- Prepared artifact `ml20m.npz`, SHA-256 `fdb17621…` (recorded in the manifest).

## Cases (fixed)

| Case | Transition | Why |
|---|---|---|
| N1-P1-float32-ml20m | NumPy 1.26.4 → 2.0.2 (dev) | tie mechanism M1 |
| N1-P2-bm25-ml20m | NumPy 1.26.4 → 2.0.2 (dev) | `log1p` mechanism M4 |
| N2-P2-als-ml20m | NumPy 2.0.2 → 2.1.3 (dev) | OpenBLAS kernel mechanism M2 |
| Q2-P1-float32-ml20m | snapshot 2024-04 → 2024-07 (held-out) | M1 in a real bundle upgrade |
| Q3-P2-als-ml20m | snapshot 2024-07 → 2024-10 (held-out) | M2 in a real bundle upgrade |
| S5-P1-float32-ml20m | SciPy 1.15.3 → 1.16.3 (held-out) | no-change control |
| N4-P2-cosine-ml20m | NumPy 2.2.6 → 2.3.5 (held-out) | no-change control |

Same grids, seeds, tolerances and verdict rules as on MovieLens 1M. Primary host only (AVX-512), so M2 is expected to be observable.
