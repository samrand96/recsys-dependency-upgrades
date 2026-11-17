# P6 Cornac pipeline declaration

Declared on 13 November 2026, after the `protocol-v1` tag and before any P6 ranking run, as [PROTOCOL_V1.md](PROTOCOL_V1.md) requires. It was written from Cornac 3.1.0's source and documentation only. Everything else follows protocols v0 and v1 unchanged.

P6 is the second independent recommender implementation planned in WP2.7. (P5 stays reserved for RecBole.)

## Why Cornac

Read from the 3.1.0 source, not from any run:
- `Recommender.rank` selects the top k with `numpy.argpartition` and `numpy.argsort`, the functions whose tie behaviour changed in NumPy 2.0.
- `EASE.fit` inverts a dense item-item matrix with `numpy.linalg.inv`, which runs in the BLAS/LAPACK library bundled with NumPy.
- `ItemKNN` builds its inputs with SciPy sparse matrices.

## Environments

Group `cornac`: the NumPy, SciPy and pandas pins of the systematic single-dependency pairs plus `cornac==3.1.0`, in the source runner image, with the upload cutoff `exclude-newer = 2026-11-13T00:00:00Z`.

Cornac 3.1.0 requires `numpy>2.0.0`. Pairs with a NumPy 1.26 side (N1, S1, S2, D0) are expected to fail resolution and are recorded as such. pandas transitions are not applicable: Cornac does not depend on pandas. Snapshot transitions do not include Cornac.

## Configuration

- **Input:** all observed training ratings as `(user, item, float rating)` tuples in frozen row order, via `Dataset.from_uir`.
- **P6-ease:** `EASE(lamb=λ)`, other settings at library defaults (`posB=True`). Grid λ ∈ (100, 500, 1000), in that order.
- **P6-knn:** `ItemKNN(k=n, num_threads=1)`, other settings at library defaults (cosine similarity, no mean centring option, no weighting). Grid n ∈ (20, 50, 100), in that order.
- Both models are deterministic. Retraining "seeds" are repetitions, as for item-kNN and SAR.
- **Scores:** `model.score(user_idx)` over all items, mapped to catalogue order.
- **Native ranking:** `model.rank(user_idx, item_indices=<unseen items in Cornac index order>, k=N)` for N = 10 and 20, the call Cornac's own ranking evaluation makes.
- **Inference mode:** the old side's parameters are loaded into a fresh model attached to the same dataset without training: `B` for EASE; similarity matrix, centred rating matrix and user means for ItemKNN.
- **Cold users or items:** refused. The declared splits contain none.

## Cases

`<transition>-P6-ease` and `<transition>-P6-knn` for NumPy and SciPy transitions. N2, N3, S3 and S4 are development cases. N4–N6 and S5–S7 are held-out cases.
