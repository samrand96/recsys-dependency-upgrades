# P3 Surprise and P4 SAR pipeline declarations

Declared on 13 November 2026, before any P3 or P4 ranking run, as required by `PROTOCOL.md` (protocol-v0): "Declare each case's pipeline configuration and model search grid before its ranking runs." Everything else follows protocol-v0 unchanged: dataset, split, seen mask, metrics, tolerances, seeds, A/A control before A/B, and verdict rules.

P3 and P4 are secondary pipelines. Their transitions are development cases, like every pilot case.

## Environments

Each pipeline reuses the pinned NumPy, SciPy and pandas versions of the systematic single-dependency pairs, adding one library:

| Group | Added requirement | Transitions | Image |
|---|---|---|---|
| `surprise` | `scikit-surprise==1.1.5` | N1–N3, S1–S4 | slim runner |
| `sar` | `recommenders==1.2.1` | N1–N3, S1–S4, D1, D2 | source runner (Recommenders installs lightgbm, which needs `libgomp`) |

These environments were defined after the protocol-v0 freeze, so their locks use the upload cutoff `exclude-newer = 2026-11-13T00:00:00Z` for reproducibility. The same three constraint pins as the earlier variants are kept.

Recommenders 1.2.1 pulls about 175 packages. A NumPy or pandas change can therefore force other packages to change. In that case the treatment is the whole resolved lock difference, as `PROTOCOL.md` and `BLUEPRINT.md` §9 define. These cases are reported as bundle-level, not single-package, changes.

Known before ranking (from a 13 November import check, not a ranking run): `recommenders` 1.2.1 `SAR.fit` raises `AttributeError: np.NaN was removed in the NumPy 2.0 release`. P4 cases with a NumPy 2 side are therefore expected to record visible runtime failures. They are run anyway so the failures are logged, not assumed. Recommenders requires `pandas<3`, so the D2 new side is expected to fail resolution. It is recorded as `resolve_failed`.

## P3: Surprise SVD

- **Input:** all observed training ratings (1–5) as floats. Rating scale (1, 5).
- **Trainset:** built as `Dataset.construct_trainset` does. Inner ids follow the first appearance in the frozen training rows, and raw ids are catalogue indices.
- **Model:** `SVD(n_factors=f, n_epochs=20, random_state=seed)`, with all other settings at library defaults.
- **Grid:** `n_factors` ∈ (50, 100), in that order. The grid order breaks validation ties.
- **Initialisation:** Surprise does not accept initial factors, so the seed drives Surprise's legacy `RandomState` stream. NumPy keeps that stream stable across versions. Any difference it caused would still be part of the measured upgrade effect.
- **Scores:** `SVD.predict(u, i).est` for every catalogue item, clipped to the rating scale by default.
- **Native ranking:** the Surprise FAQ `get_top_n` recipe. For each user, eligible (unseen) items in catalogue order are sorted by estimate with `list.sort(reverse=True)`. Ties keep catalogue order.
- **Inference mode:** the old side's `pu`, `qi`, `bu`, `bi` are loaded into a fresh `SVD` attached to the same trainset, without SGD.
- **Cold users or items:** refused. The declared splits contain none (checked 13 November: 0 of 1,007 test users, 0 of 397 validation users, all 3,655 items in training).

## P4: Recommenders SAR

- **Input:** all observed training ratings as a pandas DataFrame (`userID`, `itemID`, float64 `rating`). The released API accepts only DataFrames. This is a declared exception to the worker's import rule: pandas is a declared dependency of the library under test.
- **Model:** `SAR(similarity_type=s)`, with all other settings at library defaults: no time decay, threshold 1, no normalisation.
- **Grid:** `similarity_type` ∈ (jaccard, cosine, lift), in that order.
- **Scores:** `SAR.score(users, remove_seen=False)` in SAR's own item order.
- **Native ranking:** the protocol's seen mask is applied as `remove_seen` does it, by setting seen entries to −inf. Then `recommenders.utils.python_utils.get_top_k_scored_items(..., sort_top_k=True)` runs, as in `recommend_k_items`. That function uses NumPy `argpartition` and `argsort`. Entries at −inf are dropped.
- **Repetitions:** SAR is deterministic, so retraining "seeds" are repetitions, as for item-kNN.
- **Inference mode:** the old side's user affinity, item similarity and index maps are restored into a fresh `SAR` object.
- **Cold users or items:** refused, as for P3.

## Applicability

As for P1 and P2, a case counts only if the pipeline calls the upgraded package. Surprise imports neither SciPy nor pandas (checked 13 November), so P3 cases on SciPy transitions are expected to be recorded as not applicable.
