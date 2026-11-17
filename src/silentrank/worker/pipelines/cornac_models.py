"""P6: Cornac EASE and ItemKNN on explicit training ratings, ranked by Cornac's own ``rank``.

Declared in protocol/pipelines_p6_cornac.md before any P6 ranking run:

* ``Dataset.from_uir`` on all frozen training rows (user, item, float rating), in row order.
* ``EASE(lamb=<grid>)`` and ``ItemKNN(k=<grid>, num_threads=1)``; all other settings at
  library defaults. Both are deterministic, so retraining seeds are repetitions.
* Scores: ``model.score(user_idx)`` over all items.
* Native top-N: ``model.rank(user_idx, item_indices=<unseen items>, k=N)``, the call Cornac's
  own ranking evaluation makes. It uses ``numpy.argpartition`` and ``numpy.argsort``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from ..io import FrozenData
from ..ranking import save_rankings, seen_mask
from ..stages import Stages, csr_arrays
from ..tracing import CallTrace


def _native_call(trace: CallTrace, operation: str, function, *args, **kwargs):
    """Count Python-visible NumPy and SciPy calls made directly by Cornac code."""
    if trace.dependency not in (None, "numpy", "scipy"):
        return trace.call(operation, function, *args, **kwargs)

    def observe(frame, event, argument):
        if event == "call":
            caller, module, name = frame.f_back, frame.f_globals.get("__name__", ""), frame.f_code.co_name
        elif event == "c_call":
            caller = frame
            module = getattr(argument, "__module__", None) or type(getattr(argument, "__self__", None)).__module__
            name = getattr(argument, "__name__", "")
        else:
            return
        package = (module or "").split(".", 1)[0]
        if (package in ("numpy", "scipy") and trace.dependency in (None, package) and caller is not None
                and caller.f_globals.get("__name__", "").startswith("cornac")):
            trace.counts[f"{module}.{name}"] += 1

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        return trace.call(operation, function, *args, **kwargs)
    finally:
        sys.setprofile(previous)


def _train_set(data: FrozenData):
    from cornac.data import Dataset

    rows = [(int(u), int(i), float(r)) for u, i, r in data.train[:, :3].tolist()]
    train_set = Dataset.from_uir(rows, seed=0)
    cold_users = [int(u) for u in data.evaluation_users if int(u) not in train_set.uid_map]
    if cold_users or train_set.num_items != data.item_ids.size:
        raise ValueError(f"P6 refuses cold users ({len(cold_users)}) or a partial catalogue "
                         f"({train_set.num_items} items); the declared splits contain neither")
    return train_set


def _new_model(pipeline: str, *, lamb: float, neighbors: int):
    from cornac.models import EASE, ItemKNN

    if pipeline == "cornac_ease":
        return EASE(lamb=lamb, verbose=False)
    return ItemKNN(k=neighbors, num_threads=1, verbose=False)


def _portable(model, pipeline: str, input_sha256: str, *, lamb: float, neighbors: int) -> dict[str, np.ndarray]:
    arrays = {"training_input_sha256": np.asarray(input_sha256), "pipeline": np.asarray(pipeline)}
    if pipeline == "cornac_ease":
        arrays.update(lamb=np.asarray(lamb, dtype=np.float64), B=np.asarray(model.B, dtype=np.float64))
    else:
        similarity = csr_arrays(model.sim_mat)
        ratings = csr_arrays(model.ui_mat)
        arrays.update(
            k=np.asarray(neighbors, dtype=np.int64), mean_arr=np.asarray(model.mean_arr, dtype=np.float64),
            **{f"sim_{name}": value for name, value in similarity.items()},
            **{f"ui_{name}": value for name, value in ratings.items()},
        )
    return arrays


def _restore(path: Path, pipeline: str, train_set, input_sha256: str, *, lamb: float, neighbors: int):
    from cornac.models import Recommender

    model = _new_model(pipeline, lamb=lamb, neighbors=neighbors)
    Recommender.fit(model, train_set)  # attach id maps and rating range without training
    with np.load(path, allow_pickle=False) as arrays:
        if arrays["training_input_sha256"].item() != input_sha256 or arrays["pipeline"].item() != pipeline:
            raise ValueError("portable Cornac model does not match the requested configuration")
        if pipeline == "cornac_ease":
            if arrays["lamb"].item() != lamb or arrays["B"].shape != (train_set.num_items,) * 2:
                raise ValueError("portable EASE model has a different lambda or shape")
            model.U = train_set.matrix
            model.B = arrays["B"]
        else:
            if arrays["k"].item() != neighbors:
                raise ValueError("portable ItemKNN model has a different neighbour count")
            model.sim_mat = csr_matrix((arrays["sim_data"], arrays["sim_indices"], arrays["sim_indptr"]),
                                       shape=tuple(arrays["sim_shape"]))
            model.ui_mat = csr_matrix((arrays["ui_data"], arrays["ui_indices"], arrays["ui_indptr"]),
                                      shape=tuple(arrays["ui_shape"]))
            model.mean_arr = arrays["mean_arr"]
    return model


def _rank(model, train_set, data: FrozenData, k: int, trace: CallTrace):
    mask = seen_mask(data)
    n_items = data.item_ids.size
    # Cornac item index -> catalogue index (raw ids are catalogue indices).
    to_catalogue = np.empty(n_items, dtype=np.int64)
    for raw, index in train_set.iid_map.items():
        to_catalogue[index] = raw
    from_catalogue = np.empty(n_items, dtype=np.int64)
    from_catalogue[to_catalogue] = np.arange(n_items, dtype=np.int64)

    scores = np.empty(mask.shape, dtype=np.float64)
    cutoffs = (10, 20) if k == 20 else (k,)
    native = {n: (np.full((len(mask), n), -1, dtype=np.int64), np.full((len(mask), n), np.nan, dtype=np.float64))
              for n in cutoffs}

    def rank_all():
        for row, raw_user in enumerate(data.evaluation_users.tolist()):
            user = train_set.uid_map[raw_user]
            user_scores = np.asarray(model.score(user), dtype=np.float64).ravel()
            scores[row, to_catalogue] = user_scores
            eligible = np.sort(from_catalogue[np.flatnonzero(~mask[row])])  # unseen items, Cornac index order
            for n, (top_items, top_scores) in native.items():
                count = min(n, eligible.size)
                if not count:
                    continue
                ranked, _ = model.rank(user, item_indices=eligible, k=count)
                chosen = np.asarray(ranked[:count], dtype=np.int64)
                top_items[row, :count] = data.item_ids[to_catalogue[chosen]]
                top_scores[row, :count] = user_scores[chosen]

    _native_call(trace, "cornac.Recommender.rank", rank_all)
    return scores, native


def run(data: FrozenData, stages: Stages, trace: CallTrace, *, pipeline: str, lamb: float, neighbors: int, k: int,
        input_sha256: str, model_in: Path | None) -> None:
    stages.bundle("S1", {"user_ids": data.user_ids, "item_ids": data.item_ids}, dtype="int64")
    train_set = _train_set(data)
    matrix = train_set.matrix
    stages.bundle("S2", csr_arrays(matrix), shape=matrix.shape, dtype=f"csr:{matrix.dtype.name}")
    if model_in is None:
        model = _new_model(pipeline, lamb=lamb, neighbors=neighbors)
        _native_call(trace, f"cornac.{type(model).__name__}.fit", model.fit, train_set)
    else:
        model = _restore(model_in, pipeline, train_set, input_sha256, lamb=lamb, neighbors=neighbors)
    arrays = _portable(model, pipeline, input_sha256, lamb=lamb, neighbors=neighbors)
    np.savez(stages.output / "model.npz", **arrays)
    stages.record("S3", arrays, "model.npz", dtype="float64")
    scores, native = _rank(model, train_set, data, k, trace)
    save_rankings(stages.output, stages, data, scores, k, trace, native=native)
