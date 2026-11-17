"""P4: Microsoft Recommenders SAR (single node) on explicit training ratings.

Declared in protocol/pipelines_p3_p4.md before any P4 ranking run:

* ``SAR(similarity_type=<grid>)`` with every other setting at the library default
  (no time decay, threshold 1, no normalisation). Training rows go in as a pandas
  DataFrame with columns userID, itemID and a float64 rating column, because the
  released API accepts nothing else. pandas is a declared dependency of the library.
* Scores come from ``SAR.score(..., remove_seen=False)`` in SAR's own item order.
  The protocol's seen mask (train + validation for the test split) is then applied
  the way ``remove_seen`` does it, by setting seen entries to -inf, and the native
  top-N is ``recommenders.utils.python_utils.get_top_k_scored_items`` on that
  matrix, as ``recommend_k_items`` does. Entries at -inf are dropped, as there.
* SAR is deterministic: retraining "seeds" are repetitions, as for item-kNN.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, issparse

from ..io import FrozenData
from ..ranking import save_rankings, seen_mask
from ..stages import Stages, csr_arrays
from ..tracing import CallTrace

SIMILARITIES = ("jaccard", "cosine", "lift")
USER, ITEM, RATING = "userID", "itemID", "rating"


def _native_call(trace: CallTrace, operation: str, function, *args, **kwargs):
    """Count Python-visible NumPy, SciPy and pandas calls made directly by Recommenders code."""
    if trace.dependency not in (None, "numpy", "scipy", "pandas"):
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
        if (package in ("numpy", "scipy", "pandas") and trace.dependency in (None, package) and caller is not None
                and caller.f_globals.get("__name__", "").startswith("recommenders.")):
            trace.counts[f"{module}.{name}"] += 1

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        return trace.call(operation, function, *args, **kwargs)
    finally:
        sys.setprofile(previous)


def _new_model(similarity: str):
    from recommenders.models.sar import SAR

    if similarity not in SIMILARITIES:
        raise ValueError(f"undeclared SAR similarity {similarity!r}")
    return SAR(col_user=USER, col_item=ITEM, col_rating=RATING, similarity_type=similarity)


def _restore(path: Path, similarity: str, input_sha256: str):
    model = _new_model(similarity)
    with np.load(path, allow_pickle=False) as arrays:
        if (arrays["training_input_sha256"].item() != input_sha256
                or arrays["similarity_type"].item() != similarity):
            raise ValueError("portable SAR model does not match the requested configuration")
        users, items = arrays["index_users"], arrays["index_items"]
        affinity = csr_matrix((arrays["affinity_data"], arrays["affinity_indices"], arrays["affinity_indptr"]),
                              shape=tuple(arrays["affinity_shape"]))
        similarity_matrix = arrays["item_similarity"]
    if similarity_matrix.shape != (items.size, items.size) or affinity.shape != (users.size, items.size):
        raise ValueError("portable SAR model shapes are inconsistent")
    model.index2user = dict(enumerate(users.tolist()))
    model.index2item = dict(enumerate(items.tolist()))
    model.user2index = {value: key for key, value in model.index2user.items()}
    model.item2index = {value: key for key, value in model.index2item.items()}
    model.n_users, model.n_items = users.size, items.size
    model.user_affinity = affinity
    model.item_similarity = similarity_matrix
    return model


def _portable(model, input_sha256: str) -> dict[str, np.ndarray]:
    similarity = model.item_similarity.toarray() if issparse(model.item_similarity) else np.asarray(model.item_similarity)
    affinity = csr_arrays(model.user_affinity)
    return {
        "training_input_sha256": np.asarray(input_sha256),
        "similarity_type": np.asarray(model.similarity_type),
        "index_users": np.asarray([model.index2user[i] for i in range(model.n_users)], dtype=np.int64),
        "index_items": np.asarray([model.index2item[i] for i in range(model.n_items)], dtype=np.int64),
        "affinity_shape": affinity["shape"], "affinity_indptr": affinity["indptr"],
        "affinity_indices": affinity["indices"], "affinity_data": affinity["data"],
        "item_similarity": similarity,
    }


def _rank(model, data: FrozenData, k: int, trace: CallTrace):
    import pandas as pd
    from recommenders.utils.python_utils import get_top_k_scored_items

    columns = np.asarray([model.index2item[i] for i in range(model.n_items)], dtype=np.int64)
    cold = [int(u) for u in data.evaluation_users if int(u) not in model.user2index]
    if cold or columns.size != data.item_ids.size:
        raise ValueError(f"P4 refuses cold users ({len(cold)}) or a partial catalogue ({columns.size} items); "
                         "the declared splits contain neither")
    users = pd.DataFrame({USER: data.evaluation_users})
    native_scores = _native_call(trace, "recommenders.SAR.score", model.score, users, remove_seen=False)
    native_scores = np.asarray(native_scores, dtype=np.float64)
    mask = seen_mask(data)
    scores = np.empty(mask.shape, dtype=np.float64)
    scores[:, columns] = native_scores  # catalogue order for the evaluator

    masked = native_scores.copy()
    masked[mask[:, columns]] = -np.inf  # what remove_seen does, with the protocol's seen set
    cutoffs = (10, 20) if k == 20 else (k,)
    native = {}
    for n in cutoffs:
        top, values = _native_call(trace, "recommenders.get_top_k_scored_items", get_top_k_scored_items,
                                   scores=masked, top_k=n, sort_top_k=True)
        valid = np.isfinite(values)  # recommend_k_items drops -inf rows
        items = np.full(top.shape, -1, dtype=np.int64)
        items[valid] = data.item_ids[columns[top[valid]]]
        native[n] = (items, np.where(valid, values, np.nan))
    return scores, native


def run(data: FrozenData, stages: Stages, trace: CallTrace, *, similarity: str, k: int,
        input_sha256: str, model_in: Path | None) -> None:
    import pandas as pd

    stages.bundle("S1", {"user_ids": data.user_ids, "item_ids": data.item_ids}, dtype="int64")
    train = pd.DataFrame({USER: data.train[:, 0], ITEM: data.train[:, 1],
                          RATING: data.train[:, 2].astype(np.float64)})
    if model_in is None:
        model = _new_model(similarity)
        _native_call(trace, "recommenders.SAR.fit", model.fit, train)
    else:
        model = _restore(model_in, similarity, input_sha256)
    affinity = model.user_affinity
    stages.bundle("S2", csr_arrays(affinity), shape=affinity.shape, dtype=f"csr:{affinity.dtype.name}")
    arrays = _portable(model, input_sha256)
    np.savez(stages.output / "model.npz", **arrays)
    stages.record("S3", arrays, "model.npz", shape=arrays["item_similarity"].shape, dtype="float64")
    scores, native = _rank(model, data, k, trace)
    save_rankings(stages.output, stages, data, scores, k, trace, native=native)
