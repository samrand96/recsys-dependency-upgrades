"""CPU adapters for the implicit package's ALS, cosine and BM25 recommenders."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from ..io import FrozenData
from ..ranking import save_rankings, seen_mask
from ..stages import Stages, csr_arrays
from ..tracing import CallTrace
from .ref_itemknn import positive_matrix


def _native_call(trace: CallTrace, operation: str, function, *args, **kwargs):
    """Observe Python-visible NumPy and SciPy calls made directly by implicit."""
    if trace.dependency not in (None, "numpy", "scipy"):
        return trace.call(operation, function, *args, **kwargs)

    def observe(frame, event, argument):
        if event == "call":
            caller = frame.f_back
            module = frame.f_globals.get("__name__", "")
            name = frame.f_code.co_name
        elif event == "c_call":
            caller = frame
            bound_type = type(getattr(argument, "__self__", None))
            module = getattr(argument, "__module__", None) or bound_type.__module__
            name = getattr(argument, "__name__", "")
        else:
            return
        package = module.split(".", 1)[0]
        if (package in ("numpy", "scipy") and trace.dependency in (None, package) and caller is not None
                and caller.f_globals.get("__name__", "").startswith("implicit.")):
            trace.counts[f"{module}.{name}"] += 1

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        return trace.call(operation, function, *args, **kwargs)
    finally:
        sys.setprofile(previous)


def _factor_arrays(path: Path, shape: tuple[int, int], factor_count: int, dtype: np.dtype, **expected) -> dict:
    with np.load(path, allow_pickle=False) as arrays:
        for name, value in expected.items():
            if arrays[name].item() != value:
                raise ValueError(f"{name} in {path.name} does not match the requested run")
        weights = {name: arrays[name] for name in ("user_factors", "item_factors")}
    for array, count in zip(weights.values(), shape):
        if array.shape != (count, factor_count) or array.dtype != dtype or not np.isfinite(array).all():
            raise ValueError(f"{path.name} contains factors with an invalid shape, dtype or value")
    return weights


def _external_ids(indices: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
    valid = (indices >= 0) & (indices < item_ids.size)
    # Keep unknown native IDs visible to contract checks. The -1 value denotes padding only.
    result = np.full(indices.shape, -2, dtype=np.int64)
    result[valid] = item_ids[indices[valid]]
    return result


def _score_matrix(ids: np.ndarray, scores: np.ndarray, n_items: int) -> np.ndarray:
    matrix = np.full((len(ids), n_items), np.nan, dtype=scores.dtype)
    rows, columns = np.nonzero((ids >= 0) & (ids < n_items))
    matrix[rows, ids[rows, columns]] = scores[rows, columns]
    return matrix


def _als_ranking(model, data: FrozenData, trace: CallTrace, k: int):
    users = data.evaluation_users
    n_items = data.item_ids.size
    operation = "implicit.cpu.als.AlternatingLeastSquares.recommend"
    ids, values = _native_call(trace, operation, model.recommend, users, None, N=n_items,
                               filter_already_liked_items=False)
    scores = _score_matrix(ids, values, n_items)

    seen_users, seen_items = data.seen_interactions[:, :2].T
    history = trace.call("scipy.sparse.csr_matrix", csr_matrix,
                         (np.ones(seen_items.size, dtype=np.float32), (seen_users, seen_items)),
                         shape=(data.user_ids.size, n_items))[users]
    available = n_items - np.diff(history.indptr)
    cutoffs = (10, 20) if k == 20 else (k,)
    native = {}
    for cutoff in cutoffs:
        top_items = np.full((len(users), cutoff), -1, dtype=np.int64)
        top_scores = np.full((len(users), cutoff), np.nan, dtype=scores.dtype)
        counts = np.minimum(available, cutoff)
        for count in np.unique(counts[counts > 0]):
            selected = np.flatnonzero(counts == count)
            ids, values = _native_call(trace, operation, model.recommend, users[selected], history[selected], N=int(count),
                                       filter_already_liked_items=True)
            top_items[selected, :count] = _external_ids(ids, data.item_ids)
            top_scores[selected, :count] = values
        native[cutoff] = (top_items, top_scores)
    return scores, native


def _item_ranking(model, data: FrozenData, interactions: csr_matrix, trace: CallTrace, k: int):
    operation = "implicit.nearest_neighbours.ItemItemRecommender.recommend"
    catalogue = np.arange(data.item_ids.size, dtype=np.int64)
    mask = seen_mask(data)
    scores = np.full(mask.shape, np.nan, dtype=np.float64)
    cutoffs = (10, 20) if k == 20 else (k,)
    native = {cutoff: (np.full((len(mask), cutoff), -1, dtype=np.int64),
                       np.full((len(mask), cutoff), np.nan, dtype=scores.dtype)) for cutoff in cutoffs}
    for row, user in enumerate(data.evaluation_users):
        # Explicit catalogues use implicit's documented missing-score sentinel and native ordering.
        indices, values = _native_call(trace, operation, model.recommend, int(user), interactions[user],
                                       items=catalogue, filter_already_liked_items=False)
        valid = (indices >= 0) & (indices < catalogue.size)
        scores[row, indices[valid]] = values[valid]
        keep = ~valid
        keep[valid] = ~mask[row, indices[valid]]
        external = _external_ids(indices[keep], data.item_ids)
        values = values[keep]
        for cutoff, (top_items, top_scores) in native.items():
            count = min(cutoff, external.size)
            top_items[row, :count] = external[:count]
            top_scores[row, :count] = values[:count]
    return scores, native


def _als(
    data: FrozenData, interactions: csr_matrix, stages: Stages, trace: CallTrace, *,
    input_sha256: str, model_in: Path | None, initial_model: Path | None,
    factors: int, iterations: int, regularization: float, alpha: float, seed: int, k: int,
) -> tuple[np.ndarray, dict]:
    from implicit.cpu.als import AlternatingLeastSquares

    config = {"factors": factors, "iterations": iterations, "regularization": regularization, "alpha": alpha}
    model = AlternatingLeastSquares(**config, dtype=interactions.dtype, random_state=seed, num_threads=1)
    if model_in is None:
        if initial_model is None:
            raise ValueError("ALS retraining requires frozen initial factors")
        weights = _factor_arrays(initial_model, interactions.shape, factors, interactions.dtype, seed=seed)
    else:
        weights = _factor_arrays(model_in, interactions.shape, factors, interactions.dtype,
                                 training_input_sha256=input_sha256, **config)
    model.user_factors = weights["user_factors"]
    model.item_factors = weights["item_factors"]
    if model_in is None:
        _native_call(trace, "implicit.cpu.als.AlternatingLeastSquares.fit", model.fit, interactions, show_progress=False)
    arrays = {"user_factors": model.user_factors, "item_factors": model.item_factors,
              "training_input_sha256": np.asarray(input_sha256),
              **{key: np.asarray(value) for key, value in config.items()}}
    np.savez(stages.output / "model.npz", **arrays)
    stages.record("S3", arrays, "model.npz", dtype=interactions.dtype.name)
    return _als_ranking(model, data, trace, k)


def _item_model(
    pipeline: str, data: FrozenData, interactions: csr_matrix, stages: Stages, trace: CallTrace, *,
    input_sha256: str, model_in: Path | None, neighbors: int, k: int,
) -> tuple[np.ndarray, dict]:
    from implicit.nearest_neighbours import BM25Recommender, CosineRecommender

    model_class = CosineRecommender if pipeline == "implicit_cosine" else BM25Recommender
    operation = f"implicit.nearest_neighbours.{model_class.__name__}"
    if model_in is None:
        model = model_class(K=neighbors, num_threads=1)
        _native_call(trace, operation + ".fit", model.fit, interactions, show_progress=False)
    else:
        with np.load(model_in, allow_pickle=False) as arrays:
            if (arrays["training_input_sha256"].item() != input_sha256
                    or arrays["pipeline"].item() != pipeline or arrays["K"].item() != neighbors
                    or arrays["interaction_dtype"].item() != interactions.dtype.name
                    or tuple(arrays["shape"]) != (data.item_ids.size,) * 2):
                raise ValueError("portable item model does not match the requested configuration")
        model = _native_call(trace, operation + ".load", model_class.load, model_in)
        model.num_threads = 1
    matrix = model.similarity
    metadata = {"K": np.asarray(neighbors, dtype=np.int64), "training_input_sha256": np.asarray(input_sha256),
                "pipeline": np.asarray(pipeline), "interaction_dtype": np.asarray(interactions.dtype.name)}
    np.savez(stages.output / "model.npz", **metadata, shape=np.asarray(matrix.shape, dtype=np.int64),
             data=matrix.data, indices=matrix.indices, indptr=matrix.indptr)
    stages.record("S3", {**metadata, **csr_arrays(matrix)}, "model.npz", shape=matrix.shape, dtype=f"csr:{matrix.dtype.name}")
    return _item_ranking(model, data, interactions, trace, k)


def run(
    data: FrozenData, stages: Stages, trace: CallTrace, *, pipeline: str, dtype: np.dtype, neighbors: int, k: int,
    input_sha256: str, model_in: Path | None, initial_model: Path | None,
    factors: int, iterations: int, regularization: float, alpha: float, seed: int,
) -> None:
    stages.bundle("S1", {"user_ids": data.user_ids, "item_ids": data.item_ids}, dtype="int64")
    interactions = positive_matrix(data, dtype, trace)
    stages.bundle("S2", csr_arrays(interactions), shape=interactions.shape, dtype=f"csr:{dtype.name}")
    if pipeline == "implicit_als":
        scores, native = _als(data, interactions, stages, trace, input_sha256=input_sha256, model_in=model_in,
                              initial_model=initial_model, factors=factors, iterations=iterations,
                              regularization=regularization, alpha=alpha, seed=seed, k=k)
    else:
        scores, native = _item_model(pipeline, data, interactions, stages, trace, input_sha256=input_sha256,
                                     model_in=model_in, neighbors=neighbors, k=k)
    save_rankings(stages.output, stages, data, scores, k, trace, native=native)
