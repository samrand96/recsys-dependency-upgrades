"""Inspectable item cosine kNN over binary positive interactions."""

from __future__ import annotations

from operator import matmul
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from ..io import FrozenData
from ..ranking import native_topk, save_rankings
from ..stages import Stages, csr_arrays
from ..tracing import CallTrace


def positive_matrix(data: FrozenData, dtype: np.dtype, trace: CallTrace) -> csr_matrix:
    positives = data.train[data.train[:, 2] >= 4]
    return trace.call(
        "scipy.sparse.csr_matrix", csr_matrix,
        (np.ones(len(positives), dtype=dtype), (positives[:, 0], positives[:, 1])),
        shape=(data.user_ids.size, data.item_ids.size), dtype=dtype,
    )


def fit(interactions: csr_matrix, neighbors: int, trace: CallTrace) -> csr_matrix:
    """Store each target item's top neighbors as rows of a directed cosine matrix."""
    transpose = trace.call("scipy.sparse.csr_matrix.transpose", interactions.transpose)
    cooccurrence = trace.call("scipy.sparse.csc_matrix.__matmul__", matmul, transpose, interactions).tocsr()
    counts = trace.call("scipy.sparse.csr_matrix.sum", interactions.sum, axis=0)
    norms = np.sqrt(np.asarray(counts).ravel())
    n_items = interactions.shape[1]
    neighbor_columns = []
    similarities = []
    indptr = [0]
    for target in range(n_items):
        start, end = cooccurrence.indptr[target:target + 2]
        columns = cooccurrence.indices[start:end]
        other = columns != target
        columns = columns[other]
        values = cooccurrence.data[start:end][other] / (norms[target] * norms[columns])
        chosen = native_topk(values, neighbors, trace)
        neighbor_columns.append(columns[chosen])
        similarities.append(values[chosen])
        indptr.append(indptr[-1] + len(chosen))
    return trace.call(
        "scipy.sparse.csr_matrix", csr_matrix,
        (np.concatenate(similarities), np.concatenate(neighbor_columns), np.asarray(indptr, dtype=np.int64)),
        shape=(n_items, n_items), dtype=interactions.dtype,
    )


def _load_model(path: Path, input_sha256: str, dtype: np.dtype, neighbors: int, n_items: int) -> dict:
    with np.load(path, allow_pickle=False) as bundle:
        model = dict(bundle.items())
    if model["training_input_sha256"].item() != input_sha256:
        raise ValueError("portable model was trained on a different frozen data artifact")
    if model["neighbors"].item() != neighbors or model["data"].dtype != dtype:
        raise ValueError("portable model neighbor count or dtype differs from the requested configuration")
    if tuple(model["shape"]) != (n_items, n_items):
        raise ValueError("portable model shape differs from the frozen item catalogue")
    return model


def run(
    data: FrozenData,
    stages: Stages,
    trace: CallTrace,
    *,
    dtype: np.dtype,
    neighbors: int,
    k: int,
    input_sha256: str,
    model_in: Path | None,
) -> None:
    stages.bundle("S1", {"user_ids": data.user_ids, "item_ids": data.item_ids}, dtype="int64")
    interactions = positive_matrix(data, dtype, trace)
    stages.bundle("S2", csr_arrays(interactions), shape=interactions.shape, dtype=f"csr:{dtype.name}")

    if model_in is None:
        model = {
            **csr_arrays(fit(interactions, neighbors, trace)),
            "training_input_sha256": np.asarray(input_sha256),
            "neighbors": np.asarray(neighbors, dtype=np.int64),
        }
    else:
        model = _load_model(model_in, input_sha256, dtype, neighbors, data.item_ids.size)
    matrix = trace.call(
        "scipy.sparse.csr_matrix", csr_matrix,
        (model["data"], model["indices"], model["indptr"]), shape=tuple(model["shape"]), dtype=dtype,
    )
    matrix.check_format(full_check=True)
    np.savez(stages.output / "model.npz", **model)
    stages.record("S3", model, "model.npz", shape=matrix.shape, dtype=f"csr:{dtype.name}")

    transpose = trace.call("scipy.sparse.csr_matrix.transpose", matrix.transpose)
    scores = trace.call(
        "scipy.sparse.csr_matrix.__matmul__", matmul, interactions[data.evaluation_users], transpose,
    ).toarray()
    save_rankings(stages.output, stages, data, scores, k, trace)
