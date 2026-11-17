"""Save full candidate scores, seen masks and native top-K lists."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import FrozenData, write_json
from .stages import Stages
from .tracing import CallTrace


def native_topk(values: np.ndarray, k: int, trace: CallTrace) -> np.ndarray:
    """Use the pipeline's unstable partition and sort on the eligible values."""
    if not values.size:
        return np.empty(0, dtype=np.int64)
    k = min(k, len(values))
    selected = trace.call("numpy.argpartition", np.argpartition, -values, k - 1)[:k]
    order = trace.call("numpy.argsort", np.argsort, -values[selected])
    return selected[order].astype(np.int64, copy=False)


def seen_mask(data: FrozenData) -> np.ndarray:
    user_to_row = np.full(data.user_ids.size, -1, dtype=np.int64)
    user_to_row[data.evaluation_users] = np.arange(data.evaluation_users.size, dtype=np.int64)
    users, items = data.seen_interactions[:, :2].T
    rows = user_to_row[users]
    included = rows >= 0
    mask = np.zeros((data.evaluation_users.size, data.item_ids.size), dtype=np.bool_)
    mask[rows[included], items[included]] = True
    return mask


def _reference_ranking(scores: np.ndarray, mask: np.ndarray, item_ids: np.ndarray, k: int, trace: CallTrace):
    items = np.full((len(scores), k), -1, dtype=np.int64)
    ranked_scores = np.full((len(scores), k), np.nan, dtype=scores.dtype)
    for row, (row_scores, seen) in enumerate(zip(scores, mask)):
        eligible = np.flatnonzero(~seen)
        selected = eligible[native_topk(row_scores[eligible], k, trace)]
        items[row, :selected.size] = item_ids[selected]
        ranked_scores[row, :selected.size] = row_scores[selected]
    return items, ranked_scores


def _boundary_ties(scores: np.ndarray, mask: np.ndarray, k: int) -> dict:
    tie_size = []
    crosses_cutoff = []
    for row, seen in zip(scores, mask):
        values = row[~seen]
        if not np.isfinite(values).all():
            tie_size.append(None)
            crosses_cutoff.append(None)
        elif k > values.size:
            tie_size.append(0)
            crosses_cutoff.append(False)
        else:
            threshold = np.partition(values, values.size - k)[values.size - k]
            count = int(np.count_nonzero(values == threshold))
            above = int(np.count_nonzero(values > threshold))
            tie_size.append(count if count > 1 else 0)
            crosses_cutoff.append(above + count > k)
    return {"tie_size": tie_size, "crosses_cutoff": crosses_cutoff}


def save_rankings(
    output: Path,
    stages: Stages,
    data: FrozenData,
    scores: np.ndarray,
    k: int,
    trace: CallTrace,
    *,
    native: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    mask = seen_mask(data)
    if scores.shape != mask.shape:
        raise ValueError("scores must align to the frozen evaluation users and item catalogue")

    np.save(output / "user_ids.npy", data.evaluation_users, allow_pickle=False)
    np.save(output / "item_ids.npy", data.item_ids, allow_pickle=False)
    stages.bundle("S4", {"user_ids": data.evaluation_users, "item_ids": data.item_ids})
    np.save(output / "scores.npy", scores, allow_pickle=False)
    stages.record("S5", {"scores": scores}, "scores.npy", shape=scores.shape, dtype=scores.dtype.name)
    np.save(output / "seen_mask.npy", mask, allow_pickle=False)
    stages.record("S6", {"seen_mask": mask}, "seen_mask.npy", shape=mask.shape, dtype="bool")

    cutoffs = (10, 20) if k == 20 else (k,)
    rank_arrays = {}
    for n in cutoffs:
        items, selected_scores = (
            _reference_ranking(scores, mask, data.item_ids, n, trace) if native is None else native[n]
        )
        if items.shape != (len(scores), n) or selected_scores.shape != items.shape:
            raise ValueError(f"native top-{n} arrays do not align with evaluation users")
        suffix = "" if n == k else str(n)
        np.save(output / f"topk_native{suffix}.npy", items, allow_pickle=False)
        np.save(output / f"scores_topk{suffix}.npy", selected_scores, allow_pickle=False)
        rank_arrays[f"items_{n}"] = items
    stages.bundle("S7", rank_arrays, shape=rank_arrays[f"items_{k}"].shape, dtype="int64")
    write_json(output / "boundary_ties.json", {
        "user_ids": data.evaluation_users.tolist(),
        **{str(n): _boundary_ties(scores, mask, n) for n in cutoffs},
    })
