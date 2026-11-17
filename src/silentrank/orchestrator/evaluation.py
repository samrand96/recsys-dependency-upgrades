"""Evaluate portable worker outputs in the pinned orchestrator environment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from silentrank.evaluator.canonical import canonical_topk
from silentrank.evaluator.contracts import check_topk
from silentrank.evaluator.metrics import ndcg_at_k, recall_at_k


class ArtifactContractFailure(ValueError):
    """The worker completed but its ranking arrays violated a frozen contract."""


def array_hash(values: np.ndarray) -> dict:
    values = np.ascontiguousarray(values.astype(values.dtype.newbyteorder("<"), copy=False))
    digest = hashlib.sha256(f"{values.dtype.str}:{values.shape}\n".encode("ascii"))
    digest.update(memoryview(values).cast("B"))
    return {"canonical_sha256": digest.hexdigest(), "shape": list(values.shape), "dtype": str(values.dtype)}


def _lists(directory: Path, k: int) -> tuple[np.ndarray, np.ndarray]:
    suffix = "10" if k == 10 else ""
    return (
        np.load(directory / f"topk_native{suffix}.npy", allow_pickle=False),
        np.load(directory / f"scores_topk{suffix}.npy", allow_pickle=False),
    )


def evaluate(directory: Path, dataset: Path, split: str, metrics_path: Path) -> dict:
    """Validate shape, catalogue and seen mask, then persist independent quality metrics."""
    with np.load(dataset, allow_pickle=False) as frozen:
        train, validation, labels = frozen["train"], frozen["validation"], frozen[split]
        catalog, activity = frozen["item_ids"], frozen["activity"]
    positive = labels[labels[:, 2] >= 4]
    expected_users = np.unique(positive[:, 0])
    users = np.load(directory / "user_ids.npy", allow_pickle=False)
    if users.dtype != np.int64 or not np.array_equal(users, expected_users):
        raise ArtifactContractFailure("worker user order differs from the frozen evaluation population")
    items = np.load(directory / "item_ids.npy", allow_pickle=False)
    if items.dtype != np.int64 or not np.array_equal(items, catalog):
        raise ArtifactContractFailure("worker item order differs from the frozen catalogue")
    seen = np.zeros((len(activity), len(catalog)), dtype=bool)
    for history in (train, validation) if split == "test" else (train,):
        seen[history[:, 0], history[:, 1]] = True
    expected_mask = seen[users]
    del seen
    worker_mask = np.load(directory / "seen_mask.npy", allow_pickle=False, mmap_mode="r")
    if worker_mask.dtype != np.bool_ or not np.array_equal(worker_mask, expected_mask):
        raise ArtifactContractFailure("worker seen mask differs from the frozen observed history")
    full_scores = np.load(directory / "scores.npy", allow_pickle=False, mmap_mode="r")
    if full_scores.shape != expected_mask.shape or full_scores.dtype.kind != "f":
        raise ArtifactContractFailure("worker scores must align with evaluation users and frozen catalogue")

    rank_lists = {k: _lists(directory, k) for k in (10, 20)}
    for k, (ranks, scores) in rank_lists.items():
        if ranks.shape != (len(users), k) or scores.shape != ranks.shape:
            raise ArtifactContractFailure(f"worker top-{k} arrays have the wrong shape")

    positives_by_user: dict[int, list[int]] = {}
    for user, item, _ in positive:
        positives_by_user.setdefault(int(user), []).append(int(catalog[item]))
    rows, violations = [], []
    canonical_lists = np.full((len(users), 20), -1, dtype=np.int64)
    for row_index, user in enumerate(users):
        eligible = ~expected_mask[row_index]
        eligible_scores = full_scores[row_index, eligible]
        user_violations = []
        if np.isfinite(eligible_scores).all():
            ranking = canonical_topk(catalog[eligible], eligible_scores, 20)
            canonical_lists[row_index, :len(ranking.item_ids)] = ranking.item_ids
        else:
            user_violations.append("nonfinite_eligible_scores")
        result = {"user": int(user), "activity": int(activity[user])}
        for k, (ranks, scores) in rank_lists.items():
            present = ranks[row_index] != -1
            ranked = ranks[row_index, present]
            if np.any(present[np.count_nonzero(present):]):
                user_violations.append(f"k{k}:nonterminal_padding")
            failures = check_topk(ranked, scores[row_index, present], catalog=catalog, seen=catalog[~eligible], k=k)
            user_violations.extend(f"k{k}:{failure}" for failure in failures)
            relevance = positives_by_user[int(user)]
            result[f"ndcg{k}"] = None if user_violations else ndcg_at_k(ranked, relevance, k)
            if k == 10:
                result["recall10"] = None if user_violations else recall_at_k(ranked, relevance, k)
        if user_violations:
            violations.append({"user": int(user), "codes": user_violations})
        rows.append(result)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema({"user": pa.int64(), "activity": pa.int64(), "ndcg10": pa.float64(),
                        "recall10": pa.float64(), "ndcg20": pa.float64()})
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), metrics_path)
    np.save(directory / "topk_canonical.npy", canonical_lists, allow_pickle=False)
    (directory / "contracts.json").write_text(json.dumps(violations, indent=2) + "\n", encoding="utf-8")
    return {
        "violations": len(violations),
        "stages": {"S7c": array_hash(canonical_lists), "S8": array_hash(np.array(
            [[row[key] for key in ("ndcg10", "recall10", "ndcg20")] for row in rows], dtype=np.float64
        ))},
    }
