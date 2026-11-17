"""Synthetic NumPy tie-order reproducer. Run with the two N1 Python interpreters."""

import json

import numpy as np


def main() -> None:
    k = 10
    item_ids = np.arange(64, dtype=np.int64)
    scores = np.ones(item_ids.size, dtype=np.float32)
    selected = np.argpartition(-scores, k - 1)[:k]
    selected = selected[np.argsort(-scores[selected])]
    canonical = np.lexsort((item_ids, -scores))[:k]
    print(json.dumps({
        "input": "synthetic equal scores",
        "numpy": np.__version__,
        "candidates": item_ids.size,
        "k": k,
        "native_item_ids": item_ids[selected].tolist(),
        "native_scores": scores[selected].tolist(),
        "canonical_item_ids": item_ids[canonical].tolist(),
    }))


if __name__ == "__main__":
    main()
