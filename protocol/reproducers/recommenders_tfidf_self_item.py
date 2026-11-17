"""Check self-exclusion in the tagged Recommenders TF-IDF ranking method.

The fixture comes from PR #2391: A and B share one document, while C and D
have disjoint vocabularies. The frozen cosine scores are their idealized
float64 Gram matrix, with exact ones for ties and exact zeros for nonmatches.
The experiment supplies that matrix in place of the linear kernel output.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import numpy as np


METHOD = "__create_full_recommendation_dictionary"
ITEM_IDS = ("A", "B", "C", "D")
TEXTS = (
    "apple banana cherry",
    "apple banana cherry",
    "dog cat bird",
    "fish shark whale",
)
SCORES = (
    (1.0, 1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
K = 2


class ItemTable:
    def __getitem__(self, key):
        if key != "itemID":
            raise KeyError(key)
        return SimpleNamespace(values=ITEM_IDS)

    def __len__(self):
        return len(ITEM_IDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("phase", choices=("old", "new", "rollback", "reapply"))
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    parsed = ast.parse(source_bytes, filename=str(args.source))
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "TfidfRecommender")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == METHOD)
    matrix = np.asarray(SCORES, dtype="<f8")
    namespace = {"np": np, "linear_kernel": lambda left, right: matrix}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(args.source), "exec"), namespace)

    model = SimpleNamespace(id_col="itemID", tfidf_matrix=None)
    with warnings.catch_warnings(record=True) as emitted_warnings:
        warnings.simplefilter("always")
        namespace[METHOD](model, ItemTable())
    top_k = {item: model.recommendations[item][:K] for item in ITEM_IDS}
    violations = {
        item: [rank for rank, (_, other) in enumerate(top_k[item], start=1) if item == other]
        for item in ITEM_IDS
    }
    violations = {item: ranks for item, ranks in violations.items() if ranks}
    cpu_model = next(
        line.partition(":")[2].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    print(json.dumps({
        "phase": args.phase,
        "platform": platform.platform(),
        "container": platform.node(),
        "cpu_model": cpu_model,
        "python": sys.version,
        "numpy": np.__version__,
        "threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "score_matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "fixture": {"item_ids": ITEM_IDS, "texts": TEXTS, "scores": SCORES, "k": K},
        "ranked_scores_and_ids": top_k,
        "self_item_contract_pass": not violations,
        "violations_by_query": violations,
        "warnings": [f"{warning.category.__name__}: {warning.message}" for warning in emitted_warnings],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
