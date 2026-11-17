"""Replay direct CSR construction into the released implicit cosine normalizer."""

import argparse
import ast
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import warnings

import numpy as np
import scipy
from scipy.sparse import coo_matrix, csr_matrix


# User-item weights, in input order. User 0 interacts twice with item 0.
ROWS = (0, 0, 1, 0, 1)
COLS = (0, 0, 0, 1, 1)
WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0)
SHAPE = (2, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("phase", choices=("old", "new", "repair", "rollback"))
    args = parser.parse_args()
    raw_source = args.source.read_bytes()
    module = ast.parse(raw_source, filename=str(args.source))
    normalize = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "normalize")
    scope = {"coo_matrix": coo_matrix, "sqrt": np.sqrt, "bincount": np.bincount}
    exec(compile(ast.Module(body=[normalize], type_ignores=[]), str(args.source), "exec"), scope)
    fixture = {"rows": ROWS, "cols": COLS, "weights": WEIGHTS, "shape": SHAPE, "dtype": "float64"}
    with warnings.catch_warnings(record=True) as emitted_warnings:
        warnings.simplefilter("always")
        matrix = csr_matrix((np.asarray(WEIGHTS, dtype=np.float64), (ROWS, COLS)), shape=SHAPE)
        storage = {
            "nnz": int(matrix.nnz),
            "has_canonical_format": bool(matrix.has_canonical_format),
            "data": matrix.data.tolist(),
            "indices": matrix.indices.tolist(),
            "indptr": matrix.indptr.tolist(),
        }
        # This is the call used by implicit's CosineRecommender.fit, up to the
        # transpose back to user-item layout after normalization.
        normalized = scope["normalize"](matrix.T).toarray()
    item_norms = np.linalg.norm(normalized, axis=1)
    cpu_model = next(
        line.partition(":")[2].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    result = {
        "phase": args.phase,
        "platform": platform.platform(),
        "container": platform.node(),
        "cpu_model": cpu_model,
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "source_sha256": hashlib.sha256(raw_source).hexdigest(),
        "fixture": fixture,
        "fixture_sha256": hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest(),
        "csr_storage": storage,
        "user_item_dense": matrix.toarray().tolist(),
        "normalized_item_user_dense": normalized.tolist(),
        "normalized_item_l2": item_norms.tolist(),
        "item_dot_products": (normalized @ normalized.T).tolist(),
        "unit_l2_contract_pass": bool(np.allclose(item_norms, 1.0, rtol=0, atol=1e-12)),
        "warnings": [f"{warning.category.__name__}: {warning.message}" for warning in emitted_warnings],
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
