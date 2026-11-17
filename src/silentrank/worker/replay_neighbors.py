"""Capture or replay the reference item-kNN cosine neighbor pruning step."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .io import sha256_file, write_json
from .ranking import native_topk
from .stages import Stages, csr_arrays
from .tracing import CallTrace


def _csr(bundle) -> csr_matrix:
    matrix = csr_matrix((bundle["data"], bundle["indices"], bundle["indptr"]), shape=tuple(bundle["shape"]))
    matrix.check_format(full_check=True)
    if matrix.dtype not in (np.dtype("float32"), np.dtype("float64")) or not np.isfinite(matrix.data).all():
        raise ValueError("neighbor replay requires finite float32 or float64 CSR values")
    return matrix


def unpruned_weights(interactions: csr_matrix) -> dict[str, np.ndarray]:
    """Mirror ref_itemknn.fit up to native selection, retaining row encounter order."""
    cooccurrence = (interactions.transpose() @ interactions).tocsr()
    norms = np.sqrt(np.asarray(interactions.sum(axis=0)).ravel())
    n_items = interactions.shape[1]
    indices, weights, offsets = [], [], [0]
    for target in range(n_items):
        start, end = cooccurrence.indptr[target:target + 2]
        columns = cooccurrence.indices[start:end]
        other = columns != target
        columns = columns[other]
        values = cooccurrence.data[start:end][other] / (norms[target] * norms[columns])
        indices.append(columns)
        weights.append(values)
        offsets.append(offsets[-1] + len(columns))
    return {"shape": np.asarray([n_items, n_items], dtype=np.int64),
            "indptr": np.asarray(offsets, dtype=np.int64),
            "indices": np.concatenate(indices).astype(np.int64, copy=False),
            "data": np.concatenate(weights)}


def _prune(weights: csr_matrix, neighbors: int, trace: CallTrace) -> dict[str, np.ndarray]:
    indices, values, offsets = [], [], [0]
    for start, end in zip(weights.indptr[:-1], weights.indptr[1:]):
        row = weights.data[start:end]
        selected = native_topk(row, neighbors, trace)
        indices.append(weights.indices[start:end][selected])
        values.append(row[selected])
        offsets.append(offsets[-1] + len(selected))
    matrix = csr_matrix((np.concatenate(values), np.concatenate(indices), np.asarray(offsets, dtype=np.int64)),
                        shape=weights.shape, dtype=weights.dtype)
    return csr_arrays(matrix)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("capture", "prune"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighbors", type=int)
    parser.add_argument("--training-input-sha256")
    args = parser.parse_args(argv)
    if args.operation == "prune" and (not args.neighbors or args.neighbors < 1 or not args.training_input_sha256):
        parser.error("prune requires positive --neighbors and --training-input-sha256")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = "weights.npz" if args.operation == "capture" else "model.npz"
    output = args.output_dir / output_name
    metadata = args.output_dir / "replay_meta.json"
    if output.exists() or metadata.exists():
        raise FileExistsError(f"refusing to overwrite replay artifacts in {args.output_dir}")
    input_digest = sha256_file(args.input, args.input_sha256)
    with np.load(args.input, allow_pickle=False) as bundle:
        matrix = _csr(bundle)
    if not matrix.shape[1]:
        raise ValueError("neighbor replay requires a nonempty item catalogue")
    if args.operation == "prune" and matrix.shape[0] != matrix.shape[1]:
        raise ValueError("neighbor pruning requires a square item cosine matrix")

    start_wall, start_cpu = time.perf_counter(), time.process_time()
    stages, trace = Stages(args.output_dir), CallTrace("numpy")
    if args.operation == "capture":
        arrays = unpruned_weights(matrix)
        stages.record("encounter_order", arrays, output_name)
        stages.record("canonical", csr_arrays(_csr(arrays)), output_name)
    else:
        arrays = {**_prune(matrix, args.neighbors, trace),
                  "training_input_sha256": np.asarray(args.training_input_sha256),
                  "neighbors": np.asarray(args.neighbors, dtype=np.int64)}
        stages.record("S3", arrays, output_name)
    wall_s, cpu_s = time.perf_counter() - start_wall, time.process_time() - start_cpu
    np.savez(output, **arrays)
    write_json(metadata, {
        "operation": args.operation, "input_sha256": input_digest, "output_sha256": sha256_file(output),
        "hashes": {key: value["canonical_sha256"] for key, value in stages.rows.items()},
        "wall_s": wall_s, "cpu_s": cpu_s, "trace": dict(trace.counts),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
