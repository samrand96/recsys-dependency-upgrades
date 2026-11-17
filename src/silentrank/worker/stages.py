"""Canonical hashes for pipeline checkpoints."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .io import write_json


def csr_arrays(matrix: csr_matrix) -> dict[str, np.ndarray]:
    """Serialize a sparse matrix with sorted, summed coordinates and fixed index width."""
    canonical = matrix.tocsr(copy=True)
    canonical.sum_duplicates()
    canonical.sort_indices()
    canonical.eliminate_zeros()
    return {
        "shape": np.asarray(canonical.shape, dtype=np.int64),
        "indptr": canonical.indptr.astype(np.int64, copy=False),
        "indices": canonical.indices.astype(np.int64, copy=False),
        "data": canonical.data,
    }


class Stages:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.rows: dict[str, dict] = {}

    def input(self, digest: str) -> None:
        self.rows["S0"] = {"canonical_sha256": digest, "shape": None, "dtype": "file", "artifact": None}

    def record(
        self,
        stage: str,
        arrays: dict[str, np.ndarray],
        artifact: str,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: str | None = None,
    ) -> None:
        digest = sha256()
        for name, array in sorted(arrays.items()):
            array = np.asarray(array)
            normalized = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
            header = json.dumps([name, array.shape, normalized.dtype.str], separators=(",", ":"))
            digest.update(header.encode("ascii") + b"\0")
            if normalized.size:
                digest.update(memoryview(normalized).cast("B"))
        self.rows[stage] = {
            "canonical_sha256": digest.hexdigest(),
            "shape": None if shape is None else list(shape),
            "dtype": dtype,
            "artifact": artifact,
        }

    def bundle(
        self,
        stage: str,
        arrays: dict[str, np.ndarray],
        *,
        shape: tuple[int, ...] | None = None,
        dtype: str | None = None,
    ) -> None:
        artifact = f"stages/{stage}.npz"
        (self.output / "stages").mkdir(exist_ok=True)
        np.savez(self.output / artifact, **arrays)
        self.record(stage, arrays, artifact, shape=shape, dtype=dtype)

    def save(self) -> None:
        write_json(self.output / "stage_hashes.json", self.rows)
