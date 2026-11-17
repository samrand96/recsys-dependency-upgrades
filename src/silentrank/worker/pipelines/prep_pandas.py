"""Run temporal preparation inside the pandas environment under study."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..io import write_json
from ..stages import Stages, csr_arrays
from ..tracing import CallTrace


def run(raw: Path, cutoffs: dict, stages: Stages, trace: CallTrace) -> None:
    with ZipFile(raw) as archive, archive.open("ml-1m/ratings.dat") as stream:
        ratings = trace.call(
            "pandas.read_csv", pd.read_csv, stream, sep="::", engine="python",
            names=["user", "item", "rating", "timestamp"], dtype=np.int64,
        )
    if (
        not isinstance(ratings.index, pd.RangeIndex)
        or ratings[["user", "item"]].le(0).any().any()
        or not ratings["rating"].between(1, 5).all()
        or ratings["timestamp"].lt(0).any()
    ):
        raise ValueError("ratings have invalid fields")

    ordered = trace.call("pandas.DataFrame.sort_values", ratings.sort_values, "timestamp", kind="stable")
    ordered = trace.call("pandas.DataFrame.drop_duplicates", ordered.drop_duplicates, ["user", "item"], keep="first")
    times = trace.call("pandas.Series.to_numpy", ordered["timestamp"].to_numpy, dtype=np.int64)
    train_end, test_start = trace.call(
        "numpy.searchsorted", np.searchsorted, times,
        [cutoffs["validation_start_unix"], cutoffs["test_start_unix"]],
    )
    splits = {"train": ordered.iloc[:train_end], "validation": ordered.iloc[train_end:test_start],
              "test": ordered.iloc[test_start:]}
    if splits["train"].empty:
        raise ValueError("the chosen cutoff gives an empty training partition")

    ids = {}
    for axis in ("user", "item"):
        _, external = trace.call("pandas.factorize", pd.factorize, splits["train"][axis], sort=True)
        ids[axis] = pd.Index(external, dtype=np.int64)
    stages.bundle("S1", {f"{axis}_ids": index.to_numpy(dtype=np.int64) for axis, index in ids.items()}, dtype="int64")

    shape = (len(ids["user"]), len(ids["item"]))
    checkpoints = {}
    splits_counts = {}
    for name, split in splits.items():
        users = trace.call("pandas.Index.get_indexer", ids["user"].get_indexer, split["user"])
        items = trace.call("pandas.Index.get_indexer", ids["item"].get_indexer, split["item"])
        warm = (users >= 0) & (items >= 0)
        values = trace.call("pandas.Series.to_numpy", split["rating"].to_numpy, dtype=np.int64)
        matrix = csr_matrix((values[warm], (users[warm], items[warm])), shape=shape, dtype=np.int64)
        checkpoints.update({f"{name}_{key}": value for key, value in csr_arrays(matrix).items()})
        splits_counts[name] = {
            "interactions": len(split),
            "warm_interactions": int(warm.sum()),
            "cold_user_interactions": int((users < 0).sum()),
            "cold_item_interactions": int((items < 0).sum()),
        }
    stages.bundle("S2", checkpoints, shape=shape, dtype="csr:int64")
    write_json(stages.output / "prep_counts.json", {
        "source_rows": len(ratings), "duplicate_rows": len(ratings) - len(ordered), "splits": splits_counts,
    })
