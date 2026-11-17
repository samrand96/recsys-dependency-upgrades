"""Prepare MovieLens 20M with the protocol-v0 split rules (protocol/dataset_ml20m.md).

Reuses prepare_ml1m.py's split, catalogue and activity code unchanged. Differences, all declared:
* ratings.csv has half stars; ratings are stored as their integer part (floor). This keeps the
  relevance rule "rating >= 4" exact (3.5 -> 3 is not relevant, 4.5 -> 4 is). Only pipelines that
  use relevant events alone (reference item-kNN, implicit) are run on this dataset.
* A fixed sample of 2,000 warm relevant test users and 2,000 warm relevant validation users is
  drawn with numpy PCG64(713). Test rows are kept for sampled test users only. Validation rows are
  kept for users in either sample, so every sampled test user keeps its full seen history.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
ARCHIVE_SHA256 = "96f243c338a8665f6bcc89c53edf6ee39162a846940de6b7c8c48aeada765ff3"
ARCHIVE_MD5 = "cd245b17a1ae2cc31bb14903e1204af3"
EXPECTED_ROWS = 20_000_263
SAMPLE_USERS = 2_000
SAMPLE_SEED = 713

spec = importlib.util.spec_from_file_location("prepare_ml1m", HERE / "prepare_ml1m.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def verify_archive(path: Path) -> None:
    if base.file_hash(path) != ARCHIVE_SHA256 or base.file_hash(path, "md5") != ARCHIVE_MD5:
        raise ValueError(f"MovieLens 20M checksum mismatch for {path}; download it with resume (curl -C -) first")


def read_ratings(archive: Path) -> tuple[np.ndarray, int]:
    from zipfile import ZipFile

    with ZipFile(archive) as source, source.open("ml-20m/ratings.csv") as stream:
        frame = pd.read_csv(stream, dtype={"userId": np.int64, "movieId": np.int64, "rating": np.float64,
                                           "timestamp": np.int64})
    if list(frame.columns) != ["userId", "movieId", "rating", "timestamp"] or len(frame) != EXPECTED_ROWS:
        raise ValueError("unexpected MovieLens 20M ratings schema or row count")
    rating = frame["rating"].to_numpy()
    if np.any((rating < 0.5) | (rating > 5.0)) or np.any(np.round(rating * 2) != rating * 2):
        raise ValueError("unexpected MovieLens 20M rating values")
    rows = np.column_stack([frame["userId"].to_numpy(), frame["movieId"].to_numpy(),
                            np.floor(rating).astype(np.int64), frame["timestamp"].to_numpy()]).astype(np.int64)
    if np.any(rows[:, :2] <= 0) or np.any(rows[:, 3] < 0):
        raise ValueError("unexpected MovieLens 20M identifiers or timestamps")
    rows = rows[np.argsort(rows[:, 3], kind="stable")]
    _, first = np.unique(rows[:, :2], axis=0, return_index=True)
    return rows[np.sort(first)], len(rows) - len(first)


def sample(arrays: dict[str, np.ndarray]) -> dict:
    rng = np.random.Generator(np.random.PCG64(SAMPLE_SEED))
    test, validation = arrays["test"], arrays["validation"]
    test_pool = arrays["test_users"]
    validation_pool = np.unique(validation[validation[:, 2] >= base.RELEVANCE_RATING, 0])
    test_sample = np.sort(rng.choice(test_pool, size=min(SAMPLE_USERS, len(test_pool)), replace=False))
    validation_sample = np.sort(rng.choice(validation_pool, size=min(SAMPLE_USERS, len(validation_pool)), replace=False))
    arrays["test"] = test[np.isin(test[:, 0], test_sample)]
    arrays["validation"] = validation[np.isin(validation[:, 0], np.union1d(test_sample, validation_sample))]
    arrays["test_users"] = test_sample
    return {"rng": "PCG64", "seed": SAMPLE_SEED, "size": SAMPLE_USERS,
            "warm_relevant_test_users_before_sampling": int(len(test_pool)),
            "warm_relevant_validation_users_before_sampling": int(len(validation_pool)),
            "test_rows_kept": int(len(arrays["test"])), "validation_rows_kept": int(len(arrays["validation"]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "ml20m")
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "ml-20m.zip"
    verify_archive(archive)
    base.read_ratings = read_ratings
    arrays, manifest = base.prepare(archive)
    manifest["source"] = {"url": SOURCE, "sha256": ARCHIVE_SHA256, "publisher_md5": ARCHIVE_MD5}
    manifest["rating_rule"] = "integer part of the half-star rating; relevance is rating >= 4"
    manifest["evaluation_sample"] = sample(arrays)
    manifest["test_users_by_activity"] = dict(zip(("low", "medium", "high"),
                                                  np.bincount(arrays["activity"][arrays["test_users"]], minlength=3).tolist()))
    with tempfile.TemporaryDirectory(dir=output) as staging:
        artifact = Path(staging) / "ml20m.npz"
        np.savez(artifact, **{key: value.astype("<i8", copy=False) for key, value in arrays.items()})
        manifest["prepared"] = {"filename": artifact.name, "sha256": base.file_hash(artifact)}
        manifest_file = Path(staging) / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        artifact.replace(output / artifact.name)
        manifest_file.replace(output / manifest_file.name)
    print(f"Prepared {len(arrays['test_users'])} sampled test users. See {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
