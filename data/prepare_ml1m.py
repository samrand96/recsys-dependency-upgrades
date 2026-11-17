"""Prepare MovieLens 1M in the pinned orchestrator environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from http.client import HTTPException
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

SOURCE = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ARCHIVE_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
ARCHIVE_MD5 = "c4d9eecfca2ab87c1945afe126590906"
SPLIT_TARGETS = (80, 70)
FALLBACK_TRAIN_PERCENTAGES = range(10, 86)
VALIDATION_PERCENTAGE = 10
MIN_TEST_USERS = 1500
RELEVANCE_RATING = 4


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def verify_archive(path: Path) -> None:
    if file_hash(path) != ARCHIVE_SHA256 or file_hash(path, "md5") != ARCHIVE_MD5:
        raise ValueError(f"MovieLens 1M checksum mismatch for {path}")


def fetch_archive(path: Path) -> None:
    if path.exists():
        verify_archive(path)
        return
    with tempfile.TemporaryDirectory(dir=path.parent) as staging:
        download = Path(staging) / path.name
        for attempt in range(3):
            try:
                with urlopen(SOURCE, timeout=60) as response:
                    download.write_bytes(response.read())
                verify_archive(download)
            except (OSError, HTTPException, ValueError):
                if attempt == 2:
                    raise
            else:
                download.replace(path)
                return


def read_ratings(archive: Path) -> tuple[np.ndarray, int]:
    with ZipFile(archive) as source, source.open("ml-1m/ratings.dat") as ratings:
        rows = np.loadtxt((line.replace(b"::", b",") for line in ratings), delimiter=",", dtype=np.int64)
    if (
        rows.shape != (1_000_209, 4)
        or np.any(rows[:, :2] <= 0)
        or np.any((rows[:, 2] < 1) | (rows[:, 2] > 5))
        or np.any(rows[:, 3] < 0)
    ):
        raise ValueError("unexpected MovieLens 1M ratings schema or values")
    rows = rows[np.argsort(rows[:, 3], kind="stable")]
    _, first = np.unique(rows[:, :2], axis=0, return_index=True)
    return rows[np.sort(first)], len(rows) - len(first)


def describe_split(rows: np.ndarray, user_ids: np.ndarray, item_ids: np.ndarray) -> tuple[np.ndarray, dict]:
    warm_user = np.isin(rows[:, 0], user_ids)
    warm_item = np.isin(rows[:, 1], item_ids)
    warm = warm_user & warm_item
    relevant = rows[:, 2] >= RELEVANCE_RATING
    eligible_rows = rows[warm]
    triples = eligible_rows[:, :3].copy()
    triples[:, 0] = np.searchsorted(user_ids, triples[:, 0])
    triples[:, 1] = np.searchsorted(item_ids, triples[:, 1])
    return triples, {
        "interactions": len(rows),
        "warm_interactions": int(warm.sum()),
        "warm_users": int(np.unique(rows[warm, 0]).size),
        "cold_user_interactions": int((~warm_user).sum()),
        "cold_item_interactions": int((~warm_item).sum()),
        "cold_users": int(np.unique(rows[~warm_user, 0]).size),
        "cold_items": int(np.unique(rows[~warm_item, 1]).size),
        "excluded_relevant_interactions": int((relevant & ~warm).sum()),
        "warm_relevant_users": int(np.unique(rows[warm & relevant, 0]).size),
    }


def first_warm_time(rows: np.ndarray) -> np.ndarray:
    users, items, _, times = rows.T
    first_user = np.full(int(users.max()) + 1, np.iinfo(np.int64).max, dtype=np.int64)
    first_item = np.full(int(items.max()) + 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first_user, users, times)
    np.minimum.at(first_item, items, times)
    return np.maximum(first_user[users], first_item[items])


def global_split_user_ceiling(rows: np.ndarray, warm_time: np.ndarray) -> int:
    """Maximum warm relevant test users over global cutoffs with zero validation gap."""
    users, _, ratings, times = rows.T
    starts = warm_time + 1
    ends = times + 1
    viable = (ratings >= RELEVANCE_RATING) & (starts < ends)
    users, starts, ends = users[viable], starts[viable], ends[viable]
    order = np.lexsort((starts, users))
    events: Counter[int] = Counter()
    last_user, last_end = 0, 0
    for user, start, end in zip(users[order], starts[order], ends[order]):
        if user != last_user or start > last_end:
            events[int(start)] += 1
        else:
            if end <= last_end:
                continue
            events[int(last_end)] += 1  # Cancel the end of the previous interval.
        events[int(end)] -= 1
        last_user, last_end = user, end

    active = ceiling = 0
    for moment in sorted(events):
        active += events[moment]
        ceiling = max(ceiling, active)
    return ceiling


def split_attempt(rows: np.ndarray, warm_time: np.ndarray, train_percent: int) -> dict:
    times = rows[:, 3]
    test_percent = 100 - train_percent - VALIDATION_PERCENTAGE
    train_cut = times[len(rows) * train_percent // 100]
    test_cut = times[len(rows) * (100 - test_percent) // 100]
    eligible = (times >= test_cut) & (warm_time < train_cut) & (rows[:, 2] >= RELEVANCE_RATING)
    return {
        "target_percentages": [train_percent, VALIDATION_PERCENTAGE, test_percent],
        "validation_start_unix": int(train_cut),
        "test_start_unix": int(test_cut),
        "interactions_at_validation_start": int(np.count_nonzero(times == train_cut)),
        "interactions_at_test_start": int(np.count_nonzero(times == test_cut)),
        "warm_relevant_test_users": int(np.unique(rows[eligible, 0]).size),
    }


def prepare(archive: Path) -> tuple[dict[str, np.ndarray], dict]:
    rows, duplicate_count = read_ratings(archive)
    warm_time = first_warm_time(rows)
    ceiling = global_split_user_ceiling(rows, warm_time)
    attempts = []
    for train_percent in SPLIT_TARGETS:
        selected = split_attempt(rows, warm_time, train_percent)
        attempts.append(selected)
        if selected["warm_relevant_test_users"] >= MIN_TEST_USERS:
            break
    else:
        fallback = (split_attempt(rows, warm_time, percent) for percent in FALLBACK_TRAIN_PERCENTAGES)
        selected = max(fallback, key=lambda split: (split["warm_relevant_test_users"], split["target_percentages"][0]))
        attempts.append(selected)
    if selected["warm_relevant_test_users"] < MIN_TEST_USERS <= ceiling:
        raise ValueError(f"no planned split has {MIN_TEST_USERS} warm relevant test users: {attempts}")

    train_end, test_start = np.searchsorted(
        rows[:, 3], [selected["validation_start_unix"], selected["test_start_unix"]]
    )
    splits = (rows[:train_end], rows[train_end:test_start], rows[test_start:])
    user_ids = np.unique(splits[0][:, 0])
    item_ids = np.unique(splits[0][:, 1])
    triples, counts = zip(*(describe_split(split, user_ids, item_ids) for split in splits))
    train, validation, test = triples
    train_counts = np.bincount(train[:, 0], minlength=len(user_ids)).astype(np.int64)
    ordered_counts = np.sort(train_counts)
    low_cut, high_cut = (int(ordered_counts[(len(user_ids) * n + 2) // 3 - 1]) for n in (1, 2))
    activity = ((train_counts > low_cut).astype(np.int64) + (train_counts > high_cut)).astype(np.int64)
    test_users = np.unique(test[test[:, 2] >= RELEVANCE_RATING, 0])
    arrays = {
        "train": train,
        "validation": validation,
        "test": test,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "train_counts": train_counts,
        "activity": activity,
        "test_users": test_users,
    }
    details = {
        "format_version": 1,
        "source": {"url": SOURCE, "sha256": ARCHIVE_SHA256, "publisher_md5": ARCHIVE_MD5},
        "duplicate_user_item_interactions_removed": duplicate_count,
        "global_split_warm_relevant_test_user_ceiling_without_validation_gap": ceiling,
        "target_minimum_warm_relevant_test_users": MIN_TEST_USERS,
        "split_attempts": attempts,
        "selected_split": selected,
        "splits": dict(zip(("train", "validation", "test"), counts)),
        "training_users": len(user_ids),
        "training_items": len(item_ids),
        "activity_count_cutoffs_inclusive": [low_cut, high_cut],
        "test_users_by_activity": dict(zip(("low", "medium", "high"), np.bincount(activity[test_users], minlength=3).tolist())),
    }
    return arrays, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "ml-1m.zip"
    fetch_archive(archive)
    arrays, manifest = prepare(archive)
    with tempfile.TemporaryDirectory(dir=output) as staging:
        staged = Path(staging)
        artifact = staged / "ml1m.npz"
        np.savez(artifact, **{key: value.astype("<i8", copy=False) for key, value in arrays.items()})
        manifest["prepared"] = {"filename": artifact.name, "sha256": file_hash(artifact)}
        manifest_file = staged / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        artifact.replace(output / artifact.name)
        manifest_file.replace(output / manifest_file.name)
    print(f"Prepared {len(arrays['test_users'])} warm test users. See {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
