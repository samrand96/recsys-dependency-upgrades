"""Prepare the protocol-v0 MovieLens 1M per-user chronological split."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from prepare_ml1m import (
    ARCHIVE_MD5,
    ARCHIVE_SHA256,
    RELEVANCE_RATING,
    SOURCE,
    describe_split,
    fetch_archive,
    file_hash,
    read_ratings,
)

SPLITS = ("train", "validation", "test")


def partition_users(rows: np.ndarray) -> tuple[tuple[np.ndarray, ...], dict]:
    """Assign globally time-ordered, deduplicated rows using each user's own cutoffs."""
    if not len(rows):
        raise ValueError("MovieLens sensitivity preparation requires rating events")
    order = np.argsort(rows[:, 0], kind="stable")
    ordered_users = rows[order, 0]
    boundaries = np.flatnonzero(ordered_users[1:] != ordered_users[:-1]) + 1
    assignments = np.zeros(len(rows), dtype=np.int8)
    validation_tie_users = test_tie_users = shared_cutoff_users = 0
    for positions in np.split(order, boundaries):
        times = rows[positions, 3]
        train_floor, test_floor = len(times) * 70 // 100, len(times) * 80 // 100
        validation_time, test_time = times[[train_floor, test_floor]]
        train_end, test_start = np.searchsorted(times, (validation_time, test_time))
        assignments[positions[train_end:test_start]] = 1
        assignments[positions[test_start:]] = 2
        validation_tie_users += int(train_end < train_floor)
        test_tie_users += int(test_start < test_floor)
        shared_cutoff_users += int(validation_time == test_time)
    return tuple(rows[assignments == code] for code in range(len(SPLITS))), {
        "source_users": len(boundaries) + 1,
        "users_with_validation_boundary_moved_for_ties": validation_tie_users,
        "users_with_test_boundary_moved_for_ties": test_tie_users,
        "users_with_coincident_boundaries": shared_cutoff_users,
    }


def prepare(archive: Path) -> tuple[dict[str, np.ndarray], dict]:
    rows, duplicate_count = read_ratings(archive)
    splits, cutoff_counts = partition_users(rows)
    user_ids = np.unique(splits[0][:, 0])
    item_ids = np.unique(splits[0][:, 1])
    if not user_ids.size or not item_ids.size:
        raise ValueError("per-user chronological rules left an empty training partition")
    triples, counts = zip(*(describe_split(split, user_ids, item_ids) for split in splits))
    train, validation, test = triples
    train_counts = np.bincount(train[:, 0], minlength=len(user_ids)).astype(np.int64)
    ordered_counts = np.sort(train_counts)
    low_cut, high_cut = (int(ordered_counts[(len(user_ids) * n + 2) // 3 - 1]) for n in (1, 2))
    activity = (train_counts > low_cut).astype(np.int64) + (train_counts > high_cut).astype(np.int64)
    test_users = np.unique(test[test[:, 2] >= RELEVANCE_RATING, 0])
    arrays = {"train": train, "validation": validation, "test": test, "user_ids": user_ids, "item_ids": item_ids,
              "train_counts": train_counts, "activity": activity, "test_users": test_users}
    manifest = {
        "format_version": 1,
        "source": {"url": SOURCE, "sha256": ARCHIVE_SHA256, "publisher_md5": ARCHIVE_MD5},
        "duplicate_user_item_interactions_removed": duplicate_count,
        "selected_split": {"method": "per_user_chronological", "target_percentages": [70, 10, 20],
                           "boundary_rule": "per-user floor position, same-timestamp events enter later interval",
                           "warm_relevant_test_users": len(test_users), **cutoff_counts},
        "splits": dict(zip(SPLITS, counts)),
        "training_users": len(user_ids),
        "training_items": len(item_ids),
        "activity_count_cutoffs_inclusive": [low_cut, high_cut],
        "test_users_by_activity": dict(zip(("low", "medium", "high"), np.bincount(activity[test_users], minlength=3).tolist())),
    }
    return arrays, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path(__file__).resolve().parent
    parser.add_argument("--archive", type=Path, default=directory / "ml-1m.zip")
    parser.add_argument("--output-dir", type=Path, default=directory / "sensitivity")
    args = parser.parse_args()
    if args.output_dir.resolve() == directory:
        parser.error("use a separate output directory to preserve the primary dataset and manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    fetch_archive(args.archive)
    arrays, manifest = prepare(args.archive)
    with tempfile.TemporaryDirectory(dir=args.output_dir) as staging:
        staged = Path(staging)
        artifact = staged / "ml1m.npz"
        np.savez(artifact, **{key: value.astype("<i8", copy=False) for key, value in arrays.items()})
        manifest["prepared"] = {"filename": artifact.name, "sha256": file_hash(artifact)}
        manifest_file = staged / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        artifact.replace(args.output_dir / artifact.name)
        manifest_file.replace(args.output_dir / manifest_file.name)
    print(f"Prepared {len(arrays['test_users'])} warm sensitivity test users. See {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
