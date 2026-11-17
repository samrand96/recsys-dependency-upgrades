"""Read portable worker inputs and write explicit array/JSON artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path

import numpy as np


def sha256_file(path: Path, expected: str | None = None) -> str:
    with path.open("rb") as stream:
        actual = file_digest(stream, "sha256").hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


@dataclass(frozen=True)
class FrozenData:
    train: np.ndarray
    validation: np.ndarray
    user_ids: np.ndarray
    item_ids: np.ndarray
    evaluation_users: np.ndarray
    seen_interactions: np.ndarray


def load_data(path: Path, split: str) -> FrozenData:
    with np.load(path, allow_pickle=False) as frozen:
        train = frozen["train"]
        validation = frozen["validation"]
        user_ids = frozen["user_ids"]
        item_ids = frozen["item_ids"]
        if split == "test":
            evaluation_users = frozen["test_users"]
            seen_interactions = np.concatenate((train, validation))
        else:
            evaluation_users = np.unique(validation[validation[:, 2] >= 4, 0])
            seen_interactions = train
    if not evaluation_users.size:
        raise ValueError(f"the {split} split has no eligible evaluation users")
    return FrozenData(train, validation, user_ids, item_ids, evaluation_users, seen_interactions)
