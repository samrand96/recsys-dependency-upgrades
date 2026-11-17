"""Locations of the repo, manifests and results. Overridable for tests."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Project root: ``SILENTRANK_ROOT`` if set, else the folder holding pyproject.toml."""
    env = os.environ.get("SILENTRANK_ROOT")
    if env:
        return Path(env).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "silentrank").is_dir():
            return parent
    return Path.cwd().resolve()


def manifests_dir() -> Path:
    return repo_root() / "protocol" / "cases"


def results_dir() -> Path:
    return repo_root() / "results"


def ledger_path() -> Path:
    env = os.environ.get("SILENTRANK_LEDGER")
    return Path(env).resolve() if env else results_dir() / "ledger.duckdb"
