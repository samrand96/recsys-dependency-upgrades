"""The results ledger: one DuckDB file, append-only by convention.

Rules (README): failures are recorded, runs are never deleted or overwritten,
re-runs get a new ``repeat`` index. This module therefore offers no delete helpers.

DuckDB allows one read-write process *or* several read-only ones, so writers
should open, write and close quickly; ``connect`` retries briefly on a lock.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

from silentrank import paths

SCHEMA_VERSION = "1"

DDL = [
    "CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR)",
    """CREATE TABLE IF NOT EXISTS hosts (
        host_id VARCHAR PRIMARY KEY, cpu_model VARCHAR, cpu_flags VARCHAR, cores INTEGER,
        ram_gb DOUBLE, os VARCHAR, kernel VARCHAR, gpu VARCHAR, gpu_driver VARCHAR,
        first_seen TIMESTAMP, raw_json VARCHAR)""",
    """CREATE TABLE IF NOT EXISTS envs (
        env_id VARCHAR PRIMARY KEY, python VARCHAR, lock_sha256 VARCHAR, image_digest VARCHAR,
        status VARCHAR CHECK (status IN ('planned','ok','resolve_failed','build_failed','import_failed')),
        built_at TIMESTAMP, freeze_json VARCHAR, runtime_json VARCHAR, notes VARCHAR)""",
    """CREATE TABLE IF NOT EXISTS testsuite (
        env_id VARCHAR, package VARCHAR,
        status VARCHAR CHECK (status IN ('passed','failed','unavailable','not_runnable')),
        n_pass INTEGER, n_fail INTEGER, log_path VARCHAR, ran_at TIMESTAMP,
        PRIMARY KEY (env_id, package))""",
    """CREATE TABLE IF NOT EXISTS transitions (
        transition_id VARCHAR PRIMARY KEY, family VARCHAR,
        collection VARCHAR CHECK (collection IN ('systematic_single','systematic_snapshot','issue_guided','synthetic')),
        split VARCHAR CHECK (split IN ('dev','heldout')),
        env_old VARCHAR, env_new VARCHAR,
        direct_change_json VARCHAR, held_fixed_json VARCHAR, resolved_diff_json VARCHAR,
        selection_reason VARCHAR, status VARCHAR, manifest_file VARCHAR)""",
    """CREATE TABLE IF NOT EXISTS cases (
        case_id VARCHAR PRIMARY KEY, transition_id VARCHAR, pipeline VARCHAR, dataset VARCHAR,
        config_json VARCHAR, applicable BOOLEAN, applicability_evidence VARCHAR,
        exclusion_reason VARCHAR, created_at TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS runs (
        run_id VARCHAR PRIMARY KEY, case_id VARCHAR, env_id VARCHAR,
        side VARCHAR CHECK (side IN ('old','new','old_repeat','rollback')),
        mode VARCHAR CHECK (mode IN ('prep','inference','retrain')),
        seed INTEGER, repeat INTEGER, threads INTEGER,
        status VARCHAR CHECK (status IN ('ok','crash','timeout','contract_fail')),
        host_id VARCHAR, started_at TIMESTAMP, wall_s DOUBLE, cpu_s DOUBLE, peak_rss_mb DOUBLE,
        warnings_n INTEGER, artifact_dir VARCHAR, code_commit VARCHAR, evaluator_version VARCHAR)""",
    """CREATE TABLE IF NOT EXISTS stage_hashes (
        run_id VARCHAR, stage VARCHAR, canonical_sha256 VARCHAR, shape VARCHAR, dtype VARCHAR,
        PRIMARY KEY (run_id, stage))""",
    """CREATE TABLE IF NOT EXISTS comparisons (
        comparison_id VARCHAR PRIMARY KEY, case_id VARCHAR, run_a VARCHAR, run_b VARCHAR,
        kind VARCHAR CHECK (kind IN ('AA','AB','rollback')),
        n_users INTEGER, n_seeds INTEGER, overlap10_mean DOUBLE, users_changed_frac DOUBLE,
        canonical_equal BOOLEAN, score_maxabs DOUBLE, score_rel_l2 DOUBLE,
        d_ndcg_mean DOUBLE, ci_lo DOUBLE, ci_hi DOUBLE, method VARCHAR, created_at TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS verdicts (
        case_id VARCHAR, delta DOUBLE,
        verdict VARCHAR CHECK (verdict IN ('contract_failure','harmful','equivalent','improved','inconclusive')),
        rule_version VARCHAR, decided_at TIMESTAMP, PRIMARY KEY (case_id, delta))""",
    """CREATE TABLE IF NOT EXISTS adjudications (
        case_id VARCHAR, mechanism VARCHAR,
        cause_class VARCHAR CHECK (cause_class IN ('upstream_defect','downstream_incompatibility','intended_semantic_change','numerical_variation','unresolved')),
        evidence VARCHAR, adjudicator VARCHAR, decided_at DATE)""",
    """CREATE TABLE IF NOT EXISTS diagnostic_replays (
        replay_id VARCHAR PRIMARY KEY, case_id VARCHAR, comparison_id VARCHAR,
        phase VARCHAR, env_id VARCHAR, host_id VARCHAR, status VARCHAR,
        started_at TIMESTAMP, wall_s DOUBLE, artifact_dir VARCHAR, detail VARCHAR)""",
    "CREATE TABLE IF NOT EXISTS costs (run_id VARCHAR, phase VARCHAR, wall_s DOUBLE, cpu_s DOUBLE, notes VARCHAR)",
]

TABLES = [
    "meta", "hosts", "envs", "testsuite", "transitions", "cases", "runs",
    "stage_hashes", "comparisons", "verdicts", "adjudications", "diagnostic_replays", "costs",
]


class LedgerMissing(RuntimeError):
    """Raised when a read is attempted before ``silentrank init`` created the ledger."""


def connect(read_only: bool = False, path: Path | None = None, retries: int = 6) -> duckdb.DuckDBPyConnection:
    path = Path(path) if path else paths.ledger_path()
    if read_only and not path.exists():
        raise LedgerMissing(f"ledger not initialised at {path}; run `silentrank init` first")
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return duckdb.connect(str(path), read_only=read_only)
        except duckdb.IOException as exc:  # another process holds the file
            last = exc
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"ledger at {path} is locked by another process: {last}")


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    for statement in DDL:
        con.execute(statement)
    con.execute(
        "INSERT INTO meta VALUES ('schema_version', ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        [SCHEMA_VERSION],
    )


@contextmanager
def writer(path: Path | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    """Short-lived read-write connection with the schema ensured."""
    con = connect(read_only=False, path=path)
    try:
        init_schema(con)
        yield con
    finally:
        con.close()


@contextmanager
def reader(path: Path | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    con = connect(read_only=True, path=path)
    try:
        yield con
    finally:
        con.close()
