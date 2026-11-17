"""Operations behind the command-line interface."""

from __future__ import annotations

import datetime as dt
import decimal
from typing import Any

import duckdb

from silentrank import paths
from silentrank.evaluator import identity as evaluator_identity
from silentrank.orchestrator import ledger, manifests


def _json_safe(value: Any) -> Any:
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    return value


def _rows(con, sql: str, params: list | None = None) -> list[dict]:
    cursor = con.execute(sql, params or [])
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, (_json_safe(v) for v in row))) for row in cursor.fetchall()]


def init() -> dict:
    """Create the ledger schema if needed and sync the transition manifests into it."""
    entries = manifests.load_all()
    evaluator_version = evaluator_identity()
    with ledger.writer() as con:
        synced = manifests.sync_transitions(con, entries)
        con.execute(
            "INSERT INTO meta VALUES ('evaluator_version', ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            [evaluator_version],
        )
    return {
        "ledger": str(paths.ledger_path()),
        "schema_version": ledger.SCHEMA_VERSION,
        "evaluator_version": evaluator_version,
        "transitions_synced": synced,
        "leads_in_manifests": sum(1 for e in entries if e["kind"] == "lead"),
    }


def _ledger_counts() -> dict[str, dict] | None:
    """Per-transition progress from the ledger, or None when it does not exist yet."""
    try:
        with ledger.reader() as con:
            rows = _rows(
                con,
                """SELECT t.transition_id,
                          t.status AS ledger_status,
                          count(DISTINCT c.case_id) AS n_cases,
                          count(DISTINCT r.run_id) AS n_runs,
                          count(DISTINCT r.run_id) FILTER (WHERE r.status <> 'ok') AS n_failed_runs
                   FROM transitions t
                   LEFT JOIN cases c ON c.transition_id = t.transition_id
                   LEFT JOIN runs r ON r.case_id = c.case_id
                   GROUP BY ALL""",
            )
    except ledger.LedgerMissing:
        return None
    return {row["transition_id"]: row for row in rows}


def list_cases(collection: str | None = None, status: str | None = None) -> list[dict]:
    """Transitions and issue-guided leads from the manifests, with ledger progress."""
    counts = _ledger_counts()
    result = []
    for entry in manifests.load_all():
        if collection and entry["collection"] != collection:
            continue
        if status and entry["status"] != status:
            continue
        row = {
            "id": entry["id"],
            "kind": entry["kind"],
            "collection": entry["collection"],
            "status": entry["status"],
        }
        if entry["kind"] == "transition":
            row.update(family=entry["family"], split=entry["split"], change=entry["change"])
            progress = (counts or {}).get(entry["id"])
            row["ledger"] = (
                "not initialised" if counts is None
                else "not synced" if progress is None
                else {k: progress[k] for k in ("n_cases", "n_runs", "n_failed_runs")}
            )
        else:
            row.update(
                project=entry.get("project"),
                visibility=entry.get("visibility"),
                url=entry.get("url"),
            )
        result.append(row)
    return result


def get_case(case_id: str) -> dict:
    """Everything known about one transition, lead or ledger case id."""
    entry = next((e for e in manifests.load_all() if e["id"] == case_id), None)
    detail: dict[str, Any] = {"id": case_id, "manifest": entry}
    try:
        with ledger.reader() as con:
            detail["transition"] = _rows(con, "SELECT * FROM transitions WHERE transition_id = ?", [case_id])
            detail["cases"] = _rows(
                con, "SELECT * FROM cases WHERE transition_id = ? OR case_id = ? ORDER BY case_id", [case_id, case_id]
            )
            case_ids = [row["case_id"] for row in detail["cases"]]
            if case_ids:
                marks = ",".join("?" for _ in case_ids)
                detail["runs"] = _rows(
                    con,
                    f"""SELECT case_id, side, mode, status, count(*) AS n
                        FROM runs WHERE case_id IN ({marks}) GROUP BY ALL ORDER BY ALL""",
                    case_ids,
                )
                detail["comparisons"] = _rows(con, f"SELECT * FROM comparisons WHERE case_id IN ({marks})", case_ids)
                detail["verdicts"] = _rows(con, f"SELECT * FROM verdicts WHERE case_id IN ({marks})", case_ids)
                detail["adjudications"] = _rows(con, f"SELECT * FROM adjudications WHERE case_id IN ({marks})", case_ids)
    except ledger.LedgerMissing as exc:
        detail["ledger"] = str(exc)
    if entry is None and not detail.get("transition") and not detail.get("cases"):
        raise KeyError(f"no transition, lead or case with id {case_id!r}")
    return detail


class QueryRejected(ValueError):
    pass


def query_ledger(sql: str, max_rows: int = 200) -> dict:
    """Run exactly one SELECT against the ledger on a read-only connection.

    Three independent guards: the statement must parse as a single SELECT, the
    connection is opened read-only, and DuckDB's access to files and the network
    is switched off for the connection.
    """
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as exc:
        raise QueryRejected(f"could not parse SQL: {exc}") from exc
    if len(statements) != 1:
        raise QueryRejected("exactly one statement is allowed")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise QueryRejected(f"only SELECT is allowed, got {statements[0].type.name}")
    max_rows = max(1, min(int(max_rows), 5000))
    with ledger.reader() as con:
        con.execute("SET enable_external_access = false")
        cursor = con.execute(sql)
        columns = [col[0] for col in cursor.description]
        fetched = cursor.fetchmany(max_rows + 1)
    truncated = len(fetched) > max_rows
    rows = [[_json_safe(v) for v in row] for row in fetched[:max_rows]]
    return {"columns": columns, "rows": rows, "row_count": len(rows), "truncated": truncated}
