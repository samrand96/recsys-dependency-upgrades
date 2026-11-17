import duckdb
import pytest

from silentrank import service
from silentrank.evaluator import identity as evaluator_identity
from silentrank.orchestrator import ledger, manifests

PILOT_IDS = {"N1", "N2", "N3", "S1", "S2", "S3", "S4", "D1", "D2", "B1"}


def test_manifests_contain_the_pilot_transitions():
    entries = {e["id"]: e for e in manifests.load_all()}
    assert PILOT_IDS <= set(entries)
    assert all(entries[i]["split"] == "dev" for i in PILOT_IDS), "pilot cases are development cases"
    assert entries["N1"]["change"] == {"package": "numpy", "old": "1.26.4", "new": "2.0.2"}


def test_init_creates_schema_and_is_idempotent(tmp_ledger):
    first = service.init()
    second = service.init()
    assert first["transitions_synced"] == second["transitions_synced"] >= len(PILOT_IDS)
    with ledger.reader() as con:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        n = con.execute("SELECT count(*) FROM transitions").fetchone()[0]
        version = con.execute("SELECT value FROM meta WHERE key = 'evaluator_version'").fetchone()[0]
    assert set(ledger.TABLES) <= tables
    assert n == first["transitions_synced"]
    assert version == first["evaluator_version"] == second["evaluator_version"] == evaluator_identity()


def test_resync_keeps_screening_results(tmp_ledger):
    service.init()
    with ledger.writer() as con:
        con.execute("UPDATE transitions SET env_old = 'env-a', resolved_diff_json = '{}' WHERE transition_id = 'N1'")
    service.init()
    with ledger.reader() as con:
        row = con.execute("SELECT env_old, resolved_diff_json FROM transitions WHERE transition_id = 'N1'").fetchone()
    assert row == ("env-a", "{}")


def test_list_cases_reports_ledger_state(tmp_ledger):
    before = {row["id"]: row for row in service.list_cases()}
    assert before["N1"]["ledger"] == "not initialised"
    service.init()
    after = {row["id"]: row for row in service.list_cases(collection="systematic_single")}
    assert "B1" not in after
    assert after["N1"]["ledger"] == {"n_cases": 0, "n_runs": 0, "n_failed_runs": 0}


def test_get_case(tmp_ledger):
    service.init()
    detail = service.get_case("N1")
    assert detail["manifest"]["family"] == "numpy"
    assert detail["transition"][0]["transition_id"] == "N1"
    with pytest.raises(KeyError):
        service.get_case("does-not-exist")


def test_read_before_init_is_a_clear_error(tmp_ledger):
    with pytest.raises(ledger.LedgerMissing):
        service.query_ledger("SELECT 1")


def test_query_returns_rows_and_truncates(tmp_ledger):
    service.init()
    result = service.query_ledger("SELECT transition_id FROM transitions ORDER BY 1", max_rows=3)
    assert result["columns"] == ["transition_id"]
    assert result["row_count"] == 3 and result["truncated"] is True


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM runs",
        "DROP TABLE runs",
        "INSERT INTO meta VALUES ('x', 'y')",
        "UPDATE transitions SET status = 'done'",
        "SELECT 1; DROP TABLE runs",
        "COPY transitions TO 'out.csv'",
        "ATTACH 'other.duckdb'",
        "not sql at all",
    ],
)
def test_query_rejects_anything_but_one_select(tmp_ledger, sql):
    service.init()
    with pytest.raises(service.QueryRejected):
        service.query_ledger(sql)
    with ledger.reader() as con:
        assert con.execute("SELECT count(*) FROM transitions").fetchone()[0] >= len(PILOT_IDS)


def test_query_cannot_read_files(tmp_ledger, tmp_path):
    service.init()
    secret = tmp_path / "secret.csv"
    secret.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(duckdb.Error):
        service.query_ledger(f"SELECT * FROM read_csv('{secret.as_posix()}')")
