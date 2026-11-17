import json

from silentrank.orchestrator import env_tests, envs, ledger


def test_pilot_pairs_match_the_pinned_transitions():
    base = envs.pilot_pairs("base")
    assert base["N2"] == ("py312-np2.0.2-sp1.13.1-pd2.2.3", "py312-np2.1.3-sp1.13.1-pd2.2.3")
    assert base["S4"] == ("py312-np2.0.2-sp1.14.1-pd2.2.3", "py312-np2.0.2-sp1.15.3-pd2.2.3")
    implicit = envs.pilot_pairs("implicit")
    assert implicit["N2"] == tuple(f"{env_id}-implicit0.7.3" for env_id in base["N2"])
    assert "D1" not in implicit
    assert envs.pilot_pairs("snapshot")["B1"] == ("py312-snapshot-2024-06-01", "py312-snapshot-2026-11-01")


def test_resolved_diff_captures_transitive_changes_and_removals():
    diff = envs._resolved_diff(*envs.pilot_pairs("base")["D2"])
    assert [entry["version"] for entry in diff["pandas"]["old"]] == ["2.3.3"]
    assert [entry["version"] for entry in diff["pandas"]["new"]] == ["3.0.6"]
    assert diff["pytz"]["new"] == []
    assert "numpy" not in diff


def test_upstream_suite_report_distinguishes_failure_and_timeout(tmp_path):
    result = tmp_path / "pytest.xml"
    assert env_tests.report(124, result)["status"] == "not_runnable"
    result.write_text(
        '<testsuites><testsuite tests="10" errors="0" failures="1" skipped="2"/></testsuites>',
        encoding="utf-8",
    )
    assert env_tests.report(1, result) == {"status": "failed", "n_pass": 7, "n_fail": 1, "n_skip": 2}
    assert env_tests.report(2, result)["status"] == "not_runnable"


def test_suite_recovery_requires_a_complete_result_and_preserves_recorded_rows(tmp_path, tmp_ledger, monkeypatch):
    monkeypatch.setenv("SILENTRANK_ROOT", str(tmp_path))
    env_id = "test-env"
    image = "sha256:test-image"
    report = {"packages": {"numpy": {}, "pandas": {}}}
    envs._record_env(env_id, image_digest=image, runtime_json=json.dumps(report), status="ok")

    root = tmp_path / "results" / "screening" / env_id
    completed = root / "20261112T133200-test-numpy-complete" / "pytest.log"
    incomplete = root / "20261112T133205-test-pandas-incomplete" / "pytest.log"
    header = json.dumps({"image": image, "command": [f"/venvs/{env_id}/bin/python"]})
    for log in (completed, incomplete):
        log.parent.mkdir(parents=True)
        log.write_text(header + "\n\nstdout\n", encoding="utf-8")
    with completed.open("a", encoding="utf-8") as log:
        log.write('\nresult\n{"status": "passed", "n_pass": 5, "n_fail": 0}\n')

    result = envs.recover_suites()
    assert [row["package"] for row in result["recovered"]] == ["numpy"]
    assert [row["package"] for row in result["remaining"]] == ["pandas"]
    assert envs.recover_suites()["recovered"] == []
    with ledger.reader() as con:
        assert con.execute("SELECT package, status, n_pass FROM testsuite").fetchall() == [("numpy", "passed", 5)]
