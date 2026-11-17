"""P3 Surprise and P4 SAR: environment groups, generated definitions and case-ID rules."""

import pytest

from silentrank.orchestrator import envs, pilot


def test_surprise_pairs_skip_pandas_transitions():
    pairs = envs.pilot_pairs("surprise", "dev")
    assert set(pairs) == {"N1", "N2", "N3", "S1", "S2", "S3", "S4"}
    assert pairs["N1"] == ("py312-np1.26.4-sp1.13.1-pd2.2.3-surprise1.1.5",
                           "py312-np2.0.2-sp1.13.1-pd2.2.3-surprise1.1.5")


def test_sar_pairs_include_pandas_transitions():
    pairs = envs.pilot_pairs("sar", "dev")
    assert set(pairs) == {"N1", "N2", "N3", "S1", "S2", "S3", "S4", "D1", "D2"}
    assert pairs["D2"][1] == "py312-np2.0.2-sp1.13.1-pd3.0.6-recommenders1.2.1"


def test_implicit_pairs_unchanged():
    pairs = envs.pilot_pairs("implicit", "dev")
    assert set(pairs) == {"N1", "N2", "N3", "S1", "S2", "S3", "S4"}
    assert pairs["S1"] == ("py312-np1.26.4-sp1.11.4-pd2.2.3-implicit0.7.3",
                           "py312-np1.26.4-sp1.12.0-pd2.2.3-implicit0.7.3")


def test_define_variant_env_writes_pins_once(tmp_path, monkeypatch):
    monkeypatch.setenv("SILENTRANK_ROOT", str(tmp_path))
    (tmp_path / "envs").mkdir()
    env_id = "py312-np2.0.2-sp1.14.1-pd2.2.3-surprise1.1.5"
    project = envs.define_variant_env(env_id, "surprise")
    text = project.read_text(encoding="utf-8")
    for pin in ('"numpy==2.0.2"', '"scipy==1.14.1"', '"pandas==2.2.3"', '"scikit-surprise==1.1.5"',
                f'exclude-newer = "{envs.VARIANT_EXCLUDE_NEWER}"'):
        assert pin in text
    project.write_text("edited\n", encoding="utf-8")
    envs.define_variant_env(env_id, "surprise")
    assert project.read_text(encoding="utf-8") == "edited\n", "existing definitions are never overwritten"
    with pytest.raises(ValueError):
        envs.define_variant_env(env_id, "sar")


@pytest.mark.parametrize("case_id", ["N1-P3-sar", "D1-P3-svd", "N1-P3-svd-sensitivity", "B1-P4-sar",
                                     "N1-P4-svd", "N1-P3", "S1-P4"])
def test_invalid_p3_p4_case_ids(case_id, tmp_ledger):
    with pytest.raises(ValueError):
        pilot._pilot_state(case_id)


@pytest.mark.parametrize("case_id", ["N1-P3-svd", "S4-P3-svd", "D2-P4-sar", "S1-P4-sar"])
def test_valid_p3_p4_case_ids_parse(case_id):
    assert pilot.CASE_ID.fullmatch(case_id)


def test_worker_accepts_new_pipelines():
    from silentrank.worker import __main__ as worker

    with pytest.raises(SystemExit) as raised:
        worker.main(["--pipeline", "unknown", "--mode", "retrain", "--data", "x", "--output-dir", "y"])
    assert raised.value.code == 2
    source = worker.__file__
    text = open(source, encoding="utf-8").read()
    assert '"surprise_svd", "sar"' in text and '"--similarity"' in text
