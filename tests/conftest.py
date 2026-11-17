import pytest


@pytest.fixture
def tmp_ledger(tmp_path, monkeypatch):
    """Point the ledger at a throwaway file; manifests still come from the repo."""
    path = tmp_path / "ledger.duckdb"
    monkeypatch.setenv("SILENTRANK_LEDGER", str(path))
    return path
