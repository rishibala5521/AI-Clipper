import pytest

from config import Config


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    """Every test gets its own throwaway database, whatever the environment says."""
    monkeypatch.setattr(Config, "DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")