import pytest

from repository import AnalysisRepository, resolve_db_path


def test_resolve_db_path_rejects_other_databases():
    with pytest.raises(ValueError):
        resolve_db_path("postgresql://user@host/db")


def test_relative_path_is_inside_the_backend_folder():
    path = resolve_db_path("sqlite:///x.db")
    assert path.endswith("x.db")
    assert "backend" in path.replace("\\", "/").lower()


def test_creating_the_repository_does_not_touch_the_disk(tmp_path):
    db_file = tmp_path / "later.db"
    AnalysisRepository(str(db_file))
    assert not db_file.exists()


def test_unknown_ids_return_none(tmp_path):
    repo = AnalysisRepository(str(tmp_path / "t.db"))
    assert repo.get_analysis("nope") is None
    assert repo.get_clip("nope") is None
    assert repo.set_review("nope", "approved") is None
    assert repo.update_clip("nope", title="x", start=0.0, end=5.0) is None