import pytest

from app import create_app


@pytest.fixture
def client():
    app = create_app({"TESTING": True})
    return app.test_client()


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_unknown_route_returns_json_404(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "not_found"