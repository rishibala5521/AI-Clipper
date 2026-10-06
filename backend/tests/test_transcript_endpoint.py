import pytest

from app import create_app
from errors import AppError
from transcript_service import RawTranscript

VIDEO_ID = "dQw4w9WgXcQ"
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


class FakeProvider:
    def __init__(self, entries=None, error=None):
        self.entries = entries if entries is not None else []
        self.error = error
        self.calls = []

    def fetch(self, video_id, languages):
        self.calls.append((video_id, languages))
        if self.error:
            raise self.error
        return RawTranscript(
            video_id=video_id,
            language_code="en",
            is_generated=True,
            entries=self.entries,
        )


@pytest.fixture
def make_client():
    def _make(provider):
        app = create_app({"TESTING": True}, transcript_provider=provider)
        return app.test_client()

    return _make


def test_success_returns_segments(make_client):
    provider = FakeProvider(entries=[{"text": "hi", "start": 0, "duration": 2}])
    response = make_client(provider).post("/api/transcript", json={"url": URL})
    assert response.status_code == 200
    data = response.get_json()
    assert data["video_id"] == VIDEO_ID
    assert data["transcript_end"] == 2.0
    assert data["segments"][0]["text"] == "hi"


def test_provider_receives_video_id_not_url(make_client):
    provider = FakeProvider(entries=[{"text": "hi", "start": 0, "duration": 2}])
    make_client(provider).post("/api/transcript", json={"url": URL})
    assert provider.calls == [(VIDEO_ID, ["en"])]


def test_invalid_url_returns_400(make_client):
    response = make_client(FakeProvider()).post(
        "/api/transcript", json={"url": "https://evil.com/watch?v=dQw4w9WgXcQ"}
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_video_url"


def test_missing_body_returns_400(make_client):
    response = make_client(FakeProvider()).post("/api/transcript")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_missing_url_field_returns_400(make_client):
    response = make_client(FakeProvider()).post("/api/transcript", json={})
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_provider_error_is_returned_as_json(make_client):
    provider = FakeProvider(error=AppError("No transcript.", 404, "no_transcript_found"))
    response = make_client(provider).post("/api/transcript", json={"url": URL})
    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "no_transcript_found"


def test_empty_transcript_returns_422(make_client):
    response = make_client(FakeProvider(entries=[])).post("/api/transcript", json={"url": URL})
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "empty_transcript"