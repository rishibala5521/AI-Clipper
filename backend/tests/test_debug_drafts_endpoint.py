import pytest

from app import create_app
from errors import AppError
from llm_client import LLMResult, UsageInfo
from schemas import LLMChunkResult, LLMClip, LLMScores
from transcript_service import RawTranscript

URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
ENTRIES = [{"text": f"line {i}", "start": i * 5, "duration": 5} for i in range(40)]


class FakeProvider:
    def fetch(self, video_id, languages):
        return RawTranscript(video_id=video_id, language_code="en", is_generated=True, entries=ENTRIES)


class FakeLLM:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def generate_structured(self, prompt, schema_model):
        self.calls += 1
        if self.error:
            raise self.error
        clip = LLMClip(
            start_segment=2,
            end_segment=6,
            title="T",
            hook="H",
            reason="R",
            transcript_quote="Q",
            scores=LLMScores(hook=5, value=4, clarity=4, completeness=3),
        )
        return LLMResult(
            parsed=LLMChunkResult(clips=[clip]),
            usage=UsageInfo(input_tokens=10, output_tokens=5, thought_tokens=20, total_tokens=35),
        )


def make_client(llm):
    app = create_app({"TESTING": True}, transcript_provider=FakeProvider(), llm_client=llm)
    return app.test_client()


def test_drafts_endpoint_returns_real_timestamps():
    response = make_client(FakeLLM()).post("/api/debug/drafts", json={"url": URL})
    assert response.status_code == 200
    data = response.get_json()
    assert data["chunk_count"] == 1
    assert data["draft_count"] == 1
    assert data["drafts"][0]["start"] == 10.0
    assert data["drafts"][0]["end"] == 35.0
    assert data["usage"]["total_tokens"] == 35


def test_llm_errors_are_returned_as_json():
    llm = FakeLLM(error=AppError("slow down", 429, "llm_rate_limited"))
    response = make_client(llm).post("/api/debug/drafts", json={"url": URL})
    assert response.status_code == 429
    assert response.get_json()["error"]["code"] == "llm_rate_limited"


def test_invalid_url_never_reaches_the_llm():
    llm = FakeLLM()
    response = make_client(llm).post("/api/debug/drafts", json={"url": "https://evil.com/x"})
    assert response.status_code == 400
    assert llm.calls == 0