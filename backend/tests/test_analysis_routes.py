import subprocess

import pytest

from app import create_app
from errors import AppError
from llm_client import LLMResult, UsageInfo
from schemas import LLMChunkResult, LLMClip, LLMScores
from transcript_service import RawTranscript

URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
# 120 segments of 5 seconds = a 600 second transcript, which becomes 3 chunks.
ENTRIES = [{"text": f"line {i} words", "start": i * 5, "duration": 5} for i in range(120)]


def clip(first, last, level, quote):
    return LLMClip(
        start_segment=first,
        end_segment=last,
        title=f"Clip {first}",
        hook="hook",
        reason="reason",
        transcript_quote=quote,
        scores=LLMScores(hook=level, value=level, clarity=level, completeness=level),
    )


def default_results():
    return [
        LLMChunkResult(
            clips=[
                clip(2, 6, 5, "line 2 words"),    # 10s-35s, score 100
                clip(10, 11, 4, "line 10 words"),  # 10s long -> extended to 15s, score 80
                clip(3, 7, 3, "line 3 words"),     # overlaps the first -> duplicate
            ]
        ),
        LLMChunkResult(clips=[clip(60, 63, 3, "line 60 words")]),  # 300s-320s, score 60
        LLMChunkResult(clips=[]),
    ]


class FakeProvider:
    def fetch(self, video_id, languages):
        return RawTranscript(video_id=video_id, language_code="en", is_generated=True, entries=ENTRIES)


class ScriptedLLM:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def generate_structured(self, prompt, schema_model):
        self.calls += 1
        outcome = self.results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResult(
            parsed=outcome,
            usage=UsageInfo(input_tokens=100, output_tokens=20, thought_tokens=0, total_tokens=120),
        )


def make_client(tmp_path, llm=None):
    app = create_app(
        {"TESTING": True, "DATABASE_URL": f"sqlite:///{tmp_path / 'test.db'}"},
        transcript_provider=FakeProvider(),
        llm_client=llm or ScriptedLLM(default_results()),
    )
    return app.test_client()


def analyze(client, **extra):
    return client.post("/api/analyze", json={"url": URL, **extra})


# ---------- analyze ----------


def test_analyze_returns_a_ranked_clean_timeline(tmp_path):
    response = analyze(make_client(tmp_path))
    assert response.status_code == 201
    data = response.get_json()

    assert data["video_id"] == "dQw4w9WgXcQ"
    assert data["stats"]["chunk_count"] == 3
    assert data["stats"]["draft_count"] == 4
    assert data["stats"]["duplicates_removed"] == 1
    assert data["stats"]["total_tokens"] == 360

    clips = data["clips"]
    assert [c["rank"] for c in clips] == [1, 2, 3]
    assert [c["overall_score"] for c in clips] == [100.0, 80.0, 60.0]
    assert (clips[0]["start"], clips[0]["end"]) == (10.0, 35.0)
    assert (clips[1]["start"], clips[1]["end"]) == (50.0, 65.0)
    assert "extended" in clips[1]["flags"]
    assert clips[2]["start"] == 300.0
    assert all(c["status"] == "pending" and c["edited"] is False for c in clips)
    assert all(c["quote_verified"] for c in clips)


def test_max_clips_limits_the_result(tmp_path):
    data = analyze(make_client(tmp_path), max_clips=2).get_json()
    assert len(data["clips"]) == 2


def test_invalid_url_never_reaches_the_ai(tmp_path):
    llm = ScriptedLLM(default_results())
    response = make_client(tmp_path, llm).post("/api/analyze", json={"url": "https://evil.com/x"})
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_video_url"
    assert llm.calls == 0


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"url": URL, "max_clips": 0},
        {"url": URL, "max_clips": 500},
        {"url": URL, "min_duration": 2},
        {"url": URL, "unknown_field": 1},
    ],
)
def test_bad_analyze_bodies_return_400(tmp_path, body):
    response = make_client(tmp_path).post("/api/analyze", json=body)
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_min_duration_must_be_smaller_than_max(tmp_path):
    llm = ScriptedLLM(default_results())
    response = make_client(tmp_path, llm).post(
        "/api/analyze", json={"url": URL, "min_duration": 40, "max_duration": 30}
    )
    assert response.status_code == 400
    assert llm.calls == 0


def test_ai_errors_are_returned_as_json(tmp_path):
    llm = ScriptedLLM([AppError("slow down", 429, "llm_rate_limited")])
    response = analyze(make_client(tmp_path, llm))
    assert response.status_code == 429
    assert response.get_json()["error"]["code"] == "llm_rate_limited"


# ---------- get ----------


def test_saved_analysis_can_be_read_back(tmp_path):
    client = make_client(tmp_path)
    created = analyze(client).get_json()
    fetched = client.get(f"/api/analyses/{created['id']}")
    assert fetched.status_code == 200
    assert fetched.get_json() == created


def test_unknown_analysis_returns_404(tmp_path):
    response = make_client(tmp_path).get("/api/analyses/doesnotexist")
    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "analysis_not_found"


# ---------- patch ----------


def first_clip(client):
    return analyze(client).get_json()["clips"][0]


def test_editing_a_clip_keeps_the_original_ai_times(tmp_path):
    client = make_client(tmp_path)
    original = first_clip(client)
    response = client.patch(
        f"/api/clips/{original['id']}",
        json={"title": "  My better title ", "start": 12.0, "end": 30.0},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["title"] == "My better title"
    assert (data["start"], data["end"], data["duration"]) == (12.0, 30.0, 18.0)
    assert (data["ai_start"], data["ai_end"]) == (10.0, 35.0)
    assert data["edited"] is True


def test_editing_only_the_title_works(tmp_path):
    client = make_client(tmp_path)
    original = first_clip(client)
    data = client.patch(f"/api/clips/{original['id']}", json={"title": "New"}).get_json()
    assert data["title"] == "New"
    assert (data["start"], data["end"]) == (original["start"], original["end"])


@pytest.mark.parametrize(
    "body,code",
    [
        ({"start": 30.0, "end": 20.0}, "invalid_timestamps"),
        ({"end": 601.0}, "out_of_range"),
        ({"start": 0.0, "end": 200.0}, "invalid_duration"),
        ({"start": 10.0, "end": 10.5}, "invalid_duration"),
        ({}, "invalid_request"),
        ({"title": ""}, "invalid_request"),
        ({"status": "approved"}, "invalid_request"),
    ],
)
def test_bad_clip_edits_are_rejected(tmp_path, body, code):
    client = make_client(tmp_path)
    clip_id = first_clip(client)["id"]
    response = client.patch(f"/api/clips/{clip_id}", json=body)
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == code


def test_editing_an_unknown_clip_returns_404(tmp_path):
    response = make_client(tmp_path).patch("/api/clips/nope", json={"title": "x"})
    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "clip_not_found"


# ---------- review ----------


def test_review_status_is_saved(tmp_path):
    client = make_client(tmp_path)
    created = analyze(client).get_json()
    clip_id = created["clips"][0]["id"]

    approved = client.post(f"/api/clips/{clip_id}/review", json={"status": "approved"})
    assert approved.status_code == 200
    assert approved.get_json()["status"] == "approved"

    fetched = client.get(f"/api/analyses/{created['id']}").get_json()
    assert fetched["clips"][0]["status"] == "approved"

    pending = client.post(f"/api/clips/{clip_id}/review", json={"status": "pending"})
    assert pending.get_json()["status"] == "pending"


def test_invalid_review_status_is_rejected(tmp_path):
    client = make_client(tmp_path)
    clip_id = first_clip(client)["id"]
    response = client.post(f"/api/clips/{clip_id}/review", json={"status": "maybe"})
    assert response.status_code == 400


def test_reviewing_an_unknown_clip_returns_404(tmp_path):
    response = make_client(tmp_path).post("/api/clips/nope/review", json={"status": "approved"})
    assert response.status_code == 404


# ---------- the app never makes video ----------


def test_no_video_files_and_no_subprocesses(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("starting another program is not allowed here")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.chdir(tmp_path)

    client = make_client(tmp_path)
    data = analyze(client).get_json()
    client.patch(f"/api/clips/{data['clips'][0]['id']}", json={"title": "x"})
    client.post(f"/api/clips/{data['clips'][0]['id']}/review", json={"status": "approved"})

    video_like = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".mp3", ".m4a", ".wav"}
    created = {p.suffix.lower() for p in tmp_path.rglob("*") if p.is_file()}
    assert not (created & video_like)
    assert not any(key in data["clips"][0] for key in ("video_url", "file", "file_path"))

def test_clips_include_readable_time_labels(tmp_path):
    clip = analyze(make_client(tmp_path)).get_json()["clips"][0]
    assert (clip["start"], clip["end"]) == (10.0, 35.0)  # raw numbers are unchanged
    assert clip["start_label"] == "0:10"
    assert clip["end_label"] == "0:35"
    assert clip["time_range"] == "0:10 - 0:35"
    assert clip["duration_label"] == "0:25"