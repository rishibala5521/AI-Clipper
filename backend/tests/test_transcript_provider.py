import pytest
import requests

from errors import AppError
from transcript_service import YouTubeTranscriptProvider


class FakeSnippet:
    def __init__(self, text, start, duration):
        self.text = text
        self.start = start
        self.duration = duration


class FakeFetched(list):
    language_code = "en"
    is_generated = True


class FakeApi:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def fetch(self, video_id, languages):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def good_result():
    return FakeFetched([FakeSnippet("hello", 0.0, 2.0)])


def test_network_error_is_retried_then_succeeds():
    api = FakeApi([requests.exceptions.ConnectionError("dropped"), good_result()])
    sleeps = []
    provider = YouTubeTranscriptProvider(api=api, sleep=sleeps.append)
    raw = provider.fetch("abc", ["en"])
    assert api.calls == 2
    assert sleeps == [2]
    assert raw.entries == [{"text": "hello", "start": 0.0, "duration": 2.0}]


def test_persistent_network_error_returns_503():
    api = FakeApi([requests.exceptions.ConnectionError("dropped")] * 3)
    provider = YouTubeTranscriptProvider(api=api, sleep=lambda s: None)
    with pytest.raises(AppError) as exc_info:
        provider.fetch("abc", ["en"])
    assert exc_info.value.code == "transcript_source_unreachable"
    assert exc_info.value.status_code == 503
    assert api.calls == 3