import pytest

from errors import AppError
from llm_client import GeminiClient
from schemas import LLMChunkResult

VALID_JSON = '{"clips": []}'


class FakeUsage:
    total_input_tokens = 100
    total_output_tokens = 20
    total_thought_tokens = 300
    total_tokens = 420


class FakeInteraction:
    def __init__(self, text, usage=None):
        self.output_text = text
        self.usage = usage


class FakeApiError(Exception):
    def __init__(self, code):
        super().__init__(f"error {code}")
        self.code = code


class FakeInteractions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeSdkClient:
    def __init__(self, outcomes):
        self.interactions = FakeInteractions(outcomes)


def make_client(outcomes, max_attempts=3):
    sdk = FakeSdkClient(outcomes)
    sleeps = []
    client = GeminiClient(
        api_key="key",
        model="test-model",
        max_attempts=max_attempts,
        base_delay_seconds=1.0,
        sleep=sleeps.append,
        client=sdk,
    )
    return client, sdk, sleeps


def test_success_parses_output_and_usage():
    client, sdk, sleeps = make_client([FakeInteraction(VALID_JSON, FakeUsage())])
    result = client.generate_structured("hello", LLMChunkResult)
    assert result.parsed.clips == []
    assert result.usage.total_tokens == 420
    assert result.usage.thought_tokens == 300
    assert sdk.interactions.last_kwargs["model"] == "test-model"
    assert sdk.interactions.last_kwargs["response_format"][0]["mime_type"] == "application/json"
    assert sleeps == []


def test_missing_usage_defaults_to_zero():
    client, _, _ = make_client([FakeInteraction(VALID_JSON, usage=None)])
    assert client.generate_structured("x", LLMChunkResult).usage.total_tokens == 0


def test_transient_error_is_retried():
    client, sdk, sleeps = make_client([FakeApiError(503), FakeInteraction(VALID_JSON)])
    client.generate_structured("x", LLMChunkResult)
    assert sdk.interactions.calls == 2
    assert len(sleeps) == 1


def test_non_transient_error_fails_immediately():
    client, sdk, sleeps = make_client([FakeApiError(400)])
    with pytest.raises(AppError) as exc_info:
        client.generate_structured("x", LLMChunkResult)
    assert exc_info.value.code == "llm_request_failed"
    assert sdk.interactions.calls == 1
    assert sleeps == []


def test_persistent_server_error_gives_unavailable():
    client, sdk, sleeps = make_client([FakeApiError(503)] * 3)
    with pytest.raises(AppError) as exc_info:
        client.generate_structured("x", LLMChunkResult)
    assert exc_info.value.code == "llm_unavailable"
    assert sdk.interactions.calls == 3
    assert len(sleeps) == 2


def test_persistent_rate_limit_gives_429():
    client, _, _ = make_client([FakeApiError(429)] * 3)
    with pytest.raises(AppError) as exc_info:
        client.generate_structured("x", LLMChunkResult)
    assert exc_info.value.code == "llm_rate_limited"
    assert exc_info.value.status_code == 429


def test_invalid_json_then_valid_succeeds():
    client, sdk, _ = make_client([FakeInteraction("not json"), FakeInteraction(VALID_JSON)])
    client.generate_structured("x", LLMChunkResult)
    assert sdk.interactions.calls == 2


def test_always_invalid_output_fails():
    client, _, _ = make_client([FakeInteraction("nope")] * 3)
    with pytest.raises(AppError) as exc_info:
        client.generate_structured("x", LLMChunkResult)
    assert exc_info.value.code == "llm_invalid_response"


def test_none_output_is_treated_as_invalid():
    client, _, _ = make_client([FakeInteraction(None)] * 3)
    with pytest.raises(AppError) as exc_info:
        client.generate_structured("x", LLMChunkResult)
    assert exc_info.value.code == "llm_invalid_response"


def test_missing_configuration_is_reported():
    client = GeminiClient(api_key="", model="")
    with pytest.raises(AppError) as exc_info:
        client.generate_structured("x", LLMChunkResult)
    assert exc_info.value.code == "llm_not_configured"