from types import SimpleNamespace

import pytest

from errors import AppError
from nvidia_client import NvidiaClient, extract_json_text
from schemas import LLMChunkResult

VALID_JSON = '{"clips": []}'


def completion(content, finish_reason="stop", usage="default"):
    if usage == "default":
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        )
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class FakeApiError(Exception):
    def __init__(self, status_code, retry_after=None):
        super().__init__(f"error {status_code}")
        self.status_code = status_code
        headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
        self.response = SimpleNamespace(headers=headers)


class FakeCompletions:
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


def make_client(outcomes, max_attempts=3, json_mode="prompt_only", thinking="default"):
    completions = FakeCompletions(outcomes)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    sleeps = []
    client = NvidiaClient(
        api_key="key",
        model="test-model",
        json_mode=json_mode,
        thinking=thinking,
        max_attempts=max_attempts,
        base_delay_seconds=1.0,
        sleep=sleeps.append,
        jitter=lambda: 0.0,
        client=sdk,
    )
    return client, completions, sleeps


def test_success_parses_output_and_usage():
    client, completions, sleeps = make_client([completion(VALID_JSON)])
    result = client.generate_structured("find clips", LLMChunkResult)
    assert result.parsed.clips == []
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 20
    assert result.usage.thought_tokens == 5
    assert result.usage.total_tokens == 120
    kwargs = completions.last_kwargs
    assert kwargs["model"] == "test-model"
    assert "JSON schema" in kwargs["messages"][0]["content"]
    assert "response_format" not in kwargs and "extra_body" not in kwargs
    assert sleeps == []


def test_missing_usage_defaults_to_zero():
    client, _, _ = make_client([completion(VALID_JSON, usage=None)])
    assert client.generate_structured("x", LLMChunkResult).usage.total_tokens == 0


def test_think_block_and_code_fences_are_stripped():
    text = "<think>maybe {this} first</think>\n```json\n" + VALID_JSON + "\n```"
    client, _, _ = make_client([completion(text)])
    assert client.generate_structured("x", LLMChunkResult).parsed.clips == []


def test_json_schema_mode_sends_response_format():
    client, completions, _ = make_client([completion(VALID_JSON)], json_mode="json_schema")
    client.generate_structured("x", LLMChunkResult)
    assert completions.last_kwargs["response_format"]["type"] == "json_schema"


def test_nvext_guided_json_mode_sends_extra_body():
    client, completions, _ = make_client([completion(VALID_JSON)], json_mode="nvext_guided_json")
    client.generate_structured("x", LLMChunkResult)
    assert "guided_json" in completions.last_kwargs["extra_body"]["nvext"]


def test_guided_json_mode_sends_top_level_extra_body():
    client, completions, _ = make_client([completion(VALID_JSON)], json_mode="guided_json")
    client.generate_structured("x", LLMChunkResult)
    assert "guided_json" in completions.last_kwargs["extra_body"]


def test_unknown_json_mode_is_rejected():
    with pytest.raises(ValueError):
        NvidiaClient(api_key="k", model="m", json_mode="magic")


def test_transient_server_error_is_retried():
    client, completions, sleeps = make_client([FakeApiError(503), completion(VALID_JSON)])
    client.generate_structured("x", LLMChunkResult)
    assert completions.calls == 2
    assert sleeps == [1.0]


def test_timeout_is_retried():
    class FakeTimeout(Exception):
        pass

    client, completions, _ = make_client([FakeTimeout("Request timed out."), completion(VALID_JSON)])
    client.generate_structured("x", LLMChunkResult)
    assert completions.calls == 2


def test_persistent_server_error_gives_unavailable():
    client, completions, sleeps = make_client([FakeApiError(503)] * 3)
    with pytest.raises(AppError) as exc_info:
        client.generate_structured("x", LLMChunkResult)
    assert exc_info.value.code == "llm_unavailable"
    assert completions.calls == 3
    assert len(sleeps) == 2


def test_rate_limit_waits_for_retry_after_then_succeeds():
    client, completions, sleeps = make_client(
        [FakeApiError(429, retry_after=5), completion(VALID_JSON)]
    )
    client.generate_structured("x", LLMChunkResult)
    assert completions.calls == 2
    assert sleeps == [5.0]


def test_rate_limit_without_header_uses_default_wait():
    client, _, sleeps = make_client([FakeApiError(429), completion(VALID_JSON)])
    client.generate_structured("x", LLMChunkResult)
    assert sleeps == [10.0]


def test_long_retry_after_fails_fast():
    client, completions, sleeps = make_client([FakeApiError(429, retry_after=3600)])
    with pytest.raises(AppError) as exc_info:
        client.generate_structured("x", LLMChunkResult)
    assert exc_info.value.code == "llm_rate_limited"
    assert exc_info.value.status_code == 429
    assert completions.calls == 1
    assert sleeps == []


def test_persistent_rate_limit_gives_429_after_retries():
    client, completions, sleeps = make_client([FakeApiError(429, retry_after=2)] * 3)
    with pytest.raises(AppError) as exc_info:
        client.generate_structured("x", LLMChunkResult)
    assert exc_info.value.code == "llm_rate_limited"
    assert completions.calls == 3
    assert sleeps == [2.0, 2.0]


def test_non_transient_error_fails_immediately():
    client, completions, sleeps = make_client([FakeApiError(400)])
    with pytest.raises(AppError) as exc_info:
        client.generate_structured("x", LLMChunkResult)
    assert exc_info.value.code == "llm_request_failed"
    assert completions.calls == 1
    assert sleeps == []


def test_truncated_answer_fails_without_retry():
    client, completions, _ = make_client([completion('{"clips": [', finish_reason="length")])
    with pytest.raises(AppError) as exc_info:
        client.generate_structured("x", LLMChunkResult)
    assert exc_info.value.code == "llm_output_truncated"
    assert completions.calls == 1


def test_invalid_json_then_valid_succeeds():
    client, completions, _ = make_client([completion("not json"), completion(VALID_JSON)])
    client.generate_structured("x", LLMChunkResult)
    assert completions.calls == 2


@pytest.mark.parametrize("bad", ["nope", None, ""])
def test_always_invalid_output_fails(bad):
    client, _, _ = make_client([completion(bad)] * 3)
    with pytest.raises(AppError) as exc_info:
        client.generate_structured("x", LLMChunkResult)
    assert exc_info.value.code == "llm_invalid_response"


def test_empty_choices_are_treated_as_invalid():
    empty = SimpleNamespace(choices=[], usage=None)
    client, _, _ = make_client([empty] * 3)
    with pytest.raises(AppError) as exc_info:
        client.generate_structured("x", LLMChunkResult)
    assert exc_info.value.code == "llm_invalid_response"


def test_missing_configuration_is_reported():
    client = NvidiaClient(api_key="", model="")
    with pytest.raises(AppError) as exc_info:
        client.generate_structured("x", LLMChunkResult)
    assert exc_info.value.code == "llm_not_configured"


def test_extract_json_text():
    assert extract_json_text('noise {"a": 1} more') == '{"a": 1}'
    with pytest.raises(ValueError):
        extract_json_text("no braces here")
    with pytest.raises(ValueError):
        extract_json_text(None)

def test_thinking_default_sends_no_template_kwargs():
    client, completions, _ = make_client([completion(VALID_JSON)])
    client.generate_structured("x", LLMChunkResult)
    assert "extra_body" not in completions.last_kwargs


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("off", {"enable_thinking": False}),
        ("low", {"enable_thinking": True, "low_effort": True}),
        ("on", {"enable_thinking": True}),
    ],
)
def test_thinking_modes_send_chat_template_kwargs(mode, expected):
    client, completions, _ = make_client([completion(VALID_JSON)], thinking=mode)
    client.generate_structured("x", LLMChunkResult)
    assert completions.last_kwargs["extra_body"]["chat_template_kwargs"] == expected


def test_thinking_and_guided_json_are_combined():
    client, completions, _ = make_client(
        [completion(VALID_JSON)], json_mode="guided_json", thinking="off"
    )
    client.generate_structured("x", LLMChunkResult)
    extra = completions.last_kwargs["extra_body"]
    assert "guided_json" in extra
    assert extra["chat_template_kwargs"] == {"enable_thinking": False}


def test_unknown_thinking_mode_is_rejected():
    with pytest.raises(ValueError):
        NvidiaClient(api_key="k", model="m", thinking="maybe")