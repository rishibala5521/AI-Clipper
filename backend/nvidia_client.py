import json
import logging
import random
import re
import time

from openai import OpenAI
from pydantic import BaseModel

from errors import AppError

# Shared with the Gemini client so both providers behave the same way.
from llm_client import (
    _FINAL_ERRORS,
    LLMResult,
    UsageInfo,
    _is_rate_limited,
    _is_transient,
    _status_code,
)

logger = logging.getLogger(__name__)

JSON_MODES = {"prompt_only", "json_schema", "nvext_guided_json", "guided_json"}
THINKING_MODES = {"default", "off", "low", "on"}
MAX_RETRY_AFTER_SECONDS = 60.0
DEFAULT_RATE_LIMIT_WAIT_SECONDS = 10.0


def extract_json_text(text: str | None) -> str:
    """Return the JSON object inside a model reply.

    Removes <think>...</think> blocks and Markdown fences by taking everything
    from the first { to the last }.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the model reply")
    return cleaned[start : end + 1]


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        value = float(headers.get("retry-after"))
    except (TypeError, ValueError, AttributeError):
        return None
    return value if value >= 0 else None


def _int(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _extract_usage(completion) -> UsageInfo:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return UsageInfo()
    details = getattr(usage, "completion_tokens_details", None)
    # In this API style, completion_tokens already includes any reasoning tokens.
    return UsageInfo(
        input_tokens=_int(getattr(usage, "prompt_tokens", 0)),
        output_tokens=_int(getattr(usage, "completion_tokens", 0)),
        thought_tokens=_int(getattr(details, "reasoning_tokens", 0)),
        total_tokens=_int(getattr(usage, "total_tokens", 0)),
    )


class NvidiaClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        json_mode: str = "prompt_only",
        thinking: str = "default",
        max_output_tokens: int = 4000,
        max_attempts: int = 4,
        base_delay_seconds: float = 2.0,
        timeout_seconds: float = 120.0,
        sleep=time.sleep,
        jitter=None,
        client=None,
    ) -> None:
        if json_mode not in JSON_MODES:
            raise ValueError(
                f"NVIDIA_JSON_MODE must be one of {sorted(JSON_MODES)}, got {json_mode!r}"
            )
        if thinking not in THINKING_MODES:
            raise ValueError(
                f"NVIDIA_THINKING must be one of {sorted(THINKING_MODES)}, got {thinking!r}"
            )
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._json_mode = json_mode
        self._thinking = thinking
        self._max_output_tokens = max_output_tokens
        self._max_attempts = max(1, max_attempts)
        self._base_delay = base_delay_seconds
        self._timeout = timeout_seconds
        self._sleep = sleep
        self._jitter = jitter or (lambda: random.uniform(0, 1))
        self._client = client

    def _get_client(self):
        # Created lazily so the app still starts (and /health works) without a key.
        if not self._api_key or not self._model:
            raise AppError(
                "The AI service is not configured. Set NVIDIA_API_KEY and NVIDIA_MODEL in .env.",
                503,
                "llm_not_configured",
            )
        if self._client is None:
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=self._timeout,
                max_retries=0,  # we do our own retries
            )
        return self._client

    def _thinking_kwargs(self) -> dict | None:
        if self._thinking == "off":
            return {"enable_thinking": False}
        if self._thinking == "low":
            return {"enable_thinking": True, "low_effort": True}
        if self._thinking == "on":
            return {"enable_thinking": True}
        return None  # "default": send nothing and let the model decide

    def _build_request(self, prompt: str, schema_model: type[BaseModel]) -> dict:
        schema = schema_model.model_json_schema()
        full_prompt = (
            f"{prompt}\n\nReply with JSON only, matching this JSON schema:\n{json.dumps(schema)}"
        )
        request: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0.2,
            "max_tokens": self._max_output_tokens,
        }
        if self._json_mode == "json_schema":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_model.__name__, "schema": schema},
            }
        elif self._json_mode == "nvext_guided_json":
            request["extra_body"] = {"nvext": {"guided_json": schema}}
        elif self._json_mode == "guided_json":
            request["extra_body"] = {"guided_json": schema}

        template_kwargs = self._thinking_kwargs()
        if template_kwargs is not None:
            request.setdefault("extra_body", {})["chat_template_kwargs"] = template_kwargs
        return request

    def _retry_delay(self, attempt: int, rate_limited: bool, retry_after: float | None) -> float:
        backoff = self._base_delay * 2 ** (attempt - 1)
        if rate_limited:
            wait = retry_after if retry_after is not None else DEFAULT_RATE_LIMIT_WAIT_SECONDS
            return min(max(wait, backoff), MAX_RETRY_AFTER_SECONDS)
        return backoff

    def generate_structured(self, prompt: str, schema_model: type[BaseModel]) -> LLMResult:
        client = self._get_client()
        request = self._build_request(prompt, schema_model)

        failure = "unavailable"
        for attempt in range(1, self._max_attempts + 1):
            started = time.monotonic()
            try:
                completion = client.chat.completions.create(**request)
            except Exception as exc:
                if not _is_transient(exc):
                    logger.error(
                        "NVIDIA request failed (not retryable): %s %s: %s",
                        type(exc).__name__, _status_code(exc), str(exc)[:300],
                    )
                    raise AppError("The AI request failed.", 502, "llm_request_failed") from exc

                rate_limited = _is_rate_limited(exc)
                retry_after = _retry_after_seconds(exc) if rate_limited else None
                if retry_after is not None and retry_after > MAX_RETRY_AFTER_SECONDS:
                    # Waiting minutes or hours inside a request makes no sense: stop now.
                    logger.error("NVIDIA quota reached, provider says retry in %.0fs", retry_after)
                    message, status, code = _FINAL_ERRORS["rate_limited"]
                    raise AppError(message, status, code) from exc

                failure = "rate_limited" if rate_limited else "unavailable"
                logger.warning(
                    "NVIDIA transient error (attempt %d/%d): %s %s",
                    attempt, self._max_attempts, type(exc).__name__, _status_code(exc),
                )
                delay = self._retry_delay(attempt, rate_limited, retry_after)
            else:
                choices = getattr(completion, "choices", None) or []
                content = None
                finish_reason = None
                if choices:
                    content = getattr(choices[0].message, "content", None)
                    finish_reason = getattr(choices[0], "finish_reason", None)

                if finish_reason == "length":
                    message_obj = choices[0].message
                    reasoning_text = getattr(message_obj, "reasoning_content", None) or getattr(
                        message_obj, "reasoning", None
                    )
                    cut_usage = _extract_usage(completion)
                    logger.error(
                        "NVIDIA answer was cut off at max_tokens=%d (completion_tokens=%d, "
                        "reasoning_tokens=%d, reasoning_text_returned=%s, thinking=%s)",
                        self._max_output_tokens, cut_usage.output_tokens, cut_usage.thought_tokens,
                        bool(reasoning_text), self._thinking,
                    )
                    raise AppError(
                        "The AI's answer was cut off. Set NVIDIA_THINKING=off or increase "
                        "NVIDIA_MAX_OUTPUT_TOKENS.",
                        502,
                        "llm_output_truncated",
                    )
                try:
                    parsed = schema_model.model_validate_json(extract_json_text(content))
                except ValueError:
                    failure = "invalid_response"
                    logger.warning(
                        "NVIDIA returned unusable output (attempt %d/%d)",
                        attempt, self._max_attempts,
                    )
                    delay = self._retry_delay(attempt, False, None)
                else:
                    usage = _extract_usage(completion)
                    logger.info(
                        "NVIDIA call ok in %.1fs (%s): input=%d output=%d reasoning=%d total=%d tokens",
                        time.monotonic() - started, self._model,
                        usage.input_tokens, usage.output_tokens, usage.thought_tokens, usage.total_tokens,
                    )
                    return LLMResult(parsed=parsed, usage=usage)

            if attempt < self._max_attempts:
                self._sleep(delay + self._jitter())

        message, status, code = _FINAL_ERRORS[failure]
        raise AppError(message, status, code)