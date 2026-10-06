import logging
import random
import time
from dataclasses import dataclass
from typing import Protocol

from google import genai
from pydantic import BaseModel

from errors import AppError

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

_FINAL_ERRORS = {
    "rate_limited": (
        "The AI service quota or rate limit was reached. Please wait and try again; "
        "if it keeps happening, check your usage and credits with the AI provider.",
        429,
        "llm_rate_limited",
    ),
    "invalid_response": ("The AI returned an unusable answer. Please try again.", 502, "llm_invalid_response"),
    "unavailable": ("The AI service is unavailable right now. Please try again later.", 503, "llm_unavailable"),
}


@dataclass
class UsageInfo:
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "UsageInfo") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.thought_tokens += other.thought_tokens
        self.total_tokens += other.total_tokens


@dataclass
class LLMResult:
    parsed: BaseModel
    usage: UsageInfo


class LLMClient(Protocol):
    def generate_structured(self, prompt: str, schema_model: type[BaseModel]) -> LLMResult: ...


def _status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _is_rate_limited(exc: Exception) -> bool:
    return _status_code(exc) == 429 or "RESOURCE_EXHAUSTED" in str(exc).upper()


def _is_transient(exc: Exception) -> bool:
    code = _status_code(exc)
    if code is not None:
        return code in _TRANSIENT_STATUS_CODES
    text = f"{type(exc).__name__} {exc}".upper()
    return any(marker in text for marker in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "TIMEOUT", "TIMED OUT", "CONNECTION"))


def _extract_usage(interaction) -> UsageInfo:
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return UsageInfo()

    def number(name: str) -> int:
        value = getattr(usage, name, 0)
        return value if isinstance(value, int) else 0

    return UsageInfo(
        input_tokens=number("total_input_tokens"),
        output_tokens=number("total_output_tokens"),
        thought_tokens=number("total_thought_tokens"),
        total_tokens=number("total_tokens"),
    )


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_attempts: int = 4,
        base_delay_seconds: float = 2.0,
        sleep=time.sleep,
        client=None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_attempts = max(1, max_attempts)
        self._base_delay = base_delay_seconds
        self._sleep = sleep
        self._client = client

    def _get_client(self):
        # Created lazily so the app still starts (and /health works) without a key.
        if not self._api_key or not self._model:
            raise AppError(
                "The AI service is not configured. Set GEMINI_API_KEY and GEMINI_MODEL in .env.",
                503,
                "llm_not_configured",
            )
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate_structured(self, prompt: str, schema_model: type[BaseModel]) -> LLMResult:
        client = self._get_client()
        response_format = [
            {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema_model.model_json_schema(),
            }
        ]

        failure = "unavailable"
        for attempt in range(1, self._max_attempts + 1):
            try:
                interaction = client.interactions.create(
                    model=self._model,
                    input=prompt,
                    response_format=response_format,
                )
            except Exception as exc:
                if not _is_transient(exc):
                    logger.error("Gemini request failed (not retryable): %s: %s", type(exc).__name__, exc)
                    raise AppError("The AI request failed.", 502, "llm_request_failed") from exc
                failure = "rate_limited" if _is_rate_limited(exc) else "unavailable"
                logger.warning(
                    "Gemini transient error (attempt %d/%d): %s: %s",
                    attempt, self._max_attempts, type(exc).__name__, exc,
                )
            else:
                try:
                    parsed = schema_model.model_validate_json(interaction.output_text)
                except (ValueError, TypeError) as exc:
                    failure = "invalid_response"
                    logger.warning(
                        "Gemini returned unusable output (attempt %d/%d): %s",
                        attempt, self._max_attempts, type(exc).__name__,
                    )
                else:
                    usage = _extract_usage(interaction)
                    logger.info(
                        "Gemini call ok: input=%d output=%d thoughts=%d total=%d tokens",
                        usage.input_tokens, usage.output_tokens, usage.thought_tokens, usage.total_tokens,
                    )
                    return LLMResult(parsed=parsed, usage=usage)

            if attempt < self._max_attempts:
                self._sleep(self._base_delay * 2 ** (attempt - 1) + random.uniform(0, 1))

        message, status, code = _FINAL_ERRORS[failure]
        raise AppError(message, status, code)