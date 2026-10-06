import logging
from dataclasses import dataclass
from typing import Protocol
import time

import requests

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from errors import AppError
from schemas import TranscriptResult
from transcript_processor import normalize_segments

logger = logging.getLogger(__name__)
NETWORK_ATTEMPTS = 3


@dataclass
class RawTranscript:
    video_id: str
    language_code: str
    is_generated: bool
    entries: list[dict]  # each: {"text": str, "start": float, "duration": float}


class TranscriptProvider(Protocol):
    def fetch(self, video_id: str, languages: list[str]) -> RawTranscript: ...


class YouTubeTranscriptProvider:
    def __init__(self, api=None, sleep=time.sleep) -> None:
        self._api = api or YouTubeTranscriptApi()
        self._sleep = sleep

    def _fetch_with_retry(self, video_id: str, languages: list[str]):
        last_error: Exception | None = None
        for attempt in range(1, NETWORK_ATTEMPTS + 1):
            try:
                return self._api.fetch(video_id, languages=languages)
            except requests.exceptions.RequestException as exc:
                last_error = exc
                logger.warning(
                    "Transcript network error (attempt %d/%d): %s",
                    attempt, NETWORK_ATTEMPTS, type(exc).__name__,
                )
                if attempt < NETWORK_ATTEMPTS:
                    self._sleep(attempt * 2)
        raise AppError(
            "Could not reach the transcript source. Check your internet connection and try again.",
            503,
            "transcript_source_unreachable",
        ) from last_error

    def fetch(self, video_id: str, languages: list[str]) -> RawTranscript:
        try:
            fetched = self._fetch_with_retry(video_id, languages)  # <- changed line
        except TranscriptsDisabled:
            ...  # everything below stays exactly as it was

        entries = [
            {"text": s.text, "start": s.start, "duration": s.duration} for s in fetched
        ]
        return RawTranscript(
            video_id=video_id,
            language_code=fetched.language_code,
            is_generated=fetched.is_generated,
            entries=entries,
        )


def get_transcript(
    video_id: str, provider: TranscriptProvider, languages: list[str]
) -> TranscriptResult:
    raw = provider.fetch(video_id, languages)
    segments = normalize_segments(raw.entries)
    return TranscriptResult(
        video_id=raw.video_id,
        language_code=raw.language_code,
        is_generated=raw.is_generated,
        transcript_end=segments[-1].end,
        segments=segments,
    )