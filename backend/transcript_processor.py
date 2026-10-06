import logging
import math

from errors import AppError
from schemas import TranscriptSegment
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# If more than 20% of entries have broken timing, the whole transcript is suspect.
MAX_BAD_TIMING_RATIO = 0.2


def _parse_entry(entry) -> tuple[float, float, str] | None:
    """Return (start, end, text), None for blank text, or raise ValueError for bad timing."""
    if not isinstance(entry, dict):
        raise ValueError("entry is not a dict")
    start = float(entry["start"])
    duration = float(entry["duration"])
    if not (math.isfinite(start) and math.isfinite(duration)) or start < 0 or duration <= 0:
        raise ValueError("invalid timing")
    text = " ".join(str(entry.get("text", "")).split())  # collapse newlines/spaces
    if not text:
        return None
    return start, start + duration, text


def normalize_segments(raw_entries: list[dict]) -> list[TranscriptSegment]:
    parsed: list[tuple[float, float, str]] = []
    bad_timing = 0

    for entry in raw_entries:
        try:
            item = _parse_entry(entry)
        except (KeyError, TypeError, ValueError):
            bad_timing += 1
            continue
        if item is not None:
            parsed.append(item)

    if not parsed:
        raise AppError("The transcript is empty or unusable.", 422, "empty_transcript")
    if bad_timing / len(raw_entries) > MAX_BAD_TIMING_RATIO:
        raise AppError(
            "The transcript timestamps look unreliable, so it can't be used.",
            422,
            "unreliable_transcript",
        )
    if bad_timing:
        logger.warning("Skipped %d transcript entries with invalid timing", bad_timing)

    parsed.sort(key=lambda item: item[0])

    segments: list[TranscriptSegment] = []
    for index, (start, end, text) in enumerate(parsed):
        # Auto-captions often overlap the next line; trim so segments stay ordered.
        if index + 1 < len(parsed):
            next_start = parsed[index + 1][0]
            if start < next_start < end:
                end = next_start
        segments.append(TranscriptSegment(start=round(start, 3), end=round(end, 3), text=text))
    return segments

@dataclass(frozen=True)
class TranscriptChunk:
    index: int
    first_index: int  # position of the first segment in the FULL transcript
    last_index: int   # position of the last segment in the FULL transcript (inclusive)
    segments: list[TranscriptSegment]

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end


def build_chunks(
    segments: list[TranscriptSegment], chunk_seconds: int, overlap_seconds: int
) -> list[TranscriptChunk]:
    """Split segments into overlapping time windows without ever cutting a segment in half."""
    if chunk_seconds <= 0 or overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be >= 0 and smaller than chunk_seconds")

    chunks: list[TranscriptChunk] = []
    total = len(segments)
    i = 0
    while i < total:
        window_start = segments[i].start
        j = i
        while j < total and segments[j].start < window_start + chunk_seconds:
            j += 1  # j ends up one past the last segment in this window

        chunks.append(
            TranscriptChunk(
                index=len(chunks),
                first_index=i,
                last_index=j - 1,
                segments=segments[i:j],
            )
        )
        if j >= total:
            break

        # The next window starts `overlap_seconds` before this one ended.
        next_start_time = window_start + chunk_seconds - overlap_seconds
        k = i
        while k < total and segments[k].start < next_start_time:
            k += 1
        i = max(k, i + 1)  # always move forward
    return chunks