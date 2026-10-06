import re
from dataclasses import dataclass, field

from schemas import ClipDraft, LLMScores, TranscriptSegment

TIME_TOLERANCE = 0.01

_APOSTROPHES = str.maketrans("", "", "'’‘`")
_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(text: str) -> str:
    """Lowercase, drop apostrophes and punctuation, collapse spaces (for quote matching)."""
    text = text.lower().translate(_APOSTROPHES)
    text = _PUNCTUATION_RE.sub(" ", text)
    return " ".join(text.split())


@dataclass
class ValidatedClip:
    start: float
    end: float
    start_segment: int
    end_segment: int
    source_chunk: int
    title: str
    hook: str
    reason: str
    transcript_quote: str
    scores: LLMScores
    quote_verified: bool
    flags: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ValidationResult:
    clips: list[ValidatedClip]
    rejected: dict[str, int]  # reason -> how many drafts were rejected


def _validate_one(
    draft: ClipDraft,
    segments: list[TranscriptSegment],
    transcript_end: float,
    min_duration: float,
    max_duration: float,
) -> ValidatedClip | str:
    """Return a ValidatedClip, or a short rejection reason."""
    first, last = draft.start_segment, draft.end_segment
    if not (0 <= first <= last < len(segments)):
        return "bad_segment_range"
    if not draft.title.strip():
        return "empty_title"

    flags: list[str] = []
    start = segments[first].start

    # Too short: borrow the following segments while we stay under the maximum.
    if segments[last].end - start < min_duration:
        extended = False
        while (
            segments[last].end - start < min_duration
            and last + 1 < len(segments)
            and segments[last + 1].end - start <= max_duration
        ):
            last += 1
            extended = True
        if extended:
            flags.append("extended")
    if segments[last].end - start < min_duration:
        return "too_short"

    # Too long: drop segments from the end until it fits.
    if segments[last].end - start > max_duration:
        trimmed = False
        while last > first and segments[last].end - start > max_duration:
            last -= 1
            trimmed = True
        if trimmed:
            flags.append("trimmed")
        final_length = segments[last].end - start
        if final_length > max_duration or final_length < min_duration:
            return "too_long"

    end = segments[last].end
    if start < 0 or end <= start or end > transcript_end + TIME_TOLERANCE:
        return "out_of_range"

    # The quote must really appear in the lines of the final clip.
    clip_text = normalize_text(" ".join(s.text for s in segments[first : last + 1]))
    quote = normalize_text(draft.transcript_quote)
    if not quote:
        verified = False
        flags.append("no_quote")
    elif quote in clip_text:
        verified = True
    else:
        verified = False
        flags.append("quote_not_found")

    return ValidatedClip(
        start=start,
        end=end,
        start_segment=first,
        end_segment=last,
        source_chunk=draft.source_chunk,
        title=draft.title.strip(),
        hook=draft.hook.strip(),
        reason=draft.reason.strip(),
        transcript_quote=draft.transcript_quote.strip(),
        scores=draft.scores,
        quote_verified=verified,
        flags=flags,
    )


def validate_drafts(
    drafts: list[ClipDraft],
    segments: list[TranscriptSegment],
    transcript_end: float,
    min_duration: float,
    max_duration: float,
) -> ValidationResult:
    clips: list[ValidatedClip] = []
    rejected: dict[str, int] = {}
    for draft in drafts:
        outcome = _validate_one(draft, segments, transcript_end, min_duration, max_duration)
        if isinstance(outcome, str):
            rejected[outcome] = rejected.get(outcome, 0) + 1
        else:
            clips.append(outcome)
    return ValidationResult(clips=clips, rejected=rejected)