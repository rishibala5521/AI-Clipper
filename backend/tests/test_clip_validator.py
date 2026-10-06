from clip_validator import normalize_text, validate_drafts
from schemas import ClipDraft, LLMScores, TranscriptSegment

SCORES = LLMScores(hook=4, value=4, clarity=4, completeness=4)


def make_segments(count=20, length=5.0):
    return [
        TranscriptSegment(start=i * length, end=(i + 1) * length, text=f"line {i} words")
        for i in range(count)
    ]


def make_draft(first, last, quote="line 2 words", title="A title", length=5.0):
    return ClipDraft(
        start=first * length,
        end=(last + 1) * length,
        start_segment=first,
        end_segment=last,
        source_chunk=0,
        title=title,
        hook="hook",
        reason="reason",
        transcript_quote=quote,
        scores=SCORES,
    )


def run(drafts, segments=None, transcript_end=None, min_duration=15, max_duration=60):
    segments = segments or make_segments()
    end = transcript_end if transcript_end is not None else segments[-1].end
    return validate_drafts(drafts, segments, end, min_duration, max_duration)


def test_valid_clip_passes_unchanged():
    result = run([make_draft(2, 6)])
    clip = result.clips[0]
    assert (clip.start, clip.end) == (10.0, 35.0)
    assert clip.flags == []
    assert clip.quote_verified is True
    assert result.rejected == {}


def test_short_clip_is_extended_to_the_minimum():
    result = run([make_draft(10, 11, quote="line 10 words")])
    clip = result.clips[0]
    assert (clip.start, clip.end) == (50.0, 65.0)
    assert clip.flags == ["extended"]


def test_short_clip_at_the_end_of_the_transcript_is_rejected():
    segments = make_segments(12)
    result = run([make_draft(10, 11)], segments=segments)
    assert result.clips == []
    assert result.rejected == {"too_short": 1}


def test_long_clip_is_trimmed_to_the_maximum():
    result = run([make_draft(0, 19, quote="line 3 words")])
    clip = result.clips[0]
    assert clip.end == 60.0
    assert clip.flags == ["trimmed"]


def test_single_segment_longer_than_the_maximum_is_rejected():
    segments = [TranscriptSegment(start=0, end=100, text="a very long segment")]
    result = run([make_draft(0, 0, quote="very long", length=100.0)], segments=segments)
    assert result.rejected == {"too_long": 1}


def test_quote_not_in_the_clip_is_flagged_not_rejected():
    result = run([make_draft(2, 6, quote="something nobody said")])
    clip = result.clips[0]
    assert clip.quote_verified is False
    assert "quote_not_found" in clip.flags


def test_empty_quote_is_flagged():
    result = run([make_draft(2, 6, quote="   ")])
    assert "no_quote" in result.clips[0].flags


def test_quote_matching_ignores_punctuation_and_case():
    segments = [
        TranscriptSegment(start=0, end=10, text="Don't stop, believe."),
        TranscriptSegment(start=10, end=20, text="Keep going!"),
    ]
    result = run([make_draft(0, 1, quote="dont stop believe keep going", length=10.0)], segments=segments)
    assert result.clips[0].quote_verified is True


def test_bad_segment_range_is_rejected():
    result = run([make_draft(5, 2), make_draft(3, 99)])
    assert result.rejected == {"bad_segment_range": 2}


def test_empty_title_is_rejected():
    result = run([make_draft(2, 6, title="   ")])
    assert result.rejected == {"empty_title": 1}


def test_clip_beyond_transcript_end_is_rejected():
    result = run([make_draft(2, 6)], transcript_end=20.0)
    assert result.rejected == {"out_of_range": 1}


def test_normalize_text():
    assert normalize_text("It’s  A Test, OK?") == "its a test ok"