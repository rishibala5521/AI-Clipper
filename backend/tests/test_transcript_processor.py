import pytest

from errors import AppError
from transcript_processor import normalize_segments


def test_basic_normalization():
    raw = [
        {"text": "Hello  there\n", "start": 0.0, "duration": 2.5},
        {"text": "second line", "start": 2.5, "duration": 1.5},
    ]
    segments = normalize_segments(raw)
    assert [s.text for s in segments] == ["Hello there", "second line"]
    assert segments[0].end == 2.5
    assert segments[1].end == 4.0


def test_overlapping_segments_are_trimmed():
    raw = [
        {"text": "a", "start": 0, "duration": 5},
        {"text": "b", "start": 3, "duration": 4},
    ]
    segments = normalize_segments(raw)
    assert segments[0].end == 3.0
    assert segments[1].end == 7.0


def test_unsorted_input_is_sorted():
    raw = [
        {"text": "b", "start": 5, "duration": 1},
        {"text": "a", "start": 1, "duration": 1},
    ]
    assert [s.text for s in normalize_segments(raw)] == ["a", "b"]


def test_blank_text_is_dropped_without_counting_as_bad():
    raw = [
        {"text": "   ", "start": 0, "duration": 1},
        {"text": "real", "start": 1, "duration": 1},
    ]
    assert [s.text for s in normalize_segments(raw)] == ["real"]


def test_few_bad_entries_are_skipped():
    raw = [{"text": f"line {i}", "start": i, "duration": 1} for i in range(10)]
    raw.append({"text": "broken", "start": "abc", "duration": 1})
    assert len(normalize_segments(raw)) == 10


def test_too_many_bad_entries_rejects_transcript():
    raw = [
        {"text": "ok", "start": 0, "duration": 1},
        {"text": "bad", "start": -5, "duration": 1},
        {"text": "bad", "start": 1, "duration": 0},
    ]
    with pytest.raises(AppError) as exc_info:
        normalize_segments(raw)
    assert exc_info.value.code == "unreliable_transcript"


def test_empty_transcript_rejected():
    with pytest.raises(AppError) as exc_info:
        normalize_segments([])
    assert exc_info.value.code == "empty_transcript"