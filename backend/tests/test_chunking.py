import pytest

from schemas import TranscriptSegment
from transcript_processor import build_chunks


def make_segments(count, length=5.0):
    return [
        TranscriptSegment(start=i * length, end=(i + 1) * length, text=f"line {i}")
        for i in range(count)
    ]


def test_short_transcript_is_one_chunk():
    chunks = build_chunks(make_segments(10), chunk_seconds=100, overlap_seconds=20)
    assert [(c.first_index, c.last_index) for c in chunks] == [(0, 9)]


def test_chunks_overlap_and_cover_everything():
    chunks = build_chunks(make_segments(40), chunk_seconds=100, overlap_seconds=20)
    assert [(c.first_index, c.last_index) for c in chunks] == [(0, 19), (16, 35), (32, 39)]
    covered = set()
    for chunk in chunks:
        covered.update(range(chunk.first_index, chunk.last_index + 1))
    assert covered == set(range(40))


def test_chunk_segments_match_global_indices():
    chunks = build_chunks(make_segments(40), chunk_seconds=100, overlap_seconds=20)
    assert chunks[1].segments[0].text == "line 16"
    assert chunks[1].start == 80.0
    assert chunks[1].end == 180.0


def test_long_silence_still_makes_progress():
    segments = [
        TranscriptSegment(start=0, end=2, text="a"),
        TranscriptSegment(start=1000, end=1002, text="b"),
    ]
    chunks = build_chunks(segments, chunk_seconds=100, overlap_seconds=20)
    assert [(c.first_index, c.last_index) for c in chunks] == [(0, 0), (1, 1)]


def test_empty_input_returns_no_chunks():
    assert build_chunks([], 100, 20) == []


@pytest.mark.parametrize("chunk,overlap", [(50, 50), (50, 60), (0, 0), (100, -1)])
def test_invalid_settings_raise(chunk, overlap):
    with pytest.raises(ValueError):
        build_chunks(make_segments(5), chunk, overlap)