import pytest

from clip_ranker import (
    RankingSettings,
    ScoreWeights,
    overall_score,
    overlap_ratio,
    rank_clips,
)
from clip_validator import ValidatedClip
from schemas import LLMScores


def scores(hook=3, value=3, clarity=3, completeness=3):
    return LLMScores(hook=hook, value=value, clarity=clarity, completeness=completeness)


def clip(start, end, hook=3, title="t"):
    return ValidatedClip(
        start=start,
        end=end,
        start_segment=0,
        end_segment=1,
        source_chunk=0,
        title=title,
        hook="h",
        reason="r",
        transcript_quote="q",
        scores=scores(hook=hook, value=hook, clarity=hook, completeness=hook),
        quote_verified=True,
    )


SETTINGS = RankingSettings(weights=ScoreWeights(), dedup_overlap_ratio=0.5)


def test_score_formula():
    weights = ScoreWeights()
    assert overall_score(scores(5, 5, 5, 5), weights) == 100.0
    assert overall_score(scores(1, 1, 1, 1), weights) == 20.0
    # 20 * (0.3*5 + 0.3*4 + 0.2*3 + 0.2*2) = 20 * 3.7 = 74
    assert overall_score(scores(5, 4, 3, 2), weights) == 74.0


def test_weights_must_add_up_to_one():
    with pytest.raises(ValueError):
        ScoreWeights(hook=0.5, value=0.5, clarity=0.5, completeness=0.5)
    with pytest.raises(ValueError):
        ScoreWeights(hook=-0.1, value=0.6, clarity=0.25, completeness=0.25)


def test_dedup_setting_must_be_in_range():
    with pytest.raises(ValueError):
        RankingSettings(weights=ScoreWeights(), dedup_overlap_ratio=0)


def test_overlap_ratio():
    assert overlap_ratio(0, 10, 20, 30) == 0.0
    assert overlap_ratio(0, 10, 10, 20) == 0.0  # touching is not overlapping
    assert overlap_ratio(0, 100, 10, 30) == 1.0  # one clip inside another
    assert overlap_ratio(0, 20, 10, 30) == 0.5


def test_overlapping_clips_keep_the_higher_score():
    low = clip(10, 40, hook=3, title="low")
    high = clip(15, 45, hook=5, title="high")
    result = rank_clips([low, high], SETTINGS, max_clips=10)
    assert [r.clip.title for r in result.clips] == ["high"]
    assert result.duplicates_removed == 1


def test_clip_inside_a_longer_clip_is_a_duplicate():
    result = rank_clips([clip(0, 60, hook=5), clip(10, 25, hook=4)], SETTINGS, max_clips=10)
    assert len(result.clips) == 1


def test_touching_clips_are_both_kept():
    result = rank_clips([clip(0, 20), clip(20, 40)], SETTINGS, max_clips=10)
    assert len(result.clips) == 2
    assert result.duplicates_removed == 0


def test_ranks_are_in_score_order_and_limited():
    clips = [clip(0, 20, hook=2, title="c"), clip(30, 50, hook=5, title="a"), clip(60, 80, hook=4, title="b")]
    result = rank_clips(clips, SETTINGS, max_clips=2)
    assert [(r.rank, r.clip.title) for r in result.clips] == [(1, "a"), (2, "b")]


def test_tie_goes_to_the_earlier_clip():
    result = rank_clips([clip(60, 80, title="late"), clip(0, 20, title="early")], SETTINGS, max_clips=10)
    assert [r.clip.title for r in result.clips] == ["early", "late"]