from dataclasses import dataclass

from clip_validator import ValidatedClip
from schemas import LLMScores


@dataclass(frozen=True)
class ScoreWeights:
    hook: float = 0.30
    value: float = 0.30
    clarity: float = 0.20
    completeness: float = 0.20

    def __post_init__(self) -> None:
        weights = (self.hook, self.value, self.clarity, self.completeness)
        if any(w < 0 for w in weights):
            raise ValueError("score weights must not be negative")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("score weights must add up to 1.0")


@dataclass(frozen=True)
class RankingSettings:
    weights: ScoreWeights
    dedup_overlap_ratio: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.dedup_overlap_ratio <= 1:
            raise ValueError("DEDUP_OVERLAP_RATIO must be greater than 0 and at most 1")

    @classmethod
    def from_config(cls, config) -> "RankingSettings":
        return cls(
            weights=ScoreWeights(
                hook=config["SCORE_WEIGHT_HOOK"],
                value=config["SCORE_WEIGHT_VALUE"],
                clarity=config["SCORE_WEIGHT_CLARITY"],
                completeness=config["SCORE_WEIGHT_COMPLETENESS"],
            ),
            dedup_overlap_ratio=config["DEDUP_OVERLAP_RATIO"],
        )


@dataclass
class RankedClip:
    clip: ValidatedClip
    overall_score: float
    rank: int


@dataclass
class RankResult:
    clips: list[RankedClip]
    duplicates_removed: int


def overall_score(scores: LLMScores, weights: ScoreWeights) -> float:
    """20 * weighted average of the four 1-5 scores, so the result runs from 20 to 100."""
    total = (
        weights.hook * scores.hook
        + weights.value * scores.value
        + weights.clarity * scores.clarity
        + weights.completeness * scores.completeness
    )
    return round(20 * total, 1)


def overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Overlap as a share of the SHORTER clip (so a clip inside a longer one counts as 1.0)."""
    overlap = min(a_end, b_end) - max(a_start, b_start)
    if overlap <= 0:
        return 0.0
    shorter = min(a_end - a_start, b_end - b_start)
    return overlap / shorter if shorter > 0 else 0.0


def rank_clips(clips: list[ValidatedClip], settings: RankingSettings, max_clips: int) -> RankResult:
    scored = [(overall_score(c.scores, settings.weights), c) for c in clips]
    # Best score first; on a tie the earlier clip wins.
    scored.sort(key=lambda item: (-item[0], item[1].start))

    kept: list[tuple[float, ValidatedClip]] = []
    duplicates_removed = 0
    for score, clip in scored:
        is_duplicate = any(
            overlap_ratio(clip.start, clip.end, other.start, other.end) >= settings.dedup_overlap_ratio
            for _, other in kept
        )
        if is_duplicate:
            duplicates_removed += 1
        else:
            kept.append((score, clip))

    ranked = [
        RankedClip(clip=clip, overall_score=score, rank=index + 1)
        for index, (score, clip) in enumerate(kept[:max_clips])
    ]
    return RankResult(clips=ranked, duplicates_removed=duplicates_removed)