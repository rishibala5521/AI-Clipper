from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from timefmt import format_clock

# ---------- Transcript ----------


class TranscriptRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


class TranscriptSegment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_order(self):
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class TranscriptResult(BaseModel):
    video_id: str
    language_code: str
    is_generated: bool
    # End time of the last caption. This is NOT necessarily the video length.
    transcript_end: float = Field(ge=0)
    segments: list[TranscriptSegment]


# ---------- What the AI returns (not trusted yet) ----------


class LLMScores(BaseModel):
    hook: int = Field(ge=1, le=5, description="How strong the opening is, 1 (weak) to 5 (excellent).")
    value: int = Field(ge=1, le=5, description="How useful, interesting or entertaining it is, 1 to 5.")
    clarity: int = Field(ge=1, le=5, description="How clear it is to someone who has not seen the rest of the video, 1 to 5.")
    completeness: int = Field(ge=1, le=5, description="Whether it is a complete thought with a payoff, 1 to 5.")


class LLMClip(BaseModel):
    """One candidate exactly as the AI returns it. Not trusted yet."""

    start_segment: int = Field(ge=0, description="Number of the first segment of the clip, exactly as shown in square brackets.")
    end_segment: int = Field(ge=0, description="Number of the last segment of the clip, exactly as shown in square brackets.")
    title: str = Field(description="Short title, at most 8 words.")
    hook: str = Field(description="One sentence describing the opening hook.")
    reason: str = Field(description="One or two sentences on why this moment is worth reviewing.")
    transcript_quote: str = Field(description="Words copied exactly from the clip's own lines, at most 30 words.")
    scores: LLMScores


class LLMChunkResult(BaseModel):
    clips: list[LLMClip]


class ClipDraft(BaseModel):
    """A candidate whose timestamps come from real transcript segments.
    NOT validated or ranked yet."""

    start: float = Field(ge=0)
    end: float = Field(gt=0)
    start_segment: int
    end_segment: int
    source_chunk: int
    title: str
    hook: str
    reason: str
    transcript_quote: str
    scores: LLMScores


# ---------- API requests ----------


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=2048)
    max_clips: int = Field(default=10, ge=1, le=50)
    min_duration: int | None = Field(default=None, ge=5, le=300)
    max_duration: int | None = Field(default=None, ge=10, le=600)


class ClipUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str | None = Field(default=None, min_length=1, max_length=200)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, gt=0)


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["approved", "rejected", "pending"]


# ---------- API responses ----------


class ClipOut(BaseModel):
    id: str
    analysis_id: str
    rank: int
    start: float
    end: float
    duration: float
    ai_start: float  # what the AI originally suggested (never changes)
    ai_end: float
    title: str
    hook: str
    reason: str
    transcript_quote: str
    scores: LLMScores
    overall_score: float
    flags: list[str]
    quote_verified: bool
    status: str
    edited: bool

    # Readable versions of the numbers above, calculated whenever a clip is returned.
    @computed_field
    @property
    def start_label(self) -> str:
        return format_clock(self.start)

    @computed_field
    @property
    def end_label(self) -> str:
        return format_clock(self.end, round_up=True)

    @computed_field
    @property
    def time_range(self) -> str:
        return f"{self.start_label} - {self.end_label}"

    @computed_field
    @property
    def duration_label(self) -> str:
        return format_clock(round(self.duration))


class AnalysisSettingsOut(BaseModel):
    min_duration: int
    max_duration: int
    max_clips: int


class AnalysisStatsOut(BaseModel):
    chunk_count: int
    draft_count: int
    dropped_count: int
    rejected: dict[str, int]
    duplicates_removed: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


class AnalysisOut(BaseModel):
    id: str
    created_at: str
    video_id: str
    source_url: str
    language_code: str
    is_generated: bool
    transcript_end: float
    settings: AnalysisSettingsOut
    stats: AnalysisStatsOut
    clips: list[ClipOut]