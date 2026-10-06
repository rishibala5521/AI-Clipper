from pydantic import BaseModel, Field, model_validator


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

class LLMScores(BaseModel):
    hook: int = Field(ge=1, le=5, description="How strong the opening is, 1 (weak) to 5 (excellent).")
    value: int = Field(ge=1, le=5, description="How useful, interesting or entertaining it is, 1 to 5.")
    clarity: int = Field(ge=1, le=5, description="How clear it is to someone who has not seen the rest of the video, 1 to 5.")
    completeness: int = Field(ge=1, le=5, description="Whether it is a complete thought with a payoff, 1 to 5.")


class LLMClip(BaseModel):
    """One candidate exactly as Gemini returns it. Not trusted yet."""

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
    NOT validated or ranked yet; that is Phase 4."""

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