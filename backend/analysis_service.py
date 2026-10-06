import dataclasses

from clip_analyzer import AnalysisSettings, analyze_transcript
from clip_ranker import RankingSettings, rank_clips
from clip_validator import validate_drafts
from errors import AppError
from transcript_service import get_transcript
from video_url import extract_video_id


def analyze_video(
    *,
    url: str,
    max_clips: int,
    min_duration: int,
    max_duration: int,
    transcript_provider,
    languages: list[str],
    llm_client,
    base_settings: AnalysisSettings,
    ranking_settings: RankingSettings,
    repository,
) -> str:
    """Run the whole pipeline and save the result. Returns the new analysis id."""
    # Cheap checks first, so bad input never reaches the network or the AI.
    video_id = extract_video_id(url)
    if min_duration >= max_duration:
        raise AppError("min_duration must be smaller than max_duration.", 400, "invalid_request")
    settings = dataclasses.replace(
        base_settings, min_duration=min_duration, max_duration=max_duration
    )

    transcript = get_transcript(video_id, transcript_provider, languages)
    analysis = analyze_transcript(transcript, llm_client, settings)
    validation = validate_drafts(
        analysis.drafts,
        transcript.segments,
        transcript.transcript_end,
        min_duration,
        max_duration,
    )
    ranked = rank_clips(validation.clips, ranking_settings, max_clips)

    analysis_row = {
        "video_id": video_id,
        # Stored in a clean form, without tracking parameters such as ?si=...
        "source_url": f"https://www.youtube.com/watch?v={video_id}",
        "language_code": transcript.language_code,
        "is_generated": transcript.is_generated,
        "transcript_end": transcript.transcript_end,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "max_clips": max_clips,
        "chunk_count": analysis.chunk_count,
        "draft_count": len(analysis.drafts),
        "dropped_count": analysis.dropped_count,
        "rejected": validation.rejected,
        "duplicates_removed": ranked.duplicates_removed,
        "input_tokens": analysis.usage.input_tokens,
        "output_tokens": analysis.usage.output_tokens,
        "total_tokens": analysis.usage.total_tokens,
    }
    clip_rows = [
        {
            "rank": item.rank,
            "start": item.clip.start,
            "end": item.clip.end,
            "title": item.clip.title,
            "hook": item.clip.hook,
            "reason": item.clip.reason,
            "transcript_quote": item.clip.transcript_quote,
            "scores": item.clip.scores.model_dump(),
            "overall_score": item.overall_score,
            "flags": item.clip.flags,
            "quote_verified": item.clip.quote_verified,
        }
        for item in ranked.clips
    ]
    return repository.save_analysis(analysis_row, clip_rows)