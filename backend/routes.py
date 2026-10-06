from dataclasses import asdict

from flask import Blueprint, current_app, jsonify, request
from pydantic import ValidationError

from analysis_service import analyze_video
from clip_analyzer import analyze_transcript
from errors import AppError
from schemas import (
    AnalysisOut,
    AnalyzeRequest,
    ClipOut,
    ClipUpdate,
    ReviewRequest,
    TranscriptRequest,
)
from transcript_service import get_transcript
from video_url import extract_video_id

api_bp = Blueprint("api", __name__, url_prefix="/api")

TIME_TOLERANCE = 0.01


def _parse_body(model):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise AppError("Request body must be a JSON object.", 400, "invalid_request")
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        raise AppError(f"Invalid request: {problems}", 400, "invalid_request") from exc


def _load_transcript(url: str):
    video_id = extract_video_id(url)
    return get_transcript(
        video_id,
        current_app.extensions["transcript_provider"],
        current_app.config["TRANSCRIPT_LANGUAGES"],
    )


# ---------------- Real routes ----------------


@api_bp.post("/analyze")
def analyze():
    body = _parse_body(AnalyzeRequest)
    settings = current_app.extensions["analysis_settings"]
    repository = current_app.extensions["repository"]

    analysis_id = analyze_video(
        url=body.url,
        max_clips=body.max_clips,
        min_duration=body.min_duration or settings.min_duration,
        max_duration=body.max_duration or settings.max_duration,
        transcript_provider=current_app.extensions["transcript_provider"],
        languages=current_app.config["TRANSCRIPT_LANGUAGES"],
        llm_client=current_app.extensions["llm_client"],
        base_settings=settings,
        ranking_settings=current_app.extensions["ranking_settings"],
        repository=repository,
    )
    data = repository.get_analysis(analysis_id)
    return jsonify(AnalysisOut.model_validate(data).model_dump()), 201


@api_bp.get("/analyses/<analysis_id>")
def get_analysis(analysis_id: str):
    data = current_app.extensions["repository"].get_analysis(analysis_id)
    if data is None:
        raise AppError("Analysis not found.", 404, "analysis_not_found")
    return jsonify(AnalysisOut.model_validate(data).model_dump())


@api_bp.patch("/clips/<clip_id>")
def update_clip(clip_id: str):
    body = _parse_body(ClipUpdate)
    changes = body.model_dump(exclude_none=True)
    if not changes:
        raise AppError("Provide at least one of: title, start, end.", 400, "invalid_request")

    repository = current_app.extensions["repository"]
    clip = repository.get_clip(clip_id)
    if clip is None:
        raise AppError("Clip not found.", 404, "clip_not_found")

    title = changes.get("title", clip["title"])
    start = changes.get("start", clip["start"])
    end = changes.get("end", clip["end"])

    if end <= start:
        raise AppError("end must be greater than start.", 400, "invalid_timestamps")
    if end > clip["transcript_end"] + TIME_TOLERANCE:
        raise AppError(
            f"end is beyond the end of the transcript ({clip['transcript_end']:.1f}s).",
            400,
            "out_of_range",
        )
    config = current_app.config
    duration = end - start
    if not config["MIN_EDIT_CLIP_SECONDS"] <= duration <= config["MAX_EDIT_CLIP_SECONDS"]:
        raise AppError(
            f"Clip length must be between {config['MIN_EDIT_CLIP_SECONDS']:g} and "
            f"{config['MAX_EDIT_CLIP_SECONDS']:g} seconds.",
            400,
            "invalid_duration",
        )

    updated = repository.update_clip(clip_id, title=title, start=start, end=end)
    return jsonify(ClipOut.model_validate(updated).model_dump())


@api_bp.post("/clips/<clip_id>/review")
def review_clip(clip_id: str):
    body = _parse_body(ReviewRequest)
    updated = current_app.extensions["repository"].set_review(clip_id, body.status)
    if updated is None:
        raise AppError("Clip not found.", 404, "clip_not_found")
    return jsonify(ClipOut.model_validate(updated).model_dump())


# ---------------- Temporary debug routes (remove before deployment) ----------------


@api_bp.post("/transcript")
def transcript_debug():
    """Temporary: shows the cleaned transcript for a video."""
    body = _parse_body(TranscriptRequest)
    return jsonify(_load_transcript(body.url).model_dump())


@api_bp.post("/debug/drafts")
def drafts_debug():
    """Temporary: raw AI drafts, before validation. Uses real AI requests."""
    body = _parse_body(TranscriptRequest)
    transcript = _load_transcript(body.url)
    analysis = analyze_transcript(
        transcript,
        current_app.extensions["llm_client"],
        current_app.extensions["analysis_settings"],
    )
    return jsonify(
        {
            "video_id": transcript.video_id,
            "chunk_count": analysis.chunk_count,
            "draft_count": len(analysis.drafts),
            "dropped_count": analysis.dropped_count,
            "usage": asdict(analysis.usage),
            "drafts": [draft.model_dump() for draft in analysis.drafts],
        }
    )