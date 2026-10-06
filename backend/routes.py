from dataclasses import asdict

from flask import Blueprint, current_app, jsonify, request
from pydantic import ValidationError

from clip_analyzer import analyze_transcript
from errors import AppError
from schemas import TranscriptRequest
from transcript_service import get_transcript
from video_url import extract_video_id

api_bp = Blueprint("api", __name__, url_prefix="/api")


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


@api_bp.post("/transcript")
def transcript_debug():
    """Temporary endpoint for testing Phase 2. /api/analyze replaces it later."""
    body = _parse_body(TranscriptRequest)
    return jsonify(_load_transcript(body.url).model_dump())


@api_bp.post("/debug/drafts")
def drafts_debug():
    """Temporary endpoint for testing Phase 3. Uses real AI requests. /api/analyze replaces it."""
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