import re
from urllib.parse import parse_qs, urlparse

from errors import AppError

_VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
_PATH_PREFIXES = {"shorts", "embed", "live", "v"}


def _invalid(message: str = "Please enter a valid YouTube video URL.") -> AppError:
    return AppError(message, 400, "invalid_video_url")


def extract_video_id(url: str) -> str:
    """Return the 11-character YouTube video ID, or raise AppError."""
    candidate = url.strip()
    if "://" not in candidate:
        candidate = "https://" + candidate  # allow "youtube.com/watch?v=..."

    try:
        parsed = urlparse(candidate)
        host = (parsed.hostname or "").lower()
    except ValueError:
        raise _invalid()

    if parsed.scheme not in {"http", "https"}:
        raise _invalid()
    if parsed.username or parsed.password:
        raise _invalid()

    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif host in _YOUTUBE_HOSTS:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            video_id = parts[1] if len(parts) >= 2 and parts[0] in _PATH_PREFIXES else ""
    else:
        raise _invalid("Only YouTube URLs are supported right now.")

    if not _VIDEO_ID_RE.fullmatch(video_id):
        raise _invalid()
    return video_id