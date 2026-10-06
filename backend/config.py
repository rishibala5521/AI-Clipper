import os

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {raw!r}") from None


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


class Config:
    APP_ENV = os.getenv("APP_ENV", "development")
    DEBUG = _get_bool("FLASK_DEBUG", False)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///clip_timeline.db")

    # Reject oversized request bodies (1 MB) before they reach our code.
    MAX_CONTENT_LENGTH = 1_000_000

    # Caption languages to try, in priority order (comma-separated in .env).
    TRANSCRIPT_LANGUAGES = _get_list("TRANSCRIPT_LANGUAGES", "en")

    # Which AI provider to use: "nvidia" or "gemini"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia").strip().lower()
    LLM_MAX_ATTEMPTS = _get_int("LLM_MAX_ATTEMPTS", 4)
    LLM_BASE_DELAY_SECONDS = _get_float("LLM_BASE_DELAY_SECONDS", 2.0)
    LLM_TIMEOUT_SECONDS = _get_float("LLM_TIMEOUT_SECONDS", 120.0)

    # NVIDIA
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
    NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "").strip()
    NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip()
    NVIDIA_JSON_MODE = os.getenv("NVIDIA_JSON_MODE", "prompt_only").strip().lower()
    NVIDIA_MAX_OUTPUT_TOKENS = _get_int("NVIDIA_MAX_OUTPUT_TOKENS", 4000)
    NVIDIA_THINKING = os.getenv("NVIDIA_THINKING", "off").strip().lower()

    # Gemini (kept as an alternative)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip()

    # Analysis settings (all overridable in .env)
    CHUNK_SECONDS = _get_int("CHUNK_SECONDS", 300)
    CHUNK_OVERLAP_SECONDS = _get_int("CHUNK_OVERLAP_SECONDS", 30)
    CANDIDATES_PER_CHUNK = _get_int("CANDIDATES_PER_CHUNK", 5)
    MIN_CLIP_SECONDS = _get_int("MIN_CLIP_SECONDS", 15)
    MAX_CLIP_SECONDS = _get_int("MAX_CLIP_SECONDS", 60)
    MAX_CHUNKS_PER_ANALYSIS = _get_int("MAX_CHUNKS_PER_ANALYSIS", 10)