import logging

from flask import Flask, jsonify

from clip_analyzer import AnalysisSettings
from config import Config
from errors import register_error_handlers
from llm_client import GeminiClient
from nvidia_client import NvidiaClient
from routes import api_bp
from transcript_service import YouTubeTranscriptProvider


def configure_logging(app: Flask) -> None:
    level = logging.DEBUG if app.config["DEBUG"] else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpcore", "httpcore2", "urllib3", "hpack", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_llm_client(config):
    provider = config["LLM_PROVIDER"]
    if provider == "nvidia":
        return NvidiaClient(
            api_key=config["NVIDIA_API_KEY"],
            model=config["NVIDIA_MODEL"],
            base_url=config["NVIDIA_BASE_URL"],
            json_mode=config["NVIDIA_JSON_MODE"],
            thinking=config["NVIDIA_THINKING"],
            max_output_tokens=config["NVIDIA_MAX_OUTPUT_TOKENS"],
            max_attempts=config["LLM_MAX_ATTEMPTS"],
            base_delay_seconds=config["LLM_BASE_DELAY_SECONDS"],
            timeout_seconds=config["LLM_TIMEOUT_SECONDS"],
        )
    if provider == "gemini":
        return GeminiClient(
            api_key=config["GEMINI_API_KEY"],
            model=config["GEMINI_MODEL"],
            max_attempts=config["LLM_MAX_ATTEMPTS"],
            base_delay_seconds=config["LLM_BASE_DELAY_SECONDS"],
        )
    raise ValueError(f"LLM_PROVIDER must be 'nvidia' or 'gemini', got {provider!r}")


def create_app(
    test_config: dict | None = None,
    transcript_provider=None,
    llm_client=None,
) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    configure_logging(app)
    register_error_handlers(app)

    # Tests pass fakes so they never touch the network or spend credits.
    app.extensions["transcript_provider"] = transcript_provider or YouTubeTranscriptProvider()
    app.extensions["llm_client"] = llm_client or build_llm_client(app.config)
    # Bad settings fail here, at startup, with a clear message.
    app.extensions["analysis_settings"] = AnalysisSettings.from_config(app.config)

    app.register_blueprint(api_bp)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "ai-clip-timeline",
                "environment": app.config["APP_ENV"],
            }
        )

    return app