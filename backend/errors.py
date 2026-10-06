from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException


class AppError(Exception):
    """Base class for errors we raise on purpose and want shown to the client."""

    status_code = 400
    code = "app_error"

    def __init__(self, message: str, status_code: int | None = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


def _error_response(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(AppError)
    def handle_app_error(err: AppError):
        return _error_response(err.code, err.message, err.status_code)

    @app.errorhandler(HTTPException)
    def handle_http_error(err: HTTPException):
        code = err.name.lower().replace(" ", "_")
        return _error_response(code, err.description or err.name, err.code or 500)

    @app.errorhandler(Exception)
    def handle_unexpected_error(err: Exception):
        # Full details go to the log; the client only gets a safe message.
        app.logger.exception("Unhandled error")
        return _error_response("internal_error", "Something went wrong on our side.", 500)