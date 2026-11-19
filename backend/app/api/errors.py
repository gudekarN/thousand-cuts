"""Error handling for the FastAPI application.

Registers exception handlers that produce the standard error shape:
  {"error": {"code", "message", "details"}}
"""
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas.common import ErrorDetail, ErrorResponse


class ApiError(Exception):
    """Domain error rendered with the standard error envelope."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def _error_response(code: str, message: str, details: dict | None, status: int) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, details=details))
    return JSONResponse(status_code=status, content=body.model_dump())


def _json_safe_errors(errors: list[dict]) -> list[dict]:
    """Ensure Pydantic validation errors can be JSON-serialized (# ctx may hold exceptions)."""
    safe_errors = []
    for error in errors:
        safe = dict(error)
        ctx = safe.get("ctx")
        if isinstance(ctx, dict):
            safe["ctx"] = {key: str(value) for key, value in ctx.items()}
        safe_errors.append(safe)
    return safe_errors


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        return _error_response(exc.code, exc.message, exc.details, exc.status_code)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        code = f"HTTP_{exc.status_code}"
        return _error_response(code, str(exc.detail), None, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return _error_response(
            "VALIDATION_ERROR",
            "Request validation failed",
            {"errors": _json_safe_errors(exc.errors())},
            422,
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return _error_response(
            "INTERNAL_ERROR",
            "An unexpected server error occurred",
            None,
            500,
        )
