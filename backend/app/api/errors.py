"""Error handling for the FastAPI application.

Registers exception handlers that produce the standard error shape:
  {"error": {"code", "message", "details"}}
"""
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas.common import ErrorDetail, ErrorResponse


def _error_response(code: str, message: str, details: dict | None, status: int) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, details=details))
    return JSONResponse(status_code=status, content=body.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        code = f"HTTP_{exc.status_code}"
        return _error_response(code, str(exc.detail), None, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return _error_response(
            "VALIDATION_ERROR",
            "Request validation failed",
            {"errors": exc.errors()},
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
