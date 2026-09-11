"""Consistent, actionable API failures while preserving legacy detail fields."""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from app.core.observability import activity
from app.core.resources import ResourceBusy


def error_payload(detail, status=500):
    suggestions = []
    message = detail if isinstance(detail, str) else "Check the highlighted configuration fields."
    if isinstance(detail, dict):
        message = detail.get("message") or "; ".join(
            i["message"] for i in detail.get("validation", {}).get("issues", []) if i["level"] == "error"
        ) or message
    code = {404: "NOT_FOUND", 409: "RESOURCE_BUSY", 422: "INVALID_CONFIG", 413: "UPLOAD_TOO_LARGE"}.get(status, "REQUEST_FAILED")
    if "out of memory" in message.lower():
        code = "GPU_OUT_OF_MEMORY"
        message = "This configuration requires more GPU memory than is available."
        suggestions = ["Lower Batch Size and Context Length.", "Use QLoRA and gradient checkpointing."]
    elif "token" in message.lower() and "hugging face" in message.lower():
        code = "HF_TOKEN_REQUIRED"
        suggestions = ["Set SLMKIT_HF_TOKEN in backend/.env, then restart the backend."]
    elif status == 409:
        suggestions = ["Wait for the active GPU workload to finish, or stop it from its page."]
    elif "architecture" in message.lower() or "causal" in message.lower():
        code = "UNSUPPORTED_MODEL"
        suggestions = ["Choose a decoder-only causal language model or try the Transformers training engine."]
    if status == 500:
        message = "The backend could not complete this operation. Check System & Diagnostics for the request ID."
    return {"code": code, "message": message, "details": detail if status != 500 else None,
            "suggestions": suggestions, "detail": detail if status != 500 else message}


def install_errors(app):
    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(error_payload(exc.detail, exc.status_code), status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(ResourceBusy)
    async def busy_error(request: Request, exc: ResourceBusy):
        return JSONResponse(error_payload(str(exc), 409), status_code=409)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        messages = [f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()]
        return JSONResponse(error_payload("; ".join(messages), 422), status_code=422)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        activity.add("API", f"{request.method} {request.url.path}: {type(exc).__name__}: {exc}", "ERROR")
        return JSONResponse(error_payload(None), status_code=500)
