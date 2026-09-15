"""Consistent, actionable API failures while preserving legacy detail fields."""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from app.core.observability import activity
from app.core.resources import ResourceBusy


def normalize_failure(message: str) -> dict:
    """Classify worker/runtime failures into stable, actionable codes."""
    text = message.lower()
    rules = (
        (("out of memory", "cuda oom"), "GPU_OUT_OF_MEMORY", "RESOURCE_EXHAUSTED", ["Lower Batch Size or Context Length.", "Use QLoRA and gradient checkpointing."]),
        (("cuda is not available", "cuda unavailable", "no nvidia"), "CUDA_UNAVAILABLE", "ENVIRONMENT_ERROR", ["Use the CPU profile or install NVIDIA Container Toolkit and verify nvidia-smi."]),
        (("gated repo", "401 client", "403 client", "authentication"), "MODEL_ACCESS_DENIED", "ENVIRONMENT_ERROR", ["Accept the model license and configure SLMKIT_HF_TOKEN."]),
        (("tokenizer",), "TOKENIZER_LOAD_FAILED", "CONFIGURATION_ERROR", ["Verify tokenizer files and the adapter's base model."]),
        (("unsupported", "could not find", "architecture"), "UNSUPPORTED_MODEL", "UNSUPPORTED", ["Inspect model capabilities and choose a supported backend or architecture."]),
        (("bitsandbytes",), "QUANTIZATION_UNAVAILABLE", "ENVIRONMENT_ERROR", ["Use the GPU training image or choose LoRA/full training without 4-bit loading."]),
        (("no space left", "disk full"), "DISK_FULL", "RESOURCE_EXHAUSTED", ["Free space in the SLM Kit data volume and retry from a checkpoint."]),
        (("connection", "timed out", "network"), "NETWORK_FAILURE", "RETRYABLE", ["Check connectivity and retry; cached model files will be reused."]),
    )
    for needles, code, category, suggestions in rules:
        if any(needle in text for needle in needles):
            return {"code": code, "category": category, "message": message, "retryable": category == "RETRYABLE", "suggestions": suggestions}
    return {"code": "WORKER_FAILED", "category": "FATAL", "message": message, "retryable": False,
            "suggestions": ["Review the recent log context and environment diagnostics before retrying."]}


def error_payload(detail, status=500):
    suggestions = []
    message = detail if isinstance(detail, str) else "Check the highlighted configuration fields."
    if isinstance(detail, dict):
        message = detail.get("message") or "; ".join(
            i["message"] for i in detail.get("validation", {}).get("issues", []) if i["level"] == "error"
        ) or message
    code = {404: "NOT_FOUND", 409: "RESOURCE_BUSY", 422: "INVALID_CONFIG", 413: "UPLOAD_TOO_LARGE"}.get(status, "REQUEST_FAILED")
    normalized = normalize_failure(message)
    if normalized["code"] != "WORKER_FAILED":
        code = normalized["code"]
        suggestions = normalized["suggestions"]
    if "out of memory" in message.lower():
        message = "This configuration requires more GPU memory than is available."
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
