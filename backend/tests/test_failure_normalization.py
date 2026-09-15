from app.core.errors import normalize_failure


def test_cuda_oom_is_actionable():
    failure = normalize_failure("CUDA out of memory while allocating tensor")
    assert failure["code"] == "GPU_OUT_OF_MEMORY"
    assert failure["category"] == "RESOURCE_EXHAUSTED"
    assert failure["suggestions"]


def test_unknown_worker_failure_is_not_marked_retryable():
    failure = normalize_failure("unexpected native worker exit")
    assert failure["code"] == "WORKER_FAILED"
    assert failure["retryable"] is False
