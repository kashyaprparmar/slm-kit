"""Import-light configuration checks; installed does not imply GPU compatibility."""
import platform
import subprocess
import sys

from app.capabilities import dependency_statuses
from app.config import get_settings
from app.core.hardware import read_hardware
from app.integrations.gguf import llamacpp_available
from app.integrations.hf_hub import _token


def checks():
    settings = get_settings()
    hw = read_hardware()
    rows = [{"name": "Python", "status": "ready", "detail": platform.python_version(), "guidance": ""},
            {"name": "GPU telemetry", "status": "ready" if hw.gpu_name else "missing",
             "detail": hw.gpu_name or "No NVIDIA GPU detected", "guidance": "GPU training needs an NVIDIA driver and CUDA-compatible PyTorch. CPU scratch training remains available."},
            {"name": "Disk", "status": "ready" if (hw.disk_free_mb or 0) > 10240 else "warning",
             "detail": f"{hw.disk_free_mb} MB free", "guidance": "Allow room for base weights, checkpoints and exports."}]
    dependencies = dependency_statuses()
    for package in ("torch", "transformers", "peft", "trl", "unsloth", "bitsandbytes", "tokenizers", "datasets", "pyarrow"):
        dependency = dependencies[package]
        status = "ready" if dependency.installed else "optional" if dependency.optional else "missing"
        rows.append({"name": package, "status": status, "detail": dependency.version or "Not installed",
                     "guidance": "Package presence only; use the runtime check to verify imports and CUDA. Install the [gpu] extra for training."})
    try:
        runtime = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        values = runtime.stdout.strip().splitlines()
        cuda_ready = runtime.returncode == 0 and len(values) >= 2 and values[1] == "True"
        rows.append({"name": "PyTorch CUDA runtime", "status": "ready" if cuda_ready else "missing",
                     "detail": " · ".join(values) if values else runtime.stderr.strip()[-500:],
                     "guidance": "The CUDA-enabled PyTorch runtime must see the NVIDIA GPU for Transformers/Unsloth training."})
    except (OSError, subprocess.SubprocessError) as exc:
        rows.append({"name": "PyTorch CUDA runtime", "status": "error", "detail": str(exc),
                     "guidance": "Check the PyTorch/CUDA installation and restart the backend."})
    for name, ready, guidance in [
        ("llama.cpp", llamacpp_available(), "Set SLMKIT_LLAMACPP_DIR to a built llama.cpp checkout."),
        ("Hugging Face token", bool(_token()), "Set SLMKIT_HF_TOKEN for private repositories or publishing. Requires restart."),
        ("Judge", bool(settings.judge_api_key), "Optional: configure SLMKIT_JUDGE_API_KEY. Evaluation metrics work without it."),
        ("Ollama CLI", dependencies["ollama"].installed, "Optional: install Ollama on the host. Use Model Serving to check its HTTP service."),
    ]:
        rows.append({"name": name, "status": "ready" if ready else "optional", "detail": "Configured" if ready else "Not configured", "guidance": guidance})
    return {"checks": rows, "hardware": hw, "platform": platform.platform(), "home": str(settings.home)}
