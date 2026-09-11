# SLM Kit backend — GPU-enabled.
# Starts from an official PyTorch CUDA image so we don't download ~2GB of
# NVIDIA wheels from pypi.nvidia.com during `pip install torch`.
# For UI/API-only without a GPU, use docker-compose.cpu.yml.

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md constraints-gpu.txt ./
# Hatch needs the package marker to resolve editable metadata, but keeping the
# remaining source out of this layer means normal code edits reuse the heavy
# CUDA/Unsloth dependency installation.
COPY app/__init__.py ./app/__init__.py

# App + eval extras, then the training stack. Do NOT install the `gpu` extra
# (it lists torch) — that would let PyPI replace the CUDA build already in
# this image with a CPU wheel.
ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip install --no-cache-dir --timeout 1000 -e ".[eval]" \
    && pip install --no-cache-dir --timeout 1000 -c constraints-gpu.txt \
        "transformers>=4.43,<4.58" \
        "datasets>=2.20,<5.1" \
        "tokenizers>=0.19,<0.23" \
        "accelerate>=0.31,<1.16" \
        "peft>=0.11,<0.21" \
        "trl>=0.9,<0.24" \
        "bitsandbytes>=0.43,<0.51" \
        "unsloth>=2024.8" \
        "sentencepiece>=0.2"

# Fail the image build if pip ever resolves a superficially installable but
# import-incompatible TorchAO / Transformers / PEFT combination. These are the
# exact lazy imports used by Qwen3 training and adapter evaluation.
RUN python -c "import torchao; from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM; from transformers import BloomPreTrainedModel; from peft import PeftModel"

COPY app ./app
COPY data ./data

ENV SLMKIT_HOME=/data/slmkit \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000 8802

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
