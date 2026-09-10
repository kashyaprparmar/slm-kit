# SLM Kit backend — GPU-enabled.
# Starts from an official PyTorch CUDA image so we don't download ~2GB of
# NVIDIA wheels from pypi.nvidia.com during `pip install torch`.
# For UI/API-only without a GPU, use docker-compose.cpu.yml.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app
COPY data ./data

# App + eval extras, then the training stack. Do NOT install the `gpu` extra
# (it lists torch) — that would let PyPI replace the CUDA build already in
# this image with a CPU wheel.
ENV PIP_DEFAULT_TIMEOUT=1000
RUN pip install --no-cache-dir --timeout 1000 -e ".[eval]" \
    && pip install --no-cache-dir --timeout 1000 \
        "transformers>=4.43" \
        "datasets>=2.20" \
        "tokenizers>=0.19" \
        "accelerate>=0.31" \
        "peft>=0.11" \
        "trl>=0.9" \
        "bitsandbytes>=0.43" \
        "unsloth>=2024.8" \
        "sentencepiece>=0.2"

ENV SLMKIT_HOME=/data/slmkit \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
