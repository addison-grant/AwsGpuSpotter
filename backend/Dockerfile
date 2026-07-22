FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace/app

COPY pyproject.toml ./
RUN uv sync --no-install-project

COPY . .
RUN uv sync

EXPOSE 8000

# Deliberately NOT using uvicorn --reload: that restarts the whole process
# and would drop the SAM 3.1 weights out of VRAM. Instead, edit cv_worker.py
# on the bind-mounted volume and POST /dev-reload to hot-swap just that
# module while app.py's process (and its loaded model) keeps running.
CMD ["uv", "run", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
