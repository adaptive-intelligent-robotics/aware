# JAX and PyTorch wheels bundle CUDA, so the host only needs an NVIDIA driver and the
# NVIDIA container toolkit. Build and run from the repository root:
#   docker build -t aware .
#   docker run --gpus all -u $(id -u):$(id -g) -v $PWD/data:/app/data \
#     -v $PWD/outputs:/app/outputs -v $PWD/checkpoints:/app/checkpoints aware \
#     python scripts/evaluate.py checkpoint=checkpoints/rssm_aware_seed1
FROM python:3.10-slim-bookworm

# EGL and OpenGL for headless MuJoCo, ffmpeg for videos, a compiler for py-lz4framed
RUN apt-get update && apt-get install -y --no-install-recommends \
      libegl1 libgl1 libglib2.0-0 ffmpeg build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH \
    MUJOCO_GL=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp

WORKDIR /app

# install dependencies first, so they are cached when only the code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# install the aware package in editable mode, so REPO_ROOT resolves to /app
COPY . .
RUN uv sync --frozen

# lets entrypoint.sh register the user the container is run as
RUN chmod a+w /etc/passwd
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]
