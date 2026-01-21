FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# --------------------------------------
# System dependencies
# --------------------------------------
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

# --------------------------------------
# Dependency layer (cacheable)
# --------------------------------------
COPY ../uv.lock uv.lock
COPY ../pyproject.toml pyproject.toml

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# --------------------------------------
# Application code
# --------------------------------------
COPY ../src/ src/
COPY ../README.md README.md
#COPY ../data/ data/
#COPY ../models/ models/
# creates a directory for storing data file and  creates a directory for storing trained model files
# We mount trained weights and evaluation data at runtime
RUN mkdir -p /models /data

COPY ../reports/ reports/

# Data directory expected by corrupt_mnist()
# -p flag - creates parent directories as needed and doesn't error if directories already exist
#RUN mkdir -p /data /models # creates a directory for storing data file and  creates a directory for storing trained model files

ENTRYPOINT ["uv", "run", "src/mlops_exam_project/evaluate.py"]
