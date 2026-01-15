
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# System dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

# --------------------------------------
# Dependency layer (cacheable)
# --------------------------------------
COPY ../uv.lock uv.lock
COPY ../pyproject.toml pyproject.toml

# RUN uv sync --locked --no-cache --no-install-project
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# --------------------------------------
# Application code (frequently changing)
# --------------------------------------
COPY ../README.md README.md
COPY ../src/ src/
COPY ../data/ data/
COPY ../models/ models/
COPY ../reports/ reports/

ENTRYPOINT ["uv", "run", "src/mlops_exam_project/train.py"]