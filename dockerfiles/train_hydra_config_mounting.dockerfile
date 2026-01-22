
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
COPY ../LICENSE LICENSE    

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
# COPY ../configs/ configs/ 
RUN mkdir -p configs
# better to make  configs/  mountable.
# COPY ../models/ models/
# creates a directory for storing trained model files (avoiding copying existing model files)
RUN mkdir -p models 
#COPY ../reports/ reports/
RUN mkdir -p reports  



# If you have a config file, uncomment the following line to copy it
#COPY ../config.yaml config.yaml 


# disable Hydra changing cwd
ENV HYDRA_FULL_ERROR=1

ENTRYPOINT ["uv", "run", "src/mlops_exam_project/train.py"]

