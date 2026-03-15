# ============================================================
# Stage 1: Build environment
# ============================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=managed

WORKDIR /app

# Install Python
RUN uv python install 3.11

# Install dependencies first (layer cache)
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

# Copy source and install project
COPY src/ src/
COPY trl_fork/ trl_fork/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/trl_fork:$PYTHONPATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/trl_fork /app/trl_fork
COPY configs/ configs/
COPY scripts/ scripts/

CMD ["rmt-grpo", "--help"]
