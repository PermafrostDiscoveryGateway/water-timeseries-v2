FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    cmake \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy everything at once
COPY src/ ./src/
COPY pyproject.toml .
COPY uv.lock .
COPY README.md .

# Create virtual environment
RUN uv venv
ENV PATH="/app/.venv/bin:$PATH"

# Install h3 first
RUN uv pip install "h3>=4.0.0"

# Sync all dependencies
RUN uv sync --frozen --no-dev

# Install the package in editable mode
RUN uv pip install -e .

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV=/app/.venv

# Remove the fixed ENTRYPOINT - we'll specify commands at runtime
# Keep CMD as default for local testing, but it can be overridden
CMD ["uv", "run", "water-timeseries"]