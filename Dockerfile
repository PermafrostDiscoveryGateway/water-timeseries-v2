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
COPY . .
#COPY /src .
#COPY /google_cloud_utils .
#COPY pyproject.toml .
#COPY uv.lock .

# Create virtual environment
RUN uv venv
ENV PATH="/app/.venv/bin:$PATH"

# Install h3 first
RUN uv pip install "h3>=4.0.0"

# Sync all dependencies (now README.md exists)
RUN uv sync --frozen --no-dev

# Install the package in editable mode
RUN uv pip install -e .

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV=/app/.venv

ENTRYPOINT ["uv", "run", "water-timeseries"]