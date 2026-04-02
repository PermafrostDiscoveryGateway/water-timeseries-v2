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

COPY pyproject.toml uv.lock ./

# Use --system flag to install globally (no venv needed)
RUN uv pip install --system "h3>=4.0.0"

# Sync with --system flag
RUN uv sync --frozen --no-dev --system

COPY . .

RUN uv pip install --system -e .

ENTRYPOINT ["uv", "run", "water-timeseries"]