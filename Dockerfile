FROM python:3.12-slim

# Install only essential build tools
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy and install dependencies
COPY pyproject.toml uv.lock ./

# Pre-install h3 (the problematic package)
RUN uv pip install "h3>=4.0.0"

# Install everything else
RUN uv sync --frozen --no-dev

COPY . .
RUN uv pip install -e .

ENTRYPOINT ["uv", "run", "water-timeseries"]