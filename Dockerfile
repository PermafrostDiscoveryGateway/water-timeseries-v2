FROM python:3.12-slim

# Install required build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    cmake \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install h3 first (it needs to build from source)
RUN uv pip install "h3>=4.0.0"

# Install remaining dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Install the package
RUN uv pip install -e .

ENTRYPOINT ["uv", "run", "water-timeseries"]