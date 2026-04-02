FROM python:3.12-slim

# Install system dependencies including cmake
RUN apt-get update && apt-get install -y \
    cmake \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy pyproject.toml and uv.lock
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy the rest of the application
COPY . .

# Install the package itself
RUN uv pip install -e .

# Set the entrypoint
ENTRYPOINT ["uv", "run", "water-timeseries"]