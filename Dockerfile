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

# Add small sleep to ensure filesystem sync (logging might have caused this naturally)
RUN sleep 1 && uv pip install "h3>=4.0.0"

RUN sleep 1 && uv sync --frozen --no-dev

COPY . .

RUN uv pip install -e .

ENTRYPOINT ["uv", "run", "water-timeseries"]