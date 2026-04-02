FROM python:3.12-slim

# Install build dependencies
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

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Debug: Show Python version and pip
RUN echo "=== Python version ===" && python --version
RUN echo "=== uv version ===" && uv --version
RUN echo "=== CMake version ===" && cmake --version
RUN echo "=== System info ===" && uname -a
RUN echo "=== Architecture ===" && dpkg --print-architecture

# Debug: Show pyproject.toml h3 line
RUN echo "=== h3 dependency in pyproject.toml ===" && grep -i "h3" pyproject.toml || echo "h3 not found"

# Debug: Show uv.lock h3 info
RUN echo "=== h3 in uv.lock ===" && grep -A 5 -B 5 "name = \"h3\"" uv.lock || echo "h3 not found in lock file"

# Try to install h3 first with verbose output
RUN echo "=== Attempting to install h3 alone ===" && \
    uv pip install --verbose "h3>=4.0.0" 2>&1 | tee /tmp/h3_install.log || \
    (echo "=== h3 install failed. Retrying with binary only ===" && \
     uv pip install --verbose --only-binary h3 "h3>=4.0.0" 2>&1 | tee -a /tmp/h3_install.log || \
     echo "=== Binary install failed too ===")

# Show what packages are installed after h3 attempt
RUN echo "=== Installed packages after h3 attempt ===" && uv pip list

# Now try full sync with verbose output
RUN echo "=== Attempting full uv sync ===" && \
    uv sync --frozen --no-dev --verbose 2>&1 | tee /tmp/uv_sync.log || \
    (echo "=== Sync failed. Trying without frozen ===" && \
     uv sync --no-dev --verbose 2>&1 | tee -a /tmp/uv_sync.log)

# Copy the rest of the application
COPY . .

# Try to install the package with verbose output
RUN echo "=== Installing package in editable mode ===" && \
    uv pip install -v -e . 2>&1 | tee /tmp/pip_install.log

# Test imports with verbose errors
RUN echo "=== Testing imports ===" && \
    python -c "import sys; print('Python path:', sys.path)" && \
    python -c "import h3; print(f'h3 version: {h3.__version__}')" || \
    (echo "=== h3 import failed. Checking installed files ===" && \
     find /app -name "*h3*" -type d 2>/dev/null && \
     find /usr/local/lib -name "*h3*" -type d 2>/dev/null)

# Show the log files for debugging
RUN echo "=== Log files ===" && ls -la /tmp/*.log 2>/dev/null || echo "No log files found"

ENTRYPOINT ["uv", "run", "water-timeseries"]