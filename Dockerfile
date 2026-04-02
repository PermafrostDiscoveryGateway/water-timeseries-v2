FROM condaforge/mambaforge:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Configure conda channels
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# Install geospatial packages via conda (these are the problematic ones)
RUN mamba install -c conda-forge -y \
    python=3.12 \
    h3=4.1.2 \
    h3-pandas \
    geopandas \
    xarray \
    numpy \
    pandas \
    && mamba clean -afy

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create a script to filter out conda-installed packages from uv sync
RUN echo '#!/bin/bash\n\
# Extract dependencies from pyproject.toml excluding conda-installed ones\n\
grep -E "^[[:space:]]+\"" pyproject.toml | \n\
  sed "s/[[:space:]]*\"\([^\"]*\)\".*/\\1/" | \n\
  sed "s/[>=<].*//" | \n\
  grep -v -E "^(h3|h3pandas|geopandas|xarray|numpy|pandas)$" \n\
' > /tmp/filter_deps.sh && chmod +x /tmp/filter_deps.sh

# Install remaining dependencies with uv (excluding conda-installed ones)
RUN uv pip install $(/tmp/filter_deps.sh | head -20) || \
    (echo "Some packages failed, continuing..." && exit 0)

# Copy the rest of the application
COPY . .

# Install the package in development mode (will skip already-installed deps)
RUN uv pip install -e . --no-deps

# Set environment variables to prioritize conda packages
ENV PATH="/opt/conda/bin:$PATH"
ENV UV_SYSTEM_PYTHON=1

# Verify installations
RUN python -c "import h3; print(f'h3 version: {h3.__version__}')" && \
    python -c "import geopandas; print(f'geopandas version: {geopandas.__version__}')"

ENTRYPOINT ["uv", "run", "water-timeseries"]