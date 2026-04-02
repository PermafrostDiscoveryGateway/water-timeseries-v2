# Stage 1: Conda environment
FROM condaforge/mambaforge:latest AS conda-env

RUN mamba create -n water-env -c conda-forge -y \
    python=3.12 \
    h3=4.1.2 \
    h3-pandas \
    geopandas \
    xarray \
    numpy \
    pandas \
    && mamba clean -afy

# Stage 2: Final image
FROM python:3.12-slim

# Copy conda environment from stage 1
COPY --from=conda-env /opt/conda /opt/conda
ENV PATH="/opt/conda/bin:$PATH"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install remaining dependencies with pip
RUN uv pip install --no-deps \
    "dask[dataframe]>=2024.12.1" \
    "joblib>=1.4.2" \
    "ray[default]>=2.40.0" \
    "h5netcdf>=1.4.1" \
    "zarr>=2.18.4" \
    "opencv-python>=4.11.0.86" \
    "statsmodels>=0.14.4" \
    "scikit-learn>=1.6.1" \
    "basemap>=1.4.1" \
    "plotly-geo>=1.0.0" \
    "dask-geopandas>=0.4.2" \
    "sklearn-xarray>=0.4.0" \
    "rbeast>=0.1.23" \
    "loguru>=0.7.3" \
    "typer>=0.24.1" \
    "typer-config>=1.5.0" \
    "lz4>=4.4.5" \
    "ipykernel" \
    "jupyterlab" \
    "cyclopts>=4.8.0" \
    "rich>=14.3.3" \
    "nbconvert>=7.17.0" \
    "geemap>=0.37.1" \
    "eemont>=2025.7.1" \
    "streamlit>=1.40.0" \
    "plotly>=6.0.0" \
    "ffmpeg>=1.4" \
    "ffmpeg-python>=0.2.0"

COPY . .
RUN uv pip install -e . --no-deps

ENTRYPOINT ["uv", "run", "water-timeseries"]