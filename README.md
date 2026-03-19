# water-timeseries-v2

Automated analysis of water timeseries data from satellite imagery and remote sensing sources.

## Documentation

**📖 Full Documentation**: [View Documentation](https://PermafrostDiscoveryGateway.github.io/water-timeseries-v2/)

The documentation includes:

- Getting started guide
- API reference (auto-generated from code)
- Usage examples
- Tutorial notebooks

Documentation is automatically built and deployed on every push to `main` using GitHub Actions.

## Features

- **Dynamic World Handler**: Process Dynamic World land cover classifications
- **JRC Water Handler**: Handle JRC water occurrence and classification data
- **Earth Engine Downloader**: Download data directly from Google Earth Engine
- **Data Normalization**: Automatic normalization and scaling of time series
- **Breakpoint Detection**: Statistical (SimpleBreakpoint) and advanced (RBEAST) methods for detecting water extent changes
- **Batch Processing**: Efficient processing of multiple spatial entities
- **Comprehensive Testing**: Full test coverage including breakpoint detection, normalization, and integration tests

## Quick Start

### Python API

```python
from water_timeseries.dataset import DWDataset
import xarray as xr

# Load data
ds = xr.open_dataset("water_data.nc")

# Process with Dynamic World handler
processor = DWDataset(ds)

# Access time series
water_extent = processor.ds_normalized[processor.water_column]

# Access normalized time series
water_extent = processor.ds_normalized["water"]
```

### Download from Google Earth Engine

```python
import os
from loguru import logger
from water_timeseries.downloader import EarthEngineDownloader

# Set your EE project (or pass directly as ee_project parameter)
os.environ["EE_PROJECT"] = "your-project"

# Create downloader instance
dl = EarthEngineDownloader(ee_auth=True, logger=logger)

# Basic download - download all features
ds = dl.download_dw_monthly(
    vector_dataset="tests/data/lake_polygons.parquet",
    name_attribute="id_geohash",
    years=[2024],
    months=[7, 8],
)

# Download only specific IDs
ds = dl.download_dw_monthly(
    vector_dataset="tests/data/lake_polygons.parquet",
    name_attribute="id_geohash",
    id_list=["b7g6g1ny1mf7", "b7g4yc12k4yj", "b7g6c8gye56e"],  # Filter by specific geohash IDs
    years=[2024],
    months=[7, 8],
)

# Parallel download (faster for large datasets)
ds = dl.download_dw_monthly(
    vector_dataset="tests/data/lake_polygons.parquet",
    name_attribute="id_geohash",
    n_parallel=4,  # Use 4 parallel workers
    max_total_requests=500,  # Request limit per chunk
    years=[2024],
    months=[7, 8],
)

# Preview download without actually downloading (useful for testing)
ds = dl.download_dw_monthly(
    vector_dataset="tests/data/lake_polygons.parquet",
    name_attribute="id_geohash",
    no_download=True,  # Only logs parameters, skips actual download
)

# Save to file (auto-detects format from extension: .zarr or .nc)
ds = dl.download_dw_monthly(
    vector_dataset="tests/data/lake_polygons.parquet",
    name_attribute="id_geohash",
    years=[2024],
    months=[7, 8],
    save_to_file="data.zarr",  # Saves to downloads/data.zarr (relative path)
)

# Absolute path example
ds = dl.download_dw_monthly(
    vector_dataset="tests/data/lake_polygons.parquet",
    name_attribute="id_geohash",
    years=[2024],
    months=[7, 8],
    save_to_file="/path/to/output/data.nc",  # Saves to absolute path as NetCDF
)
```

### Command Line Interface

```bash
uv run water-timeseries breakpoint-analysis \
    data.zarr \
    output.parquet \
    --chunksize 100 \
    --n-jobs 4
```

#### Using a Config File

You can also use a YAML configuration file:

```bash
uv run water-timeseries breakpoint-analysis --config-file configs/config.yaml
```

Example config file:

```yaml
# config.yaml
water_dataset_file: /path/to/data.zarr
output_file: /path/to/output.parquet

# Optional: vector dataset for bbox filtering
vector_dataset_file: /path/to/lakes.parquet

# Bounding box filter (optional)
bbox_west: -160
bbox_east: -155
bbox_north: 68
bbox_south: 66

# Processing options
chunksize: 100
n_jobs: 20
min_chunksize: 10
```

#### CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|--------|
| `water_dataset_file` | | Path to water dataset (zarr or parquet) | Required* |
| `output_file` | | Path to output parquet file | Required* |
| `--config-file` | | Path to config YAML/JSON file | None |
| `--vector-dataset-file` | `-v` | Path to vector dataset (gpkg, shp, geojson) | None |
| `--chunksize` | `-c` | Number of IDs per chunk | 100 |
| `--n-jobs` | `-j` | Number of parallel jobs (>1 for Ray) | 1 |
| `--min-chunksize` | `-m` | Minimum chunk size | 10 |
| `--bbox-west` | | Minimum longitude (west) | -180 |
| `--bbox-south` | | Minimum latitude (south) | 60 |
| `--bbox-east` | | Maximum longitude (east) | 180 |
| `--bbox-north` | | Maximum latitude (north) | 90 |
| `--output-geometry` | | Export output with geometries | True |
| `--output-geometry-all` | | Export output all geometries including non breakpoints | True |

*Can also be provided via config file

#### Plot Timeseries

Plot time series for a specific lake:

```bash
# Plot lake timeseries
uv run water-timeseries plot-timeseries data.zarr --lake-id b7uefy0bvcrc

# Save figure to file
uv run water-timeseries plot-timeseries data.zarr --lake-id b7uefy0bvcrc --output-figure plot.png

# Save only (no popup window)
uv run water-timeseries plot-timeseries data.zarr --lake-id b7uefy0bvcrc --output-figure plot.png --no-show

# Use config file
uv run water-timeseries plot-timeseries --config-file configs/plot_config.yaml

# Plot lake timeseries
uv run water-timeseries plot-timeseries tests/data/lakes_dw_test.zarr --lake-id b7uefy0bvcrc --output-figure examples/dw_example_b7uefy0bvcrc.png --break-method beast

```

![Example Timeseries Plot](examples/dw_example_b7uefy0bvcrc.png)

```python
# Plot lake timeseries
uv run water-timeseries plot-timeseries tests/data/lakes_jrc_test.zarr --lake-id b7uefy0bvcrc --output-figure examples/jrc_example_b7uefy0bvcrc.png --break-method beast
```

![Example Timeseries Plot](examples/jrc_example_b7uefy0bvcrc.png)

Plot options:

| Option | Short | Description | Default |
|--------|-------|-------------|--------|
| `water_dataset_file` | | Path to water dataset (zarr or netCDF) | Required* |
| `--lake-id` | | Geohash ID of the lake | Required* |
| `--output-figure` | | Path to save output figure | None |
| `--break-method` | | Break method to overlay (beast or simple) | None |
| `--no-show` | | Don't show popup window, only save if output-figure is provided | False |
| `--config-file` | | Path to config YAML/JSON file | None |

*Can also be provided via config file

## Installation

```bash
git clone https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2
cd water-timeseries-v2
```

`pip install .` or `uv sync`

Or for development:

```bash
git clone https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2
cd water-timeseries-v2
pip install -e ".[dev]"
```

## Main Classes

### Datasets

- **DWDataset**: Dynamic World land cover processor
- **JRCDataset**: JRC water classification processor

### Download

- **EarthEngineDownloader**: Download data from Google Earth Engine

### Breakpoints

- **SimpleBreakpoint**: Statistical breakpoint detection
- **BeastBreakpoint**: Advanced RBEAST-based detection

## Testing

The package includes comprehensive tests covering:

- Dataset normalization and masking
- Breakpoint detection methods (Simple and RBEAST)
- Batch processing functionality
- Integration tests with real and synthetic data

Run tests with: `pytest`

## Contributing

We welcome contributions! Please ensure you:

1. Add docstrings to new functions and classes (Google style)
2. Update documentation in the `docs/` folder
3. Run tests before submitting PRs

## License

[Add your license here]

## Author

Ingmar Nitze
