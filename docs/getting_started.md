# Getting Started

## Installation

For development with all testing and documentation tools:

```bash
git clone https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2
cd water-timeseries-v2
pip install -e ".[dev]"
```

Or with uv:

```bash
git clone https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2
cd water-timeseries-v2
uv sync
```

For installing just the runtime dependencies:

```bash
pip install .
```

## Quick Example

```python
from water_timeseries.dataset import DWDataset
import xarray as xr

# Load your data
ds = xr.open_dataset("your_data.nc")

# Create a DWDataset instance
dataset = DWDataset(ds)

# Access normalized data
normalized_data = dataset.ds_normalized

# Access the preprocessed dataset
preprocessed_ds = dataset.ds
```

## Downloading from Google Earth Engine

The `EarthEngineDownloader` class allows you to download Dynamic World land cover data directly from Google Earth Engine.

### Initialization

```python
import os
from loguru import logger
from water_timeseries.downloader import EarthEngineDownloader

# Set your EE project via environment variable
os.environ["EE_PROJECT"] = "your-project"

# Or pass directly as parameter
dl = EarthEngineDownloader(ee_project="your-project", ee_auth=True, logger=logger)
```

### Download Parameters

The `download_dw_monthly()` method supports the following parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `vector_dataset` | str/Path | Path to input vector dataset (Parquet format) | Required |
| `name_attribute` | str | Column name in vector dataset for grouping | Required |
| `years` | List[int] | Years to process | [2017-2025] |
| `months` | List[int] | Months to process (default: June-September) | [6,7,8,9] |
| `bbox_west/east/north/south` | float | Bounding box for spatial filtering | Global (-180 to 180, -90 to 90) |
| `id_list` | List[str] | Filter by specific IDs (from name_attribute column) | None (all) |
| `scale` | float | Pixel scale in meters | 10 |
| `max_total_requests` | int | Max requests per chunk (controls chunking) | 500 |
| `n_parallel` | int | Number of parallel workers (1 = sequential) | 1 |
| `no_download` | bool | If True, only log parameters without downloading | False |
| `save_to_file` | str | Path to save dataset (.zarr or .nc). Relative paths go to output dir | None |

### Usage Examples

#### Basic Download

Download all features from the test dataset:

```python
dl = EarthEngineDownloader(ee_project="your-project", ee_auth=True, logger=logger)

ds = dl.download_dw_monthly(
    vector_dataset="tests/data/lake_polygons.parquet",
    name_attribute="id_geohash",
    years=[2024],
    months=[7, 8],
)
```

#### Filter by Specific IDs

Download only specific lakes using their geohash IDs:

```python
ds = dl.download_dw_monthly(
    vector_dataset="tests/data/lake_polygons.parquet",
    name_attribute="id_geohash",
    id_list=["b7g6g1ny1mf7", "b7g4yc12k4yj", "b7g6c8gye56e"],
    years=[2024],
    months=[7, 8],
)
```

#### Parallel Download

Speed up large downloads by using multiple parallel workers:

```python
ds = dl.download_dw_monthly(
    vector_dataset="tests/data/lake_polygons.parquet",
    name_attribute="id_geohash",
    n_parallel=4,  # Use 4 parallel workers
    max_total_requests=500,  # Control chunk size
    years=[2024, 2025],
    months=[6, 7, 8, 9],
)
```

#### Spatial Bounding Box Filter

Filter by geographic region:

```python
ds = dl.download_dw_monthly(
    vector_dataset="tests/data/lake_polygons.parquet",
    name_attribute="id_geohash",
    bbox_west=-165,
    bbox_east=-164,
    bbox_south=66.2,
    bbox_north=66.6,
    years=[2024],
    months=[7, 8],
)
```

#### Preview Mode (No Download)

Test your parameters without actually downloading data:

```python
ds = dl.download_dw_monthly(
    vector_dataset="tests/data/lake_polygons.parquet",
    name_attribute="id_geohash",
    years=[2024],
    months=[7, 8],
    no_download=True,  # Only logs parameters, skips actual download
)
```

#### Save to File

Automatically save the downloaded dataset to file. The format is determined by the file extension:

```python
# Save to Zarr format (relative path goes to output directory)
ds = dl.download_dw_monthly(
    vector_dataset="tests/data/lake_polygons.parquet",
    name_attribute="id_geohash",
    years=[2024],
    months=[7, 8],
    save_to_file="data.zarr",  # Saves to downloads/data.zarr
)

# Save to NetCDF format (absolute path)
ds = dl.download_dw_monthly(
    vector_dataset="tests/data/lake_polygons.parquet",
    name_attribute="id_geohash",
    years=[2024],
    months=[7, 8],
    save_to_file="/path/to/output/data.nc",
)
```

### Test Dataset

The package includes a test dataset at `tests/data/lake_polygons.parquet` with 118 lake polygons in Alaska.

### Saving and Loading Datasets

The package provides utility functions for saving and loading xarray datasets:

```python
from water_timeseries.utils import save_xarray_dataset, load_xarray_dataset

# Save to Zarr format
save_xarray_dataset(ds, "output.zarr", output_dir="./data")

# Save to NetCDF format
save_xarray_dataset(ds, "/full/path/output.nc")

# Load from Zarr
ds = load_xarray_dataset("output.zarr")

# Load from NetCDF
ds = load_xarray_dataset("output.nc", format="netcdf")
```

## Command Line Interface

The package includes a hierarchical CLI tool `water-timeseries` for running breakpoint detection from the command line.

### Installation

The CLI is installed automatically with the package:

```bash
uv sync
```

### Basic Usage

```bash
# Show all options
uv run water-timeseries --help

# Show breakpoint-analysis subcommand help
uv run water-timeseries breakpoint-analysis --help

# Show plot-timeseries subcommand help
uv run water-timeseries plot-timeseries --help

# Show dashboard subcommand help
uv run water-timeseries dashboard --help

# Run breakpoint analysis
uv run water-timeseries breakpoint-analysis data.zarr output.parquet

# Run with optional parameters
uv run water-timeseries breakpoint-analysis \
    data.zarr \
    output.parquet \
    --chunksize 100 \
    --n-jobs 4

# Run with a config file
uv run water-timeseries breakpoint-analysis --config-file configs/config.yaml

# Run with parallel backend (joblib or ray)
uv run water-timeseries breakpoint-analysis \
    data.zarr \
    output.parquet \
    --parallel-backend joblib \
    --chunksize 50 \
    --n-jobs 20

# Plot lake timeseries
uv run water-timeseries plot-timeseries data.zarr --lake-id b7uefy0bvcrc

# Save figure to file
uv run water-timeseries plot-timeseries data.zarr --lake-id b7uefy0bvcrc --output-figure plot.png

# Save only (no popup window)
uv run water-timeseries plot-timeseries data.zarr --lake-id b7uefy0bvcrc --output-figure plot.png --no-show

# Launch the Streamlit dashboard (default port 8501)
uv run water-timeseries dashboard

# Launch dashboard on a custom port
uv run water-timeseries dashboard --port 8502
```

### Using a Config File

Create a YAML configuration file:

```yaml
# config.yaml
water_dataset_file: /path/to/your/data.zarr
output_file: /path/to/output.parquet

# Optional: vector dataset for bbox filtering
vector_dataset_file: /path/to/lakes.parquet

# Bounding box (optional)
bbox_west: -160
bbox_east: -155
bbox_north: 68
bbox_south: 66

# Processing options
chunksize: 100
n_jobs: 20
min_chunksize: 10
parallel_backend: joblib  # or "ray"
```

CLI arguments take priority over config file values.

## Key Classes

### `LakeDataset`
Base class for lake dataset handling. Provides preprocessing, normalization, and masking functionality.

### `DWDataset`
Handles Dynamic World land cover data with classes for water, bare, snow, trees, grass, and more.

### `JRCDataset`
Handles Joint Research Centre (JRC) water data with permanent and seasonal water classifications.

## Testing

The package includes comprehensive tests for all functionality. Tests are organized by module and cover:

- **Dataset processing**: Normalization, masking, and preprocessing
- **Breakpoint detection**: Simple and RBEAST-based methods
- **Integration tests**: End-to-end functionality with real and synthetic data

## Interactive Dashboard

The package includes an interactive Streamlit dashboard for visualizing lake polygons and time series data.

### Running the Dashboard

```bash
# Launch via CLI (recommended)
uv run water-timeseries dashboard

# Or with a custom port
uv run water-timeseries dashboard --port 8502

# Alternative: Run directly with streamlit
streamlit run src/water_timeseries/dashboard/app.py
```

### Dashboard Features

The dashboard provides a graphical interface for:

1. **Map Visualization**: Interactive map showing lake polygons from a parquet file
   - Hover over polygons to see attributes (id_geohash, area, net change)
   - Click on a polygon to select it

2. **Time Series Plotting**: Automatically plots water extent over time for the selected lake
   - Shows a preview below the map
   - Click "Open Time Series in Popup" for a larger view

3. **Automatic Download**: If the selected lake's data is not in the cached dataset:
   - Shows "Downloading..." status
   - Automatically fetches data from Google Earth Engine
   - Displays the time series plot after download completes

4. **Google Earth Engine Configuration**:
   - Enter your EE project in the sidebar
   - Click "Set EE Project" to save it

### Dashboard Arguments

The `create_app()` function accepts these parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `data_path` | str/Path | Path to parquet file with lake polygons | `tests/data/lake_polygons.parquet` |
| `zarr_path` | str/Path | Path to zarr file with cached time series | `tests/data/lakes_dw_test.zarr` |

### Using with Custom Data

```python
from water_timeseries.dashboard.map_viewer import create_app

# Create dashboard with custom paths
create_app(
    data_path="/path/to/your/lakes.parquet",
    zarr_path="/path/to/your/data.zarr"
)
```

Or run directly with custom paths:

```bash
# Modify the app.py or create a custom launcher
streamlit run your_custom_app.py
```

### Running Tests

To run the test suite:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test modules
pytest tests/test_breakpoints.py
pytest tests/test_normalization.py

# Run with coverage
pytest --cov=water_timeseries
```

### Test Data

Tests use both real and synthetic datasets:
- **Real data**: Located in `tests/data/` (DW and JRC test datasets)
- **Synthetic data**: Generated programmatically for predictable breakpoint testing

## Next Steps

- See [API Reference](api/index.md) for detailed class documentation
- Check out the [Examples](examples.md) for more use cases
- Visit the [GitHub repository](https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2)
