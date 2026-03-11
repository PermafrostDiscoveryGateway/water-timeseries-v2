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

### Command Line Interface

The package provides a hierarchical CLI tool `water-timeseries` for running breakpoint detection on water datasets.

#### Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install .
```

#### Basic Usage

```bash
# Show help
uv run water-timeseries --help

# Show breakpoint-analysis subcommand help
uv run water-timeseries breakpoint-analysis --help

# Run with required arguments
uv run water-timeseries breakpoint-analysis data.zarr output.parquet

# Run with optional parameters
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
| `--bbox-west` | | Minimum longitude (west) | None |
| `--bbox-south` | | Minimum latitude (south) | None |
| `--bbox-east` | | Maximum longitude (east) | None |
| `--bbox-north` | | Maximum latitude (north) | None |

*Can also be provided via config file

#### Plot Lake Timeseries

Plot time series for a specific lake:

```bash
# Plot lake timeseries
uv run water-timeseries plot-lake-timeseries data.zarr --lake-id abc123

# Save figure to file
uv run water-timeseries plot-lake-timeseries data.zarr --lake-id abc123 --output-figure plot.png

# Use config file
uv run water-timeseries plot-lake-timeseries --config-file configs/plot_config.yaml
```

Plot options:

| Option | Short | Description | Default |
|--------|-------|-------------|--------|
| `water_dataset_file` | | Path to water dataset (zarr) | Required* |
| `--lake-id` | | Geohash ID of the lake | Required* |
| `--output-figure` | | Path to save output figure | None |
| `--break-method` | | Break method to overlay | None |
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
