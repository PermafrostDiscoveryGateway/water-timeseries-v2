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

##### Show Help for cli

```bash
# Show help
uv run water-timeseries --help
```

##### Main CLI tools

```bash
# Show breakpoint-analysis-historical subcommand help
uv run water-timeseries breakpoint-analysis-historical

# Show breakpoint-analysis-historical subcommand help
uv run water-timeseries breakpoint-analysis-nrt

# Show plot-timeseries subcommand help
uv run water-timeseries plot-timeseries

# run dashboard
uv run water-timeseries dashboard
```

#### Using a Config File

You can also use a YAML configuration file:

```bash
uv run water-timeseries breakpoint-analysis-historical --config-file configs/config.yaml
```

Example config file for breakpoint-analysis-historical:

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
| --------| ------- | ------------- | -------- |
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

# Plot lake timeseries with break detection
uv run water-timeseries plot-timeseries tests/data/lakes_dw_test.zarr --lake-id b7uefy0bvcrc --output-figure examples/dw_example_b7uefy0bvcrc.png --break-method beast
```

![Example Timeseries Plot](../examples/dw_example_b7uefy0bvcrc.png)

```bash
# Plot lake timeseries with JRC data
uv run water-timeseries plot-timeseries tests/data/lakes_jrc_test.zarr --lake-id b7uefy0bvcrc --output-figure examples/jrc_example_b7uefy0bvcrc.png --break-method beast
```

![Example Timeseries Plot](../examples/jrc_example_b7uefy0bvcrc.png)

Plot options:

| Option | Short | Description | Default |
| -------- | ------- | ------------- | -------- |
| `water_dataset_file` | | Path to water dataset (zarr or netCDF) | Required* |
| `--lake-id` | | Geohash ID of the lake | Required* |
| `--output-figure` | | Path to save output figure | None |
| `--break-method` | | Break method to overlay (beast or simple) | None |
| `--no-show` | | Don't show popup window, only save if output-figure is provided | False |
| `--config-file` | | Path to config YAML/JSON file | None |

*Can also be provided via config file

## Main Classes

### Datasets

- **DWDataset**: Dynamic World land cover processor
- **JRCDataset**: JRC water classification processor

### Download

- **EarthEngineDownloader**: Download data from Google Earth Engine

### Breakpoints

- **SimpleBreakpoint**: Statistical breakpoint detection
- **BeastBreakpoint**: Advanced RBEAST-based detection

## Interactive Dashboard

The package includes an interactive Streamlit dashboard for visualizing lake polygons and time series data.

### Running the Dashboard

```bash
uv run water-timeseries dashboard
```

### Features

- **Map Viewer**: Interactive map displaying lake polygons from parquet files
- **Hover Tooltips**: View attributes (id_geohash, Area_start_ha, Area_end_ha, NetChange_ha, NetChange_perc)
- **Click Selection**: Click on a polygon to select it and view its time series
- **Time Series Plot**: Automatic visualization of water extent over time
- **Automatic Download**: If the selected lake's data is not in the cached dataset, it automatically downloads from Google Earth Engine
- **Popup View**: Click "Open Time Series in Popup" for a larger view
- **EE Project Config**: Set your Google Earth Engine project in the sidebar

### Configuration

The dashboard accepts the following optional arguments:

| Parameter | Description | Default |
| ----------- | ------------- | --------- |
| `vector_file` | Path to vector dataset file (GeoParquet) | `tests/data/lake_polygons.parquet` |
| `dw_dataset_file` | Path to Dynamic World dataset file (zarr or nc) | `tests/data/lakes_dw_test.zarr` |
| `jrc_dataset_file` | Path to JRC dataset file (zarr or nc) | `tests/data/lakes_jrc_test.zarr` |
| `precomputed_nrt_dir` | Directory with pre-computed NRT parquet files. Auto-detected from `precomputed/nrt/` in the repo root when present | `None` |
| `offline_mode` | If set, disables Google Earth Engine download functionality. Use this when running without internet access or Earth Engine authentication | `False` |
| `ee_project` | Google Earth Engine project ID. Required for EE downloads | `None` |
| `dw_start_year` | Start year for Dynamic World dataset time series | `2017` |
| `dw_end_year` | End year for Dynamic World dataset time series | `2025` |
| `dw_start_month` | Start month (1-12) for Dynamic World dataset time series filtering | `6` (June) |
| `dw_end_month` | End month (1-12) for Dynamic World dataset time series filtering | `9` (September) |
| `port` | Port to run the dashboard on | `8501` |
| `logfile` | Path to log file. If not provided, a default logfile is created with the format `{subcommand}_{timestamp}.log` | Auto-generated |
| `verbose` | Verbosity level (`-v` for DEBUG) | `0` (INFO) |
| `config_file` | Path to a YAML or JSON configuration file containing default parameters. CLI arguments take priority over config file values | `None` |

#### Example usage

```bash
uv run water-timeseries dashboard
uv run water-timeseries dashboard --vector-file data/lakes.parquet --dw-dataset-file data/lakes.zarr
uv run water-timeseries dashboard --offline-mode
uv run water-timeseries dashboard --ee-project my-ee-project
uv run water-timeseries dashboard --dw-start-year 2017 --dw-end-year 2026
uv run water-timeseries dashboard --config-file configs/dashboard_config.yaml
```

#### Example config file

```yaml
# dashboard with downloading time-series on the fly for 2017-2026 (May-October)
vector_file: notebooks_temporary/drain_2026-05_high_confidence.parquet
ee_project: my-ee-project
dw_start_year: 2017
dw_end_year: 2026
dw_start_month: 5
dw_end_month: 10
```

## Documentation Links

- [Getting Started](getting_started.md) - Installation and setup guide
- [Examples](examples.md) - Usage examples and tutorials
- [API Reference](api/index.md) - Complete API documentation

## Contributing

We welcome contributions! Please ensure you:

1. Add docstrings to new functions and classes (Google style)
2. Update documentation in the `docs/` folder
3. Run tests before submitting PRs

## License

[Add your license here]

## Author

Ingmar Nitze
