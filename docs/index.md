# water-timeseries

Automated analysis of water timeseries data from satellite imagery and remote sensing sources.

## Features

- **Dynamic World Handler**: Process Dynamic World land cover classifications
- **JRC Water Handler**: Handle JRC water occurrence and classification data
- **Data Normalization**: Automatic normalization and scaling of time series
- **Breakpoint Detection**: Statistical (SimpleBreakpoint) and advanced (RBEAST) methods for detecting water extent changes
- **Batch Processing**: Efficient processing of multiple spatial entities
- **Comprehensive Testing**: Full test coverage including breakpoint detection, normalization, and integration tests
- **Command Line Interface**: Run breakpoint detection from the command line with CLI tool

## Quick Start

```python
from water_timeseries.dataset import DWDataset
import xarray as xr

# Load data
ds = xr.open_dataset("water_data.nc")

# Process with Dynamic World handler
processor = DWDataset(ds)

# Access normalized time series
water_extent = processor.ds_normalized["water"]
```

## Installation

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

## Documentation

- [Getting Started](getting_started.md) - Installation and setup guide
- [Examples](examples.md) - Usage examples and tutorials
- [API Reference](api/index.md) - Complete API documentation

## Command Line Interface

The package provides a CLI tool `water-timeseries-bp` for running breakpoint detection on water datasets.

### Quick Start

```bash
# Show help
uv run water-timeseries-bp --help

# Run with required arguments
uv run water-timeseries-bp --water-dataset-file data.zarr --output-file output.parquet

# Run with config file
uv run water-timeseries-bp -C configs/config.yaml
```

### Configuration File

Create a YAML config file:

```yaml
water_dataset_file: /path/to/data.zarr
output_file: /path/to/output.parquet
vector_dataset_file: /path/to/lakes.parquet
chunksize: 100
n_jobs: 20
bbox_west: -160
bbox_east: -155
bbox_north: 68
bbox_south: 66
```

## Contributing

We welcome contributions! Please ensure you:

1. Add docstrings to new functions and classes (Google style)
2. Update documentation in the `docs/` folder
3. Run tests before submitting PRs

## Author

Ingmar Nitze
