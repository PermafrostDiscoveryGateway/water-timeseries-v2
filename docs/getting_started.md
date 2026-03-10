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

## Command Line Interface

The package includes a CLI tool `water-timeseries-bp` for running breakpoint detection from the command line.

### Installation

The CLI is installed automatically with the package:

```bash
uv sync
```

### Basic Usage

```bash
# Show all options
uv run water-timeseries-bp --help

# Run with required arguments
uv run water-timeseries-bp --water-dataset-file data.zarr --output-file output.parquet

# Run with optional parameters
uv run water-timeseries-bp \
    --water-dataset-file data.zarr \
    --output-file output.parquet \
    --chunksize 100 \
    --n-jobs 4

# Run with a config file
uv run water-timeseries-bp -C configs/config.yaml
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
