# Getting Started

## Installation

Install the package using pip:

```bash
pip install water-timeseries
```

Or if you're developing locally:

```bash
git clone https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2
cd water-timeseries-v2
pip install -e ".[dev]"
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

## Key Classes

### `LakeDataset`
Base class for lake dataset handling. Provides preprocessing, normalization, and masking functionality.

### `DWDataset`
Handles Dynamic World land cover data with classes for water, bare, snow, trees, grass, and more.

### `JRCDataset`
Handles Joint Research Centre (JRC) water data with permanent and seasonal water classifications.

## Next Steps

- See [API Reference](api/index.md) for detailed class documentation
- Check out the [Examples](examples.md) for more use cases
- Visit the [GitHub repository](https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2)
