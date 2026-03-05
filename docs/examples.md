# Examples

## Working with DWDataset

The `DWDataset` class processes Dynamic World land cover classifications.

```python
import xarray as xr
from water_timeseries.dataset import DWDataset

# Load Dynamic World data
ds = xr.open_dataset("dynamic_world_data.nc")

# Initialize the dataset processor
processor = DWDataset(ds)

# Access data
original_data = processor.ds
normalized_data = processor.ds_normalized

# Check available data columns
print(processor.data_columns)
# Output: ['water', 'bare', 'snow_and_ice', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built']

# Water column
print(processor.water_column)  # 'water'
```

## Working with JRCDataset

The `JRCDataset` class handles JRC water classification data.

```python
from water_timeseries.dataset import JRCDataset

# Load JRC data
ds = xr.open_dataset("jrc_water_data.nc")

# Initialize the processor
processor = JRCDataset(ds)

# Access permanent and seasonal water data
print(processor.data_columns)
# Output: ['area_water_permanent', 'area_water_seasonal', 'area_land']

# Get preprocessed data with calculated total area
total_area = processor.ds["area_data"]
```

## Data Normalization

Both dataset classes automatically normalize data by the maximum area:

```python
processor = DWDataset(ds)

# Original data
original = processor.ds["water"]

# Normalized data (0-1 scale)
normalized = processor.ds_normalized["water"]

# The normalized data is scaled by max area across all dates
```

## Masking Invalid Data

Invalid data is automatically masked based on quality criteria:

```python
# After initialization, you have access to masked datasets
masked_data = processor.ds  # Original masked
masked_normalized = processor.ds_normalized  # Normalized masked

# Check if data is masked
print(processor.ds_ismasked_)
print(processor.ds_normalized_ismasked_)
```
