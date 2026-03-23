# Dataset Module

::: water_timeseries.dataset
    options:
      show_source: true
      docstring_style: google

## Merge Functionality

The `LakeDataset` class and its subclasses (`DWDataset`, `JRCDataset`) provide a `merge()` method to combine two datasets. This is useful for:

- Combining datasets from different time periods
- Adding new lakes to an existing dataset
- Combining partial datasets into a complete one

### Merge Strategies

The `merge()` method accepts a `how` parameter with three options:

| Strategy | Description | Requirements |
|----------|-------------|--------------|
| `"both"` | Merge along both dimensions (date and id_geohash). Combines all unique data from both datasets. | Same variables |
| `"date"` | Merge along the date dimension only. Adds new dates for the same lakes. | Same id_geohash values, same variables |
| `"id_geohash"` | Merge along the id_geohash dimension only. Adds new lakes with the same dates. | Same dates, same variables |

### Examples

```python
from water_timeseries.dataset import DWDataset
import xarray as xr

# Load two datasets
ds1 = xr.open_dataset("data_2020_2022.zarr")
dataset1 = DWDataset(ds1)

ds2 = xr.open_dataset("data_2023_2024.zarr")
dataset2 = DWDataset(ds2)

# Merge along both dimensions
merged = dataset1.merge(dataset2, how="both")

# Add new dates to existing time series (same lakes)
# Both datasets must have the same id_geohash values
merged = dataset1.merge(dataset2, how="date")

# Add new lakes with the same temporal coverage
# Both datasets must have the same dates
merged = dataset1.merge(dataset2, how="id_geohash")
```

### Warnings

When there are overlapping values, a warning is issued:

- **`how="date"`**: Warns if there are duplicate dates between datasets
- **`how="id_geohash"`**: Warns if there are duplicate id_geohash values

In both cases, data from the second dataset will overwrite the first for overlapping values.

### Requirements

- Both datasets must be of the same type (both `DWDataset` or both `JRCDataset`)
- Both datasets must have the same variables
- The specific merge strategy may have additional requirements (see table above)

### Return Value

The `merge()` method returns a new `LakeDataset` instance (of the same type as the first dataset) with the combined data. The returned dataset is fully preprocessed and normalized.

---

## Plot Time Series

Both `DWDataset` and `JRCDataset` provide a `plot_timeseries()` method to visualize water extent over time for a specific lake.

### DWDataset.plot_timeseries()

```python
from water_timeseries.dataset import DWDataset
import xarray as xr

# Load data
ds = xr.open_zarr("lakes_dw.zarr")
dataset = DWDataset(ds)

# Plot time series for a specific lake
fig = dataset.plot_timeseries(
    id_geohash="b7uefy0bvcrc",
    breakpoints=None  # Optional: pass BreakpointMethod to overlay detected breaks
)

# Show the plot
fig.show()
```

### JRCDataset.plot_timeseries()

```python
from water_timeseries.dataset import JRCDataset
import xarray as xr

# Load data
ds = xr.open_zarr("lakes_jrc.zarr")
dataset = JRCDataset(ds)

# Plot time series
fig = dataset.plot_timeseries(
    id_geohash="b7uefy0bvcrc",
    breakpoints=None  # Optional: BreakpointMethod to overlay detected breaks
)

fig.show()
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `id_geohash` | str | The geohash identifier for the lake to plot |
| `breakpoints` | BreakpointMethod, optional | Breakpoint detection result to overlay on the plot (e.g., from `SimpleBreakpoint` or `BeastBreakpoint`) |

### Return Value

Returns a `matplotlib.figure.Figure` object that can be displayed or saved.

### With Breakpoint Overlay

```python
from water_timeseries.dataset import DWDataset
from water_timeseries.breakpoint import SimpleBreakpoint

# Initialize dataset
dataset = DWDataset(xr.open_zarr("lakes_dw.zarr"))

# Detect breakpoints
bp = SimpleBreakpoint()
breaks = bp.calculate_break(dataset, object_id="b7uefy0bvcrc")

# Plot with breakpoint overlay
fig = dataset.plot_timeseries(
    id_geohash="b7uefy0bvcrc",
    breakpoints=breaks
)

fig.show()
```

### Visual Output

**DWDataset Time Series**

![DW Time Series Example](../../tests/data/figures/example_dw_timeseries.png)

The DWDataset plot shows land cover class proportions as a stacked area chart:
- **Water (blue)**: Primary water extent indicator
- **Vegetation classes** (trees, grass, crops, shrub): Grouped in green tones
- **Other classes** (built, bare, snow): Shown in distinct colors
- Values are normalized to total area (0-1 scale)

**JRCDataset Time Series**

![JRC Time Series Example](../../tests/data/figures/example_jrc_timeseries.png)

The JRCDataset plot shows permanent vs seasonal water as a line chart:
- **Permanent water (blue)**: Water present year-round
- **Seasonal water (light blue)**: Water present seasonally
- **Land**: Dry land area (shown in brown/green)
- Values are percentages (0-100)

**With Breakpoint Overlay**

When a breakpoint is detected, a vertical dashed line marks when significant water extent changes occurred.
