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

Both `DWDataset` and `JRCDataset` provide two methods for visualization:
- `plot_timeseries()` - Static matplotlib plots
- `plot_timeseries_interactive()` - Interactive Plotly plots

### DWDataset.plot_timeseries() / plot_timeseries_interactive()

```python
from water_timeseries.dataset import DWDataset
import xarray as xr

# Load data
ds = xr.open_zarr("lakes_dw.zarr")
dataset = DWDataset(ds)

# Get a geohash from the dataset
geohash = dataset.object_ids_[0]

# Static matplotlib plot
fig = dataset.plot_timeseries(
    id_geohash=geohash,
    breakpoints=None  # Optional: see breakpoints section below
)

# Interactive Plotly plot (returns go.Figure)
fig_interactive = dataset.plot_timeseries_interactive(
    id_geohash=geohash,
    breakpoints=None
)
```

### JRCDataset.plot_timeseries() / plot_timeseries_interactive()

```python
from water_timeseries.dataset import JRCDataset
import xarray as xr

# Load data
ds = xr.open_zarr("lakes_jrc.zarr")
dataset = JRCDataset(ds)

# Get a geohash from the dataset
geohash = dataset.object_ids_[0]

# Static matplotlib plot
fig = dataset.plot_timeseries(
    id_geohash=geohash,
    breakpoints=None
)

# Interactive Plotly plot
fig_interactive = dataset.plot_timeseries_interactive(
    id_geohash=geohash
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `id_geohash` | str | The geohash identifier for the lake to plot |
| `breakpoints` | BreakpointMethod, pd.Timestamp, str, list, optional | Breakpoint(s) to overlay on the plot. Can be a `BreakpointMethod` instance (e.g., `SimpleBreakpoint()`), a single date string ("YYYY-MM-DD") or `pd.Timestamp`, or a list of dates. Only the first date is used. |
| `plot_variables` | list, optional | List of variables to plot. If None, uses all variables. For DWDataset: `["water", "bare", "vegetation", "snow_and_ice"]`. For JRCDataset: `["area_water_permanent", "area_water_seasonal", "area_land"]`. |
| `save_path` | str, Path, optional | If provided, saves the plot to this path (`.png` for static, `.html` for interactive) |

### Return Values

| Method | Return Type | Description |
|--------|-------------|-------------|
| `plot_timeseries()` | `matplotlib.figure.Figure` | Static matplotlib figure |
| `plot_timeseries_interactive()` | `plotly.graph_objects.Figure` | Interactive Plotly figure (can be displayed in notebooks, saved as HTML, or used with Streamlit) |

### With Breakpoint Detection

The `breakpoints` parameter accepts a `BreakpointMethod` instance which will automatically detect and visualize the breakpoint:

```python
from water_timeseries.dataset import DWDataset
from water_timeseries.breakpoint import SimpleBreakpoint
import xarray as xr

# Initialize dataset
dataset = DWDataset(xr.open_zarr("lakes_dw.zarr"))
geohash = dataset.object_ids_[0]

# Create breakpoint method and pass directly to plot
bp = SimpleBreakpoint()

# Static plot with breakpoint
fig = dataset.plot_timeseries(
    id_geohash=geohash,
    breakpoints=bp  # Pass the BreakpointMethod, not the result!
)

# Interactive plot with breakpoint
fig_interactive = dataset.plot_timeseries_interactive(
    id_geohash=geohash,
    breakpoints=bp
)
```

### With Specific Date

Alternatively, you can pass a specific date or list of dates:

```python
# Single date (string)
fig = dataset.plot_timeseries(
    id_geohash=geohash,
    breakpoints="2023-06-15"
)

# Single date (pd.Timestamp)
import pandas as pd
fig = dataset.plot_timeseries(
    id_geohash=geohash,
    breakpoints=pd.Timestamp("2023-06-15")
)

# List of dates (only first is used)
fig = dataset.plot_timeseries(
    id_geohash=geohash,
    breakpoints=["2023-06-15", "2020-09-01"]
)
```

### With Custom plot_variables

You can customize which variables are displayed using the `plot_variables` parameter:

```python
# DWDataset: plot only water and bare (exclude vegetation and snow_and_ice)
fig = dataset.plot_timeseries_interactive(
    id_geohash=geohash,
    plot_variables=["water", "bare"]
)

# JRCDataset: plot only permanent water
fig = dataset.plot_timeseries_interactive(
    id_geohash=geohash,
    plot_variables=["area_water_permanent"]
)
```

### Visual Output

**DWDataset Time Series**

![DW Time Series Example](../../tests/data/figures/example_dw_timeseries.png)

The DWDataset plot shows land cover class proportions:
- **Water (blue)**: Primary water extent indicator
- **Vegetation (green)**: Combined trees, grass, crops, shrub/scrub, flooded vegetation
- **Bare (brown)**: Bare soil
- **Snow and Ice (black)**: Snow/ice coverage
- Values are shown in hectares (left axis) with optional percentage scale (right axis)

Use `plot_variables` to select which classes to display (e.g., `plot_variables=["water", "bare", "vegetation"]` to exclude snow_and_ice).

**JRCDataset Time Series**

![JRC Time Series Example](../../tests/data/figures/example_jrc_timeseries.png)

The JRCDataset plot shows permanent vs seasonal water:
- **Permanent water (blue)**: Water present year-round
- **Seasonal water (light blue)**: Water present seasonally
- **Land (brown)**: Dry land area
- Gray shading indicates no-data regions
- Values are shown in hectares (left axis) with optional percentage scale (right axis)

**With Breakpoint Overlay**

When a breakpoint is provided, a vertical dashed black line marks the date:
- Static plots: Includes "Breakpoint" entry in the legend
- Interactive plots: Includes "Breakpoint" entry in the legend for hover inspection
