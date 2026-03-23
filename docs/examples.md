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

## Merging Datasets

The `merge()` method allows combining two LakeDataset instances. This is useful for combining data from different time periods or adding new lakes to an existing dataset.

### Merge Along Both Dimensions

The default strategy merges along both date and id_geohash dimensions:

```python
from water_timeseries.dataset import DWDataset
import xarray as xr

# Load two datasets from different time periods
ds1 = xr.open_zarr("data_2020_2022.zarr")
ds2 = xr.open_zarr("data_2023_2024.zarr")

dataset1 = DWDataset(ds1)
dataset2 = DWDataset(ds2)

# Merge along both dimensions
merged = dataset1.merge(dataset2, how="both")

print(f"Original dates: {len(dataset1.dates_)} + {len(dataset2.dates_)}")
print(f"Merged dates: {len(merged.dates_)}")
```

### Add New Dates (Same Lakes)

Use `how="date"` to extend the time series of existing lakes:

```python
# Both datasets must have the same lakes (id_geohash)
ds_early = xr.open_zarr("data_2020_2021.zarr")
ds_late = xr.open_zarr("data_2022_2023.zarr")

dataset_early = DWDataset(ds_early)
dataset_late = DWDataset(ds_late)

# Add new dates to existing time series
merged = dataset_early.merge(dataset_late, how="date")

# Check that we have all dates
print(f"Total dates: {len(merged.dates_)}")
# Output: Combined dates from both datasets
```

### Add New Lakes (Same Time Period)

Use `how="id_geohash"` to add new lakes with the same temporal coverage:

```python
# Both datasets must have the same dates
ds_region1 = xr.open_zarr("region_a.zarr")  # Lakes in region A
ds_region2 = xr.open_zarr("region_b.zarr")  # Lakes in region B

dataset1 = DWDataset(ds_region1)
dataset2 = DWDataset(ds_region2)

# Add new lakes
merged = dataset1.merge(dataset2, how="id_geohash")

print(f"Total lakes: {len(merged.object_ids_)}")
# Output: Combined lake count from both datasets
```

### Handling Overlapping Data

When there are overlapping dates or id_geohash values, a warning is issued:

```python
import warnings

# Enable warnings to be displayed
warnings.filterwarnings("default")

# This will issue a warning about overlapping dates
merged = dataset1.merge(dataset2, how="date")
# UserWarning: Datasets have X overlapping dates...
```

### Requirements

- Both datasets must be the same type (both `DWDataset` or both `JRCDataset`)
- Both datasets must have the same variables
- For `how="date"`: Same id_geohash values required
- For `how="id_geohash"`: Same dates required

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

## Breakpoint Detection

```python
from water_timeseries.breakpoint import SimpleBreakpoint, BeastBreakpoint
from water_timeseries.dataset import DWDataset
import xarray as xr

# Load a small test dataset
xr_ds = xr.open_zarr('tests/data/lakes_dw_test.zarr')
# Wrap in the dataset class
ds = DWDataset(xr_ds)

# Simple method – one lake
simple = SimpleBreakpoint()
print(simple.calculate_break(ds, 'b7uefy0bvcrc'))

# Beast method – batch processing
beast = BeastBreakpoint()
print(beast.calculate_breaks_batch(ds, progress_bar=False).head())
```


Detect changes in water extent over time using statistical or advanced methods:

```python
from water_timeseries.breakpoint import SimpleBreakpoint, BeastBreakpoint

# Initialize dataset
processor = DWDataset(ds)

# Simple statistical breakpoint detection
simple_bp = SimpleBreakpoint()
breaks_simple = simple_bp.calculate_break(processor, geohash_id="your_geohash")

# Advanced RBEAST-based detection
beast_bp = BeastBreakpoint()
breaks_beast = beast_bp.calculate_break(processor, geohash_id="your_geohash")

# Batch processing for all geohashes
all_breaks = simple_bp.calculate_breaks_batch(processor, progress_bar=True)

# Results include:
# - date_break: When the break was detected
# - date_before_break: Date immediately before the break
# - break_method: "simple" or "rbeast"
# - break_number: Sequential numbering (Beast only)
# - proba_rbeast: Probability score (Beast only)
```

### Breakpoint Methods

- **SimpleBreakpoint**: Statistical method using rolling window comparisons
- **BeastBreakpoint**: Advanced Bayesian analysis using RBEAST library

## Command Line Interface

The hierarchical CLI tool provides a convenient way to run breakpoint detection without writing Python code.

### Running from Command Line

```bash
# Basic usage with required arguments
uv run water-timeseries breakpoint-analysis \
    /path/to/lakes.zarr \
    /path/to/breaks.parquet

# With parallel processing
uv run water-timeseries breakpoint-analysis \
    /path/to/lakes.zarr \
    /path/to/breaks.parquet \
    --chunksize 100 \
    --n-jobs 20

# With bounding box filter
uv run water-timeseries breakpoint-analysis \
    /path/to/lakes.zarr \
    /path/to/breaks.parquet \
    --vector-dataset-file /path/to/lakes.gpkg \
    --bbox-west -160 \
    --bbox-east -155 \
    --bbox-north 68 \
    --bbox-south 66
```

### Using Configuration Files

For complex workflows, use a config file:

```yaml
# config.yaml
water_dataset_file: /path/to/lakes.zarr
output_file: /path/to/breaks.parquet
vector_dataset_file: /path/to/lakes.gpkg
chunksize: 100
n_jobs: 20
bbox_west: -160
bbox_east: -155
bbox_north: 68
bbox_south: 66
min_chunksize: 10
```

```bash
# Run with config file
uv run water-timeseries breakpoint-analysis --config-file config.yaml

# Override specific config values from CLI
uv run water-timeseries breakpoint-analysis --config-file config.yaml --n-jobs 8
```

### CLI Options Reference

| Option | Short | Description | Default |
|--------|-------|-------------|--------|
| `water_dataset_file` | | Path to water dataset (zarr) | Required* |
| `output_file` | | Path to output parquet | Required* |
| `--config-file` | | Path to config file | None |
| `--vector-dataset-file` | `-v` | Path to vector dataset | None |
| `--chunksize` | `-c` | IDs per chunk | 100 |
| `--n-jobs` | `-j` | Parallel jobs | 1 |
| `--min-chunksize` | `-m` | Min chunk size | 10 |
| `--bbox-west` | | Min longitude | None |
| `--bbox-south` | | Min latitude | None |
| `--bbox-east` | | Max longitude | None |
| `--bbox-north` | | Max latitude | None |
| `--output-geometry` | | Include geometry in output (default: True) | True |
| `--output-geometry-all` | | Include geometry for all lakes (default: False) | False |

*Can also be provided via config file

## Plot Timeseries

Plot time series for a specific lake using the CLI:

```bash
# Plot lake timeseries
uv run water-timeseries plot-timeseries data.zarr --lake-id b7uefy0bvcrc

# Save figure to file
uv run water-timeseries plot-timeseries data.zarr --lake-id b7uefy0bvcrc --output-figure plot.png

# Save only (no popup window)
uv run water-timeseries plot-timeseries data.zarr --lake-id b7uefy0bvcrc --output-figure plot.png --no-show

# Use config file
uv run water-timeseries plot-timeseries --config-file configs/plot_config.yaml
```

### Plot Options

| Option | Short | Description | Default |
|--------|-------|-------------|--------|
| `water_dataset_file` | | Path to water dataset (zarr or netCDF) | Required* |
| `--lake-id` | | Geohash ID of the lake | Required* |
| `--output-figure` | | Path to save output figure | None |
| `--break-method` | | Break method to overlay (beast or simple) | None |
| `--no-show` | | Don't show popup window | False |

// The `--no-show` flag suppresses the interactive plot window; use it when running headless or when only saving the figure.
| `--config-file` | | Path to config YAML/JSON file | None |

*Can also be provided via config file

## Interactive Dashboard

The package includes a Streamlit dashboard for interactive visualization of lake polygons and time series data.

### Running the Dashboard

```bash
# Install streamlit and plotly if not already installed
pip install streamlit plotly

# Run the dashboard
streamlit run src/water_timeseries/dashboard/app.py
```

### Dashboard Workflow

1. **Map View**: The dashboard loads lake polygons from a parquet file and displays them on an interactive map
2. **Selection**: Click on any lake polygon to select it
3. **Time Series**: The dashboard automatically loads/creates a DWDataset and plots the time series for the selected lake
4. **Automatic Download**: If the selected lake's data is not in the cached zarr file, it automatically downloads from Google Earth Engine

### Sidebar Settings

- **Google Earth Engine Project**: Enter your EE project ID and click "Set EE Project"
- **Parquet File Path**: Path to the lake polygons file (default: `tests/data/lake_polygons.parquet`)
- **Zarr Path**: Path to cached time series data (default: `tests/data/lakes_dw_test.zarr`)
- **ID Column**: The column name containing geohash IDs (default: `id_geohash`)
- **Zoom Level**: Initial map zoom (1-20)

### Features

- **Hover**: View lake attributes (id_geohash, area, net change) on hover
- **Click**: Select a lake to view its time series
- **Popup**: Click "Open Time Series in Popup" for a larger plot view
- **Automatic Download**: Missing data is fetched automatically from GEE

### Python API

```python
from water_timeseries.dashboard.map_viewer import create_app

# Basic usage with defaults
create_app()

# Custom paths
create_app(
    data_path="/path/to/lakes.parquet",
    zarr_path="/path/to/timeseries.zarr"
)
```
