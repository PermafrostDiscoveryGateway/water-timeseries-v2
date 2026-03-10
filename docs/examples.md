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

## Breakpoint Detection

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

The CLI tool provides a convenient way to run breakpoint detection without writing Python code.

### Running from Command Line

```bash
# Basic usage with required arguments
uv run water-timeseries-bp \
    --water-dataset-file /path/to/lakes.zarr \
    --output-file /path/to/breaks.parquet

# With parallel processing
uv run water-timeseries-bp \
    --water-dataset-file /path/to/lakes.zarr \
    --output-file /path/to/breaks.parquet \
    --chunksize 100 \
    --n-jobs 20

# With bounding box filter
uv run water-timeseries-bp \
    --water-dataset-file /path/to/lakes.zarr \
    --output-file /path/to/breaks.parquet \
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
uv run water-timeseries-bp -C config.yaml

# Override specific config values from CLI
uv run water-timeseries-bp -C config.yaml --n-jobs 8
```

### CLI Options Reference

| Option | Short | Description | Default |
|--------|-------|-------------|--------|
| `--water-dataset-file` | | Path to water dataset (zarr) | Required |
| `--output-file` | | Path to output parquet | Required |
| `--vector-dataset-file` | `-v` | Path to vector dataset | None |
| `--config` | `-C` | Path to config file | None |
| `--chunksize` | `-c` | IDs per chunk | 100 |
| `--n-jobs` | `-j` | Parallel jobs | 1 |
| `--min-chunksize` | `-m` | Min chunk size | 10 |
| `--bbox-west` | | Min longitude | -180 |
| `--bbox-south` | | Min latitude | -90 |
| `--bbox-east` | | Max longitude | 180 |
| `--bbox-north` | | Max latitude | 90 |
