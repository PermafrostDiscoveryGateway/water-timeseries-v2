# Test Data Directory

This directory stores test datasets used by the test suite.

## Directory Structure

```
tests/data/
├── lakes_dw_test.nc      # Dynamic World test dataset (NetCDF)
├── lakes_dw_test.zarr/   # Dynamic World test dataset (Zarr)
├── lakes_jrc_test.nc     # JRC water test dataset (NetCDF)
├── lakes_jrc_test.zarr/  # JRC water test dataset (Zarr)
└── README.md             # This file
```

## Adding Test Data

Place your test datasets in this directory. The current test data is provided in multiple formats:

1. **Dynamic World Data** (`lakes_dw_test.*`)
   - Should contain variables: water, bare, snow_and_ice, trees, grass, flooded_vegetation, crops, shrub_and_scrub, built
   - Dimensions: time, id_geohash
   - Formats: NetCDF (.nc), Zarr (.zarr)

2. **JRC Water Data** (`lakes_jrc_test.*`)
   - Should contain variables: area_water_permanent, area_water_seasonal, area_land
   - Dimensions: time, id_geohash
   - Formats: NetCDF (.nc), Zarr (.zarr)

## Loading Test Data in Tests

To use the test data files in your tests:

```python
import os
from pathlib import Path

# Get path to test data directory
test_data_dir = Path(__file__).parent / "data"

# Load with xarray - choose your preferred format
import xarray as xr

# NetCDF format (most compatible)
dw_file_nc = test_data_dir / "lakes_dw_test.nc"
ds_dw_nc = xr.open_dataset(dw_file_nc)

# Zarr format (cloud-optimized)
dw_file_zarr = test_data_dir / "lakes_dw_test.zarr"
ds_dw_zarr = xr.open_zarr(dw_file_zarr)
```

Or use a pytest fixture:

```python
@pytest.fixture
def dw_test_data():
    test_data_dir = Path(__file__).parent / "data"
    return xr.open_dataset(test_data_dir / "lakes_dw_test.nc")

@pytest.fixture
def jrc_test_data():
    test_data_dir = Path(__file__).parent / "data"
    return xr.open_dataset(test_data_dir / "lakes_jrc_test.nc")
```

## Notes

- Keep test data files small for fast test execution
- Use `.gitignore` if you don't want to commit large files
- Consider generating synthetic data dynamically in conftest.py (as currently done)
