# Tests for water-timeseries

This directory contains the test suite for the water-timeseries package.

## Test Structure

- **conftest.py** - Pytest fixtures that create synthetic test datasets
  - `dw_test_dataset`: Small DW (Dynamic World) test dataset
  - `jrc_test_dataset`: Small JRC water classification test dataset

- **test_loading.py** - Tests for dataset loading and initialization
  - `TestDWDatasetLoading`: Tests for DWDataset initialization
  - `TestJRCDatasetLoading`: Tests for JRCDataset initialization

- **test_normalization.py** - Tests for data normalization
  - `TestDWDatasetNormalization`: Tests for DW data normalization
  - `TestJRCDatasetNormalization`: Tests for JRC data normalization

- **test_plotting.py** - Tests for plotting functionality
  - `TestDWDatasetPlotting`: Tests for DW time series plots
  - `TestJRCDatasetPlotting`: Tests for JRC time series plots

## Test Data

The test datasets are created as pytest fixtures in `conftest.py`:

- **DW Test Dataset**: 10 time steps × 2 geohashes with 9 land cover classes
- **JRC Test Dataset**: 10 time steps × 2 geohashes with permanent water, seasonal water, and land

## Running Tests

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_loading.py
```

Run specific test class:
```bash
pytest tests/test_loading.py::TestDWDatasetLoading
```

Run specific test:
```bash
pytest tests/test_loading.py::TestDWDatasetLoading::test_dw_dataset_initialization
```

Run with verbose output:
```bash
pytest -v
```

Run with coverage report:
```bash
pytest --cov=water_timeseries tests/
```

## Test Coverage

The test suite covers:

1. **Loading**: Ensures datasets are loaded and initialized correctly
2. **Normalization**: Verifies data is normalized to [0, 1] range
3. **Plotting**: Confirms plots are generated without errors

## Adding New Tests

When adding new tests:

1. Create test methods starting with `test_`
2. Use descriptive names that explain what is being tested
3. Add docstrings explaining the test purpose
4. Use appropriate fixtures from `conftest.py`
5. Clean up matplotlib figures with `plt.close()`

## Dependencies

Tests require:
- pytest
- pytest-cov (for coverage reports)
- All dependencies from pyproject.toml
