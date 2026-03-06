"""Test fixtures for water-timeseries tests.

This module provides pytest fixtures that load real test datasets
from the tests/data directory.
"""

from pathlib import Path

import pytest
import xarray as xr


@pytest.fixture
def dw_test_dataset():
    """Load the real Dynamic World test dataset.

    Returns the actual DW dataset from tests/data/lakes_dw_test.nc
    """
    test_data_dir = Path(__file__).parent / "data"
    dw_file = test_data_dir / "lakes_dw_test.nc"
    return xr.open_dataset(dw_file)


@pytest.fixture
def jrc_test_dataset():
    """Load the real JRC water test dataset.

    Returns the actual JRC dataset from tests/data/lakes_jrc_test.nc
    """
    test_data_dir = Path(__file__).parent / "data"
    jrc_file = test_data_dir / "lakes_jrc_test.nc"
    return xr.open_dataset(jrc_file)
