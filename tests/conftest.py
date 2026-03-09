"""Test fixtures for water-timeseries tests.

This module provides pytest fixtures that load real test datasets
from the tests/data directory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
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


@pytest.fixture
def synthetic_dw_dataset_with_break():
    """Create synthetic DW dataset with a known breakpoint pattern.

    Creates a dataset with:
    - High water values (0.8-0.9) for first half
    - Sharp drop to low values (0.1-0.2) for second half
    - This should trigger breakpoint detection
    """
    # Create time series
    dates = pd.date_range("2020-01-01", periods=20, freq="D")

    # Create geohash IDs
    geohashes = ["u4pru", "u4prv", "u4pry"]

    # Create data with breakpoint: high values then sharp drop
    n_dates = len(dates)
    water_high = np.random.uniform(0.8, 0.9, n_dates//2)
    water_low = np.random.uniform(0.1, 0.2, n_dates - n_dates//2)
    water_values = np.concatenate([water_high, water_low])

    # Create other land cover classes (simplified)
    bare = np.random.uniform(0.0, 0.1, n_dates)
    snow_ice = np.random.uniform(0.0, 0.05, n_dates)
    trees = np.random.uniform(0.0, 0.1, n_dates)
    grass = np.random.uniform(0.0, 0.1, n_dates)
    flooded_veg = np.random.uniform(0.0, 0.05, n_dates)
    crops = np.random.uniform(0.0, 0.1, n_dates)
    shrub = np.random.uniform(0.0, 0.1, n_dates)
    built = np.random.uniform(0.0, 0.05, n_dates)

    # Create dataset
    ds = xr.Dataset(
        {
            "water": (["date", "id_geohash"], [water_values] * len(geohashes)),
            "bare": (["date", "id_geohash"], [bare] * len(geohashes)),
            "snow_and_ice": (["date", "id_geohash"], [snow_ice] * len(geohashes)),
            "trees": (["date", "id_geohash"], [trees] * len(geohashes)),
            "grass": (["date", "id_geohash"], [grass] * len(geohashes)),
            "flooded_vegetation": (["date", "id_geohash"], [flooded_veg] * len(geohashes)),
            "crops": (["date", "id_geohash"], [crops] * len(geohashes)),
            "shrub_and_scrub": (["date", "id_geohash"], [shrub] * len(geohashes)),
            "built": (["date", "id_geohash"], [built] * len(geohashes)),
        },
        coords={
            "date": dates,
            "id_geohash": geohashes,
        }
    )

    return ds


@pytest.fixture
def synthetic_dw_dataset_no_break():
    """Create synthetic DW dataset with no breakpoint pattern.

    Creates a dataset with stable water values throughout the time series.
    """
    # Create time series
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    geohashes = ["u4pru", "u4prv"]

    # Create stable water values (no breakpoint)
    water_values = np.random.uniform(0.7, 0.8, len(dates))

    # Create other land cover classes (simplified)
    bare = np.random.uniform(0.0, 0.1, len(dates))
    snow_ice = np.random.uniform(0.0, 0.05, len(dates))
    trees = np.random.uniform(0.0, 0.1, len(dates))
    grass = np.random.uniform(0.0, 0.1, len(dates))
    flooded_veg = np.random.uniform(0.0, 0.05, len(dates))
    crops = np.random.uniform(0.0, 0.1, len(dates))
    shrub = np.random.uniform(0.0, 0.1, len(dates))
    built = np.random.uniform(0.0, 0.05, len(dates))

    # Create dataset
    ds = xr.Dataset(
        {
            "water": (["date", "id_geohash"], [water_values] * len(geohashes)),
            "bare": (["date", "id_geohash"], [bare] * len(geohashes)),
            "snow_and_ice": (["date", "id_geohash"], [snow_ice] * len(geohashes)),
            "trees": (["date", "id_geohash"], [trees] * len(geohashes)),
            "grass": (["date", "id_geohash"], [grass] * len(geohashes)),
            "flooded_vegetation": (["date", "id_geohash"], [flooded_veg] * len(geohashes)),
            "crops": (["date", "id_geohash"], [crops] * len(geohashes)),
            "shrub_and_scrub": (["date", "id_geohash"], [shrub] * len(geohashes)),
            "built": (["date", "id_geohash"], [built] * len(geohashes)),
        },
        coords={
            "date": dates,
            "id_geohash": geohashes,
        }
    )

    return ds


@pytest.fixture
def synthetic_jrc_dataset_with_break():
    """Create synthetic JRC dataset with a known breakpoint pattern.

    Creates a dataset with:
    - High permanent water values (0.8-0.9) for first half
    - Sharp drop to low values (0.1-0.2) for second half
    """
    # Create time series
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    geohashes = ["u4pru", "u4prv"]

    n_dates = len(dates)
    # Create breakpoint pattern
    perm_high = np.random.uniform(0.8, 0.9, n_dates//2)
    perm_low = np.random.uniform(0.1, 0.2, n_dates - n_dates//2)
    permanent_values = np.concatenate([perm_high, perm_low])

    seasonal_values = np.random.uniform(0.0, 0.1, n_dates)
    land_values = np.random.uniform(0.0, 0.2, n_dates)

    # Create dataset
    ds = xr.Dataset(
        {
            "area_water_permanent": (["date", "id_geohash"], [permanent_values] * len(geohashes)),
            "area_water_seasonal": (["date", "id_geohash"], [seasonal_values] * len(geohashes)),
            "area_land": (["date", "id_geohash"], [land_values] * len(geohashes)),
        },
        coords={
            "date": dates,
            "id_geohash": geohashes,
        }
    )

    return ds
