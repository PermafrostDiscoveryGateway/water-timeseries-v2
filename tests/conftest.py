"""Test fixtures for water-timeseries tests.

This module provides pytest fixtures that create small synthetic datasets
for testing DW and JRC dataset processors.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def dw_test_dataset():
    """Create a small test xarray Dataset for Dynamic World data.
    
    Returns a synthetic DW dataset with 10 time steps and 2 spatial locations.
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=10, freq="MS")
    geohashes = ["ezs42", "ezs43"]
    
    # Create realistic land cover proportions (sum to ~100)
    n_dates = len(dates)
    n_geo = len(geohashes)
    
    data_vars = {}
    
    # Water increases over time
    data_vars["water"] = (["date", "id_geohash"], np.random.rand(n_dates, n_geo) * 10 + np.arange(n_dates).reshape(-1, 1))
    
    # Other land cover classes
    data_vars["bare"] = (["date", "id_geohash"], np.random.rand(n_dates, n_geo) * 20)
    data_vars["snow_and_ice"] = (["date", "id_geohash"], np.random.rand(n_dates, n_geo) * 2)
    data_vars["trees"] = (["date", "id_geohash"], np.random.rand(n_dates, n_geo) * 25)
    data_vars["grass"] = (["date", "id_geohash"], np.random.rand(n_dates, n_geo) * 25)
    data_vars["flooded_vegetation"] = (["date", "id_geohash"], np.random.rand(n_dates, n_geo) * 5)
    data_vars["crops"] = (["date", "id_geohash"], np.random.rand(n_dates, n_geo) * 10)
    data_vars["shrub_and_scrub"] = (["date", "id_geohash"], np.random.rand(n_dates, n_geo) * 15)
    data_vars["built"] = (["date", "id_geohash"], np.random.rand(n_dates, n_geo) * 5)
    
    ds = xr.Dataset(
        data_vars,
        coords={
            "date": dates,
            "id_geohash": geohashes,
        }
    )
    
    return ds


@pytest.fixture
def jrc_test_dataset():
    """Create a small test xarray Dataset for JRC water data.
    
    Returns a synthetic JRC dataset with 10 time steps and 2 spatial locations.
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=10, freq="MS")
    geohashes = ["ezs42", "ezs43"]
    
    n_dates = len(dates)
    n_geo = len(geohashes)
    
    # JRC has permanent water, seasonal water, and land
    data_vars = {}
    
    # Permanent water increases slightly
    data_vars["area_water_permanent"] = (
        ["date", "id_geohash"],
        np.random.rand(n_dates, n_geo) * 5 + np.arange(n_dates).reshape(-1, 1) * 0.5,
    )
    
    # Seasonal water varies
    data_vars["area_water_seasonal"] = (
        ["date", "id_geohash"],
        np.random.rand(n_dates, n_geo) * 8,
    )
    
    # Land area
    data_vars["area_land"] = (
        ["date", "id_geohash"],
        np.ones((n_dates, n_geo)) * 87,  # Most of the area is land
    )
    
    ds = xr.Dataset(
        data_vars,
        coords={
            "date": dates,
            "id_geohash": geohashes,
        }
    )
    
    return ds
