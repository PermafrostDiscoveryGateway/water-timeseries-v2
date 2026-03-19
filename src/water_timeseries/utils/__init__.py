"""Utility functions for water timeseries."""

from water_timeseries.utils.data import (
    calculate_temporal_stats,
    calculate_water_area_after,
    calculate_water_area_before,
    get_water_dataset_type,
)
from water_timeseries.utils.io import (
    load_vector_dataset,
    load_xarray_dataset,
    save_xarray_dataset,
)

__all__ = [
    "calculate_temporal_stats",
    "calculate_water_area_after",
    "calculate_water_area_before",
    "get_water_dataset_type",
    "load_vector_dataset",
    "load_xarray_dataset",
    "save_xarray_dataset",
]
