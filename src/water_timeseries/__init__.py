"""Water Timeseries Analysis Package.

This package provides tools for analyzing water-related time series data from satellite
imagery and other sources. It includes utilities for processing Dynamic World and JRC
land cover classifications with specialized handling for water-related features.

Main Components:
    - LakeDataset: Base class for processing lake and water body datasets
    - DWDataset: Specialized handler for Dynamic World land cover data
    - JRCDataset: Specialized handler for JRC water classification data
    - BreakpointMethod: Framework for detecting changes in water extent over time

Example:
    >>> from water_timeseries.dataset import DWDataset
    >>> import xarray as xr
    >>> 
    >>> ds = xr.open_dataset("land_cover_data.nc")
    >>> processor = DWDataset(ds)
    >>> 
    >>> # Access normalized water extent time series
    >>> water_data = processor.ds_normalized["water"]
"""

__version__ = "0.1.0"
__author__ = "Ingmar Nitze"


def main() -> None:
    """Main entry point for the water-timeseries package."""
    print("Hello from water-timeseries!")
