"""
Google Earth Engine Downloader

This module provides a Downloader class to download data from Google Earth Engine
into Dynamic World or JRC format.
"""

from pathlib import Path
from typing import List, Optional

import ee
import eemont  # noqa: F401
import geemap
import geopandas as gpd
import pandas as pd

import os

from water_timeseries.utils.earthengine import calc_monthly_dw, create_dw_classes_mask, drop_z_from_gdf


def setup_monthly_dates(years: List[int], months: List[int]) -> List[str]:
    """Generate a list of monthly dates from the given years and months.

    Creates dates starting from the first day of each specified month.

    Args:
        years: List of years (e.g., [2017, 2018]).
        months: List of months as integers (1-12).

    Returns:
        List[str]: Formatted dates in 'YYYY-MM-DD' format (e.g., '2017-06-01').

    Example:
        >>> setup_monthly_dates([2023], [1, 2])
        ['2023-01-01', '2023-02-01']
    """
    dates = []
    for year in years:
        for month in months:
            dates.append(f"{year}-{month:02d}-01")
    return dates


class EarthEngineDownloader:
    """
    A class to download data from Google Earth Engine into various formats.

    Attributes:
        ee_project: The Google Earth Engine project identifier.
        output_dir: Directory to save downloaded data.
        ee_auth: Whether to authenticate with Earth Engine.
    """

    def __init__(self, ee_project: Optional[str] = None, output_dir: Optional[str] = None, ee_auth: bool = False):
        """
        Initialize the Earth Engine Downloader.

        Args:
            ee_project: Google Earth Engine project ID. If None, will check
                the EE_PROJECT environment variable.
            output_dir: Output directory for downloaded data (optional).
            ee_auth: Whether to authenticate with Earth Engine (default: False).

        Raises:
            ValueError: If ee_project is empty or invalid.
        """
        # Use _check_ee_initialization to handle project ID resolution
        self.ee_project = ee_project
        self.ee_project = self._check_ee_initialization()
        self.output_dir = Path(output_dir) if output_dir else Path("downloads")
        self.ee_auth = ee_auth
        self.dw_bandnames = [
            "water",
            "trees",
            "grass",
            "flooded_vegetation",
            "crops",
            "shrub_and_scrub",
            "built",
            "bare",
            "snow_and_ice",
        ]

        # Initialize Earth Engine
        if ee_auth:
            geemap.ee_initialize(project=self.ee_project)

    def _ensure_output_dir(self) -> Path:
        """Create output directory if it doesn't exist.

        Returns:
            Path: The output directory path.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def _check_ee_initialization(self) -> str:
        """Check and return the Earth Engine project ID.

        Checks the ee_project attribute first, and if None, falls back to
        checking the EE_PROJECT environment variable.

        Returns:
            str: The Earth Engine project ID.

        Raises:
            ValueError: If neither ee_project nor EE_PROJECT env var is set, or
                if the project ID is an empty string.
        """
        # First check if ee_project is set and non-empty
        if self.ee_project is not None and isinstance(self.ee_project, str):
            if self.ee_project.strip() == "":
                raise ValueError(
                    "ee_project must be provided or set as EE_PROJECT environment variable"
                )
            return self.ee_project

        # If ee_project is None, check environment variable
        ee_project_env = os.getenv("EE_PROJECT")
        if ee_project_env is not None and isinstance(ee_project_env, str):
            if ee_project_env.strip() == "":
                raise ValueError(
                    "ee_project must be provided or set as EE_PROJECT environment variable"
                )
            return ee_project_env

        # If neither is set, raise an error
        raise ValueError(
            "ee_project must be provided or set as EE_PROJECT environment variable"
        )

    def download_dw_monthly(
        self,
        vector_dataset: str | Path,
        name_attribute: str,
        years: List[int] = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        months: List[int] = [6, 7, 8, 9],
        bbox_west: float = -180,
        bbox_east: float = 180,
        bbox_north: float = 90,
        bbox_south: float = -90,
    ) -> pd.DataFrame:
        """Download monthly Dynamic World land cover data for specified periods.

        Extracts land cover class areas from Google Earth Engine for each
        polygon in the vector dataset, grouped by the specified date periods.

        Args:
            vector_dataset: Path to the input vector dataset (Parquet format).
            name_attribute: Column name in the vector dataset to use for grouping.
            years: List of years to process (default: 2017-2025).
            months: List of months to process as integers (default: June-September).
            bbox_west: Western boundary for spatial filtering (default: -180).
            bbox_east: Eastern boundary for spatial filtering (default: 180).
            bbox_north: Northern boundary for spatial filtering (default: 90).
            bbox_south: Southern boundary for spatial filtering (default: -90).

        Returns:
            xr.Dataset: Xarray dataset with land cover areas indexed by name
                attribute and date.

        Raises:
            KeyError: If the specified name_attribute column is not found in the
                vector dataset.
        """
        # Read vector data from parquet file
        gdf = gpd.read_parquet(vector_dataset)
        if name_attribute not in gdf.columns:
            raise KeyError(f"The designated column '{name_attribute}' is not present in the vector dataset.")

        fc = geemap.gdf_to_ee(drop_z_from_gdf(gdf[:]))

        # Configuration for reduction operation
        feature_index_name = name_attribute
        reducer = ee.Reducer.sum()
        CRS = "EPSG:3572"  # Coordinate reference system
        SCALE = 10  # Pixel scale in meters
        reducer_dict = {
            "reducer": reducer,
            "collection": fc.select(feature_index_name),
            "crs": CRS,
            "scale": SCALE,
            "bands": self.dw_bandnames,
        }
        # Generate date range based on years and months
        dates = setup_monthly_dates(years=years, months=months)
        imlist = []

        print("Processing")
        # Iterate through each date and process monthly land cover data
        for date in dates:
            print(date)
            try:
                # Calculate monthly Dynamic World land cover for the date
                im = calc_monthly_dw(start_date=date, polygons=fc)
                # assert isinstance(im, ee.Image)
                if im is None:
                    print(f"No data for date: {date}")
                    continue
                # Create masks for land cover classes
                im_classes = create_dw_classes_mask(ee.Image(im))
                imlist.append(im_classes)
            except Exception:
                # Skip dates with errors
                continue

        print(len(imlist))
        # Create an ImageCollection from the processed images
        ic_classes = ee.ImageCollection(imlist)

        # Extract time series data by regions
        fc_out = ic_classes.getTimeSeriesByRegions(**reducer_dict)

        # Convert FeatureCollection to pandas DataFrame
        df_out = geemap.ee_to_df(fc_out)

        # Convert to xarray dataset with proper indexing
        ds = df_out.set_index([feature_index_name, "date"]).to_xarray()
        # df_final = calculate_data_area(ds).to_dataframe().reset_index()
        # except:
        #     print("crashed")

        return ds.drop_vars('reducer')


# Example usage
if __name__ == "__main__":
    # Create downloader instance with ee-project flag
    downloader = EarthEngineDownloader(ee_project="your-ee-project-id")
