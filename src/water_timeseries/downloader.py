"""
Google Earth Engine Downloader

This module provides a Downloader class to download data from Google Earth Engine
into Dynamic World or JRC format.
"""

import os
from pathlib import Path
from typing import List, Optional

import ee
import eemont  # noqa: F401
import geemap
import geopandas as gpd
import joblib
import xarray as xr
from loguru import logger
from tqdm import tqdm

from water_timeseries.utils.data import load_vector_dataset
from water_timeseries.utils.earthengine import calc_monthly_dw, create_dw_classes_mask, drop_z_from_gdf
from water_timeseries.utils.spatial import filter_gdf_by_bbox
import pandas as pd


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
        logger: Optional logger instance for logging operations.
    """

    def __init__(
        self,
        ee_project: Optional[str] = None,
        output_dir: Optional[str] = None,
        ee_auth: bool = True,
        logger: Optional[logger] = None,
    ):
        """
        Initialize the Earth Engine Downloader.

        Args:
            ee_project: Google Earth Engine project ID. If None, will check
                the EE_PROJECT environment variable.
            output_dir: Output directory for downloaded data (optional).
            ee_auth: Whether to authenticate with Earth Engine (default: True).
            logger: Optional logger instance. If provided, will be used for logging.
                If None, print statements will be used as fallback.

        Raises:
            ValueError: If ee_project is empty or invalid.
        """
        self.logger = logger

        # Use _check_ee_product_name_setup to handle project ID resolution
        self.ee_project = self._check_ee_product_name_setup(ee_project)

        # Log initialization
        self._log_info(f"Initializing EarthEngineDownloader with project: {self.ee_project}")

        # Check and record EE initialization status
        self._check_ee_initialization_status()

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
            self._check_ee_initialization_status()

        self._log_info(f"EarthEngineDownloader initialized successfully. Output directory: {self.output_dir}")

    def _log_info(self, message: str):
        """Log an info message using the provided logger or print."""
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def _log_warning(self, message: str):
        """Log a warning message using the provided logger or print."""
        if self.logger is not None:
            self.logger.warning(message)
        else:
            print(f"Warning: {message}")

    def _log_error(self, message: str):
        """Log an error message using the provided logger or print."""
        if self.logger is not None:
            self.logger.error(message)
        else:
            print(f"Error: {message}")

    def _ensure_output_dir(self) -> Path:
        """Create output directory if it doesn't exist.

        Returns:
            Path: The output directory path.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def _check_ee_product_name_setup(self, ee_project: Optional[str] = None) -> str:
        """Check and return the Earth Engine project ID.

        Checks the ee_project parameter first, and if None, falls back to
        checking the EE_PROJECT environment variable.

        Args:
            ee_project: The ee_project parameter to check.

        Returns:
            str: The Earth Engine project ID.

        Raises:
            ValueError: If neither ee_project nor EE_PROJECT env var is set, or
                if the project ID is an empty string.
        """
        # First check if ee_project parameter is set and non-empty
        if ee_project is not None and isinstance(ee_project, str):
            if ee_project.strip() == "":
                raise ValueError("ee_project must be provided or set as EE_PROJECT environment variable")
            return ee_project

        # If ee_project is None, check environment variable
        ee_project_env = os.getenv("EE_PROJECT")
        if ee_project_env is not None and isinstance(ee_project_env, str):
            if ee_project_env.strip() == "":
                raise ValueError("ee_project must be provided or set as EE_PROJECT environment variable")
            return ee_project_env

        # If neither is set, raise an error
        raise ValueError("ee_project must be provided or set as EE_PROJECT environment variable")

    def _check_ee_initialization_status(self):
        """Check and record if Earth Engine is properly initialized.

        Returns:
            bool: True if EE is initialized, False otherwise.
        """
        try:
            # Check if ee.data is accessible (indicates successful initialization)
            self.ee_is_initialized = ee.data is not None and hasattr(ee.data, "getInfo")
            if not self.ee_is_initialized:
                self._log_warning("Earth Engine may not be properly initialized")
        except Exception:
            self.ee_is_initialized = False
            self._log_warning("Earth Engine initialization check failed")
        return self.ee_is_initialized

    def _setup_gee_reducer(self, gdf, feature_index_name: str, scale: float = 10) -> tuple:
        """Setup Google Earth Engine reducer configuration.

        Args:
            gdf: GeoDataFrame with features to process.
            feature_index_name: Column name to use as the feature index.
            scale: Pixel scale in meters (default: 10).

        Returns:
            tuple: (fc, reducer_dict) - GEE FeatureCollection and reducer configuration.
        """
        fc = geemap.gdf_to_ee(drop_z_from_gdf(gdf[:]))

        reducer = ee.Reducer.sum()
        CRS = "EPSG:3572"  # Coordinate reference system
        reducer_dict = {
            "reducer": reducer,
            "collection": fc.select(feature_index_name),
            "crs": CRS,
            "scale": scale,
            "bands": self.dw_bandnames,
        }
        return fc, reducer_dict

    def _chunk_gdf(self, gdf, max_total_requests: int, n_dates: int = 1) -> list:
        """Split a GeoDataFrame into chunks based on max_total_requests.

        Chunks the GeoDataFrame into smaller subsets to manage API request limits.
        Each chunk will contain approximately max_total_requests / n_dates features.

        Args:
            gdf: GeoDataFrame to chunk.
            max_total_requests: Maximum number of total requests per chunk (features * dates).
            n_dates: Number of dates/time periods being processed.

        Returns:
            List of GeoDataFrames, each representing a chunk.
        """
        n_features = len(gdf)

        # Calculate chunk size based on max_total_requests and number of dates
        # Each chunk should have max_total_requests / n_dates features
        features_per_chunk = max_total_requests // n_dates if n_dates > 0 else max_total_requests
        chunk_size = max(1, min(n_features, features_per_chunk))

        self._log_info(
            f"Chunking: n_features={n_features}, max_total_requests={max_total_requests}, n_dates={n_dates}, features_per_chunk={chunk_size}"
        )

        chunks = []
        for i in range(0, n_features, chunk_size):
            chunk = gdf.iloc[i : i + chunk_size]
            chunks.append(chunk)

        self._log_info(f"Split {n_features} features into {len(chunks)} chunks (chunk_size={chunk_size})")
        return chunks

    def _apply_id_filter(self, gdf, id_list: Optional[List], name_attribute: str) -> gpd.GeoDataFrame:
        """Apply ID filter to the GeoDataFrame.
        
        Args:
            gdf: GeoDataFrame to filter.
            id_list: Optional list of IDs to filter by.
            name_attribute: Column name to use for filtering.
            
        Returns:
            Filtered GeoDataFrame.
        """
        if id_list is None or len(id_list) == 0:
            self._log_info("No ID filtering applied")
            return gdf
            
        self._log_info(f"Applying ID filter: {len(id_list)} IDs specified")

        # Check which IDs are available in the dataset
        available_ids = set(gdf[name_attribute].values)
        requested_ids = set(id_list)
        found_ids = available_ids.intersection(requested_ids)
        missing_ids = requested_ids - found_ids

        if len(missing_ids) > 0:
            if len(found_ids) == 0:
                # None of the requested IDs found
                raise ValueError(
                    f"None of the {len(requested_ids)} requested IDs found in the dataset. "
                    f"Missing IDs: {list(missing_ids)[:10]}{'...' if len(missing_ids) > 10 else ''}"
                )
            else:
                # Some IDs found, some missing - log warning
                self._log_warning(
                    f"Only {len(found_ids)} of {len(requested_ids)} requested IDs found in dataset. "
                    f"Missing IDs ({len(missing_ids)}): {list(missing_ids)[:10]}{'...' if len(missing_ids) > 10 else ''}"
                )

        gdf_filtered = gdf[gdf[name_attribute].isin(id_list)]
        n_after_id_filter = len(gdf_filtered)
        self._log_info(f"After ID filter: {n_after_id_filter} features")
        return gdf_filtered

    def _extract_time_series(self, imlist: list, gdf_chunk, name_attribute: str, scale: float = 10) -> pd.DataFrame:
        """Extract time series data for a single chunk.

        Args:
            imlist: List of EE images to process.
            gdf_chunk: GeoDataFrame chunk to extract data for.
            name_attribute: Column name to use as the feature index.
            scale: Pixel scale in meters.

        Returns:
            pd.DataFrame: DataFrame with land cover areas for this chunk, indexed by name_attribute and date.
        """

        fc, reducer_dict = self._setup_gee_reducer(gdf_chunk, name_attribute, scale=scale)

        # Create an ImageCollection from the processed images
        ic_classes = ee.ImageCollection(imlist)

        # Extract time series data by regions
        fc_out = ic_classes.getTimeSeriesByRegions(**reducer_dict)

        # Convert FeatureCollection to pandas DataFrame
        df_out = geemap.ee_to_df(fc_out)

        # Return DataFrame with multi-index
        return df_out  # .set_index([name_attribute, "date"])

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
        id_list: Optional[List] = None,
        scale: float = 10,
        max_total_requests: int = 500,
        n_parallel: int = 1,
        no_download: bool = False,
    ) -> xr.Dataset:
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
            id_list: Optional list of IDs to filter by (values from name_attribute column).
                If provided, only features matching these IDs will be processed.
                Default is None (no ID filtering).
            no_download: If True, only log the download parameters without actually 
                downloading data (default: False).

        Returns:
            xr.Dataset: Xarray dataset with land cover areas indexed by name
                attribute and date.

        Raises:
            KeyError: If the specified name_attribute column is not found in the
                vector dataset.
        """
        # Announce no_download mode at the top
        if no_download:
            self._log_info("=== NO DOWNLOAD MODE - Will skip after preprocessing ===")

        # Read vector data using the reusable function
        gdf = load_vector_dataset(vector_dataset, logger=self.logger)
        if name_attribute not in gdf.columns:
            raise KeyError(f"The designated column '{name_attribute}' is not present in the vector dataset.")

        # Log initial number of features
        n_features_initial = len(gdf)
        self._log_info(f"Initial dataset has {n_features_initial} features")

        # Apply ID filter if id_list is provided
        gdf = self._apply_id_filter(gdf, id_list, name_attribute)

        # Apply spatial bbox filter if any bbox parameter is provided and differs from defaults
        if any(v is not None for v in [bbox_west, bbox_south, bbox_east, bbox_north]) and not (
            bbox_west == -180 and bbox_east == 180 and bbox_north == 90 and bbox_south == -90
        ):
            n_before_bbox_filter = len(gdf)
            self._log_info(
                f"Applying bbox filter: west={bbox_west}, south={bbox_south}, east={bbox_east}, north={bbox_north}"
            )
            gdf = filter_gdf_by_bbox(
                gdf,
                bbox_west=bbox_west,
                bbox_south=bbox_south,
                bbox_east=bbox_east,
                bbox_north=bbox_north,
            )
            n_features_filtered = len(gdf)
            self._log_info(
                f"After bbox filter: {n_features_filtered} features (removed {n_before_bbox_filter - n_features_filtered})"
            )
        else:
            self._log_info("No spatial bbox filtering applied (using default global bounds)")

        n_features = len(gdf)
        self._log_info(f"Processing {n_features} features")

        # Generate date range based on years and months
        dates = setup_monthly_dates(years=years, months=months)
        n_dates = len(dates)
        n_total_requests = n_features * n_dates
        self._log_info(f"Processing {n_features} features x {n_dates} dates = {n_total_requests} total requests")
        self._log_info(f"Processing dates: {dates}")

        # Setup GEE processing
        fc, reducer_dict = self._setup_gee_reducer(gdf, name_attribute, scale=scale)
        imlist = []
        self._log_info("Start downloading process")
        # Iterate through each date and process monthly land cover data
        for date in dates:
            try:
                # Calculate monthly Dynamic World land cover for the date
                im = calc_monthly_dw(start_date=date, polygons=fc)
                # assert isinstance(im, ee.Image)
                if im is None:
                    self._log_warning(f"No data for date: {date}")
                    continue
                # Create masks for land cover classes
                im_classes = create_dw_classes_mask(ee.Image(im))
                imlist.append(im_classes)
            except Exception:
                # Skip dates with errors
                continue
        self._log_info(f"Total images collected: {len(imlist)}")

        # Chunk the GeoDataFrame into smaller pieces based on number of dates
        n_dates = len(imlist)
        gdf_chunks = self._chunk_gdf(gdf, max_total_requests, n_dates=n_dates)
        
        # Return early if no_download is True - skip downloading but show summary
        if no_download:
            self._log_info(f"Would process {len(gdf)} features with {len(gdf_chunks)} chunks")
            self._log_info("=== END NO DOWNLOAD MODE ===")
            return None
        
        # Ensure n_parallel is not greater than number of chunks
        n_parallel_effective = min(n_parallel, len(gdf_chunks))
        
        # Process chunks with joblib (sequential if n_parallel=1, parallel otherwise)
        self._log_info(f"Processing {len(gdf_chunks)} chunks with {n_parallel_effective} workers")
        
        # Use tqdm for progress bar
        df_out_list = joblib.Parallel(n_jobs=n_parallel_effective, prefer="threads")(
            joblib.delayed(self._extract_time_series)(
                imlist=imlist, 
                gdf_chunk=chunk, 
                name_attribute=name_attribute,
                scale=scale
            )
            for chunk in tqdm(gdf_chunks, desc="Downloading chunks", unit="chunk")
        )
        # Filter out None/empty results
        df_out_list = [df for df in df_out_list if df is not None and not df.empty]

        # Combine all chunks
        if not df_out_list:
            raise ValueError("No data was extracted from any chunk. Check GEE request parameters.")

        df_out = pd.concat(df_out_list)
        ds = df_out.set_index([name_attribute, "date"]).to_xarray()

        # Log summary statistics using the dataset coordinates (indexes)
        n_items = len(ds.coords[name_attribute])
        n_dates = len(ds.coords["date"])
        self._log_info(f"Download complete: {n_items} items, {n_dates} dates collected")

        return ds.drop_vars("reducer")


# Example usage
if __name__ == "__main__":
    # Create downloader instance with ee-project flag
    downloader = EarthEngineDownloader(ee_project="your-ee-project-id")
