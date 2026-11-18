"""
Google Earth Engine Downloader

This module provides a Downloader class to download data from Google Earth Engine
into Dynamic World or JRC format.
"""

import os
from pathlib import Path

import ee
import eemont  # noqa: F401
import geemap
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from water_timeseries.utils.data import annotate_xr_dataset_dw, annotate_xr_dataset_jrc, dw_bandnames, jrc_bandnames
from water_timeseries.utils.earthengine import (
    NoDynamicWorldDataError,
    calc_monthly_dw,
    call_with_retry,
    create_dw_classes_mask,
    drop_z_from_gdf,
    dw_acquisitions,
    month_slices,
    month_windows,
    season_windows,
    setup_annual_dates,
    setup_dates_from_options,
    setup_monthly_dates,
    windows_from_date_list,
)
from water_timeseries.utils.io import load_vector_dataset, save_xarray_dataset
from water_timeseries.utils.spatial import chunk_gdf_simple, chunk_gdf_spatial_kmeans, filter_gdf_by_bbox

# Typical number of Dynamic World acquisitions per month in an Arctic May-Sep season
# (measured: 42 acquisitions over the 2022 season on Alaska test lakes). Used only to size
# download chunks and to log the projected volume before the first Earth Engine call.
EST_ACQUISITIONS_PER_MONTH = 7

# Re-export for backward compatibility
__all__ = [
    "EarthEngineDownloader",
    "NoDynamicWorldDataError",
    "setup_annual_dates",
    "setup_dates_from_options",
    "setup_monthly_dates",
]


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
        ee_project: str | None = None,
        output_dir: str | None = None,
        ee_auth: bool = True,
        logger=None,
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

        if output_dir is not None:
            self._log_warning("Setting an output_dir is being ignored now and will be removed eventually!")
        self.output_dir = None
        self.ee_auth = ee_auth
        self.dw_bandnames = dw_bandnames
        self.jrc_bandnames = jrc_bandnames

        # Initialize Earth Engine
        if ee_auth:
            from water_timeseries.utils.earthengine import initialize_earth_engine

            initialize_earth_engine(project=self.ee_project)
            self._check_ee_initialization_status()

        self._log_info("EarthEngineDownloader initialized successfully.")

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

    def _check_ee_product_name_setup(self, ee_project: str | None = None) -> str:
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
        except AttributeError:
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

    def _setup_jrc_reducer(self, gdf, feature_index_name: str, scale: float = 30) -> tuple:
        """Setup Google Earth Engine reducer configuration for JRC data.

        Args:
            gdf: GeoDataFrame with features to process.
            feature_index_name: Column name to use as the feature index.
            scale: Pixel scale in meters (default: 30 for JRC).

        Returns:
            tuple: (fc, reducer_dict) - GEE FeatureCollection and reducer configuration.
        """
        fc = geemap.gdf_to_ee(drop_z_from_gdf(gdf[:]))

        reducer = ee.Reducer.sum()
        CRS = "EPSG:4326"  # Use WGS84 (matching original JRC code)
        reducer_dict = {
            "reducer": reducer,
            "collection": fc.select(feature_index_name),
            "crs": CRS,
            "scale": scale,
            "bands": self.jrc_bandnames,
        }
        return fc, reducer_dict

    def _chunk_gdf(self, gdf, max_total_requests: int, n_dates: int = 1, chunk_method: str = "spatial_kmeans") -> list:
        """Split a GeoDataFrame into chunks based on max_total_requests.

        Chunks the GeoDataFrame into smaller subsets to manage API request limits.
        Each chunk will contain approximately max_total_requests / n_dates features.

        Args:
            gdf: GeoDataFrame to chunk.
            max_total_requests: Maximum number of total requests per chunk (features * dates).
            n_dates: Number of dates/time periods being processed.
            chunk_method: Method to use for chunking ("spatial_kmeans" or "simple").

        Returns:
            List of GeoDataFrames, each representing a chunk.
        """
        if chunk_method not in ["spatial_kmeans", "simple"]:
            raise ValueError("Invalid chunk_method. Must be 'spatial_kmeans' or 'simple'.")

        n_features = len(gdf)

        # Calculate chunk size based on max_total_requests and number of dates
        # Each chunk should have max_total_requests / n_dates features
        features_per_chunk = max_total_requests // n_dates if n_dates > 0 else max_total_requests
        chunk_size = max(1, min(n_features, features_per_chunk))

        # spatial chunking
        if chunk_method == "spatial_kmeans":
            self._log_info(
                f"Chunking with spatial KMeans: n_features={n_features}, max_total_requests={max_total_requests}, n_dates={n_dates}, features_per_chunk={chunk_size}"
            )
            chunks = chunk_gdf_spatial_kmeans(gdf=gdf, chunk_size=chunk_size, epsg=3995)
            self._log_info(
                f"Dataset was split into {len(chunks)} spatially coherent chunks using KMeans clustering (chunk_size={chunk_size})"
            )
        else:
            self._log_info(
                f"Chunking: n_features={n_features}, max_total_requests={max_total_requests}, n_dates={n_dates}, features_per_chunk={chunk_size}"
            )
            chunks = chunk_gdf_simple(gdf=gdf, chunk_size=chunk_size)
            self._log_info(
                f"Dataset was split into {len(chunks)} spatially coherent chunks using simple chunking (chunk_size={chunk_size})"
            )

        self._log_info(f"Split {n_features} features into {len(chunks)} chunks (chunk_size={chunk_size})")
        return chunks

    def _apply_id_filter(self, gdf, id_list: list | None, name_attribute: str) -> gpd.GeoDataFrame:
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

    def _extract_time_series(self, dates: list[str], gdf_chunk, name_attribute: str, scale: float = 10) -> pd.DataFrame:
        """Extract time series data for a single chunk.

        Processes each date within this method to avoid memory issues with large imlists.

        Args:
            dates: List of dates to process.
            gdf_chunk: GeoDataFrame chunk to extract data for.
            name_attribute: Column name to use as the feature index.
            scale: Pixel scale in meters.

        Returns:
            pd.DataFrame: DataFrame with land cover areas for this chunk, indexed by name_attribute and date.
        """

        fc, reducer_dict = self._setup_gee_reducer(gdf_chunk, name_attribute, scale=scale)

        # Process each date inside this method to avoid large imlist in memory
        imlist = []
        for date in dates:
            # Calculate monthly Dynamic World land cover for the date
            im = calc_monthly_dw(start_date=date, polygons=fc)
            if im is None:
                self._log_warning(f"No data for date: {date}")
                continue
            # Create masks for land cover classes
            im_classes = create_dw_classes_mask(ee.Image(im))
            imlist.append(im_classes)

        if not imlist:
            self._log_warning("No images processed for chunk")
            return pd.DataFrame()

        # Create an ImageCollection from the processed images
        ic_classes = ee.ImageCollection(imlist)

        # Extract time series data by regions
        fc_out = ic_classes.getTimeSeriesByRegions(**reducer_dict)

        # Convert FeatureCollection to pandas DataFrame (this is where the computation runs)
        df_out = call_with_retry(geemap.ee_to_df, fc_out)

        # Return DataFrame with multi-index
        return df_out  # .set_index([name_attribute, "date"])

    def _polygon_areas_ha(self, gdf, name_attribute: str = "id_geohash") -> dict:
        """Geodesic polygon areas in hectares, keyed by ``name_attribute``.

        EPSG:6933 is an equal-area projection; measuring in a conformal or stereographic
        projection instead (e.g. EPSG:3995) inflates area by roughly 3 % at Arctic
        latitudes, which would make any small max_nodata tolerance reject every observation.

        Args:
            gdf: GeoDataFrame with one row per polygon.
            name_attribute: Column holding the polygon identifier.

        Returns:
            dict: ``{id: area_ha}``.

        Raises:
            ValueError: If the GeoDataFrame has no CRS, since polygon area cannot be measured
                without one (coverage is relative to it).
        """
        if gdf.crs is None:
            raise ValueError(
                "The GeoDataFrame has no CRS, so its polygon area cannot be measured. Set one "
                "(e.g. gdf = gdf.set_crs(4326)) - per-acquisition coverage is computed relative to "
                "the polygon area, which the monthly downloader does not need."
            )
        areas_ha = gdf.to_crs(6933).geometry.area / 1e4
        if areas_ha.isna().any():
            self._log_warning(f"{int(areas_ha.isna().sum())} polygon(s) have no valid geometry; their coverage is undefined")
        return dict(zip(gdf[name_attribute].values, areas_ha.values, strict=True))

    def _extract_acquisitions(
        self,
        windows: list[tuple[str, str]],
        gdf_chunk,
        name_attribute: str,
        scale: float = 10,
    ) -> pd.DataFrame:
        """Extract per-acquisition class areas for one chunk over several date windows.

        Processes one calendar month per Earth Engine request, so no single request carries
        a whole season of images.

        Args:
            windows: List of ``(start, end)`` date-range pairs; the end is exclusive.
            gdf_chunk: GeoDataFrame chunk to extract data for.
            name_attribute: Column name used as the feature index.
            scale: Pixel scale in meters.

        Returns:
            pd.DataFrame: One row per (polygon, acquisition); empty if nothing was found.
        """
        fc, reducer_dict = self._setup_gee_reducer(gdf_chunk, name_attribute, scale=scale)

        frames = []
        for window_start, window_end in windows:
            for slice_start, slice_end in month_slices(window_start, window_end):
                collection = dw_acquisitions(fc, slice_start, slice_end, scale=scale)
                n_images = call_with_retry(collection.size().getInfo)
                if n_images == 0:
                    self._log_info(f"No Dynamic World acquisitions between {slice_start} and {slice_end}")
                    continue
                self._log_info(f"{slice_start}..{slice_end}: {n_images} acquisitions")

                image_collection = collection.map(create_dw_classes_mask)
                fc_out = image_collection.getTimeSeriesByRegions(**reducer_dict)
                df = call_with_retry(geemap.ee_to_df, fc_out)
                if df is None or df.empty:
                    continue
                frames.append(df)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _monthly_median_frame(
        self,
        dates: list[str],
        gdf_chunks: list,
        name_attribute: str,
        scale: float,
        n_parallel: int,
        snow_free: bool,
        max_snow_fraction: float,
    ) -> pd.DataFrame:
        """Aggregate one median row per (polygon, month) from per-acquisition class areas.

        With ``snow_free=False`` every acquisition with at least one classified pixel
        contributes. With ``snow_free=True`` a scene is discarded for a polygon when
        ``snow_and_ice / area_data > max_snow_fraction``, i.e. the lake surface in that scene
        is not (nearly) snow-free; the test is per polygon, so each lake keeps only its own
        snow-free acquisitions. The survivors are combined by median, which stays meaningful
        in a transition month where only one to three scenes qualify, and a month with no
        usable acquisition is absent rather than reported as zero.

        Args:
            dates: Month-start dates to aggregate (``'YYYY-MM-01'``).
            gdf_chunks: Polygons already split into chunks.
            name_attribute: Column name used as the polygon identifier.
            scale: Pixel scale in meters.
            n_parallel: Number of chunk workers.
            snow_free: Whether to keep only snow-free acquisitions.
            max_snow_fraction: Allowed snow/ice share of the polygon for a scene to count as
                snow-free; ignored when ``snow_free`` is False.

        Returns:
            pd.DataFrame: One row per ``(name_attribute, date)`` with the nine class areas,
                ``date`` being the month start.

        Raises:
            NoDynamicWorldDataError: If Earth Engine returned no acquisitions, or if the snow
                filter left nothing to aggregate.
        """
        windows = [
            (
                pd.Timestamp(date).strftime("%Y-%m-%d"),
                (pd.Timestamp(date) + pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
            )
            for date in dates
        ]

        frames = joblib.Parallel(n_jobs=n_parallel, prefer="threads")(
            joblib.delayed(self._extract_acquisitions)(
                windows=windows, gdf_chunk=chunk, name_attribute=name_attribute, scale=scale
            )
            for chunk in tqdm(gdf_chunks, desc="Downloading DW acquisitions (median)", unit="chunk")
        )
        frames = [frame for frame in frames if frame is not None and not frame.empty]
        if not frames:
            raise NoDynamicWorldDataError(
                f"No data was extracted from any chunk: Earth Engine returned no Dynamic World data "
                f"for any of the {len(gdf_chunks)} chunks and {len(windows)} months requested."
            )

        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop(columns=["reducer"], errors="ignore")

        df["area_data"] = df[dw_bandnames].sum(axis=1, skipna=False)
        has_observation = df["area_data"] > 0
        n_total = len(df)

        if snow_free:
            snow_fraction = df["snow_and_ice"] / df["area_data"]
            keep = has_observation & (snow_fraction <= max_snow_fraction)
            self._log_info(
                f"Snow-free scene filter: kept {int(keep.sum())} of {n_total} acquisitions "
                f"(<= {max_snow_fraction:.0%} snow/ice within the polygon)"
            )
        else:
            keep = has_observation
            self._log_info(f"No snow filter: kept {int(keep.sum())} of {n_total} acquisitions with data")

        df = df[keep]
        if df.empty:
            reason = (
                f"none of the {n_total} acquisitions had <= {max_snow_fraction:.0%} snow/ice"
                if snow_free
                else f"none of the {n_total} acquisitions contained classified pixels"
            )
            raise NoDynamicWorldDataError(f"No data was extracted: {reason}, so no month could be aggregated.")

        month = df["date"].dt.to_period("M").dt.to_timestamp().rename("date")
        dates_col = df.drop(columns=["date"]).assign(date=month)
        grouped = dates_col.groupby([name_attribute, "date"], as_index=False)
        aggregated = grouped[dw_bandnames].median()
        scenes_per_month = grouped.size()["size"]
        self._log_info(
            f"Monthly snow-free aggregation: {len(aggregated)} (polygon, month) rows; scenes per month "
            f"median {scenes_per_month.median():.0f} (min {scenes_per_month.min()}, max {scenes_per_month.max()})"
        )
        return aggregated

    def _extract_jrc_time_series(
        self, gdf_chunk, name_attribute: str, years: list[int], scale: float = 30
    ) -> pd.DataFrame:
        """Extract JRC time series data for a single chunk.

        Processes each year within this method to avoid memory issues with large imlists.

        Args:
            gdf_chunk: GeoDataFrame chunk to extract data for.
            name_attribute: Column name to use as the feature index.
            years: List of years to process (e.g., [2017, 2018]).
            scale: Pixel scale in meters (default: 30 for JRC data).

        Returns:
            pd.DataFrame: DataFrame with JRC water classification areas for this chunk.
        """
        image_collection = ee.ImageCollection("JRC/GSW1_4/YearlyHistory")

        # Filter by specific years if provided
        if years is not None and len(years) > 0:
            image_collection = image_collection.filter(ee.Filter.inList("year", years))
            self._log_info(f"Filtering JRC image collection to specific years: {years}")

        def waterMaskArea(image):
            for i in range(4):
                # calculate pixel area in ha for seasonal water surface
                water_mask = (
                    image.select("waterClass")
                    .eq(i)
                    .multiply(ee.Image.pixelArea())
                    .multiply(1e-4)
                    .rename(self.jrc_bandnames[i])
                )
                image = image.addBands(water_mask.select(self.jrc_bandnames[i]))
            return image

        _geom, reducer_dict = self._setup_jrc_reducer(gdf=gdf_chunk, feature_index_name=name_attribute, scale=scale)
        ic_water = image_collection.map(waterMaskArea)
        df = geemap.ee_to_gdf(ic_water.getTimeSeriesByRegions(**reducer_dict))
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        return df.drop(columns=["geometry"])

    def _validate_years_dw(self, years) -> list[int]:
        """Validate the years parameter for Dynamic World download.

        Args:
            years: The years parameter to validate
        Returns:
            List[int]: Validated list of years to process
        """
        current_year = pd.Timestamp.now().year
        # Check if years are empty and create default or check if they are in the correct format
        if years is None:
            # Default years for Dynamic World data (2016-2021)
            years = list(range(2016, current_year + 1))
        elif isinstance(years, list):
            pass  # already in correct format
        elif isinstance(years, (np.ndarray, range)):
            years = list(years)
        else:
            raise ValueError(
                "Years must be provided as a list, numpy array, or range object or as None for default years (2016-"
                + str(current_year)
                + ")"
            )

        # validate years
        years_cleaned = []
        for year in years:
            if isinstance(year, int) and 2016 <= year <= current_year:
                years_cleaned.append(year)
            else:
                self._log_warning(
                    f"Year {year} is invalid, must be between 2016 and {current_year} and will be skipped!"
                )
        return years_cleaned

    def download_dw_monthly(
        self,
        vector_dataset: str | Path | None = None,
        gdf: gpd.GeoDataFrame | None = None,
        name_attribute: str = "id_geohash",
        years: list[int] | None = None,
        months: list[int] | None = None,
        date_list: list[str] | None = None,
        bbox_west: float = -180,
        bbox_east: float = 180,
        bbox_north: float = 90,
        bbox_south: float = -90,
        id_list: list | None = None,
        scale: float = 10,
        max_total_requests: int = 500,
        n_parallel: int = 1,
        chunk_method: str = "simple",
        no_download: bool = False,
        save_to_file: str | None = None,
        aggregation: str = "mode",
        snow_free: bool = False,
        max_snow_fraction: float = 0.05,
    ) -> xr.Dataset:
        """Download monthly Dynamic World land cover data for specified periods.

        Extracts land cover class areas from Google Earth Engine for each
        polygon in the vector dataset, grouped by the specified date periods.

        Either provide `date_list` OR (`years` AND `months`). These options are
        mutually exclusive.

        Args:
            vector_dataset: Path to the input vector dataset (Parquet format). Mutually exclusive with `gdf`.
            gdf: Optional GeoDataFrame to use directly instead of loading from vector_dataset path. Mutually exclusive with `vector_dataset`.
            name_attribute: Column name in the vector dataset to use for grouping.
            years: List of years to process. Must be provided together with `months`
                if `date_list` is not used. Default: [2017-2025].
            months: List of months to process as integers (1-12). Must be provided
                together with `years` if `date_list` is not used. Default: [6,7,8,9].
            date_list: Optional list of dates in 'YYYY-MM' format (e.g., ['2017-06', '2018-07']).
                If provided, `years` and `months` are ignored. Mutually exclusive with
                (years AND months).
            bbox_west: Western boundary for spatial filtering (default: -180).
            bbox_east: Eastern boundary for spatial filtering (default: 180).
            bbox_north: Northern boundary for spatial filtering (default: 90).
            bbox_south: Southern boundary for spatial filtering (default: -90).
            id_list: Optional list of IDs to filter by (values from name_attribute column).
                If provided, only features matching these IDs will be processed.
                Default is None (no ID filtering).
            no_download: If True, only log the download parameters without actually
                downloading data (default: False).
            save_to_file: Optional path to save the downloaded dataset. If provided, the
                dataset will be saved to this path. The format is determined by the file
                extension: '.zarr' for Zarr format, '.nc' for NetCDF format. If a relative
                path is provided, it will be saved in the output directory (default: None).
            aggregation: `"mode"` (default) builds one mode composite per month and reports its
                class areas - unchanged historical behaviour. `"median"` extracts every
                acquisition of the month and reports the median class areas across them.
                Documented under "Aggregation methods" below.
            snow_free: With `aggregation="median"`, keep only acquisitions that are snow-free
                within the polygon (default: False). Rejected for `"mode"`, which cannot filter
                scenes per polygon.
            max_snow_fraction: With `snow_free=True`, the share of a polygon that may be
                snow/ice for the acquisition to count as snow-free (default: 0.05, matching
                `DWDataset._mask_invalid`).

        Returns:
            xr.Dataset: Xarray dataset with land cover areas indexed by name
                attribute and date.

        Aggregation methods:
            The statistic and the scene filter are separate arguments:

            * ``aggregation="mode"`` (default, unchanged since the beginning) builds one mode
              composite per month with ``ee.Reducer.mode()`` and reports its class areas. In a
              snow-covered month the modal class is ``snow_and_ice``, so such a month comes back
              as ``water = 0`` even where the lake opened up mid-month.
            * ``aggregation="median"`` extracts every acquisition of the month and reports the
              **median** class areas. It costs roughly what ``download_dw_daily`` costs, because
              it reuses that extraction path.
            * ``snow_free=True`` (only with ``aggregation="median"``) keeps just the acquisitions
              that are snow-free *within each polygon* before the median is taken. A scene is
              snow-free when ``snow_and_ice / area_data <= max_snow_fraction`` (default 0.05, the
              same threshold ``DWDataset._mask_invalid`` uses), so ice is included. Months where
              no acquisition is snow-free are **absent** from the output rather than reported as
              zero. This is the recommended setting for snow-affected lakes and the fix for the
              ``0 ha`` artefacts.

            ``snow_free=True`` with ``aggregation="mode"`` is rejected: filtering scenes per
            polygon would require one composite per polygon, which the mode path does not build.

        Raises:
            KeyError: If the specified name_attribute column is not found in the
                vector dataset.
            ValueError: If neither (years and months) nor date_list is provided,
                or if both are provided, or if ``aggregation`` is not one of
                ``"mode"`` / ``"median"``, or if ``snow_free=True`` is combined with
                ``aggregation="mode"``.
            NoDynamicWorldDataError: If Earth Engine confirms there is no Dynamic World
                data for any chunk and date requested, or (with ``snow_free=True``)
                if no requested month contains a snow-free acquisition.
            ee.EEException, requests.exceptions.RequestException, ...: If an Earth Engine
                or network request fails after retrying transient errors.
        """
        # check mutually exclusive vector_dataset versus gdf
        if vector_dataset is None and gdf is None:
            raise ValueError("Either vector_dataset or gdf must be provided")
        elif vector_dataset is not None and gdf is not None:
            raise ValueError("Only one of vector_dataset or gdf should be provided, not both")

        if aggregation not in ("mode", "median"):
            raise ValueError(f"aggregation must be 'mode' or 'median', got {aggregation!r}")
        if aggregation == "mode" and snow_free:
            raise ValueError(
                "snow_free=True requires aggregation='median': filtering scenes per polygon needs one "
                "composite per polygon, which the mode path does not build. Use aggregation='median'."
            )
        if not 0 <= max_snow_fraction <= 1:
            raise ValueError(
                f"max_snow_fraction must be between 0 and 1 (share of the polygon allowed to be "
                f"snow/ice), got {max_snow_fraction}"
            )

        # Announce no_download mode at the top
        if no_download:
            self._log_info("=== NO DOWNLOAD MODE - Will skip after preprocessing ===")

        # Check if years are empty and create default or check if they are in the correct format

        if date_list is not None and (years is not None or months is not None):
            self._log_warning("years and/or months parameters will be ignored since date_list is provided")
        elif date_list is None:
            years = self._validate_years_dw(years)

        # Read vector data using the reusable function
        if vector_dataset is not None:
            self._log_info(f"Loading vector dataset from: {vector_dataset}")
            gdf = load_vector_dataset(vector_dataset, logger=self.logger)

        elif gdf is not None:
            gdf = gdf.copy()

        if name_attribute not in gdf.columns:
            raise KeyError(f"The designated column '{name_attribute}' is not present in the vector dataset.")

        # Log initial number of features
        n_features_initial = len(gdf)
        self._log_info(f"Initial dataset has {n_features_initial} features")

        # Apply ID filter if id_list is provided
        gdf = self._apply_id_filter(gdf, id_list, name_attribute)

        if bbox_west > bbox_east or bbox_south > bbox_north:
            raise ValueError(
                f"Invalid bbox parameters: west={bbox_west}, east={bbox_east}, south={bbox_south}, north={bbox_north}. "
                "Ensure that west <= east and south <= north."
            )

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

        # Generate dates from options (validates and handles mutually exclusive parameters)
        dates = setup_dates_from_options(years=years, months=months, date_list=date_list)
        n_dates = len(dates)
        n_total_requests = n_features * n_dates
        self._log_info(f"Processing {n_features} features x {n_dates} dates = {n_total_requests} total requests")
        self._log_info(f"Processing dates: {dates}")

        # Chunk the GeoDataFrame into smaller pieces based on number of dates
        gdf_chunks = self._chunk_gdf(gdf, max_total_requests, n_dates=n_dates, chunk_method=chunk_method)

        # Return early if no_download is True - skip downloading but show summary
        if no_download:
            self._log_info(f"Would process {len(gdf)} features with {len(gdf_chunks)} chunks")
            self._log_info("=== END NO DOWNLOAD MODE ===")
            return None

        # Ensure n_parallel is not greater than number of chunks
        n_parallel_effective = min(n_parallel, len(gdf_chunks))

        # Process chunks with joblib (sequential if n_parallel=1, parallel otherwise)
        self._log_info(f"Processing {len(gdf_chunks)} chunks with {n_parallel_effective} workers")

        if aggregation == "median":
            self._log_info(
                "Aggregation 'median': extracting every acquisition of each month"
                + (
                    f" and keeping only those with <= {max_snow_fraction:.0%} snow/ice within the polygon"
                    if snow_free
                    else " (no snow filter)"
                )
            )
            df_out = self._monthly_median_frame(
                dates=dates,
                gdf_chunks=gdf_chunks,
                name_attribute=name_attribute,
                scale=scale,
                n_parallel=n_parallel_effective,
                snow_free=snow_free,
                max_snow_fraction=max_snow_fraction,
            )
        else:
            # Use tqdm for progress bar
            df_out_list = joblib.Parallel(n_jobs=n_parallel_effective, prefer="threads")(
                joblib.delayed(self._extract_time_series)(
                    dates=dates, gdf_chunk=chunk, name_attribute=name_attribute, scale=scale
                )
                for chunk in tqdm(gdf_chunks, desc="Downloading DW chunks", unit="chunk")
            )
            # Filter out None/empty results
            df_out_list = [df for df in df_out_list if df is not None and not df.empty]

            # Combine all chunks. Errors have already been raised above, so an empty list
            # here means Earth Engine confirmed there is no Dynamic World data at all.
            if not df_out_list:
                raise NoDynamicWorldDataError(
                    f"No data was extracted from any chunk: Earth Engine returned no Dynamic World data "
                    f"for any of the {len(gdf_chunks)} chunks and {n_dates} dates requested."
                )

            df_out = pd.concat(df_out_list)

        ds = df_out.set_index([name_attribute, "date"]).to_xarray()

        # Log summary statistics using the dataset coordinates (indexes)
        n_items = len(ds.coords[name_attribute])
        n_dates = len(ds.coords["date"])
        self._log_info(f"Download complete: {n_items} items, {n_dates} dates collected")

        # Remove the 'reducer' variable if present (the snow_free_median path has dropped it already)
        ds = ds.drop_vars("reducer", errors="ignore")

        # force date to pd.datetime format
        ds["date"] = pd.to_datetime(ds["date"])

        # Annotate dataset with metadata
        ds = annotate_xr_dataset_dw(ds, input_vector_file=vector_dataset)
        ds.attrs["aggregation"] = aggregation
        ds.attrs["snow_free"] = bool(snow_free)
        if snow_free:
            ds.attrs["max_snow_fraction"] = float(max_snow_fraction)

        # Save to file if requested
        if save_to_file is not None:
            # Determine output_dir for relative paths
            save_path = Path(save_to_file).absolute()
            save_xarray_dataset(ds=ds, save_path=save_path, logger=self.logger)

        return ds

    def download_dw_daily(
        self,
        vector_dataset: str | Path | None = None,
        gdf: gpd.GeoDataFrame | None = None,
        name_attribute: str = "id_geohash",
        years: list[int] | None = None,
        months: list[int] | None = None,
        date_list: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        max_nodata: float = 0.02,
        bbox_west: float = -180,
        bbox_east: float = 180,
        bbox_north: float = 90,
        bbox_south: float = -90,
        id_list: list | None = None,
        scale: float = 10,
        max_total_requests: int = 500,
        n_parallel: int = 1,
        chunk_method: str = "simple",
        no_download: bool = False,
        save_to_file: str | None = None,
    ) -> xr.Dataset:
        """Download Dynamic World land cover areas per acquisition (unaggregated).

        Returns one row per Dynamic World acquisition (Sentinel-2 overpass) inside the selected
        date windows, instead of the monthly mode composite produced by
        ``download_dw_monthly``. This is **not** a gap-filled daily series: only dates with
        an actually classified acquisition appear, so missing dates are cloud/coverage gaps.

        The date selection accepts the same ``months`` / ``date_list`` arguments as
        ``download_dw_monthly``, so that call can be pointed at per-acquisition data by
        changing the method name only. Alternatively give a recurring within-year window with
        ``start_date`` / ``end_date`` (``MM-DD``), which is the only option for an arbitrary
        period such as "01-05 to 15-09".

        Observations are masked when more than ``max_nodata`` of the polygon has no data,
        and acquisitions with no classified pixels at all are dropped (a fully masked image
        means "no observation", not "zero water area").

        The result has the same variables and dimension names as the monthly product, so it
        can be handed to :class:`water_timeseries.dataset.DWDataset` and its plotting methods
        unchanged. Note that with ``mask_data=False`` snow-covered acquisitions keep their
        ``water = 0`` values; the default ``DWDataset`` masking removes them.

        Either provide `vector_dataset` OR `gdf`. These options are mutually exclusive.

        Args:
            vector_dataset: Path to the input vector dataset (Parquet format).
            gdf: Optional GeoDataFrame to use directly instead of loading from a path.
            name_attribute: Column name in the vector dataset to use for grouping.
            years: Years to process. Default: 2016 up to the current year.
            months: Month numbers (1-12) to download, like ``download_dw_monthly``. Each month
                becomes its own window, so ``[6, 7, 8, 9]`` means June through September.
                Mutually exclusive with ``date_list`` and with ``start_date``/``end_date``.
            date_list: Month strings in ``'YYYY-MM'`` format (e.g. ``['2020-06']``), like
                ``download_dw_monthly``. Mutually exclusive with ``months``.
            start_date: Within-year window start as ``'MM-DD'``, inclusive (default: '05-01'
                when neither ``months`` nor ``date_list`` is given). Mutually exclusive with
                ``months`` / ``date_list``.
            end_date: Within-year window end as ``'MM-DD'``, inclusive (default: '09-30').
            max_nodata: Mask an observation when more than this fraction of the polygon has
                no data (default: 0.02, i.e. 2 %). Use 1.0 to keep everything.
            bbox_west: Western boundary for spatial filtering (default: -180).
            bbox_east: Eastern boundary for spatial filtering (default: 180).
            bbox_north: Northern boundary for spatial filtering (default: 90).
            bbox_south: Southern boundary for spatial filtering (default: -90).
            id_list: Optional list of IDs to filter by (values from name_attribute column).
            scale: Pixel scale in meters (default: 10).
            max_total_requests: Maximum number of (feature, acquisition) requests per chunk.
            n_parallel: Number of parallel workers for processing chunks (default: 1).
            chunk_method: Method for chunking the data: "simple" or "spatial_kmeans".
            no_download: If True, only log the download parameters without downloading.
            save_to_file: Optional path to save the dataset ('.zarr' or '.nc').

        Returns:
            xr.Dataset: Class areas in hectares indexed by (name_attribute, date), with
                ``date`` on the acquisition day.

        Raises:
            ValueError: If both or neither of vector_dataset/gdf is given, if max_nodata is
                outside 0..1, or if the season window is malformed.
            KeyError: If name_attribute is not present in the vector dataset.
            NoDynamicWorldDataError: If Earth Engine returned no usable observation, or every
                observation was masked by the no-data tolerance.
            ee.EEException, requests.exceptions.RequestException, ...: If an Earth Engine or
                network request fails after retrying transient errors.

        Performance:
            Chunks are the unit of parallelism: ``n_parallel`` workers can only run as many
            chunks at once, so provide more chunks than workers. ``max_total_requests`` sets
            the chunk size (features x one month of images); lowering it produces more,
            smaller chunks. Measured on 118 lakes, 11 seasons, 05-01..09-30: defaults
            (``max_total_requests=500``, ``n_parallel=1``) took 7 m 54 s; with
            ``max_total_requests=150`` and ``n_parallel=6`` the same download took 1 m 58 s,
            with identical dates and rows. Neither setting changes the data materially
            (values agree to ~1e-10 ha), but a fixed ``max_total_requests`` with
            ``n_parallel=1`` is the only bit-reproducible configuration. For very large lake
            sets keep ``max_total_requests`` near its default: chunks beyond what the workers
            can run only add per-request overhead.

        Example:
            >>> downloader = EarthEngineDownloader()
            >>> ds = downloader.download_dw_daily(
            ...     vector_dataset="tests/data/lake_polygons.parquet",
            ...     years=[2022, 2023],
            ...     start_date="05-01",
            ...     end_date="09-30",
            ...     max_nodata=0.02,
            ...     max_total_requests=150,
            ...     n_parallel=6,
            ... )
        """
        # check mutually exclusive vector_dataset versus gdf
        if vector_dataset is None and gdf is None:
            raise ValueError("Either vector_dataset or gdf must be provided")
        elif vector_dataset is not None and gdf is not None:
            raise ValueError("Only one of vector_dataset or gdf should be provided, not both")

        if not 0 <= max_nodata <= 1:
            raise ValueError(
                f"max_nodata must be between 0 and 1 (share of the polygon allowed to be no-data), got {max_nodata}"
            )

        if no_download:
            self._log_info("=== NO DOWNLOAD MODE - Will skip after preprocessing ===")

        years = self._validate_years_dw(years)

        # Date selection: monthly-style (months / date_list) or a within-year MM-DD window.
        # The monthly-style arguments exist so a download_dw_monthly call can be pointed at
        # per-acquisition data by changing the method name only.
        if months is not None and date_list is not None:
            raise ValueError("Provide either 'months' or 'date_list', not both.")
        if (months is not None or date_list is not None) and (start_date is not None or end_date is not None):
            raise ValueError(
                "Provide either 'months'/'date_list' or 'start_date'/'end_date', not both - "
                "they select the same thing in two different ways."
            )
        if date_list is not None:
            windows = windows_from_date_list(date_list)
            window_label = f"date_list({len(date_list)} month(s))"
        elif months is not None:
            windows = month_windows(years, months)
            window_label = "months(" + ",".join(str(int(m)) for m in months) + ")"
        else:
            start_date = start_date or "05-01"
            end_date = end_date or "09-30"
            windows = season_windows(years, start_date=start_date, end_date=end_date)
            window_label = f"{start_date}..{end_date}"

        if not windows:
            raise ValueError("No valid date windows to download - check years, months and date_list.")

        # Read vector data using the reusable function
        if vector_dataset is not None:
            self._log_info(f"Loading vector dataset from: {vector_dataset}")
            gdf = load_vector_dataset(vector_dataset, logger=self.logger)
        else:
            gdf = gdf.copy()

        if name_attribute not in gdf.columns:
            raise KeyError(f"The designated column '{name_attribute}' is not present in the vector dataset.")

        self._log_info(f"Initial dataset has {len(gdf)} features")
        gdf = self._apply_id_filter(gdf, id_list, name_attribute)

        if bbox_west > bbox_east or bbox_south > bbox_north:
            raise ValueError(
                f"Invalid bbox parameters: west={bbox_west}, east={bbox_east}, south={bbox_south}, north={bbox_north}. "
                "Ensure that west <= east and south <= north."
            )

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
            self._log_info(
                f"After bbox filter: {len(gdf)} features (removed {n_before_bbox_filter - len(gdf)})"
            )
        else:
            self._log_info("No spatial bbox filtering applied (using default global bounds)")

        n_features = len(gdf)
        self._log_info(f"Processing {n_features} features")

        n_month_slices = sum(len(month_slices(start, end)) for start, end in windows)
        est_total_acquisitions = max(1, n_month_slices * EST_ACQUISITIONS_PER_MONTH)
        self._log_info(
            f"Date selection {window_label}: {len(years)} year(s), {len(windows)} window(s); "
            f"~{est_total_acquisitions} acquisitions per polygon (estimate), "
            f"~{n_features * est_total_acquisitions} (polygon, acquisition) requests"
        )

        areas = self._polygon_areas_ha(gdf, name_attribute)
        # Each reduce call carries one month of acquisitions, not the whole request, so the
        # chunk budget is sized by that. Sizing by the multi-year total would collapse chunks
        # to a single polygon and multiply the number of Earth Engine calls by the year count.
        gdf_chunks = self._chunk_gdf(
            gdf, max_total_requests, n_dates=EST_ACQUISITIONS_PER_MONTH, chunk_method=chunk_method
        )

        n_parallel_effective = min(n_parallel, len(gdf_chunks))
        if n_parallel_effective < n_parallel:
            target_chunks = max(1, n_parallel * 2)
            target_features = -(-n_features // target_chunks)  # ceil division
            suggested_budget = max(1, target_features * EST_ACQUISITIONS_PER_MONTH)
            self._log_warning(
                f"Only {len(gdf_chunks)} chunk(s) for {n_parallel} workers, so "
                f"{n_parallel - n_parallel_effective} will idle. max_total_requests={suggested_budget} "
                f"would split these {n_features} features into about {target_chunks} chunks."
            )

        if no_download:
            self._log_info(f"Would process {len(gdf)} features with {len(gdf_chunks)} chunks")
            self._log_info("=== END NO DOWNLOAD MODE ===")
            return None

        self._log_info(f"Processing {len(gdf_chunks)} chunks with {n_parallel_effective} workers")

        df_out_list = joblib.Parallel(n_jobs=n_parallel_effective, prefer="threads")(
            joblib.delayed(self._extract_acquisitions)(
                windows=windows, gdf_chunk=chunk, name_attribute=name_attribute, scale=scale
            )
            for chunk in tqdm(gdf_chunks, desc="Downloading DW acquisitions", unit="chunk")
        )
        df_out_list = [df for df in df_out_list if df is not None and not df.empty]

        if not df_out_list:
            raise NoDynamicWorldDataError(
                f"No data was extracted from any chunk: Earth Engine returned no Dynamic World data "
                f"for any of the {len(gdf_chunks)} chunks and {est_total_acquisitions} requested acquisitions."
            )

        df = pd.concat(df_out_list, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop(columns=["reducer"], errors="ignore")

        # Coverage per observation, relative to the polygon's own area.
        # skipna=False keeps a partially-NaN row NaN so it is dropped below instead of
        # being mistaken for a genuine zero-area observation.
        df["area_data"] = df[dw_bandnames].sum(axis=1, skipna=False)
        df["area_ha"] = df[name_attribute].map(areas)
        df["coverage"] = (df["area_data"] / df["area_ha"]).clip(upper=1.0)

        n_rows = len(df)
        df = df[df["area_data"] > 0]
        if n_rows - len(df):
            self._log_info(
                f"Dropped {n_rows - len(df)} acquisition(s) with no classified pixels "
                "(no observation, not a zero water area)"
            )

        n_rows = len(df)
        df = df.sort_values("coverage").drop_duplicates([name_attribute, "date"], keep="last")
        if n_rows - len(df):
            self._log_info(
                f"Collapsed {n_rows - len(df)} duplicate (polygon, date) pair(s), keeping the higher-coverage acquisition"
            )

        n_rows = len(df)
        df = df[df["coverage"] >= 1 - max_nodata]
        self._log_info(f"Masked {n_rows - len(df)} acquisition(s) with more than {max_nodata:.1%} no-data")

        if df.empty:
            raise NoDynamicWorldDataError(
                "No data was extracted: every acquisition was fully masked or exceeded the "
                f"max_nodata tolerance of {max_nodata:.1%}."
            )

        ds = (
            df.drop(columns=["area_data", "area_ha", "coverage"])
            .set_index([name_attribute, "date"])
            .to_xarray()
        )
        ds["date"] = pd.to_datetime(ds["date"])

        per_polygon = ds["water"].notnull().sum("date").to_pandas()
        self._log_info(
            f"Download complete: {len(ds.coords[name_attribute])} polygons, "
            f"{len(ds.coords['date'])} acquisitions, {int(ds['water'].notnull().sum())} observations "
            f"(per polygon: min {int(per_polygon.min())}, median {per_polygon.median():.0f}, max {int(per_polygon.max())})"
        )
        if float(per_polygon.median()) < 5:
            self._log_warning(
                "Median observations per polygon is below 5 - that is a cloud/coverage limit, not a download failure"
            )

        ds = annotate_xr_dataset_dw(ds, input_vector_file=vector_dataset, temporal_resolution="per_acquisition")
        ds.attrs["season_window"] = window_label
        ds.attrs["max_nodata"] = float(max_nodata)
        ds.attrs["n_acquisitions"] = len(ds.coords["date"])

        if save_to_file is not None:
            save_path = Path(save_to_file).absolute()
            save_xarray_dataset(ds=ds, save_path=save_path, logger=self.logger)

        return ds

    def _validate_years_jrc(self, years) -> list[int]:
        """Validate the years parameter for JRC download.

        Args:
            years: The years parameter to validate
        Returns:
            List[int]: Validated list of years to process
        """
        # Check if years are empty and create default or check if they are in the correct format
        if years is None:
            # Default years for JRC data (1984-2021)
            years = list(range(1984, 2022))
        elif isinstance(years, list):
            pass  # already in correct format
        elif isinstance(years, (np.ndarray, range)):
            years = list(years)
        else:
            raise ValueError(
                "Years must be provided as a list, numpy array, or range object or as None for default years (1984-2021)"
            )

        # validate years
        years_cleaned = []
        for year in years:
            if isinstance(year, int) and 1984 <= year <= 2021:
                years_cleaned.append(year)
            else:
                self._log_warning(f"Year {year} is invalid, must be between 1984 and 2021 and will be skipped!")
        return years_cleaned

    def download_jrc_annual(
        self,
        vector_dataset: str | Path | None = None,
        gdf: gpd.GeoDataFrame | None = None,
        name_attribute: str = "id_geohash",
        years: list[int] | np.ndarray | range | None = None,
        bbox_west: float = -180,
        bbox_east: float = 180,
        bbox_north: float = 90,
        bbox_south: float = -90,
        id_list: list | None = None,
        scale: float = 30,
        max_total_requests: int = 500,
        n_parallel: int = 1,
        chunk_method: str = "simple",
        no_download: bool = False,
        save_to_file: str | None = None,
    ) -> xr.Dataset:
        """Download annual JRC (Joint Research Centre) water classification data.

        Extracts JRC water classification areas from Google Earth Engine for each
        polygon in the vector dataset, grouped by year.

        The JRC dataset provides annual water classification with the following bands:
        - area_nodata: Areas with no data
        - area_land: Land areas
        - area_water_seasonal: Seasonally flooded water areas
        - area_water_permanent: Permanently flooded water areas

        Either provide `vector_dataset` or `gdf`. These options are mutually exclusive.

        Args:
            vector_dataset: Path to the input vector dataset (Parquet format). Mutually exclusive with `gdf`.
            gdf: Optional GeoDataFrame to use directly instead of loading from vector_dataset path. Mutually exclusive with `vector_dataset`.
            name_attribute: Column name in the vector dataset to use for grouping.
            years: List of years to process (e.g., [2017, 2018]). Default: [1984-2021].
            bbox_west: Western boundary for spatial filtering (default: -180).
            bbox_east: Eastern boundary for spatial filtering (default: 180).
            bbox_north: Northern boundary for spatial filtering (default: 90).
            bbox_south: Southern boundary for spatial filtering (default: -90).
            id_list: Optional list of IDs to filter by (values from name_attribute column).
                If provided, only features matching these IDs will be processed.
                Default is None (no ID filtering).
            scale: Pixel scale in meters (default: 30 for JRC data).
            max_total_requests: Maximum number of total requests per chunk (features * years).
                Default: 500.
            n_parallel: Number of parallel workers for processing chunks (default: 1).
            chunk_method: Method for chunking the data. Options are "simple" or "spatial_kmeans".
            no_download: If True, only log the download parameters without actually
                downloading data (default: False).
            save_to_file: Optional path to save the downloaded dataset. If provided, the
                dataset will be saved to this path. The format is determined by the file
                extension: '.zarr' for Zarr format, '.nc' for NetCDF format. If a relative
                path is provided, it will be saved in the output directory (default: None).

        Returns:
            xr.Dataset: Xarray dataset with JRC water classification areas indexed by
                name attribute and date (stored as 'YYYY-01-01' format).

        Raises:
            KeyError: If the specified name_attribute column is not found in the
                vector dataset.
            ValueError: If neither vector_dataset nor gdf is provided, or if both are provided.

        Example:
            >>> downloader = EarthEngineDownloader()
            >>> ds = downloader.download_jrc_annual(
            ...     vector_dataset="lakes.parquet",
            ...     name_attribute="lake_id",
            ...     years=[2017, 2018, 2019]
            ... )
        """
        # check mutually exclusive vector_dataset versus gdf
        if vector_dataset is None and gdf is None:
            raise ValueError("Either vector_dataset or gdf must be provided")
        elif vector_dataset is not None and gdf is not None:
            raise ValueError("Only one of vector_dataset or gdf should be provided, not both")

        # Announce no_download mode at the top
        if no_download:
            self._log_info("=== NO DOWNLOAD MODE - Will skip after preprocessing ===")

        # Check if years are empty and create default or check if they are in the correct format
        years = self._validate_years_jrc(years)

        # Read vector data using the reusable function
        if vector_dataset is not None:
            self._log_info(f"Loading vector dataset from: {vector_dataset}")
            gdf = load_vector_dataset(vector_dataset, logger=self.logger)

        else:
            self._log_info("Using provided GeoDataFrame directly, no further filtering applied")
            gdf = gdf.copy()

        if name_attribute not in gdf.columns:
            raise KeyError(f"The designated column '{name_attribute}' is not present in the vector dataset.")

        # Log initial number of features
        n_features_initial = len(gdf)
        self._log_info(f"Initial dataset has {n_features_initial} features")

        # Apply ID filter if id_list is provided
        gdf = self._apply_id_filter(gdf, id_list, name_attribute)

        if bbox_west > bbox_east or bbox_south > bbox_north:
            raise ValueError(
                f"Invalid bbox parameters: west={bbox_west}, east={bbox_east}, south={bbox_south}, north={bbox_north}. "
                "Ensure that west <= east and south <= north."
            )

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

        # Generate annual dates from years (stored as 'YYYY-01-01' format internally)

        dates = [f"{year}-01-01" for year in years]
        n_dates = len(dates)
        n_total_requests = n_features * n_dates
        self._log_info(f"Processing {n_features} features x {n_dates} years = {n_total_requests} total requests")
        self._log_info(f"Processing years: {[d.split('-')[0] for d in dates]}")

        # Chunk the GeoDataFrame into smaller pieces based on number of dates
        gdf_chunks = self._chunk_gdf(gdf, max_total_requests, n_dates=n_dates, chunk_method=chunk_method)

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
            joblib.delayed(self._extract_jrc_time_series)(
                gdf_chunk=chunk, name_attribute=name_attribute, years=years, scale=scale
            )
            for chunk in tqdm(gdf_chunks, desc="Downloading JRC annual chunks", unit="chunk")
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
        n_dates_out = len(ds.coords["date"])
        self._log_info(f"Download complete: {n_items} items, {n_dates_out} years collected")

        # Remove the 'reducer' variable if present
        ds = ds.drop_vars("reducer")

        # Annotate dataset with metadata
        ds = annotate_xr_dataset_jrc(ds, input_vector_file=vector_dataset)

        # Save to file if requested
        if save_to_file is not None:
            # Determine output_dir for relative paths
            save_path = Path(save_to_file).absolute()
            save_xarray_dataset(ds=ds, save_path=save_path, logger=self.logger)

        return ds


# Example usage
if __name__ == "__main__":
    # Create downloader instance with ee-project flag
    downloader = EarthEngineDownloader(ee_project="your-ee-project-id")
