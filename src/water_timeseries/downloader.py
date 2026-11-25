"""
Google Earth Engine Downloader

This module provides a Downloader class to download data from Google Earth Engine
into Dynamic World or JRC format.
"""

import os
import warnings
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

from water_timeseries.utils.data import (
    annotate_xr_dataset_dswx,
    annotate_xr_dataset_dswx_hls,
    annotate_xr_dataset_dw,
    annotate_xr_dataset_jrc,
    dswx_bandnames,
    dswx_hls_bandnames,
    dw_bandnames,
    jrc_bandnames,
)
from water_timeseries.utils.earthengine import (
    DSWX_ARCHIVE_START,
    DSWX_CRS,
    DSWX_HLS_ARCHIVE_START,
    DSWX_MIN_MAPPING_AREA_HA,
    DSWX_SCALE,
    NoDSWxDataError,
    NoDSWxHLSDataError,
    NoDynamicWorldDataError,
    calc_monthly_dw,
    call_with_retry,
    create_dw_classes_mask,
    drop_z_from_gdf,
    dswx_acquisitions,
    dswx_class_areas,
    dswx_collection,
    dswx_hls_acquisitions,
    dswx_hls_class_areas,
    dswx_hls_collection,
    dswx_hls_month_composite,
    dswx_month_composite,
    dw_acquisitions,
    dw_mode_composite,
    dw_snow_valid_areas,
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

# The same figure for OPERA DSWx-S1: a 6-12 day revisit puts ~2-3 scenes per month per
# polygon, and 1.4 MGRS tiles per date, so ~15 dates x 1.4 = ~21 images per (polygon, month)
# (measured on 2 lakes, 2025: 8-38 (lake, tile, date) rows per lake-month).
EST_DSWX_ACQUISITIONS_PER_MONTH = 20

# Re-export for backward compatibility
__all__ = [
    "EarthEngineDownloader",
    "NoDSWxDataError",
    "NoDSWxHLSDataError",
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
        self.dswx_bandnames = dswx_bandnames
        self.dswx_hls_bandnames = dswx_hls_bandnames

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

    @staticmethod
    def _snow_free_groups(
        survey: pd.DataFrame, name_attribute: str, max_snow_fraction: float
    ) -> dict[tuple, list]:
        """Group polygon ids by the acquisitions that are snow-free for them.

        Args:
            survey: One row per (polygon, acquisition) with ``snow_area`` and ``valid_area``
                in hectares.
            name_attribute: Column holding the polygon identifier.
            max_snow_fraction: Allowed snow/ice share of the observed surface.

        Returns:
            dict: ``{tuple_of_time_starts_ms: [polygon ids]}``. Polygons with no snow-free
                acquisition in the month are omitted, so their month is absent rather than 0.
        """
        fraction = survey["snow_area"] / survey["valid_area"]
        kept = survey[fraction <= max_snow_fraction]
        groups: dict[tuple, list] = {}
        for polygon_id, rows in kept.groupby(name_attribute):
            key = tuple(sorted(int(value) for value in rows["date"]))
            groups.setdefault(key, []).append(polygon_id)
        return groups

    def _monthly_mode_snow_free_frame(
        self,
        dates: list[str],
        gdf_chunks: list,
        name_attribute: str,
        scale: float,
        n_parallel: int,
        max_snow_fraction: float,
    ) -> pd.DataFrame:
        """Mode composite per month over each polygon's own snow-free acquisitions.

        For every (chunk, month) an extra pass measures the snow/ice share of each polygon in
        each acquisition. Polygons are then grouped by the set of acquisitions that are
        snow-free for *them*, and one mode composite is built per distinct set - so a lake
        whose breakup happens mid-month is composited from its own snow-free scenes only,
        while a lake still frozen in the same month drops out entirely.

        Args:
            dates: Month-start dates to aggregate (``'YYYY-MM-01'``).
            gdf_chunks: Polygons already split into chunks.
            name_attribute: Column name used as the polygon identifier.
            scale: Pixel scale in meters.
            n_parallel: Number of chunk workers.
            max_snow_fraction: Allowed snow/ice share of the observed surface for a scene to
                count as snow-free.

        Returns:
            pd.DataFrame: One row per ``(name_attribute, date)`` with the nine class areas,
                ``date`` being the month start.

        Raises:
            NoDynamicWorldDataError: If Earth Engine returned no acquisitions, or if no polygon
                had a snow-free acquisition in any requested month.
        """

        def per_chunk(chunk: gpd.GeoDataFrame) -> pd.DataFrame:
            fc = geemap.gdf_to_ee(drop_z_from_gdf(chunk[:]))
            frames = []
            for date in dates:
                window_start = pd.Timestamp(date).strftime("%Y-%m-%d")
                window_end = (pd.Timestamp(date) + pd.DateOffset(months=1)).strftime("%Y-%m-%d")

                survey_collection = dw_snow_valid_areas(fc, window_start, window_end, scale=scale)
                survey_fc = survey_collection.getTimeSeriesByRegions(
                    reducer=ee.Reducer.sum(),
                    collection=fc.select(name_attribute),
                    bands=["snow_area", "valid_area"],
                    crs="EPSG:3572",
                    scale=scale,
                    dateFormat="ms",
                )
                survey = call_with_retry(geemap.ee_to_df, survey_fc)
                if survey is None or survey.empty:
                    self._log_info(f"{window_start}: no acquisitions")
                    continue

                groups = self._snow_free_groups(survey, name_attribute, max_snow_fraction)
                polygons_kept = sum(len(ids) for ids in groups.values())
                self._log_info(
                    f"{window_start}: {len(groups)} snow-free scene set(s) covering {polygons_kept} polygon(s)"
                )

                for timestamps, ids in groups.items():
                    composite = dw_mode_composite(fc, window_start, window_end, list(timestamps), scale=scale)
                    classes = create_dw_classes_mask(composite)
                    group_fc = fc.filter(ee.Filter.inList(name_attribute, ids)).select(name_attribute)
                    fc_out = ee.ImageCollection([classes]).getTimeSeriesByRegions(
                        reducer=ee.Reducer.sum(),
                        collection=group_fc,
                        bands=self.dw_bandnames,
                        crs="EPSG:3572",
                        scale=scale,
                    )
                    frame = call_with_retry(geemap.ee_to_df, fc_out)
                    if frame is not None and not frame.empty:
                        frames.append(frame)

            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)

        frames = joblib.Parallel(n_jobs=n_parallel, prefer="threads")(
            joblib.delayed(per_chunk)(chunk)
            for chunk in tqdm(gdf_chunks, desc="Downloading DW chunks (snow-free mode)", unit="chunk")
        )
        frames = [frame for frame in frames if frame is not None and not frame.empty]
        if not frames:
            raise NoDynamicWorldDataError(
                f"No data was extracted: no polygon had a snow-free acquisition (<= "
                f"{max_snow_fraction:.0%} snow/ice) in any of the {len(dates)} months requested."
            )
        return pd.concat(frames, ignore_index=True)

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
            snow_free: Keep only acquisitions that are snow-free within the polygon before
                compositing (default: False). Each scene is judged per polygon, so every lake
                is composited from its own snow-free acquisitions; a lake whose month has no
                such acquisition is absent from it rather than reported as zero. Must be a
                bool, not a string.
            max_snow_fraction: With `snow_free=True`, the share of the *observed* polygon
                surface that may be snow/ice for the acquisition to count as snow-free
                (default: 0.05, matching `DWDataset._mask_invalid`).

        Returns:
            xr.Dataset: Xarray dataset with land cover areas indexed by name
                attribute and date.

        Snow-free scene filtering:
            By default one mode composite of the month is built with ``ee.Reducer.mode()`` and
            its class areas are reported. In a snow-covered month the modal class is
            ``snow_and_ice``, so such a month comes back as ``water = 0`` even where the lake
            opened up mid-month.

            With ``snow_free=True`` the acquisitions are filtered first, **per polygon**: a
            scene counts as snow-free for a polygon when
            ``snow_and_ice / valid_area <= max_snow_fraction`` (default 0.05, the same
            threshold ``DWDataset._mask_invalid`` uses, so ice is included). Polygons are
            grouped by the acquisitions that are snow-free for them and one mode composite is
            built per distinct set, so each lake is composited from its own snow-free scenes.
            A month with no snow-free acquisition for a polygon is **absent** for that polygon
            rather than reported as zero - the fix for the ``0 ha`` artefacts.

            Cost: one survey request plus one mode composite per distinct scene set per
            (chunk, month) - usually one or two sets inside a spatially coherent chunk.

        Raises:
            KeyError: If the specified name_attribute column is not found in the
                vector dataset.
            TypeError: If ``snow_free`` is not a bool.
            ValueError: If neither (years and months) nor date_list is provided,
                or if both are provided.
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

        if not isinstance(snow_free, bool):
            raise TypeError(f"snow_free must be True or False, got {snow_free!r}")
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

        if snow_free:
            self._log_info(
                "snow_free=True: per-polygon snow-free scene selection, then one mode composite "
                f"per distinct scene set (<= {max_snow_fraction:.0%} snow/ice)"
            )
            df_out = self._monthly_mode_snow_free_frame(
                dates=dates,
                gdf_chunks=gdf_chunks,
                name_attribute=name_attribute,
                scale=scale,
                n_parallel=n_parallel_effective,
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
        ds = annotate_xr_dataset_dw(ds, input_vector_file=vector_dataset, id_field=name_attribute)
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

        ds = annotate_xr_dataset_dw(
            ds, input_vector_file=vector_dataset, temporal_resolution="per_acquisition", id_field=name_attribute
        )
        ds.attrs["season_window"] = window_label
        ds.attrs["max_nodata"] = float(max_nodata)
        ds.attrs["n_acquisitions"] = len(ds.coords["date"])

        if save_to_file is not None:
            save_path = Path(save_to_file).absolute()
            save_xarray_dataset(ds=ds, save_path=save_path, logger=self.logger)

        return ds

    def _setup_dswx_reducer(self, gdf, feature_index_name: str, scale: float = DSWX_SCALE) -> tuple:
        """Setup Google Earth Engine reducer configuration for OPERA DSWx-S1.

        Args:
            gdf: GeoDataFrame with features to process.
            feature_index_name: Column name to use as the feature index.
            scale: Pixel scale in meters (default: 30, the product's resolution).

        Returns:
            tuple: (fc, reducer_dict) - GEE FeatureCollection and reducer configuration.
        """
        fc = geemap.gdf_to_ee(drop_z_from_gdf(gdf[:]))

        reducer_dict = {
            "reducer": ee.Reducer.sum(),
            "collection": fc.select(feature_index_name),
            "crs": DSWX_CRS,
            "scale": scale,
            "bands": self.dswx_bandnames,
        }
        return fc, reducer_dict

    def _validate_years_dswx(self, years, first_year: int | None = None, archive_start: str = DSWX_ARCHIVE_START) -> list[int]:
        """Validate the years parameter for an OPERA DSWx-S1 download.

        Production starts in August 2024, so earlier years are dropped rather than
        downloaded as empty ranges.

        Args:
            years: The years parameter to validate.

        Returns:
            List[int]: Validated list of years to process.
        """
        current_year = pd.Timestamp.now().year
        if first_year is None:
            first_year = int(archive_start[:4])
        if years is None:
            years = list(range(first_year, current_year + 1))
        elif isinstance(years, list):
            pass  # already in correct format
        elif isinstance(years, (np.ndarray, range)):
            years = list(years)
        else:
            raise ValueError(
                "Years must be provided as a list, numpy array, or range object or as None for default years ("
                + str(first_year)
                + "-"
                + str(current_year)
                + ")"
            )

        years_cleaned = []
        for year in years:
            if isinstance(year, int) and first_year <= year <= current_year:
                years_cleaned.append(year)
            else:
                self._log_warning(
                    f"Year {year} is invalid, must be between {first_year} and {current_year} "
                    f"(production starts {archive_start}) and will be skipped!"
                )
        return years_cleaned

    @staticmethod
    def _dswx_add_coverage(
        frame: pd.DataFrame,
        areas: dict,
        name_attribute: str,
        scale: float,
        cover_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Add classified area, its share of the polygon, and the polygon's pixel count.

        ``coverage`` is the share of the polygon carrying any classification, i.e. the
        quantity the no-data tolerance is applied to. It is measured against the polygon's
        geodesic area rather than against another observation, because a per-acquisition
        download carries no reference scene.

        Args:
            frame: One row per (polygon, acquisition).
            areas: ``{id: area_ha}`` from :meth:`_polygon_areas_ha`.
            name_attribute: Column holding the polygon identifier.
            scale: Pixel scale in meters, used to express the polygon area in pixels.

        Returns:
            pd.DataFrame: A copy with ``area_data``, ``area_ha``, ``coverage`` and
            ``polygon_pixels`` added.
        """
        df = frame.copy()
        # skipna=False keeps a partially-NaN row NaN so it is dropped below instead of being
        # mistaken for a genuine zero-area observation.
        columns = list(cover_columns) if cover_columns is not None else list(dswx_bandnames)
        df["area_data"] = df[columns].sum(axis=1, skipna=False)
        df["area_ha"] = df[name_attribute].map(areas)
        df["coverage"] = (df["area_data"] / df["area_ha"]).clip(upper=1.0)
        # A 30 m product puts ~11 pixels in a hectare, so a sub-hectare pond is a handful of
        # pixels and a percentage tolerance is meaningless there.
        df["polygon_pixels"] = df["area_ha"] * 1e4 / (scale**2)
        return df

    @staticmethod
    def _dswx_apply_tolerance(frame: pd.DataFrame, max_nodata: float, min_masked_pixels: float) -> pd.DataFrame:
        """Keep observations whose no-data share is within tolerance.

        The tolerance is ``max_nodata``, raised to ``min_masked_pixels`` worth of pixels where
        the polygon is small enough that the fraction is stricter than a couple of pixels
        (a 0.5 ha pond at 30 m spans ~6 pixels, so 2 % would reject every observation that
        misses a single edge pixel).

        Args:
            frame: Rows with ``coverage`` and ``polygon_pixels`` from
                :meth:`_dswx_add_coverage`.
            max_nodata: Allowed no-data share of the polygon.
            min_masked_pixels: Absolute floor, in pixels, for that tolerance.

        Returns:
            pd.DataFrame: The rows within tolerance.
        """
        tolerance = np.maximum(max_nodata, min_masked_pixels / frame["polygon_pixels"])
        return frame[frame["coverage"] >= 1 - tolerance]

    def _extract_dswx_daily(
        self,
        windows: list[tuple[str, str]],
        gdf_chunk,
        name_attribute: str,
        scale: float = DSWX_SCALE,
    ) -> pd.DataFrame:
        """Extract per-acquisition DSWx class areas for one chunk over several date windows.

        Processes one calendar month per Earth Engine request, so no single request carries a
        whole season of images.

        Args:
            windows: List of ``(start, end)`` date-range pairs; the end is exclusive.
            gdf_chunk: GeoDataFrame chunk to extract data for.
            name_attribute: Column name used as the feature index.
            scale: Pixel scale in meters.

        Returns:
            pd.DataFrame: One row per (polygon, tile, acquisition); empty if nothing was found.
        """
        fc, reducer_dict = self._setup_dswx_reducer(gdf_chunk, name_attribute, scale=scale)

        frames = []
        for window_start, window_end in windows:
            for slice_start, slice_end in month_slices(window_start, window_end):
                collection = dswx_acquisitions(fc, slice_start, slice_end, scale=scale)
                n_images = call_with_retry(collection.size().getInfo)
                if n_images == 0:
                    self._log_info(f"No OPERA DSWx-S1 acquisitions between {slice_start} and {slice_end}")
                    continue
                self._log_info(f"{slice_start}..{slice_end}: {n_images} acquisitions")

                fc_out = collection.getTimeSeriesByRegions(**reducer_dict)
                df = call_with_retry(geemap.ee_to_df, fc_out)
                if df is None or df.empty:
                    continue
                frames.append(df)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _extract_dswx_monthly(
        self,
        dates: list[str],
        gdf_chunk,
        name_attribute: str,
        scale: float = DSWX_SCALE,
    ) -> pd.DataFrame:
        """Extract monthly DSWx class areas for one chunk.

        One mode composite per month in a single ``getTimeSeriesByRegions`` call, like the
        Dynamic World monthly product. A month with no acquisition is skipped rather than
        reduced, because a mode over an empty collection is not an observation.

        Args:
            dates: Month starts in ``'YYYY-MM-DD'`` format.
            gdf_chunk: GeoDataFrame chunk to extract data for.
            name_attribute: Column name used as the feature index.
            scale: Pixel scale in meters.

        Returns:
            pd.DataFrame: One row per (polygon, month); empty if nothing was found.
        """
        fc, reducer_dict = self._setup_dswx_reducer(gdf_chunk, name_attribute, scale=scale)

        imlist = []
        for date in dates:
            month_start = pd.Timestamp(date)
            month_end = month_start + pd.DateOffset(months=1)
            slice_start = month_start.strftime("%Y-%m-%d")
            slice_end = month_end.strftime("%Y-%m-%d")

            n_images = call_with_retry(dswx_collection(fc, slice_start, slice_end).size().getInfo)
            if n_images == 0:
                self._log_info(f"No OPERA DSWx-S1 acquisitions in {slice_start[:7]}")
                continue
            composite = dswx_month_composite(fc, slice_start, slice_end, scale=scale)
            imlist.append(dswx_class_areas(composite))

        if not imlist:
            self._log_warning("No images processed for chunk")
            return pd.DataFrame()

        image_collection = ee.ImageCollection(imlist)
        fc_out = image_collection.getTimeSeriesByRegions(**reducer_dict)
        return call_with_retry(geemap.ee_to_df, fc_out)

    def download_dswx_daily(
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
        min_masked_pixels: float = 2,
        bbox_west: float = -180,
        bbox_east: float = 180,
        bbox_north: float = 90,
        bbox_south: float = -90,
        id_list: list | None = None,
        scale: float = DSWX_SCALE,
        max_total_requests: int = 500,
        n_parallel: int = 1,
        chunk_method: str = "simple",
        no_download: bool = False,
        save_to_file: str | None = None,
    ) -> xr.Dataset:
        """Download OPERA DSWx-S1 class areas per acquisition (unaggregated).

        Returns one row per DSWx-S1 acquisition inside the selected date windows instead of
        the monthly mode composite produced by :meth:`download_dswx_monthly`. Only dates with
        an acquisition appear - a missing date is a revisit/coverage gap, not zero water.

        The date arguments are identical to :meth:`download_dw_monthly` /
        :meth:`download_dw_daily`: either ``months`` / ``date_list`` (so a Dynamic World call
        is repointed by changing the method name), or a recurring within-year window with
        ``start_date`` / ``end_date`` as ``MM-DD``. The two forms are mutually exclusive.

        Production starts in August 2024, so earlier years are skipped.

        Each MGRS tile reduces the polygon over only the part it covers, so a lake can return
        several rows for the same date. Those fragments are collapsed to the best-covered tile
        - never summed. Overlapping tiles of this product often return the same classified
        area, in which case the tie-break is the tile with the larger not-water area; that is
        deterministic, but it does bias the class split towards the drier of two disagreeing
        tiles. Rows with no classified pixels at all are dropped (a fully masked scene is "no
        observation", not "zero water"), and observations whose no-data share exceeds the
        tolerance are masked.

        Note that DSWx-S1 has no snow/ice class: a frozen basin is classified as inundated
        vegetation, so winter rows are an ice signal. Gate them against Dynamic World's
        ``snow_and_ice`` for the same polygon and date, or restrict the window.

        Either provide `vector_dataset` OR `gdf`. These options are mutually exclusive.

        Args:
            vector_dataset: Path to the input vector dataset (Parquet format).
            gdf: Optional GeoDataFrame to use directly instead of loading from a path.
            name_attribute: Column name in the vector dataset to use for grouping.
            years: Years to process. Default: 2024 up to the current year.
            months: Month numbers (1-12) to download, like ``download_dw_monthly``. Each month
                becomes its own window, so ``[6, 7, 8, 9]`` means June through September.
                Mutually exclusive with ``date_list`` and with ``start_date``/``end_date``.
            date_list: Month strings in ``'YYYY-MM'`` format (e.g. ``['2025-06']``), like
                ``download_dw_monthly``. Mutually exclusive with ``months``.
            start_date: Within-year window start as ``'MM-DD'``, inclusive (default: '05-01'
                when neither ``months`` nor ``date_list`` is given). Mutually exclusive with
                ``months`` / ``date_list``.
            end_date: Within-year window end as ``'MM-DD'``, inclusive (default: '09-30').
            max_nodata: Mask an observation when more than this share of the polygon has no
                data (default: 0.02, i.e. 2 %). Use 1.0 to keep everything.
            min_masked_pixels: Absolute floor, in pixels, for that tolerance, so a polygon
                spanning a handful of 30 m pixels is not rejected for one missing edge pixel
                (default: 2). One pixel is ~0.09 ha at 30 m.
            bbox_west: Western boundary for spatial filtering (default: -180).
            bbox_east: Eastern boundary for spatial filtering (default: 180).
            bbox_north: Northern boundary for spatial filtering (default: 90).
            bbox_south: Southern boundary for spatial filtering (default: -90).
            id_list: Optional list of IDs to filter by (values from name_attribute column).
            scale: Pixel scale in meters (default: 30, the product's resolution).
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
                outside 0..1, if min_masked_pixels is negative, or if the date selection is
                ambiguous.
            KeyError: If name_attribute is not present in the vector dataset.
            NoDSWxDataError: If Earth Engine returned no usable observation, or every
                observation was masked by the no-data tolerance.
            ee.EEException, requests.exceptions.RequestException, ...: If an Earth Engine or
                network request fails after retrying transient errors.

        Example:
            >>> downloader = EarthEngineDownloader()
            >>> ds = downloader.download_dswx_daily(
            ...     vector_dataset="tests/data/lake_polygons.parquet",
            ...     years=[2025],
            ...     start_date="05-01",
            ...     end_date="09-30",
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
        if min_masked_pixels < 0:
            raise ValueError(f"min_masked_pixels must be zero or positive, got {min_masked_pixels}")

        if no_download:
            self._log_info("=== NO DOWNLOAD MODE - Will skip after preprocessing ===")

        years = self._validate_years_dswx(years)

        # Date selection: monthly-style (months / date_list) or a within-year MM-DD window.
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
            self._log_info(f"After bbox filter: {len(gdf)} features (removed {n_before_bbox_filter - len(gdf)})")
        else:
            self._log_info("No spatial bbox filtering applied (using default global bounds)")

        n_features = len(gdf)
        self._log_info(f"Processing {n_features} features")

        n_month_slices = sum(len(month_slices(start, end)) for start, end in windows)
        est_total_acquisitions = max(1, n_month_slices * EST_DSWX_ACQUISITIONS_PER_MONTH)
        self._log_info(
            f"Date selection {window_label}: {len(years)} year(s), {len(windows)} window(s); "
            f"~{est_total_acquisitions} acquisitions per polygon (estimate), "
            f"~{n_features * est_total_acquisitions} (polygon, acquisition) requests"
        )

        areas = self._polygon_areas_ha(gdf, name_attribute)
        n_small = sum(1 for value in areas.values() if pd.notna(value) and value < DSWX_MIN_MAPPING_AREA_HA)
        if n_small:
            self._log_warning(
                f"{n_small} of {n_features} polygon(s) are below the ~{DSWX_MIN_MAPPING_AREA_HA:.0f} ha minimum "
                "mapping unit of DSWx-S1 and will read as no-water by design"
            )

        # One reduce call carries one month of acquisitions plus tile duplicates, not the whole
        # request, so the chunk budget is sized by that.
        gdf_chunks = self._chunk_gdf(
            gdf, max_total_requests, n_dates=EST_DSWX_ACQUISITIONS_PER_MONTH, chunk_method=chunk_method
        )

        n_parallel_effective = min(n_parallel, len(gdf_chunks))
        if n_parallel_effective < n_parallel:
            target_chunks = max(1, n_parallel * 2)
            target_features = -(-n_features // target_chunks)  # ceil division
            suggested_budget = max(1, target_features * EST_DSWX_ACQUISITIONS_PER_MONTH)
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
            joblib.delayed(self._extract_dswx_daily)(
                windows=windows, gdf_chunk=chunk, name_attribute=name_attribute, scale=scale
            )
            for chunk in tqdm(gdf_chunks, desc="Downloading DSWx acquisitions", unit="chunk")
        )
        df_out_list = [df for df in df_out_list if df is not None and not df.empty]

        if not df_out_list:
            raise NoDSWxDataError(
                f"No data was extracted from any chunk: Earth Engine returned no OPERA DSWx-S1 data "
                f"for any of the {len(gdf_chunks)} chunks and {est_total_acquisitions} requested acquisitions."
            )

        df = pd.concat(df_out_list, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop(columns=["reducer"], errors="ignore")
        df = self._dswx_add_coverage(df, areas=areas, name_attribute=name_attribute, scale=scale)

        n_rows = len(df)
        df = df[df["area_data"] > 0]
        if n_rows - len(df):
            self._log_info(
                f"Dropped {n_rows - len(df)} acquisition(s) with no classified pixels "
                "(no observation, not a zero water area)"
            )

        n_rows = len(df)
        # Largest coverage first; overlapping tiles of this product return the same classified
        # area up to float noise, so coverage must be compared at a rounded precision - sorting
        # on the raw value lets a 1e-10 difference outrank the tie-break, which silently picks
        # a tile with a very different class split. The tie-break is the larger not-water area,
        # the rule the per-acquisition probe used and the only deterministic choice left
        # (row order is not reproducible).
        df["coverage_rank"] = df["coverage"].round(6)
        df = df.sort_values(["coverage_rank", "area_not_water"]).drop_duplicates([name_attribute, "date"], keep="last")
        df = df.drop(columns=["coverage_rank"])
        if n_rows - len(df):
            self._log_info(
                f"Collapsed {n_rows - len(df)} duplicate (polygon, date) row(s) from overlapping MGRS tiles, "
                "keeping the best-covered tile (tie-break: larger not-water area)"
            )

        n_rows = len(df)
        df = self._dswx_apply_tolerance(df, max_nodata=max_nodata, min_masked_pixels=min_masked_pixels)
        self._log_info(
            f"Masked {n_rows - len(df)} acquisition(s) with more than {max_nodata:.1%} no-data "
            f"(or {min_masked_pixels:g} pixel(s) on polygons too small for a percentage)"
        )

        if df.empty:
            raise NoDSWxDataError(
                "No data was extracted: every acquisition was fully masked or exceeded the "
                f"max_nodata tolerance of {max_nodata:.1%}."
            )

        ds = (
            df.drop(columns=["area_data", "area_ha", "coverage", "polygon_pixels"])
            .set_index([name_attribute, "date"])
            .to_xarray()
        )
        ds["date"] = pd.to_datetime(ds["date"])

        per_polygon = ds["area_open_water"].notnull().sum("date").to_pandas()
        self._log_info(
            f"Download complete: {len(ds.coords[name_attribute])} polygons, "
            f"{len(ds.coords['date'])} acquisitions, {int(ds['area_open_water'].notnull().sum())} observations "
            f"(per polygon: min {int(per_polygon.min())}, median {per_polygon.median():.0f}, max {int(per_polygon.max())})"
        )
        if float(per_polygon.median()) < 5:
            self._log_warning(
                "Median observations per polygon is below 5 - that is the 6-12 day revisit and the "
                "coverage limit, not a download failure"
            )

        ds = annotate_xr_dataset_dswx(
            ds, input_vector_file=vector_dataset, temporal_resolution="per_acquisition", id_field=name_attribute
        )
        ds.attrs["season_window"] = window_label
        ds.attrs["max_nodata"] = float(max_nodata)
        ds.attrs["min_masked_pixels"] = float(min_masked_pixels)
        ds.attrs["n_acquisitions"] = len(ds.coords["date"])

        if save_to_file is not None:
            save_path = Path(save_to_file).absolute()
            save_xarray_dataset(ds=ds, save_path=save_path, logger=self.logger)

        return ds

    def download_dswx_acquisitions(self, *args, **kwargs):
        """Deprecated alias for :meth:`download_dswx_daily`.

        The per-acquisition downloaders are named ``_daily`` for every source; this alias keeps the
        0.17.x name working and will be removed in a later release.
        """
        warnings.warn(
            "download_dswx_acquisitions is deprecated; use download_dswx_daily",
            FutureWarning,
            stacklevel=2,
        )
        return self.download_dswx_daily(*args, **kwargs)

    def download_dswx_monthly(
        self,
        vector_dataset: str | Path | None = None,
        gdf: gpd.GeoDataFrame | None = None,
        name_attribute: str = "id_geohash",
        years: list[int] | None = None,
        months: list[int] | None = None,
        date_list: list[str] | None = None,
        max_nodata: float = 0.02,
        min_masked_pixels: float = 2,
        bbox_west: float = -180,
        bbox_east: float = 180,
        bbox_north: float = 90,
        bbox_south: float = -90,
        id_list: list | None = None,
        scale: float = DSWX_SCALE,
        max_total_requests: int = 500,
        n_parallel: int = 1,
        chunk_method: str = "simple",
        no_download: bool = False,
        save_to_file: str | None = None,
    ) -> xr.Dataset:
        """Download monthly OPERA DSWx-S1 class areas for specified periods.

        One per-pixel mode composite of the month's acquisitions, reduced to the three class
        areas - the monthly counterpart of :meth:`download_dswx_daily`, and the same
        compositing rule as :meth:`download_dw_monthly`, so the two sources share a cadence
        and a statistic.

        The date arguments are identical to :meth:`download_dw_monthly`: either
        ``years`` with ``months``, or ``date_list`` in ``'YYYY-MM'`` format. Production starts
        in August 2024, so earlier years are skipped.

        Args:
            vector_dataset: Path to the input vector dataset (Parquet format). Mutually exclusive with `gdf`.
            gdf: Optional GeoDataFrame to use instead of loading from a path. Mutually exclusive with `vector_dataset`.
            name_attribute: Column name in the vector dataset to use for grouping.
            years: List of years to process. Default: 2024 up to the current year.
            months: List of months to process as integers (1-12). Required unless `date_list` is given.
            date_list: Optional list of months in 'YYYY-MM' format (e.g. ['2025-06']). Mutually
                exclusive with (years AND months).
            max_nodata: Mask a month when more than this share of the polygon has no data
                (default: 0.02).
            min_masked_pixels: Absolute floor, in pixels, for that tolerance (default: 2).
            bbox_west: Western boundary for spatial filtering (default: -180).
            bbox_east: Eastern boundary for spatial filtering (default: 180).
            bbox_north: Northern boundary for spatial filtering (default: 90).
            bbox_south: Southern boundary for spatial filtering (default: -90).
            id_list: Optional list of IDs to filter by (values from name_attribute column).
            scale: Pixel scale in meters (default: 30).
            max_total_requests: Maximum number of (feature, month) requests per chunk.
            n_parallel: Number of parallel workers for processing chunks (default: 1).
            chunk_method: Method for chunking the data: "simple" or "spatial_kmeans".
            no_download: If True, only log the download parameters without downloading.
            save_to_file: Optional path to save the dataset ('.zarr' or '.nc').

        Returns:
            xr.Dataset: Class areas in hectares indexed by (name_attribute, date), with
                ``date`` on the first of the month. A lake-month with no acquisition is
                absent, not zero.

        Raises:
            ValueError: If both or neither of vector_dataset/gdf is given, if max_nodata is
                outside 0..1, if min_masked_pixels is negative, or if the date arguments are
                ambiguous or incomplete.
            KeyError: If name_attribute is not present in the vector dataset.
            NoDSWxDataError: If Earth Engine returned no usable month.
            ee.EEException, requests.exceptions.RequestException, ...: If an Earth Engine or
                network request fails after retrying transient errors.

        Example:
            >>> downloader = EarthEngineDownloader()
            >>> ds = downloader.download_dswx_monthly(
            ...     vector_dataset="tests/data/lake_polygons.parquet",
            ...     years=[2025],
            ...     months=[6, 7, 8, 9],
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
        if min_masked_pixels < 0:
            raise ValueError(f"min_masked_pixels must be zero or positive, got {min_masked_pixels}")

        if no_download:
            self._log_info("=== NO DOWNLOAD MODE - Will skip after preprocessing ===")

        if date_list is not None and (years is not None or months is not None):
            self._log_warning("years and/or months parameters will be ignored since date_list is provided")
        elif date_list is None:
            years = self._validate_years_dswx(years)

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
            self._log_info(f"After bbox filter: {len(gdf)} features (removed {n_before_bbox_filter - len(gdf)})")
        else:
            self._log_info("No spatial bbox filtering applied (using default global bounds)")

        n_features = len(gdf)
        self._log_info(f"Processing {n_features} features")

        dates = setup_dates_from_options(years=years, months=months, date_list=date_list)
        n_dates = len(dates)
        self._log_info(f"Processing {n_features} features x {n_dates} months = {n_features * n_dates} total requests")
        self._log_info(f"Processing dates: {dates}")

        areas = self._polygon_areas_ha(gdf, name_attribute)
        n_small = sum(1 for value in areas.values() if pd.notna(value) and value < DSWX_MIN_MAPPING_AREA_HA)
        if n_small:
            self._log_warning(
                f"{n_small} of {n_features} polygon(s) are below the ~{DSWX_MIN_MAPPING_AREA_HA:.0f} ha minimum "
                "mapping unit of DSWx-S1 and will read as no-water by design"
            )

        gdf_chunks = self._chunk_gdf(gdf, max_total_requests, n_dates=n_dates, chunk_method=chunk_method)

        if no_download:
            self._log_info(f"Would process {len(gdf)} features with {len(gdf_chunks)} chunks")
            self._log_info("=== END NO DOWNLOAD MODE ===")
            return None

        n_parallel_effective = min(n_parallel, len(gdf_chunks))
        self._log_info(f"Processing {len(gdf_chunks)} chunks with {n_parallel_effective} workers")

        df_out_list = joblib.Parallel(n_jobs=n_parallel_effective, prefer="threads")(
            joblib.delayed(self._extract_dswx_monthly)(
                dates=dates, gdf_chunk=chunk, name_attribute=name_attribute, scale=scale
            )
            for chunk in tqdm(gdf_chunks, desc="Downloading DSWx months", unit="chunk")
        )
        df_out_list = [df for df in df_out_list if df is not None and not df.empty]

        if not df_out_list:
            raise NoDSWxDataError(
                f"No data was extracted from any chunk: Earth Engine returned no OPERA DSWx-S1 data "
                f"for any of the {len(gdf_chunks)} chunks and {n_dates} months requested."
            )

        df = pd.concat(df_out_list, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop(columns=["reducer"], errors="ignore")
        df = self._dswx_add_coverage(df, areas=areas, name_attribute=name_attribute, scale=scale)

        n_rows = len(df)
        df = df[df["area_data"] > 0]
        if n_rows - len(df):
            self._log_info(f"Dropped {n_rows - len(df)} month(s) with no classified pixels")

        n_rows = len(df)
        df = self._dswx_apply_tolerance(df, max_nodata=max_nodata, min_masked_pixels=min_masked_pixels)
        self._log_info(
            f"Masked {n_rows - len(df)} month(s) with more than {max_nodata:.1%} no-data "
            f"(or {min_masked_pixels:g} pixel(s) on polygons too small for a percentage)"
        )

        if df.empty:
            raise NoDSWxDataError(
                "No data was extracted: every month was fully masked or exceeded the "
                f"max_nodata tolerance of {max_nodata:.1%}."
            )

        ds = (
            df.drop(columns=["area_data", "area_ha", "coverage", "polygon_pixels"])
            .set_index([name_attribute, "date"])
            .to_xarray()
        )
        ds["date"] = pd.to_datetime(ds["date"])

        n_items = len(ds.coords[name_attribute])
        self._log_info(
            f"Download complete: {n_items} items, {len(ds.coords['date'])} months collected"
        )

        ds = annotate_xr_dataset_dswx(
            ds, input_vector_file=vector_dataset, temporal_resolution="monthly", id_field=name_attribute
        )
        ds.attrs["monthly_composite"] = "mode"
        ds.attrs["max_nodata"] = float(max_nodata)
        ds.attrs["min_masked_pixels"] = float(min_masked_pixels)

        if save_to_file is not None:
            save_path = Path(save_to_file).absolute()
            save_xarray_dataset(ds=ds, save_path=save_path, logger=self.logger)

        return ds

    def _setup_dswx_hls_reducer(self, gdf, feature_index_name: str, scale: float = DSWX_SCALE) -> tuple:
        """Setup the reducer configuration for OPERA DSWx-HLS.

        Same shape as :meth:`_setup_dswx_reducer`, but the band list includes the LAND layer band
        even though it is not part of ``area_data`` - it is published alongside the water classes.

        Args:
            gdf: GeoDataFrame with features to process.
            feature_index_name: Column name to use as the feature index.
            scale: Pixel scale in meters (default: 30).

        Returns:
            tuple: (fc, reducer_dict) - GEE FeatureCollection and reducer configuration.
        """
        fc = geemap.gdf_to_ee(drop_z_from_gdf(gdf[:]))
        reducer_dict = {
            "reducer": ee.Reducer.sum(),
            "collection": fc.select(feature_index_name),
            "crs": DSWX_CRS,
            "scale": scale,
            "bands": self.dswx_hls_bandnames,
        }
        return fc, reducer_dict

    def _extract_dswx_hls_daily(
        self,
        windows: list[tuple[str, str]],
        gdf_chunk,
        name_attribute: str,
        scale: float = DSWX_SCALE,
    ) -> pd.DataFrame:
        """Extract per-acquisition DSWx-HLS class areas for one chunk over several date windows.

        One calendar month per Earth Engine request, as in the DSWx-S1 extractor.

        Args:
            windows: List of ``(start, end)`` date-range pairs; the end is exclusive.
            gdf_chunk: GeoDataFrame chunk to extract data for.
            name_attribute: Column name used as the feature index.
            scale: Pixel scale in meters.

        Returns:
            pd.DataFrame: One row per (polygon, tile, acquisition); empty if nothing was found.
        """
        fc, reducer_dict = self._setup_dswx_hls_reducer(gdf_chunk, name_attribute, scale=scale)

        frames = []
        for window_start, window_end in windows:
            for slice_start, slice_end in month_slices(window_start, window_end):
                collection = dswx_hls_acquisitions(fc, slice_start, slice_end, scale=scale)
                n_images = call_with_retry(collection.size().getInfo)
                if n_images == 0:
                    self._log_info(f"No OPERA DSWx-HLS acquisitions between {slice_start} and {slice_end}")
                    continue
                self._log_info(f"{slice_start}..{slice_end}: {n_images} acquisitions")

                fc_out = collection.getTimeSeriesByRegions(**reducer_dict)
                df = call_with_retry(geemap.ee_to_df, fc_out)
                if df is None or df.empty:
                    continue
                frames.append(df)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _extract_dswx_hls_monthly(
        self,
        dates: list[str],
        gdf_chunk,
        name_attribute: str,
        scale: float = DSWX_SCALE,
    ) -> pd.DataFrame:
        """Extract monthly DSWx-HLS class areas for one chunk, one mode composite per month.

        Args:
            dates: Month starts in ``'YYYY-MM-DD'`` format.
            gdf_chunk: GeoDataFrame chunk to extract data for.
            name_attribute: Column name used as the feature index.
            scale: Pixel scale in meters.

        Returns:
            pd.DataFrame: One row per (polygon, month); empty if nothing was found.
        """
        fc, reducer_dict = self._setup_dswx_hls_reducer(gdf_chunk, name_attribute, scale=scale)

        imlist = []
        for date in dates:
            month_start = pd.Timestamp(date)
            month_end = month_start + pd.DateOffset(months=1)
            slice_start = month_start.strftime("%Y-%m-%d")
            slice_end = month_end.strftime("%Y-%m-%d")

            n_images = call_with_retry(dswx_hls_collection(fc, slice_start, slice_end).size().getInfo)
            if n_images == 0:
                self._log_info(f"No OPERA DSWx-HLS acquisitions in {slice_start[:7]}")
                continue
            composite = dswx_hls_month_composite(fc, slice_start, slice_end, scale=scale)
            imlist.append(dswx_hls_class_areas(composite))

        if not imlist:
            self._log_warning("No images processed for chunk")
            return pd.DataFrame()

        image_collection = ee.ImageCollection(imlist)
        fc_out = image_collection.getTimeSeriesByRegions(**reducer_dict)
        return call_with_retry(geemap.ee_to_df, fc_out)

    def download_dswx_hls_daily(
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
        min_masked_pixels: float = 2,
        bbox_west: float = -180,
        bbox_east: float = 180,
        bbox_north: float = 90,
        bbox_south: float = -90,
        id_list: list | None = None,
        scale: float = DSWX_SCALE,
        max_total_requests: int = 500,
        n_parallel: int = 1,
        chunk_method: str = "simple",
        no_download: bool = False,
        save_to_file: str | None = None,
    ) -> xr.Dataset:
        """Download OPERA DSWx-HLS class areas per acquisition (unaggregated).

        The optical counterpart of :meth:`download_dswx_daily`: Harmonized Landsat 8/9 +
        Sentinel-2, 30 m, one row per scene inside the selected windows. The date arguments are the
        same as for the Dynamic World and DSWx-S1 downloaders.

        The class set differs from DSWx-S1: open water (WTR 1), **partial surface water** (WTR 2)
        and not water (WTR 0); there is no inundated-vegetation class, and 253 is the mask/cloud
        value. ``area_water`` is derived later by :class:`DSWxHLSDataset` as open + partial, which
        is the product's own ``BWTR_Binary_water``. ``area_land_cover`` comes from the LAND layer and
        is an independent quantity, not the complement of the water classes.

        Being optical, cloud and snow remove observations: a scene can be almost entirely 253 and
        still exist in the collection. Tiles are collapsed to the best-covered one per (polygon,
        date) - here the fragments differ a lot, so this matters more than for S1 - fully masked
        rows are dropped, and the no-data tolerance is the same lake-size-aware rule.

        The archive has two eras with a 28-month hole (2016-03 .. 2020-09, then 2023-02 onwards);
        all years are accepted, the gap simply produces no rows.

        Either provide `vector_dataset` OR `gdf`. These options are mutually exclusive.

        Args:
            vector_dataset: Path to the input vector dataset (Parquet format).
            gdf: Optional GeoDataFrame to use directly instead of loading from a path.
            name_attribute: Column name in the vector dataset to use for grouping.
            years: Years to process. Default: 2016 up to the current year.
            months: Month numbers (1-12) to download. Mutually exclusive with ``date_list`` and
                with ``start_date``/``end_date``.
            date_list: Month strings in ``'YYYY-MM'`` format. Mutually exclusive with ``months``.
            start_date: Within-year window start as ``'MM-DD'`` (default: '05-01').
            end_date: Within-year window end as ``'MM-DD'`` (default: '09-30').
            max_nodata: Mask an observation when more than this share of the polygon has no data.
            min_masked_pixels: Absolute floor, in pixels, for that tolerance (default: 2).
            bbox_west: Western boundary for spatial filtering (default: -180).
            bbox_east: Eastern boundary for spatial filtering (default: 180).
            bbox_north: Northern boundary for spatial filtering (default: 90).
            bbox_south: Southern boundary for spatial filtering (default: -90).
            id_list: Optional list of IDs to filter by (values from name_attribute column).
            scale: Pixel scale in meters (default: 30).
            max_total_requests: Maximum number of (feature, acquisition) requests per chunk.
            n_parallel: Number of parallel workers for processing chunks (default: 1).
            chunk_method: Method for chunking the data: "simple" or "spatial_kmeans".
            no_download: If True, only log the download parameters without downloading.
            save_to_file: Optional path to save the dataset ('.zarr' or '.nc').

        Returns:
            xr.Dataset: Class areas in hectares indexed by (name_attribute, date).

        Raises:
            ValueError: If both or neither of vector_dataset/gdf is given, if max_nodata is outside
                0..1, if min_masked_pixels is negative, or if the date selection is ambiguous.
            KeyError: If name_attribute is not present in the vector dataset.
            NoDSWxHLSDataError: If Earth Engine returned no usable observation.
        """
        if vector_dataset is None and gdf is None:
            raise ValueError("Either vector_dataset or gdf must be provided")
        elif vector_dataset is not None and gdf is not None:
            raise ValueError("Only one of vector_dataset or gdf should be provided, not both")

        if not 0 <= max_nodata <= 1:
            raise ValueError(
                f"max_nodata must be between 0 and 1 (share of the polygon allowed to be no-data), got {max_nodata}"
            )
        if min_masked_pixels < 0:
            raise ValueError(f"min_masked_pixels must be zero or positive, got {min_masked_pixels}")

        if no_download:
            self._log_info("=== NO DOWNLOAD MODE - Will skip after preprocessing ===")

        years = self._validate_years_dswx(years, archive_start=DSWX_HLS_ARCHIVE_START)

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
            self._log_info(f"After bbox filter: {len(gdf)} features (removed {n_before_bbox_filter - len(gdf)})")
        else:
            self._log_info("No spatial bbox filtering applied (using default global bounds)")

        n_features = len(gdf)
        self._log_info(f"Processing {n_features} features")

        n_month_slices = sum(len(month_slices(start, end)) for start, end in windows)
        est_total_acquisitions = max(1, n_month_slices * EST_DSWX_ACQUISITIONS_PER_MONTH)
        self._log_info(
            f"Date selection {window_label}: {len(years)} year(s), {len(windows)} window(s); "
            f"~{est_total_acquisitions} acquisitions per polygon (estimate), "
            f"~{n_features * est_total_acquisitions} (polygon, acquisition) requests"
        )

        areas = self._polygon_areas_ha(gdf, name_attribute)
        n_small = sum(1 for value in areas.values() if pd.notna(value) and value < DSWX_MIN_MAPPING_AREA_HA)
        if n_small:
            self._log_warning(
                f"{n_small} of {n_features} polygon(s) are below the ~{DSWX_MIN_MAPPING_AREA_HA:.0f} ha minimum "
                "mapping unit of DSWx and will read as no-water by design"
            )

        gdf_chunks = self._chunk_gdf(
            gdf, max_total_requests, n_dates=EST_DSWX_ACQUISITIONS_PER_MONTH, chunk_method=chunk_method
        )

        n_parallel_effective = min(n_parallel, len(gdf_chunks))
        if n_parallel_effective < n_parallel:
            target_chunks = max(1, n_parallel * 2)
            target_features = -(-n_features // target_chunks)  # ceil division
            suggested_budget = max(1, target_features * EST_DSWX_ACQUISITIONS_PER_MONTH)
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
            joblib.delayed(self._extract_dswx_hls_daily)(
                windows=windows, gdf_chunk=chunk, name_attribute=name_attribute, scale=scale
            )
            for chunk in tqdm(gdf_chunks, desc="Downloading DSWx-HLS acquisitions", unit="chunk")
        )
        df_out_list = [df for df in df_out_list if df is not None and not df.empty]

        if not df_out_list:
            raise NoDSWxHLSDataError(
                f"No data was extracted from any chunk: Earth Engine returned no OPERA DSWx-HLS data "
                f"for any of the {len(gdf_chunks)} chunks and {est_total_acquisitions} requested acquisitions."
            )

        cover_columns = [c for c in self.dswx_hls_bandnames if c != "area_land_cover"]
        df = pd.concat(df_out_list, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop(columns=["reducer"], errors="ignore")
        df = self._dswx_add_coverage(
            df, areas=areas, name_attribute=name_attribute, scale=scale, cover_columns=cover_columns
        )

        n_rows = len(df)
        df = df[df["area_data"] > 0]
        if n_rows - len(df):
            self._log_info(
                f"Dropped {n_rows - len(df)} acquisition(s) with no classified pixels "
                "(cloud-covered or a tile that only clips the polygon)"
            )

        n_rows = len(df)
        df["coverage_rank"] = df["coverage"].round(6)
        df = df.sort_values(["coverage_rank", "area_not_water"]).drop_duplicates([name_attribute, "date"], keep="last")
        df = df.drop(columns=["coverage_rank"])
        if n_rows - len(df):
            self._log_info(
                f"Collapsed {n_rows - len(df)} duplicate (polygon, date) row(s) from overlapping MGRS tiles, "
                "keeping the best-covered tile (tie-break: larger not-water area)"
            )

        n_rows = len(df)
        df = self._dswx_apply_tolerance(df, max_nodata=max_nodata, min_masked_pixels=min_masked_pixels)
        self._log_info(
            f"Masked {n_rows - len(df)} acquisition(s) with more than {max_nodata:.1%} no-data "
            f"(or {min_masked_pixels:g} pixel(s) on polygons too small for a percentage)"
        )

        if df.empty:
            raise NoDSWxHLSDataError(
                "No data was extracted: every acquisition was fully masked (cloud) or exceeded the "
                f"max_nodata tolerance of {max_nodata:.1%}."
            )

        ds = (
            df.drop(columns=["area_data", "area_ha", "coverage", "polygon_pixels"])
            .set_index([name_attribute, "date"])
            .to_xarray()
        )
        ds["date"] = pd.to_datetime(ds["date"])

        per_polygon = ds["area_open_water"].notnull().sum("date").to_pandas()
        self._log_info(
            f"Download complete: {len(ds.coords[name_attribute])} polygons, "
            f"{len(ds.coords['date'])} acquisitions, {int(ds['area_open_water'].notnull().sum())} observations "
            f"(per polygon: min {int(per_polygon.min())}, median {per_polygon.median():.0f}, max {int(per_polygon.max())})"
        )

        ds = annotate_xr_dataset_dswx_hls(
            ds,
            input_vector_file=vector_dataset,
            temporal_resolution="per_acquisition",
            id_field=name_attribute,
        )
        ds.attrs["season_window"] = window_label
        ds.attrs["max_nodata"] = float(max_nodata)
        ds.attrs["min_masked_pixels"] = float(min_masked_pixels)
        ds.attrs["n_acquisitions"] = len(ds.coords["date"])

        if save_to_file is not None:
            save_path = Path(save_to_file).absolute()
            save_xarray_dataset(ds=ds, save_path=save_path, logger=self.logger)

        return ds

    def download_dswx_hls_acquisitions(self, *args, **kwargs):
        """Deprecated alias for :meth:`download_dswx_hls_daily`.

        The per-acquisition downloaders are named ``_daily`` for every source; this alias keeps the
        0.17.x name working and will be removed in a later release.
        """
        warnings.warn(
            "download_dswx_hls_acquisitions is deprecated; use download_dswx_hls_daily",
            FutureWarning,
            stacklevel=2,
        )
        return self.download_dswx_hls_daily(*args, **kwargs)

    def download_dswx_hls_monthly(
        self,
        vector_dataset: str | Path | None = None,
        gdf: gpd.GeoDataFrame | None = None,
        name_attribute: str = "id_geohash",
        years: list[int] | None = None,
        months: list[int] | None = None,
        date_list: list[str] | None = None,
        max_nodata: float = 0.02,
        min_masked_pixels: float = 2,
        bbox_west: float = -180,
        bbox_east: float = 180,
        bbox_north: float = 90,
        bbox_south: float = -90,
        id_list: list | None = None,
        scale: float = DSWX_SCALE,
        max_total_requests: int = 500,
        n_parallel: int = 1,
        chunk_method: str = "simple",
        no_download: bool = False,
        save_to_file: str | None = None,
    ) -> xr.Dataset:
        """Download monthly OPERA DSWx-HLS class areas for specified periods.

        One per-pixel mode composite per month, the same statistic as
        :meth:`download_dswx_monthly` and :meth:`download_dw_monthly`. A month without a usable
        acquisition is absent, not zero - with an optical product that happens often in the Arctic,
        so a lake's series can have sizeable holes that are sky, not hydrology.

        Args:
            vector_dataset: Path to the input vector dataset (Parquet format).
            gdf: Optional GeoDataFrame to use instead of loading from a path.
            name_attribute: Column name in the vector dataset to use for grouping.
            years: List of years to process. Default: 2016 up to the current year.
            months: List of months as integers (1-12). Required unless `date_list` is given.
            date_list: Optional list of months in 'YYYY-MM' format. Mutually exclusive with
                (years AND months).
            max_nodata: Mask a month when more than this share of the polygon has no data.
            min_masked_pixels: Absolute floor, in pixels, for that tolerance (default: 2).
            bbox_west: Western boundary for spatial filtering (default: -180).
            bbox_east: Eastern boundary for spatial filtering (default: 180).
            bbox_north: Northern boundary for spatial filtering (default: 90).
            bbox_south: Southern boundary for spatial filtering (default: -90).
            id_list: Optional list of IDs to filter by (values from name_attribute column).
            scale: Pixel scale in meters (default: 30).
            max_total_requests: Maximum number of (feature, month) requests per chunk.
            n_parallel: Number of parallel workers for processing chunks (default: 1).
            chunk_method: Method for chunking the data: "simple" or "spatial_kmeans".
            no_download: If True, only log the download parameters without downloading.
            save_to_file: Optional path to save the dataset ('.zarr' or '.nc').

        Returns:
            xr.Dataset: Class areas in hectares indexed by (name_attribute, date), with ``date`` on
                the first of the month.

        Raises:
            ValueError: If both or neither of vector_dataset/gdf is given, if max_nodata is outside
                0..1, if min_masked_pixels is negative, or if the date arguments are incomplete.
            KeyError: If name_attribute is not present in the vector dataset.
            NoDSWxHLSDataError: If Earth Engine returned no usable month.
        """
        if vector_dataset is None and gdf is None:
            raise ValueError("Either vector_dataset or gdf must be provided")
        elif vector_dataset is not None and gdf is not None:
            raise ValueError("Only one of vector_dataset or gdf should be provided, not both")

        if not 0 <= max_nodata <= 1:
            raise ValueError(
                f"max_nodata must be between 0 and 1 (share of the polygon allowed to be no-data), got {max_nodata}"
            )
        if min_masked_pixels < 0:
            raise ValueError(f"min_masked_pixels must be zero or positive, got {min_masked_pixels}")

        if no_download:
            self._log_info("=== NO DOWNLOAD MODE - Will skip after preprocessing ===")

        if date_list is not None and (years is not None or months is not None):
            self._log_warning("years and/or months parameters will be ignored since date_list is provided")
        elif date_list is None:
            years = self._validate_years_dswx(years, archive_start=DSWX_HLS_ARCHIVE_START)

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
            self._log_info(f"After bbox filter: {len(gdf)} features (removed {n_before_bbox_filter - len(gdf)})")
        else:
            self._log_info("No spatial bbox filtering applied (using default global bounds)")

        n_features = len(gdf)
        self._log_info(f"Processing {n_features} features")

        dates = setup_dates_from_options(years=years, months=months, date_list=date_list)
        n_dates = len(dates)
        self._log_info(f"Processing {n_features} features x {n_dates} months = {n_features * n_dates} total requests")
        self._log_info(f"Processing dates: {dates}")

        areas = self._polygon_areas_ha(gdf, name_attribute)

        gdf_chunks = self._chunk_gdf(gdf, max_total_requests, n_dates=n_dates, chunk_method=chunk_method)

        if no_download:
            self._log_info(f"Would process {len(gdf)} features with {len(gdf_chunks)} chunks")
            self._log_info("=== END NO DOWNLOAD MODE ===")
            return None

        n_parallel_effective = min(n_parallel, len(gdf_chunks))
        self._log_info(f"Processing {len(gdf_chunks)} chunks with {n_parallel_effective} workers")

        df_out_list = joblib.Parallel(n_jobs=n_parallel_effective, prefer="threads")(
            joblib.delayed(self._extract_dswx_hls_monthly)(
                dates=dates, gdf_chunk=chunk, name_attribute=name_attribute, scale=scale
            )
            for chunk in tqdm(gdf_chunks, desc="Downloading DSWx-HLS months", unit="chunk")
        )
        df_out_list = [df for df in df_out_list if df is not None and not df.empty]

        if not df_out_list:
            raise NoDSWxHLSDataError(
                f"No data was extracted from any chunk: Earth Engine returned no OPERA DSWx-HLS data "
                f"for any of the {len(gdf_chunks)} chunks and {n_dates} months requested."
            )

        cover_columns = [c for c in self.dswx_hls_bandnames if c != "area_land_cover"]
        df = pd.concat(df_out_list, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop(columns=["reducer"], errors="ignore")
        df = self._dswx_add_coverage(
            df, areas=areas, name_attribute=name_attribute, scale=scale, cover_columns=cover_columns
        )

        n_rows = len(df)
        df = df[df["area_data"] > 0]
        if n_rows - len(df):
            self._log_info(f"Dropped {n_rows - len(df)} month(s) with no classified pixels")

        n_rows = len(df)
        df = self._dswx_apply_tolerance(df, max_nodata=max_nodata, min_masked_pixels=min_masked_pixels)
        self._log_info(
            f"Masked {n_rows - len(df)} month(s) with more than {max_nodata:.1%} no-data "
            f"(or {min_masked_pixels:g} pixel(s) on polygons too small for a percentage)"
        )

        if df.empty:
            raise NoDSWxHLSDataError(
                "No data was extracted: every month was fully masked or exceeded the "
                f"max_nodata tolerance of {max_nodata:.1%}."
            )

        ds = (
            df.drop(columns=["area_data", "area_ha", "coverage", "polygon_pixels"])
            .set_index([name_attribute, "date"])
            .to_xarray()
        )
        ds["date"] = pd.to_datetime(ds["date"])

        n_items = len(ds.coords[name_attribute])
        self._log_info(f"Download complete: {n_items} items, {len(ds.coords['date'])} months collected")

        ds = annotate_xr_dataset_dswx_hls(
            ds, input_vector_file=vector_dataset, temporal_resolution="monthly", id_field=name_attribute
        )
        ds.attrs["monthly_composite"] = "mode"
        ds.attrs["max_nodata"] = float(max_nodata)
        ds.attrs["min_masked_pixels"] = float(min_masked_pixels)

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
        ds = annotate_xr_dataset_jrc(ds, input_vector_file=vector_dataset, id_field=name_attribute)

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
