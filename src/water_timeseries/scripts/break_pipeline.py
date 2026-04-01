# imports
import json
import os
import threading
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import ray
import xarray as xr
import yaml
from loguru import logger
from tqdm import tqdm

from water_timeseries.breakpoint import BeastBreakpoint
from water_timeseries.dataset import DWDataset, JRCDataset
from water_timeseries.utils.io import load_vector_dataset
from water_timeseries.utils.spatial import filter_gdf_by_bbox


def load_config(config_path: Optional[Path]) -> dict:
    """Load configuration from YAML or JSON file.

    Args:
        config_path: Path to config file.

    Returns:
        Dictionary with configuration values.
    """
    if not config_path or not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            if config_path.suffix in (".yaml", ".yml"):
                return yaml.safe_load(f) or {}
            elif config_path.suffix == ".json":
                return json.load(f)
    except Exception as e:
        if logger:
            logger.warning(f"Failed to load config file {config_path}: {e}")
    return {}


def merge_config_with_args(config: dict, **kwargs) -> dict:
    """Merge config with CLI args, CLI args take priority.

    Args:
        config: Configuration dictionary from config file.
        **kwargs: CLI arguments (None values are ignored).

    Returns:
        Merged dictionary with CLI args taking priority.
    """
    result = config.copy()
    for key, value in kwargs.items():
        if value is not None:
            result[key] = value
    return result


@ray.remote
def process_chunk_remote(chunk: xr.Dataset, water_dataset_type: str, break_method: str = "beast") -> pd.DataFrame:
    """Ray remote function to process a single chunk."""
    from water_timeseries.breakpoint import SimpleBreakpoint

    if water_dataset_type == "dynamic_world":
        ds = DWDataset(chunk)
    elif water_dataset_type == "jrc":
        ds = JRCDataset(chunk)
    else:
        raise ValueError(f"Unknown water dataset type: {water_dataset_type}")

    if break_method == "beast":
        bp = BeastBreakpoint()
    elif break_method == "simple":
        bp = SimpleBreakpoint()
    else:
        raise ValueError(f"Unknown break method: {break_method}")

    return bp.calculate_breaks_batch(ds)


def process_chunk(chunk: xr.Dataset, water_dataset_type: str, break_method: str = "beast") -> pd.DataFrame:
    """Process a single chunk (for use with joblib)."""
    from water_timeseries.breakpoint import SimpleBreakpoint

    if water_dataset_type == "dynamic_world":
        ds = DWDataset(chunk)
    elif water_dataset_type == "jrc":
        ds = JRCDataset(chunk)
    else:
        raise ValueError(f"Unknown water dataset type: {water_dataset_type}")

    if break_method == "beast":
        bp = BeastBreakpoint()
    elif break_method == "simple":
        bp = SimpleBreakpoint()
    else:
        raise ValueError(f"Unknown break method: {break_method}")

    return bp.calculate_breaks_batch(ds)


class BreakpointPipeline:
    """Pipeline for running Rbeast break detection on water dataset time series.

    The pipeline handles loading data from zarr or parquet, optional bounding box
    filtering, chunking the dataset, and running break detection either sequentially
    or in parallel using Ray.

    Args:
        water_dataset_file: Path to water dataset (zarr or parquet format).
        output_file: Path to output parquet file for results.
        vector_dataset_file: Optional path to vector dataset (gpkg, shp, geojson).
        chunksize: Number of IDs per chunk (default: 100).
        parallel_backend: Parallelization backend (default: ray)
        break_method: Breakpoint detection method (default: "beast", also "simple").
        n_jobs: Number of parallel jobs for Ray (default: 1, sequential).
        min_chunksize: Minimum chunk size (default: 10).
        bbox_west: Minimum longitude for bbox filter.
        bbox_south: Minimum latitude for bbox filter.
        bbox_east: Maximum longitude for bbox filter.
        bbox_north: Maximum latitude for bbox filter.
        logger: Optional logger instance.
    """

    def __init__(
        self,
        water_dataset_file: str,
        output_file: str,
        vector_dataset_file: Optional[str] = None,
        n_chunks: int = 1,
        chunksize: int = 100,
        parallel_backend: str = "ray",
        break_method: str = "beast",
        n_jobs: int = 1,
        logger: Optional[logger] = None,
        min_chunksize: int = 10,
        bbox_west: Optional[float] = None,
        bbox_south: Optional[float] = None,
        bbox_east: Optional[float] = None,
        bbox_north: Optional[float] = None,
        output_geometry: bool = True,
        output_geometry_all: bool = False,
    ):
        self.water_dataset_file = water_dataset_file
        self.output_file = output_file
        self.vector_dataset_file = vector_dataset_file
        self.n_chunks = n_chunks
        self.chunksize = chunksize
        # Handle n_jobs=-1 to use all available CPUs
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        # Store initial n_jobs for logging before we potentially reduce it
        self._requested_n_jobs = n_jobs
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.min_chunksize = min_chunksize
        self.bbox_west = bbox_west
        self.bbox_south = bbox_south
        self.bbox_east = bbox_east
        self.bbox_north = bbox_north
        self.logger = logger
        self.process_ids = None
        self.breaks = None
        self.output_geometry = output_geometry
        self.output_geometry_all = output_geometry_all

        self.input_ds = self.load_water_data()
        self.get_water_dataset_type()
        self.has_vector_dataset = False
        self.gdf = self.load_vector_data()

        # Apply bbox filter if provided
        if any(v is not None for v in [bbox_west, bbox_south, bbox_east, bbox_north]):
            self.input_ds = self.apply_bbox_filter()

        # Store break method
        self.break_method = break_method

        self.chunked_ds = self.chunk_dataset()

        # Limit n_jobs to the number of chunks if it exceeds
        if self.n_jobs > len(self.chunked_ds):
            self.n_jobs = len(self.chunked_ds)
            if logger:
                logger.info(f"Reduced n_jobs to {self.n_jobs} (number of chunks)")

        # Log initialization if logger is provided
        if logger:
            # Show requested n_jobs if it was reduced
            n_jobs_info = self._requested_n_jobs
            if self._requested_n_jobs > self.n_jobs:
                n_jobs_info = f"{self._requested_n_jobs} -> {self.n_jobs} (limited by chunks)"
            logger.info(
                f"Initialized BreakpointPipeline with:\n"
                f"water_dataset={self.water_dataset_file}\n"
                f"output_file={self.output_file}\n"
                f"n_chunks={self.n_chunks}\n"
                f"chunksize={self.chunksize}\n"
                f"n_jobs={n_jobs_info}\n"
                f"parallel_backend={self.parallel_backend}\n"
                f"break_method={self.break_method}\n"
            )

    def load_water_data(self) -> xr.Dataset:
        """Load water dataset from zarr file.

        Returns:
            xarray Dataset with water time series data.
        """
        ds = xr.open_zarr(self.water_dataset_file)
        self.process_ids = ds.id_geohash.values
        return ds

    def save_to_parquet(self):
        """Save break detection results to parquet file."""
        output_file = Path(self.output_file)

        # join geospatial data
        if self.gdf is not None and "id_geohash" in self.gdf.columns and self.output_geometry:
            if self.logger:
                self.logger.info("Joining break results with vector dataset geometries")
            local_gdf = self.gdf.set_index("id_geohash").loc[self.process_ids][["geometry"]]
            if self.output_geometry_all:
                joined = local_gdf.join(self.breaks, how="left").reset_index(drop=False)
            else:
                joined = local_gdf.join(self.breaks, how="inner").reset_index(drop=False)
            joined.to_parquet(output_file)
        else:
            if self.logger:
                self.logger.info("Saving break results without vector dataset geometries")
            self.breaks.to_parquet(output_file)

    def get_water_dataset_type(self) -> str:
        """Determine the water dataset type based on the presence of specific variables in the dataset."""
        if "area_water_permanent" in self.input_ds.data_vars:
            self.water_dataset_type = "jrc"
        elif "water" in self.input_ds.data_vars:
            self.water_dataset_type = "dynamic_world"
        else:
            raise ValueError("Unknown water dataset type")
        if self.logger:
            self.logger.info(f"Determined water dataset type: {self.water_dataset_type}")

    def load_vector_data(self):
        """Load vector dataset from file.

        Uses the reusable load_vector_dataset function from utils.data.
        """
        if self.vector_dataset_file is not None:
            vector_ds = load_vector_dataset(self.vector_dataset_file, logger=self.logger)

            self.has_vector_dataset_ = True
            return vector_ds
        else:
            if self.logger:
                self.logger.info("No vector dataset file provided, skipping vector data loading.")
            return None

    #  optionally restrict to lakes whose centroids fall inside the provided bbox
    def apply_bbox_filter(self) -> xr.Dataset:
        """Apply bounding box filter to the dataset.

        Filters the dataset based on the provided bounding box coordinates.
        Uses the vector dataset geometry if available, otherwise returns the full dataset.
        """
        if self.logger:
            self.logger.info(
                f"Applying bbox filter: west={self.bbox_west}, south={self.bbox_south}, "
                f"east={self.bbox_east}, north={self.bbox_north}"
            )

        # Check if we have a vector dataset with geometry to filter by
        if self.gdf is not None and self.has_vector_dataset_:
            gdf = self.gdf

            # Find overlapping IDs between gdf and ds
            overlap_ids = (
                gdf[["id_geohash"]]
                .set_index("id_geohash")
                .join(self.input_ds.coords["id_geohash"].to_dataframe(), how="inner")["id_geohash"]
                .tolist()
            )

            # Apply bbox filter on the geodataframe using the reusable function
            if any(v is not None for v in (self.bbox_west, self.bbox_east, self.bbox_south, self.bbox_north)):
                filtered_gdf = filter_gdf_by_bbox(
                    gdf,
                    bbox_west=self.bbox_west,
                    bbox_south=self.bbox_south,
                    bbox_east=self.bbox_east,
                    bbox_north=self.bbox_north,
                    id_column="id_geohash",
                )

                # Get overlapping IDs after bbox filter
                filtered_overlap_ids = (
                    filtered_gdf[["id_geohash"]]
                    .set_index("id_geohash")
                    .join(self.input_ds.coords["id_geohash"].to_dataframe(), how="inner")["id_geohash"]
                    .tolist()
                )
                geohash_ids = filtered_overlap_ids
            else:
                geohash_ids = overlap_ids

            # Filter the xarray dataset
            self.input_ds = self.input_ds.sel(id_geohash=geohash_ids)
            self.process_ids = geohash_ids

            if self.logger:
                self.logger.info(f"Filtered dataset to {len(geohash_ids)} geohashes")
        else:
            # No vector dataset available - log warning
            if self.logger:
                self.logger.warning("BBox filtering requires vector dataset with geometries - skipping filter")

        return self.input_ds

    def chunk_dataset(self) -> list[xr.Dataset]:
        """Split xarray dataset into chunks along id_geohash dimension.

        Uses chunksize to determine the size of each chunk. The total number
        of chunks is calculated automatically based on the dataset size.

        Returns:
            List of xarray Datasets, one per chunk.
        """
        n_ids = len(self.input_ds.id_geohash)

        # Check if n_ids is smaller than chunksize
        if n_ids <= self.chunksize:
            # Create only one chunk containing all data
            self.n_chunks = 1
            chunk = self.input_ds.isel(id_geohash=slice(0, n_ids))
            if self.logger:
                self.logger.info(
                    f"Dataset has only {n_ids} ids (less than or equal to chunksize={self.chunksize}). Creating single chunk."
                )
            return [chunk]

        # Calculate number of chunks based on chunksize
        self.n_chunks = max(1, (n_ids + self.chunksize - 1) // self.chunksize)
        chunk_size = self.chunksize
        chunks = []

        for i in range(self.n_chunks):
            start_idx = i * chunk_size
            if i == self.n_chunks - 1:
                # Last chunk gets remaining ids
                end_idx = n_ids
            else:
                end_idx = (i + 1) * chunk_size

            chunk = self.input_ds.isel(id_geohash=slice(start_idx, end_idx))
            if len(chunk.id_geohash) > 0:
                chunks.append(chunk)

        self.n_chunks = len(chunks)
        if self.logger:
            self.logger.info(f"Chunking dataset with {n_ids} ids into {self.n_chunks} chunks of size {chunk_size}")
        return chunks

    def run_breaks(self):
        """Run break detection on chunked dataset.

        Processes the dataset either sequentially or in parallel using Ray,
        depending on the n_jobs setting. Shows progress with rich progress bar.
        """
        # Determine processing mode based on n_jobs
        use_parallel = self.n_jobs > 1

        # Initialize Ray if using parallel processing
        if use_parallel and (self.parallel_backend == "ray"):
            if not ray.is_initialized():
                # Copy environment and remove VIRTUAL_ENV to avoid the warning
                env_vars = os.environ.copy()
                # Remove VIRTUAL_ENV to let Ray create its own environment
                env_vars.pop("VIRTUAL_ENV", None)
                # Also set UV_LINK_MODE to avoid hardlink issues
                env_vars["UV_LINK_MODE"] = "copy"

                ray.init(
                    ignore_reinit_error=True,
                    num_cpus=self.n_jobs,
                    runtime_env={
                        "env_vars": env_vars,
                        "RAY_LOG_TO_STDERR": "0",
                        "RAY_DEDUP_LOGS": 0,
                    },  # Suppress Ray process PID lines
                    logging_level="WARNING",  # Only show WARNING and ERROR level logs from Ray workers
                    include_dashboard=False,  # Disable Ray dashboard to reduce output
                    log_to_driver=False,
                    object_store_memory=10*1e9
                )
            if self.logger:
                logger.info(
                    f"Starting parallel processing with Ray using {self.n_jobs} jobs (backend={self.parallel_backend})"
                )
        elif use_parallel and (self.parallel_backend == "joblib"):
            if self.logger:
                logger.info(
                    f"Starting parallel processing with joblib using {self.n_jobs} jobs (backend={self.parallel_backend})"
                )
        else:
            if self.logger:
                logger.info(f"Starting sequential processing (n_jobs={self.n_jobs}, backend={self.parallel_backend})")

        # load data
        break_list = []

        # For joblib, use tqdm with callback; for ray/sequential, use rich Progress
        if use_parallel and self.parallel_backend == "joblib":
            # joblib parallel processing with tqdm progress bar
            # Use threading backend so tqdm progress bar works across workers
            pbar = tqdm(total=len(self.chunked_ds), desc="Processing chunks", unit="chunk")
            pbar_lock = threading.Lock()

            def process_with_pbar(chunk):
                result = process_chunk(chunk, self.water_dataset_type, self.break_method)
                with pbar_lock:
                    pbar.update(1)
                return result

            break_list = joblib.Parallel(n_jobs=self.n_jobs, backend="threading")(
                joblib.delayed(process_with_pbar)(chunk) for chunk in self.chunked_ds
            )
            pbar.close()
        else:
            # Progress bar with tqdm for Ray, rich for sequential
            if use_parallel and self.parallel_backend == "ray":
                # Ray parallel processing with tqdm progress bar
                pbar = tqdm(total=len(self.chunked_ds), desc="Processing chunks", unit="chunk")

                # Submit all tasks
                futures = [
                    process_chunk_remote.remote(chunk, self.water_dataset_type, self.break_method)
                    for chunk in self.chunked_ds
                ]
                # Collect results with progress updates
                for future in futures:
                    result = ray.get(future)
                    break_list.append(result)
                    pbar.update(1)
                pbar.close()
            else:
                # Sequential processing with tqdm
                from water_timeseries.breakpoint import SimpleBreakpoint

                pbar = tqdm(total=len(self.chunked_ds), desc="Processing chunks", unit="chunk")

                # Sequential processing
                for chunk in self.chunked_ds:
                    if self.water_dataset_type == "dynamic_world":
                        ds = DWDataset(chunk)
                    elif self.water_dataset_type == "jrc":
                        ds = JRCDataset(chunk)

                    if self.break_method == "beast":
                        bp = BeastBreakpoint()
                    elif self.break_method == "simple":
                        bp = SimpleBreakpoint()

                    break_list.append(bp.calculate_breaks_batch(ds))
                    pbar.update(1)
                pbar.close()

        self.breaks = pd.concat(break_list, axis=0)
        if self.logger:
            self.logger.info(f"Processed {len(self.breaks)} breakpoints")
