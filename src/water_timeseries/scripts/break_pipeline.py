# imports
# imports
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import ray
import typer
import xarray as xr
from loguru import logger
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from water_timeseries.breakpoint import BeastBreakpoint
from water_timeseries.dataset import DWDataset, JRCDataset

# configure logger: writes to rbeast_batch.log in current working dir
_log_file = Path.cwd() / "rbeast_batch.log"
logger.add(
    _log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} {level} {extra[script]}:{function}:{line} - {message}",
    mode="a",
)
# Bind script filename so logs show script name instead of __main__
logger = logger.bind(script=Path(__file__).name)
app = typer.Typer(help="Run Rbeast break detection on Dynamic World lakes")


@ray.remote
def process_chunk_remote(chunk, water_dataset_type):
    """Ray remote function to process a single chunk."""
    if water_dataset_type == "dynamic_world":
        ds = DWDataset(chunk)
    elif water_dataset_type == "jrc":
        ds = JRCDataset(chunk)
    else:
        raise ValueError(f"Unknown water dataset type: {water_dataset_type}")

    bp = BeastBreakpoint()
    return bp.calculate_breaks_batch(ds)


class BreakpointPipeline:
    def __init__(
        self,
        # lake_vector_file: str,
        water_dataset_file: str,
        output_file: str,
        vector_dataset_file: Optional[str] = None,
        n_chunks: Optional[int] = 1,
        logger: Optional[logger] = None,
        min_chunksize: int = 10,
    ):
        self.water_dataset_file = water_dataset_file
        self.output_file = output_file
        self.vector_dataset_file = vector_dataset_file
        self.n_chunks = n_chunks
        self.min_chunksize = min_chunksize
        self.logger = logger
        if logger:
            self.logger.info(
                f"Initialized BreakpointPipeline with \nwater dataset: {self.water_dataset_file} \noutput file: {self.output_file} \nn_chunks: {self.n_chunks}"
            )
        self.input_ds = self.load_water_data()
        self.get_water_dataset_type()
        self.has_vector_dataset = False
        self.gdf = self.load_vector_data()

        self.chunked_ds = self.chunk_dataset()

    def load_water_data(self):
        # load data
        return xr.open_zarr(self.water_dataset_file)

    def save_to_parquet(self):
        output_file = Path(self.output_file)
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

        Supports gpkg, shp, and other geopandas formats.
        """
        if self.vector_dataset_file is not None:
            vector_path = Path(self.vector_dataset_file)
            suffix = vector_path.suffix.lower()
            
            if self.logger:
                self.logger.info(f"Loading vector dataset from {self.vector_dataset_file}")

            if suffix in [".gpkg", ".shp", ".geojson", ".gjson"]:
                vector_ds = gpd.read_file(self.vector_dataset_file)
            elif suffix in [".parquet"]:
                vector_ds = gpd.read_parquet(self.vector_dataset_file)
            else:
                if self.logger:
                    self.logger.warning(f"Unsupported vector file format: {suffix}")
                return None

            self.has_vector_dataset_ = True
            return vector_ds
        else:
            if self.logger:
                self.logger.info("No vector dataset file provided, skipping vector data loading.")
            return None

    #  optionally restrict to lakes whose centroids fall inside the provided bbox
    def bbox_filter(
        gdf: gpd.GeoDataFrame,
        bbox_west: float = None,
        bbox_east: float = None,
        bbox_south: float = None,
        bbox_north: float = None,
    ) -> gpd.GeoDataFrame:
        if any(v is not None for v in (bbox_west, bbox_east, bbox_south, bbox_north)):
            cent = gdf.geometry.centroid
            mask = True
            if bbox_west is not None:
                mask &= cent.x >= bbox_west
            if bbox_east is not None:
                mask &= cent.x <= bbox_east
            if bbox_south is not None:
                mask &= cent.y >= bbox_south
            if bbox_north is not None:
                mask &= cent.y <= bbox_north
            filtered = gdf[mask]
        else:
            filtered = gdf
        return filtered

    def chunk_dataset(self) -> list[xr.Dataset]:
        """Split xarray dataset into n chunks along id_geohash dimension.

        Args:
            ds: xarray Dataset with id_geohash dimension
            n_chunks: Number of chunks to create

        Returns:
            List of xarray Datasets
        """

        n_ids = len(self.input_ds.id_geohash)

        # Check if n_ids is smaller than min_chunksize
        if n_ids < self.min_chunksize:
            # Create only one chunk containing all data
            self.n_chunks = 1
            chunk = self.input_ds.isel(id_geohash=slice(0, n_ids))
            if self.logger:
                self.logger.info(
                    f"Dataset has only {n_ids} ids (less than min_chunksize={self.min_chunksize}). Creating single chunk."
                )
            return [chunk]

        # Proceed with normal chunking
        chunk_size = max(self.min_chunksize, n_ids // self.n_chunks)
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
        # Initialize Ray if using parallel processing
        if self.n_chunks > 1:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            if self.logger:
                self.logger.info(f"Starting parallel processing with Ray using {self.n_chunks} chunks")
        else:
            if self.logger:
                self.logger.info("Starting sequential processing (n_chunks=1)")

        # load data
        break_list = []

        # Progress bar with rich
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Processing chunks...", total=len(self.chunked_ds))

            if self.n_chunks > 1:
                # Parallel processing with Ray
                # Submit all tasks
                futures = [process_chunk_remote.remote(chunk, self.water_dataset_type) for chunk in self.chunked_ds]
                # Collect results with progress updates
                for future in futures:
                    result = ray.get(future)
                    break_list.append(result)
                    progress.advance(task)
            else:
                # Sequential processing
                for chunk in self.chunked_ds:
                    if self.water_dataset_type == "dynamic_world":
                        ds = DWDataset(chunk)
                    elif self.water_dataset_type == "jrc":
                        ds = JRCDataset(chunk)
                    bp = BeastBreakpoint()
                    break_list.append(bp.calculate_breaks_batch(ds))
                    progress.advance(task)

        self.breaks = pd.concat(break_list, axis=0)
        print(self.breaks)


@app.command()
def main(
    water_dataset_file: str = None,
    output_file: str = None,
    vector_dataset_file: str = None,
):
    """
    Example usage:
    """
    # Example usage
    pipeline = BreakpointPipeline(
        water_dataset_file=water_dataset_file,
        output_file=output_file,
        vector_dataset_file=vector_dataset_file,
        n_chunks=10,
        logger=logger,
    )
    pipeline.run_breaks()
    pipeline.save_to_parquet()


if __name__ == "__main__":
    app()
