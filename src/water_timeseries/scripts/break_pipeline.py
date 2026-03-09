# imports
# imports
from pathlib import Path
from typing import Optional

import geopandas as gpd
import typer
import xarray as xr
from loguru import logger

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


class BreakpointPipeline:
    def __init__(
        self,
        # lake_vector_file: str,
        water_dataset_file: str,
        output_file: str,
        n_chunks: Optional[int] = None,
    ):
        self.water_dataset_file = water_dataset_file
        self.output_file = output_file
        self.n_chunks = n_chunks
        self.input_ds = self.load_water_data()
        self.get_water_dataset_type()

    def load_water_data(self):
        # load data
        return xr.open_zarr(self.water_dataset_file)

    def run_breaks(self):
        # load data
        if self.water_dataset_type == "dynamic_world":
            ds = DWDataset(self.input_ds)
        elif self.water_dataset_type == "jrc":
            ds = JRCDataset(self.input_ds)
        bp = BeastBreakpoint()
        self.breaks = bp.calculate_breaks_batch(ds)
        print(self.breaks)

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
        print(f"Determined water dataset type: {self.water_dataset_type}")

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


@app.command()
def main(
    water_dataset_file: str = "/isipd/projects/Response/GIS_RS_projects/Ingmar_other/water-timeseries-v2/tests/data/lakes_dw_test.zarr",
    output_file: str = "/isipd/projects/Response/GIS_RS_projects/Ingmar_other/water-timeseries-v2/test.parquet",
):
    """
    Example usage:
    """
    # Example usage
    pipeline = BreakpointPipeline(
        water_dataset_file=water_dataset_file,
        output_file=output_file,
        n_chunks=10,
    )
    pipeline.run_breaks()
    pipeline.save_to_parquet()


if __name__ == "__main__":
    app()
