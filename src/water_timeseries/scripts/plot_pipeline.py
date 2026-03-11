# imports

# Try to use interactive backend for popup windows
# try:
#     matplotlib.use("TkAgg")
# except:
#     pass  # Fall back to default if TkAgg not available
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import typer
import xarray as xr
from loguru import logger

from water_timeseries.breakpoint import BeastBreakpoint, SimpleBreakpoint
from water_timeseries.dataset import DWDataset, JRCDataset
from water_timeseries.utils.data import get_water_dataset_type

# configure logger: writes to rbeast_batch.log in current working dir
_log_file = Path.cwd() / "rbeast_batch.log"
logger.add(
    _log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} {level} {name}:{function}:{line} - {message}",
    mode="a",
    diagnose=False,
)
# Bind script filename so logs show script name instead of __main__
logger = logger.bind(script=Path(__file__).name)
app = typer.Typer(help="Plot time series of dataset")


@app.command()
def main(
    water_dataset_file: str,
    lake_id: Optional[str] = None,
    output_figure: Optional[str] = None,
    break_method: Optional[str] = None,
    show: bool = True,
):
    """Plot time series for a lake.

    Args:
        water_dataset_file: Path to water dataset file (zarr or NetCDF).
        lake_id: Geohash ID of the lake to plot.
        output_figure: Path to save the output figure.
        break_method: Break method to overlay (optional).
        show: Whether to display the figure (default: True).
    """
    plot_lake_timeseries(water_dataset_file, lake_id, output_figure, break_method, show)


def plot_lake_timeseries(
    water_dataset_file: str,
    lake_id: Optional[str] = None,
    output_figure: Optional[str] = None,
    break_method: Optional[str] = None,
    show: bool = True,
):
    """Plot time series for a specific lake.

    This function can be imported and used programmatically.

    Args:
        water_dataset_file: Path to water dataset file (zarr or netCDF).
        lake_id: Geohash ID of the lake to plot.
        output_figure: Path to save the output figure.
        break_method: Break method to overlay (optional).
        show: Whether to display the figure (default: True).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    ds_xr = xr.load_dataset(water_dataset_file)

    # Check if id exists in dataset
    if lake_id not in ds_xr.coords["id_geohash"]:
        logger.error(f"ID {lake_id} not found in dataset coordinates")
        raise ValueError(f"ID {lake_id} not found in dataset coordinates")

    # Get dataset type
    water_dataset_type = get_water_dataset_type(ds_xr)
    if water_dataset_type == "jrc":
        ds = JRCDataset(ds_xr)
    elif water_dataset_type == "dynamic_world":
        ds = DWDataset(ds_xr)
    else:
        logger.error(f"Unknown water dataset type: {water_dataset_type}")
        raise ValueError(f"Unknown water dataset type: {water_dataset_type}")

    # Plot timeseries
    if break_method == "beast":
        breakpoints = BeastBreakpoint()
    elif break_method == "simple":
        breakpoints = SimpleBreakpoint()
    else:
        breakpoints = None
    fig = ds.plot_timeseries(id_geohash=lake_id, breakpoints=breakpoints)

    # Save figure if output path provided
    if output_figure:
        fig.savefig(output_figure)
        logger.info(f"Saved figure to {output_figure}")

    # Show figure if requested
    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    app()
