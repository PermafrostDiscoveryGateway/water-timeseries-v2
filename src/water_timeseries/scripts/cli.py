# Hierarchical CLI for water-timeseries using cyclopts
"""Hierarchical CLI for water-timeseries.

Usage:
    water-timeseries breakpoint-analysis data.zarr output.parquet
    water-timeseries breakpoint-analysis data.zarr output.parquet -c 100 -j 20
    water-timeseries plot-lake-timeseries data.zarr --lake-id abc123
"""

from pathlib import Path
from typing import Optional

import cyclopts
from loguru import logger

# Import pipeline and utilities from break_pipeline
from water_timeseries.scripts.break_pipeline import (
    BreakpointPipeline,
    load_config,
    merge_config_with_args,
)

# Import plotting function from plot_pipeline
from water_timeseries.scripts.plot_pipeline import plot_lake_timeseries

# Create the main app
app = cyclopts.App(name="water-timeseries", help="Water timeseries analysis tools")


# Subcommand: breakpoint analysis
@app.command(group="Analysis")
def breakpoint_analysis(
    water_dataset_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    config_file: Optional[Path] = None,
    vector_dataset_file: Optional[Path] = None,
    chunksize: Optional[int] = None,
    n_jobs: Optional[int] = None,
    min_chunksize: Optional[int] = None,
    bbox_west: Optional[float] = None,
    bbox_south: Optional[float] = None,
    bbox_east: Optional[float] = None,
    bbox_north: Optional[float] = None,
):
    """Run breakpoint analysis on water dataset.

    Args:
        water_dataset_file: Path to water dataset file (zarr or parquet)
        output_file: Path to output parquet file
        config_file: Path to config YAML/JSON file
        vector_dataset_file: Path to vector dataset file
        chunksize: Number of IDs per chunk
        n_jobs: Number of parallel jobs (use >1 for Ray)
        min_chunksize: Minimum chunk size
        bbox_west: Minimum longitude (west)
        bbox_south: Minimum latitude (south)
        bbox_east: Maximum longitude (east)
        bbox_north: Maximum latitude (north)

    Example usage:
        water-timeseries breakpoint-analysis data.zarr output.parquet
        water-timeseries breakpoint-analysis data.zarr output.parquet -c 100 -j 20
        water-timeseries breakpoint-analysis --config-file configs/config.yaml
    """
    # Load config file if provided
    config_dict = load_config(config_file) if config_file else {}

    # Merge config with CLI args (CLI takes priority)
    config_dict = merge_config_with_args(
        config_dict,
        water_dataset_file=str(water_dataset_file) if water_dataset_file else None,
        output_file=str(output_file) if output_file else None,
        vector_dataset_file=str(vector_dataset_file) if vector_dataset_file else None,
        chunksize=chunksize,
        n_jobs=n_jobs,
        min_chunksize=min_chunksize,
        bbox_west=bbox_west,
        bbox_south=bbox_south,
        bbox_east=bbox_east,
        bbox_north=bbox_north,
    )

    # Get water_dataset_file and output_file from merged config
    water_ds = config_dict.get("water_dataset_file")
    output_ds = config_dict.get("output_file")

    # Validate required arguments
    if not water_ds or not output_ds:
        logger.error("water_dataset_file and output_file are required. Provide via CLI arguments or config file.")
        raise SystemExit(1)

    # Run the pipeline
    pipeline = BreakpointPipeline(
        water_dataset_file=water_ds,
        output_file=output_ds,
        vector_dataset_file=config_dict.get("vector_dataset_file"),
        chunksize=config_dict.get("chunksize") or 100,
        n_jobs=config_dict.get("n_jobs") or 1,
        min_chunksize=config_dict.get("min_chunksize") or 10,
        bbox_west=config_dict.get("bbox_west"),
        bbox_south=config_dict.get("bbox_south"),
        bbox_east=config_dict.get("bbox_east"),
        bbox_north=config_dict.get("bbox_north"),
        logger=logger,
    )
    pipeline.run_breaks()
    pipeline.save_to_parquet()


# Subcommand: plot lake timeseries
@app.command(group="Plotting")
def plot_lake(
    water_dataset_file: Optional[Path] = None,
    lake_id: Optional[str] = None,
    output_figure: Optional[Path] = None,
    break_method: Optional[str] = None,
    config_file: Optional[Path] = None,
    show: bool = True,
):
    """Plot time series for a specific lake.

    Args:
        water_dataset_file: Path to water dataset file (zarr or netCDF)
        lake_id: Geohash ID of the lake to plot
        output_figure: Path to save the output figure
        break_method: Break method to overlay (optional)
        config_file: Path to config YAML/JSON file

    Example usage:
        water-timeseries plot-lake-timeseries data.zarr --lake-id abc123
        water-timeseries plot-lake-timeseries data.zarr --lake-id abc123 --output-figure plot.png
        water-timeseries plot-lake-timeseries --config-file configs/plot_config.yaml
    """
    # Load config file if provided
    config_dict = load_config(config_file) if config_file else {}

    # Merge config with CLI args (CLI takes priority)
    # Note: show is handled separately since it's a bool
    config_dict = merge_config_with_args(
        config_dict,
        water_dataset_file=str(water_dataset_file) if water_dataset_file else None,
        lake_id=lake_id,
        output_figure=str(output_figure) if output_figure else None,
        break_method=break_method,
    )

    # Get values from merged config
    water_ds = config_dict.get("water_dataset_file")
    lake_id_val = config_dict.get("lake_id")
    output_fig = config_dict.get("output_figure")
    break_method_val = config_dict.get("break_method")

    # Validate required arguments
    if not water_ds or not lake_id_val:
        logger.error("water_dataset_file and lake_id are required. Provide via CLI arguments or config file.")
        raise SystemExit(1)

    # Log key parameters
    logger.info(
        f"Plotting lake timeseries with parameters: "
        f"water_dataset_file={water_ds}, "
        f"lake_id={lake_id_val}, "
        f"output_figure={output_fig}, "
        f"break_method={break_method_val}, "
        f"show={show}"
    )

    # Use the imported function
    plot_lake_timeseries(
        water_dataset_file=water_ds,
        lake_id=lake_id_val,
        output_figure=output_fig,
        break_method=break_method_val,
        show=show,
    )


if __name__ == "__main__":
    app()
