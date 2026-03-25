# Hierarchical CLI for water-timeseries using cyclopts
"""Hierarchical CLI for water-timeseries.

Usage:
    water-timeseries breakpoint-analysis data.zarr output.parquet
    water-timeseries breakpoint-analysis data.zarr output.parquet -c 100 -j 20
    water-timeseries plot-timeseries data.zarr --lake-id b7uefy0bvcrc
"""

import sys
from datetime import datetime
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


# Helper function to configure logging


def setup_logging(logfile: Optional[str] = None, verbose: int = 0):
    """Configure logging with verbosity control.

    Args:
        logfile: Path to log file. If not provided, logs to console only.
        verbose: Verbosity level (0=INFO, 1=DEBUG)

    Verbosity flags:
        - No flag or -v: INFO level (default)
        - -v: DEBUG level
    """
    # Determine log level based on verbosity count
    if verbose >= 1:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    # Generate default logfile name from subcommand and timestamp
    if logfile is None:
        try:
            # sys.argv[0] is the script name, sys.argv[1] is the subcommand
            if len(sys.argv) >= 2:
                subcommand = sys.argv[1].replace("-", "_")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                logfile = f"{subcommand}_{timestamp}.log"
                print(f"Using default logfile: {logfile}")  # Use print to avoid circular logging
        except Exception:
            pass
        # If no logfile set, log to console only
        if logfile is None:
            return

    logger.add(logfile, rotation="10 MB", retention="1 week", level=log_level)
    print(f"Logging to file: {logfile} with level: {log_level}")  # Use print to avoid circular logging


# Subcommand: dashboard
@app.command(group="Visualization")
def dashboard(
    port: int = 8501,
    logfile: Optional[str] = None,
    verbose: int = 0,
):
    """Launch the Streamlit dashboard.

    Args:
        port: Port to run the dashboard on (default: 8501)
        logfile: Path to log file
        verbose: Verbosity level (-v for DEBUG)

    Example usage:
        water-timeseries dashboard
        water-timeseries dashboard --port 8502
    """
    import subprocess
    import sys

    # Setup logging
    setup_logging(logfile=logfile, verbose=verbose)

    # Build streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).parent.parent / "dashboard" / "app.py"),
        "--server.port",
        str(port),
    ]

    logger.info(f"Starting dashboard with command: {' '.join(cmd)}")
    subprocess.run(cmd)


# Subcommand: breakpoint analysis
@app.command(group="Analysis")
def breakpoint_analysis(
    water_dataset_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    config_file: Optional[Path] = None,
    vector_dataset_file: Optional[Path] = None,
    chunksize: Optional[int] = None,
    parallel_backend: Optional[str] = None,
    n_jobs: Optional[int] = None,
    min_chunksize: Optional[int] = None,
    bbox_west: Optional[float] = None,
    bbox_south: Optional[float] = None,
    bbox_east: Optional[float] = None,
    bbox_north: Optional[float] = None,
    output_geometry: bool = True,
    output_geometry_all: bool = True,
    logfile: Optional[str] = None,
    verbose: int = 0,
):
    """Run breakpoint analysis on water dataset.

    Args:
        water_dataset_file: Path to water dataset file (zarr or parquet)
        output_file: Path to output parquet file
        config_file: Path to config YAML/JSON file
        vector_dataset_file: Path to vector dataset file
        chunksize: Number of IDs per chunk
        n_jobs: Number of parallel jobs (use >1 for Ray)
        parallel_backend: Parallelization backend (joblib or ray)
        min_chunksize: Minimum chunk size
        bbox_west: Minimum longitude (west)
        bbox_south: Minimum latitude (south)
        bbox_east: Maximum longitude (east)
        bbox_north: Maximum latitude (north)
        output_geometry: Whether to include geometry in output (default: True)
        output_geometry_all: Whether to include geometry for all lakes (default: True)
        logfile: Path to log file
        verbose: Verbosity level (-v for DEBUG)

    Example usage:
        water-timeseries breakpoint-analysis tests/data/lakes_dw_test.zarr output.parquet
        water-timeseries breakpoint-analysis tests/data/lakes_dw_test.zarr output.parquet -c 100 -j 20
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
        parallel_backend=parallel_backend,
        min_chunksize=min_chunksize,
        bbox_west=bbox_west,
        bbox_south=bbox_south,
        bbox_east=bbox_east,
        bbox_north=bbox_north,
        output_geometry=output_geometry,
        output_geometry_all=output_geometry_all,
        logfile=logfile,
        verbose=verbose,
    )

    # Get values from merged config
    water_ds = config_dict.get("water_dataset_file")
    output_ds = config_dict.get("output_file")
    logfile_val = config_dict.get("logfile")
    verbose_val = config_dict.get("verbose", 0)

    # Validate required arguments
    if not water_ds or not output_ds:
        logger.error("water_dataset_file and output_file are required. Provide via CLI arguments or config file.")
        raise SystemExit(1)

    # Setup logging AFTER config is loaded
    setup_logging(logfile=logfile_val, verbose=verbose_val)

    # Run the pipeline
    pipeline = BreakpointPipeline(
        water_dataset_file=water_ds,
        output_file=output_ds,
        vector_dataset_file=config_dict.get("vector_dataset_file"),
        chunksize=config_dict.get("chunksize") or 100,
        parallel_backend=config_dict.get("parallel_backend") or "joblib",
        n_jobs=config_dict.get("n_jobs") or 1,
        min_chunksize=config_dict.get("min_chunksize") or 10,
        bbox_west=config_dict.get("bbox_west"),
        bbox_south=config_dict.get("bbox_south"),
        bbox_east=config_dict.get("bbox_east"),
        bbox_north=config_dict.get("bbox_north"),
        output_geometry=config_dict.get("output_geometry", True),
        output_geometry_all=config_dict.get("output_geometry_all", False),
        logger=logger,
    )
    pipeline.run_breaks()
    pipeline.save_to_parquet()


# Subcommand: plot timeseries
@app.command(group="Plotting")
def plot_timeseries(
    water_dataset_file: Optional[Path] = None,
    lake_id: Optional[str] = None,
    output_figure: Optional[Path] = None,
    break_method: Optional[str] = None,
    config_file: Optional[Path] = None,
    show: bool = True,
    logfile: Optional[str] = None,
    verbose: int = 0,
):
    """Plot time series for a specific lake.

    Args:
        water_dataset_file: Path to water dataset file (zarr or netCDF)
        lake_id: Geohash ID of the lake to plot
        output_figure: Path to save the output figure
        break_method: Break method to overlay (optional)
        config_file: Path to config YAML/JSON file
        logfile: Path to log file
        verbose: Verbosity level (-v for DEBUG)

    Example usage:
        water-timeseries plot-timeseries data.zarr --lake-id b7uefy0bvcrc
        water-timeseries plot-timeseries data.zarr --lake-id b7uefy0bvcrc --output-figure plot.png
        water-timeseries plot-timeseries --config-file configs/plot_config.yaml
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
        logfile=logfile,
        verbose=verbose,
    )

    # Get values from merged config
    water_ds = config_dict.get("water_dataset_file")
    lake_id_val = config_dict.get("lake_id")
    output_fig = config_dict.get("output_figure")
    break_method_val = config_dict.get("break_method")
    logfile_val = config_dict.get("logfile")
    verbose_val = config_dict.get("verbose", 0)

    # Validate required arguments
    if not water_ds or not lake_id_val:
        logger.error("water_dataset_file and lake_id are required. Provide via CLI arguments or config file.")
        raise SystemExit(1)

    # Setup logging AFTER config is loaded
    setup_logging(logfile=logfile_val, verbose=verbose_val)

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
