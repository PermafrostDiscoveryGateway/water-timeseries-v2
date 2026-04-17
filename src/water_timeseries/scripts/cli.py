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
import yaml
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
    vector_file: Optional[str] = None,
    dw_dataset_file: Optional[str] = None,
    logfile: Optional[str] = None,
    verbose: int = 0,
):
    """Launch the Streamlit dashboard.

    Args:
        port: Port to run the dashboard on (default: 8501)
        vector_file: Path to vector dataset file (GeoParquet)
        dw_dataset_file: Path to water dataset file (zarr)
        logfile: Path to log file
        verbose: Verbosity level (-v for DEBUG)

    Example usage:
        water-timeseries dashboard
        water-timeseries dashboard --port 8502
        water-timeseries dashboard --vector-file data/lakes.parquet --dw-dataset-file data/lakes.zarr
    """
    import subprocess
    import sys

    # Setup logging
    setup_logging(logfile=logfile, verbose=verbose)

    # Build streamlit command with optional arguments
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).parent.parent / "dashboard" / "app.py"),
        "--server.port",
        str(port),
    ]

    # Add optional data file arguments
    if vector_file:
        cmd.extend(["--", "--vector-file", vector_file])
    if dw_dataset_file:
        cmd.extend(["--", "--dw-dataset-file", dw_dataset_file])

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
    break_method: Optional[str] = None,
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

    This command performs breakpoint detection on lake water area time series
    data to identify significant changes in water availability. It supports
    multiple detection methods (simple statistical and Bayesian RBEAST) and
    can process datasets in parallel using Ray or Joblib.

    The analysis identifies points where water area undergoes significant
    changes, which can indicate events like drought, water diversion, or
    land use changes affecting the lake.

    Parameters
    ----------
    water_dataset_file : Path, optional
        Path to water dataset file in zarr or parquet format. Can be
        specified via CLI argument or config file.
    output_file : Path, optional
        Path to output parquet file where results will be saved.
        A YAML config file with the same name will also be created
        with the parameters used.
    config_file : Path, optional
        Path to a YAML or JSON configuration file containing default
        parameters. CLI arguments take priority over config file values.
    vector_dataset_file : Path, optional
        Path to vector dataset file (e.g., GeoParquet) containing
        lake boundary geometries for spatial analysis.
    chunksize : int, optional
        Number of lake IDs to process per chunk. Controls memory
        usage during parallel processing. Default is 100.
    parallel_backend : str, optional
        Parallelization backend to use. Options: "joblib" or "ray".
        Default is "ray" for better performance with large datasets.
    break_method : str, optional
        Breakpoint detection method. Options: "simple" (rolling window
        statistical detector) or "beast" (Bayesian RBEAST-based detector).
        Default is "beast".
    n_jobs : int, optional
        Number of parallel jobs. Use >1 for parallel processing.
        Default is 1 (sequential).
    min_chunksize : int, optional
        Minimum chunk size for parallel processing. Default is 10.
    bbox_west : float, optional
        Western boundary of bounding box for spatial filtering
        (minimum longitude).
    bbox_south : float, optional
        Southern boundary of bounding box for spatial filtering
        (minimum latitude).
    bbox_east : float, optional
        Eastern boundary of bounding box for spatial filtering
        (maximum longitude).
    bbox_north : float, optional
        Northern boundary of bounding box for spatial filtering
        (maximum latitude).
    output_geometry : bool, optional
        Whether to include geometry data in the output. Default is True.
    output_geometry_all : bool, optional
        Whether to include geometry for all lakes (not just those with
        breakpoints). Default is True.
    logfile : str, optional
        Path to log file. If not provided, a default logfile is created
        with the format `{subcommand}_{timestamp}.log`.
    verbose : int, optional
        Verbosity level. 0 = INFO (default), 1 or more = DEBUG.

    Returns
    -------
    None
        Results are written directly to the output parquet file.
        A companion YAML file with the same name (but .yaml extension)
        is also created containing the parameters used for the run.

    Raises
    ------
    SystemExit
        If required arguments (water_dataset_file and output_file) are
        not provided via CLI or config file.

    Notes
    -----
    The SimpleBreakpoint method uses a rolling window statistical approach
    comparing current values against rolling mean/median/max to detect drops
    in water area.

    The BeastBreakpoint method uses the RBEAST library for Bayesian
    change-point detection, which can identify more nuanced changes in
    time series properties.

    Example usage
    ------------
    Basic usage with required arguments::

        water-timeseries breakpoint-analysis tests/data/lakes_dw_test.zarr output.parquet

    With custom chunk size and parallel jobs::

        water-timeseries breakpoint-analysis tests/data/lakes_dw_test.zarr output.parquet -c 100 -j 20

    Using a configuration file::

        water-timeseries breakpoint-analysis --config-file configs/config.yaml

    Spatial filtering with bounding box::

        water-timeseries breakpoint-analysis data.zarr output.parquet \\
            --bbox-west 100 --bbox-south 20 --bbox-east 110 --bbox-north 30
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
        break_method=break_method,
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
    water_dataset_file = config_dict.get("water_dataset_file")
    output_file = config_dict.get("output_file")
    logfile_val = config_dict.get("logfile")
    verbose_val = config_dict.get("verbose", 0)

    # Validate required arguments
    if not water_dataset_file or not output_file:
        logger.error("water_dataset_file and output_file are required. Provide via CLI arguments or config file.")
        raise SystemExit(1)

    # Setup logging AFTER config is loaded
    setup_logging(logfile=logfile_val, verbose=verbose_val)

    # Run the pipeline
    pipeline = BreakpointPipeline(
        water_dataset_file=water_dataset_file,
        output_file=output_file,
        vector_dataset_file=config_dict.get("vector_dataset_file"),
        chunksize=config_dict.get("chunksize") or 100,
        parallel_backend=config_dict.get("parallel_backend") or "ray",
        break_method=config_dict.get("break_method") or "beast",
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

    # Save the used parameters to a config file next to the output file
    output_path = Path(output_file)
    config_output_path = output_path.with_suffix(".yaml")
    used_config = {
        "water_dataset_file": water_dataset_file,
        "output_file": output_file,
        "vector_dataset_file": config_dict.get("vector_dataset_file"),
        "chunksize": config_dict.get("chunksize"),
        "parallel_backend": config_dict.get("parallel_backend"),
        "break_method": config_dict.get("break_method"),
        "n_jobs": pipeline.n_jobs,  # Use actual n_jobs (may have been reduced)
        "min_chunksize": config_dict.get("min_chunksize"),
        "bbox_west": config_dict.get("bbox_west"),
        "bbox_south": config_dict.get("bbox_south"),
        "bbox_east": config_dict.get("bbox_east"),
        "bbox_north": config_dict.get("bbox_north"),
        "output_geometry": config_dict.get("output_geometry"),
        "output_geometry_all": config_dict.get("output_geometry_all"),
    }
    with open(config_output_path, "w") as f:
        yaml.dump(used_config, f, default_flow_style=False)
    logger.info(f"Saved used parameters to {config_output_path}")


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
    water_dataset_file = config_dict.get("water_dataset_file")
    lake_id = config_dict.get("lake_id")
    output_figure = config_dict.get("output_figure")
    break_method = config_dict.get("break_method")
    logfile_val = config_dict.get("logfile")
    verbose_val = config_dict.get("verbose", 0)

    # Validate required arguments
    if not water_dataset_file or not lake_id:
        logger.error("water_dataset_file and lake_id are required. Provide via CLI arguments or config file.")
        raise SystemExit(1)

    # Setup logging AFTER config is loaded
    setup_logging(logfile=logfile_val, verbose=verbose_val)

    # Log key parameters
    logger.info(
        f"Plotting lake timeseries with parameters: "
        f"water_dataset_file={water_dataset_file}, "
        f"lake_id={lake_id}, "
        f"output_figure={output_figure}, "
        f"break_method={break_method}, "
        f"show={show}"
    )

    # Use the imported function
    plot_lake_timeseries(
        water_dataset_file=water_dataset_file,
        lake_id=lake_id,
        output_figure=output_figure,
        break_method=break_method,
        show=show,
    )


if __name__ == "__main__":
    app()
