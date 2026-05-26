# Hierarchical CLI for water-timeseries using cyclopts
"""Hierarchical CLI for water-timeseries.

Usage:
    water-timeseries breakpoint-analysis-historical data.zarr output.parquet
    water-timeseries breakpoint-analysis-historical data.zarr output.parquet -c 100 -j 20
    water-timeseries plot-timeseries data.zarr --lake-id b7uefy0bvcrc
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cyclopts
import pandas as pd
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

# Import NRT pre-computation
from water_timeseries.scripts.precompute_nrt_monthly import precompute_nrt_monthly

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
    jrc_dataset_file: Optional[str] = None,
    precomputed_nrt_dir: Optional[str] = None,
    logfile: Optional[str] = None,
    verbose: int = 0,
):
    """Launch the Streamlit dashboard.

    Args:
        port: Port to run the dashboard on (default: 8501)
        vector_file: Path to vector dataset file (GeoParquet)
        dw_dataset_file: Path to Dynamic World dataset file (zarr)
        jrc_dataset_file: Path to JRC dataset file (zarr)
        precomputed_nrt_dir: Directory with pre-computed NRT parquet files.
            Auto-detected from ``precomputed/nrt/`` in the repo root when present.
        logfile: Path to log file
        verbose: Verbosity level (-v for DEBUG)

    Example usage:
        water-timeseries dashboard
        water-timeseries dashboard --port 8502
        water-timeseries dashboard --vector-file data/lakes.parquet --dw-dataset-file data/lakes.zarr
        water-timeseries dashboard --vector-file tests/data/lake_polygons.parquet \\
            --dw-dataset-file downloads/data.zarr \\
            --jrc-dataset-file downloads/downloads/lakes_jrc_viz.zarr \\
            --precomputed-nrt-dir precomputed/nrt-demo
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

    # Add optional data file arguments (single "--" separates streamlit args from script args)
    script_args = []
    if vector_file:
        script_args.extend(["--vector-file", vector_file])
    if dw_dataset_file:
        script_args.extend(["--dw-dataset-file", dw_dataset_file])
    if jrc_dataset_file:
        script_args.extend(["--jrc-dataset-file", jrc_dataset_file])
    if precomputed_nrt_dir:
        script_args.extend(["--precomputed-nrt-dir", precomputed_nrt_dir])
    if script_args:
        cmd.extend(["--"] + script_args)

    logger.info(f"Starting dashboard with command: {' '.join(cmd)}")
    subprocess.run(cmd)


# Subcommand: breakpoint analysis
@app.command(group="Analysis")
def breakpoint_analysis_historical(
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
    """Run historical breakpoint analysis on water dataset.

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

        water-timeseries breakpoint-analysis-historical tests/data/lakes_dw_test.zarr output.parquet

    With custom chunk size and parallel jobs::

        water-timeseries breakpoint-analysis-historical tests/data/lakes_dw_test.zarr output.parquet -c 100 -j 20

    Using a configuration file::

        water-timeseries breakpoint-analysis-historical --config-file configs/config.yaml

    Spatial filtering with bounding box::

        water-timeseries breakpoint-analysis-historical data.zarr output.parquet \\
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


# Subcommand: NRT monthly pre-computation
@app.command(group="Analysis")
def breakpoint_analysis_nrt(
    dataset_file: Path,
    analysis_date: Optional[str] = None,
    analysis_date_start: Optional[str] = None,
    analysis_date_end: Optional[str] = None,
    output_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    no_resume: bool = False,
    drain_threshold: float = -0.25,
    data_aggregation_period: str = "all",
    lake_chunk_size: int = 5000,
    n_jobs: int = 4,
    vector_file: Optional[Path] = None,
    logfile: Optional[str] = None,
    verbose: int = 0,
):
    """Pre-compute near real-time drained-lake results for one month or a date range.

    **Single-month mode** (``--analysis-date``): runs for exactly one month
    and writes results to ``--output-file``.

    **Range mode** (``--analysis-date-start`` + ``--analysis-date-end``): runs
    for every month in the inclusive range and writes one parquet file per
    month to ``--output-dir``, auto-named
    ``nrt_<YYYY-MM>_drain_breaks.parquet``.  Already-present files are skipped
    unless ``--no-resume`` is set.

    Both modes produce per-lake NRT break results (drained lakes only) with an
    ``analysis_month`` column, ready for the dashboard map overlay.

    Parameters
    ----------
    dataset_file:
        Path to the DW dataset file (``.ncin`` / ``.nc`` NetCDF or ``.zarr``).
    analysis_date:
        Single month to analyse, as ``YYYY-MM`` (e.g. ``2024-01``).
        Mutually exclusive with ``--analysis-date-start`` /
        ``--analysis-date-end``.
    analysis_date_start:
        First month of an inclusive range, as ``YYYY-MM``.
        Must be used together with ``--analysis-date-end``.
    analysis_date_end:
        Last month of an inclusive range, as ``YYYY-MM``.
        Must be used together with ``--analysis-date-start``.
    output_file:
        Destination parquet file (single-month mode only).  Defaults to
        ``nrt_<analysis_date>_drain_breaks.parquet`` next to *dataset_file*.
    output_dir:
        Destination directory for range mode.  One file per month is written
        as ``nrt_<YYYY-MM>_drain_breaks.parquet``.  Defaults to the parent
        directory of *dataset_file*.
    no_resume:
        Range mode only.  When set, re-process months even if their output
        file already exists in ``--output-dir``.
    drain_threshold:
        ``water_residual`` threshold below which a lake is classified as
        drained (default ``-0.25``).
    data_aggregation_period:
        Passed to ``NRTBreakpoint.calculate_break`` (default ``"all"``).
    lake_chunk_size:
        Lakes processed per chunk. Smaller = less RAM. Default 5000.
    n_jobs:
        Parallel ARIMA workers per chunk. Reduce if RAM is tight. Default 4.
    vector_file:
        Optional GeoParquet vector file.  When provided, only the
        ``id_geohash`` values present in that file are processed.
    logfile:
        Path to log file. Auto-generated if not provided.
    verbose:
        Verbosity level (0 = INFO, 1 = DEBUG).

    Example usage
    -------------
    .. code-block:: bash

        # Single month
        water-timeseries breakpoint-analysis-nrt downloads/lakes_dw_V2d.nc \\
            --analysis-date 2024-01 \\
            --output-file precomputed/nrt/nrt_2024-01_drain_breaks.parquet

        # Date range (one file per month written to --output-dir)
        water-timeseries breakpoint-analysis-nrt downloads/lakes_dw_V2d.nc \\
            --analysis-date-start 2024-01 \\
            --analysis-date-end 2024-06 \\
            --output-dir precomputed/nrt

        # Resume a previously interrupted range run
        water-timeseries breakpoint-analysis-nrt downloads/lakes_dw_V2d.nc \\
            --analysis-date-start 2024-01 \\
            --analysis-date-end 2024-12 \\
            --output-dir precomputed/nrt

        # Force re-process all months in range
        water-timeseries breakpoint-analysis-nrt downloads/lakes_dw_V2d.nc \\
            --analysis-date-start 2024-01 \\
            --analysis-date-end 2024-12 \\
            --output-dir precomputed/nrt \\
            --no-resume
    """
    setup_logging(logfile=logfile, verbose=verbose)

    # --- Validate mode selection -------------------------------------------
    range_mode = analysis_date_start is not None or analysis_date_end is not None
    single_mode = analysis_date is not None

    if single_mode and range_mode:
        logger.error(
            "--analysis-date cannot be combined with "
            "--analysis-date-start / --analysis-date-end"
        )
        raise SystemExit(1)

    if not single_mode and not range_mode:
        logger.error(
            "Provide either --analysis-date (single month) or both "
            "--analysis-date-start and --analysis-date-end (range)."
        )
        raise SystemExit(1)

    if range_mode and (analysis_date_start is None or analysis_date_end is None):
        logger.error(
            "--analysis-date-start and --analysis-date-end must both be provided."
        )
        raise SystemExit(1)

    # --- Resolve lake IDs from vector file ----------------------------------
    lake_ids = None
    if vector_file is not None:
        import geopandas as gpd
        gdf = gpd.read_parquet(vector_file)
        if "id_geohash" not in gdf.columns:
            logger.error("vector_file %s does not contain an 'id_geohash' column", vector_file)
            raise SystemExit(1)
        lake_ids = gdf["id_geohash"].dropna().unique().tolist()
        logger.info("Loaded %d lake IDs from vector file: %s", len(lake_ids), vector_file)

    shared_kwargs = dict(
        dataset_file=dataset_file,
        drain_threshold=drain_threshold,
        data_aggregation_period=data_aggregation_period,
        lake_chunk_size=lake_chunk_size,
        n_jobs=n_jobs,
        lake_ids=lake_ids,
    )

    # --- Single-month mode --------------------------------------------------
    if single_mode:
        resolved_output_file = (
            output_file
            if output_file is not None
            else Path(dataset_file).parent / f"nrt_{analysis_date}_drain_breaks.parquet"
        )
        logger.info(
            "Starting NRT pre-computation (single month):\n"
            f"  dataset_file      = {dataset_file}\n"
            f"  analysis_date     = {analysis_date}\n"
            f"  output_file       = {resolved_output_file}\n"
            f"  drain_threshold   = {drain_threshold}\n"
            f"  data_aggregation  = {data_aggregation_period}\n"
            f"  lake_chunk_size   = {lake_chunk_size}\n"
            f"  n_jobs            = {n_jobs}\n"
            f"  lake_ids filter   = {len(lake_ids) if lake_ids is not None else 'all'}"
        )
        breaks_df = precompute_nrt_monthly(
            output_file=resolved_output_file,
            analysis_date=analysis_date,
            **shared_kwargs,
        )
        logger.info(
            "Pre-computation complete.\n"
            f"  analysis_date  : {analysis_date}\n"
            f"  drained lakes  : {len(breaks_df)}\n"
            f"  output_file    : {resolved_output_file}"
        )
        return

    # --- Range mode ---------------------------------------------------------
    try:
        start_ts = pd.Timestamp(analysis_date_start)
        end_ts = pd.Timestamp(analysis_date_end)
    except Exception as exc:
        logger.error("Invalid date range: %s", exc)
        raise SystemExit(1) from exc

    if end_ts < start_ts:
        logger.error(
            "--analysis-date-end (%s) must be >= --analysis-date-start (%s)",
            analysis_date_end, analysis_date_start,
        )
        raise SystemExit(1)

    months = pd.period_range(start=start_ts, end=end_ts, freq="M")
    month_strs = [str(m) for m in months]  # "YYYY-MM" format

    resolved_output_dir = output_dir if output_dir is not None else Path(dataset_file).parent
    resolved_output_dir = Path(resolved_output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting NRT pre-computation (range mode):\n"
        f"  dataset_file      = {dataset_file}\n"
        f"  range             = {analysis_date_start} – {analysis_date_end} "
        f"({len(month_strs)} months)\n"
        f"  output_dir        = {resolved_output_dir}\n"
        f"  resume            = {not no_resume}\n"
        f"  drain_threshold   = {drain_threshold}\n"
        f"  data_aggregation  = {data_aggregation_period}\n"
        f"  lake_chunk_size   = {lake_chunk_size}\n"
        f"  n_jobs            = {n_jobs}\n"
        f"  lake_ids filter   = {len(lake_ids) if lake_ids is not None else 'all'}"
    )

    total_drained = 0
    skipped = 0
    failed = 0

    for month_str in month_strs:
        month_file = resolved_output_dir / f"nrt_{month_str}_drain_breaks.parquet"

        if not no_resume and month_file.exists():
            logger.info("Skipping %s — output already exists: %s", month_str, month_file)
            skipped += 1
            continue

        try:
            breaks_df = precompute_nrt_monthly(
                output_file=month_file,
                analysis_date=month_str,
                **shared_kwargs,
            )
            total_drained += len(breaks_df)
        except Exception as exc:
            logger.warning("Failed for %s: %s", month_str, exc)
            failed += 1

    logger.info(
        "Range pre-computation complete.\n"
        f"  months in range   : {len(month_strs)}\n"
        f"  skipped (existing): {skipped}\n"
        f"  failed            : {failed}\n"
        f"  processed         : {len(month_strs) - skipped - failed}\n"
        f"  total drained rows: {total_drained}\n"
        f"  output_dir        : {resolved_output_dir}"
    )


if __name__ == "__main__":
    app()
