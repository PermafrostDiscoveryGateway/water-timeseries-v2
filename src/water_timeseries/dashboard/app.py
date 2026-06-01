"""Run the Streamlit dashboard."""

import argparse
import warnings
from pathlib import Path

from water_timeseries.dashboard.map_viewer import create_app

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_DEFAULT_NRT_DIR = _REPO_ROOT / "precomputed" / "nrt"


def parse_args():
    """Parse command line arguments for the dashboard."""
    parser = argparse.ArgumentParser(description="Run the Water Timeseries Dashboard")
    parser.add_argument(
        "--vector-file",
        type=str,
        default=None,
        help="Path to vector dataset file (GeoParquet). If not provided, uses default test data.",
    )
    parser.add_argument(
        "--dw-dataset-file",
        type=str,
        default=None,
        help="Path to Dynamic World dataset file (zarr). If not provided, uses default test data.",
    )
    parser.add_argument(
        "--jrc-dataset-file",
        type=str,
        default=None,
        help="Path to JRC water dataset file (zarr). If not provided, uses default test data.",
    )
    parser.add_argument(
        "--precomputed-nrt-dir",
        type=str,
        default=None,
        help=(
            "Directory containing pre-computed NRT parquet files "
            "(nrt_monthly_drain_counts.parquet / nrt_monthly_drain_breaks.parquet). "
            f"Defaults to {_DEFAULT_NRT_DIR} when that directory exists."
        ),
    )
    parser.add_argument(
        "--offline-mode",
        action="store_true",
        default=False,
        help="Disable Google Earth Engine download functionality. Use when running without internet access or EE authentication.",
    )
    parser.add_argument(
        "--ee-project",
        type=str,
        default=None,
        help="Google Earth Engine project ID. Required for EE downloads.",
    )
    parser.add_argument(
        "--dw-start-year",
        type=int,
        default=2017,
        help="Start year for Dynamic World time series (inclusive). Default is 2017.",
    )
    parser.add_argument(
        "--dw-end-year",
        type=int,
        default=2025,
        help="End year for Dynamic World time series (inclusive). Default is 2025.",
    )
    parser.add_argument(
        "--dw-start-month",
        type=int,
        default=6,
        help="Start month for Dynamic World time series (inclusive). Default is 6.",
    )
    parser.add_argument(
        "--dw-end-month",
        type=int,
        default=9,
        help="End month for Dynamic World time series (inclusive). Default is 9.",
    )
    parser.add_argument(
        "--viz-configuration",
        type=str,
        default="colored_historical",
        help=(
            "Visualization configuration name for the map viewer. "
            "Options include 'colored_historical' (default) and 'nrt_drainage'. "
            "This controls the styling and color scheme of the map layers."
        ),
    )

    return parser.parse_args()


def main(
    vector_file: str | Path = None,
    dw_dataset_file: str | Path = None,
    jrc_dataset_file: str | Path = None,
    precomputed_nrt_dir: str | Path = None,
    offline_mode: bool = False,
    ee_project: str = None,
    viz_configuration: str = None,
    dw_start_year: int = None,
    dw_end_year: int = None,
    dw_start_month: int = None,
    dw_end_month: int = None,
):
    """Run the dashboard app.

    Args:
        vector_file: Path to vector dataset file (GeoParquet). Defaults to test data.
        dw_dataset_file: Path to Dynamic World dataset file (zarr). Defaults to test data.
        jrc_dataset_file: Path to JRC dataset file (zarr). Defaults to test data.
        precomputed_nrt_dir: Directory with pre-computed NRT parquet files.
            Auto-detected from ``precomputed/nrt/`` in the repo root when present.
        offline_mode: If True, disables Google Earth Engine download functionality.
        viz_configuration: The visualization configuration name for the map viewer.
    """
    # Default paths to test data
    default_vector_file = _REPO_ROOT / "tests" / "data" / "lake_polygons.parquet"
    default_dw_dataset_file = _REPO_ROOT / "tests" / "data" / "lakes_dw_test.zarr"
    default_jrc_dataset_file = _REPO_ROOT / "tests" / "data" / "lakes_jrc_test.zarr"

    # Validate provided file paths and fall back to defaults if they don't exist
    if vector_file is not None:
        path = Path(vector_file)
        if not path.exists():
            warnings.warn(f"Vector file not found: {vector_file}. Falling back to default test data.")
            vector_file = None

    if dw_dataset_file is not None:
        path = Path(dw_dataset_file)
        if not path.exists():
            warnings.warn(f"DW dataset file not found: {dw_dataset_file}. Falling back to default test data.")
            dw_dataset_file = None

    if jrc_dataset_file is not None:
        path = Path(jrc_dataset_file)
        if not path.exists():
            warnings.warn(f"JRC dataset file not found: {jrc_dataset_file}. Falling back to default test data.")
            jrc_dataset_file = None

    # Use provided paths or defaults
    if vector_file is None:
        vector_file = default_vector_file
    if dw_dataset_file is None:
        dw_dataset_file = default_dw_dataset_file
    if jrc_dataset_file is None:
        jrc_dataset_file = default_jrc_dataset_file

    # Auto-detect precomputed NRT directory if not explicitly provided
    if precomputed_nrt_dir is None:
        counts_file = _DEFAULT_NRT_DIR / "nrt_monthly_drain_counts.parquet"
        breaks_file = _DEFAULT_NRT_DIR / "nrt_monthly_drain_breaks.parquet"
        if counts_file.exists() or breaks_file.exists():
            precomputed_nrt_dir = _DEFAULT_NRT_DIR

    if viz_configuration is None:
        viz_configuration = "colored_historical"

    create_app(
        data_path=vector_file,
        zarr_path=dw_dataset_file,
        zarr_path_jrc=jrc_dataset_file,
        precomputed_nrt_dir=precomputed_nrt_dir,
        offline_mode=offline_mode,
        ee_project=ee_project,
        dw_start_year=dw_start_year,
        dw_end_year=dw_end_year,
        dw_start_month=dw_start_month,
        dw_end_month=dw_end_month,
        viz_configuration_name=viz_configuration,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        vector_file=args.vector_file,
        dw_dataset_file=args.dw_dataset_file,
        jrc_dataset_file=args.jrc_dataset_file,
        precomputed_nrt_dir=args.precomputed_nrt_dir,
        offline_mode=args.offline_mode,
        ee_project=args.ee_project,
        dw_start_year=args.dw_start_year,
        dw_end_year=args.dw_end_year,
        dw_start_month=args.dw_start_month,
        dw_end_month=args.dw_end_month,
        viz_configuration=args.viz_configuration,
    )
