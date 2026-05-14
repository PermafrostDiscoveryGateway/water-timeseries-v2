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
        help="Path to water dataset file (zarr). If not provided, uses default test data.",
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
    return parser.parse_args()


def main(
    vector_file: str | Path = None,
    dw_dataset_file: str | Path = None,
    precomputed_nrt_dir: str | Path = None,
):
    """Run the dashboard app.

    Args:
        vector_file: Path to vector dataset file (GeoParquet). Defaults to test data.
        dw_dataset_file: Path to water dataset file (zarr). Defaults to test data.
        precomputed_nrt_dir: Directory with pre-computed NRT parquet files.
            Auto-detected from ``precomputed/nrt/`` in the repo root when present.
    """
    # Default paths to test data
    default_vector_file = _REPO_ROOT / "tests" / "data" / "lake_polygons.parquet"
    default_dw_dataset_file = _REPO_ROOT / "tests" / "data" / "lakes_dw_test.zarr"

    # Validate provided file paths and fall back to defaults if they don't exist
    if vector_file is not None:
        path = Path(vector_file)
        if not path.exists():
            warnings.warn(f"Vector file not found: {vector_file}. Falling back to default test data.")
            vector_file = None

    if dw_dataset_file is not None:
        path = Path(dw_dataset_file)
        if not path.exists():
            warnings.warn(f"Water dataset file not found: {dw_dataset_file}. Falling back to default test data.")
            dw_dataset_file = None

    # Use provided paths or defaults
    if vector_file is None:
        vector_file = default_vector_file
    if dw_dataset_file is None:
        dw_dataset_file = default_dw_dataset_file

    # Auto-detect precomputed NRT directory if not explicitly provided
    if precomputed_nrt_dir is None:
        counts_file = _DEFAULT_NRT_DIR / "nrt_monthly_drain_counts.parquet"
        breaks_file = _DEFAULT_NRT_DIR / "nrt_monthly_drain_breaks.parquet"
        if counts_file.exists() or breaks_file.exists():
            precomputed_nrt_dir = _DEFAULT_NRT_DIR

    create_app(data_path=vector_file, zarr_path=dw_dataset_file, precomputed_nrt_dir=precomputed_nrt_dir)


if __name__ == "__main__":
    args = parse_args()
    main(
        vector_file=args.vector_file,
        dw_dataset_file=args.dw_dataset_file,
        precomputed_nrt_dir=args.precomputed_nrt_dir,
    )
