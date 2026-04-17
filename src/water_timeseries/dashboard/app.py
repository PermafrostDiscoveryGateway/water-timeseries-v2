"""Run the Streamlit dashboard."""

import argparse
import warnings
from pathlib import Path

from water_timeseries.dashboard.map_viewer import create_app


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
    return parser.parse_args()


def main(vector_file: str | Path = None, dw_dataset_file: str | Path = None):
    """Run the dashboard app.

    Args:
        vector_file: Path to vector dataset file (GeoParquet). Defaults to test data.
        dw_dataset_file: Path to water dataset file (zarr). Defaults to test data.
    """
    # Default paths to test data
    default_vector_file = Path(__file__).parent.parent.parent.parent / "tests" / "data" / "lake_polygons.parquet"
    default_dw_dataset_file = Path(__file__).parent.parent.parent.parent / "tests" / "data" / "lakes_dw_test.zarr"

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

    create_app(data_path=vector_file, zarr_path=dw_dataset_file)


if __name__ == "__main__":
    args = parse_args()
    main(
        vector_file=args.vector_file,
        dw_dataset_file=args.dw_dataset_file,
    )
