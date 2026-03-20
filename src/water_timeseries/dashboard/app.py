"""Run the Streamlit dashboard."""

from pathlib import Path

from water_timeseries.dashboard.map_viewer import create_app


def main():
    """Run the dashboard app."""
    # Default paths to test data
    default_path = Path(__file__).parent.parent.parent.parent / "tests" / "data" / "lake_polygons.parquet"
    default_zarr_path = Path(__file__).parent.parent.parent.parent / "tests" / "data" / "lakes_dw_test.zarr"
    create_app(data_path=default_path, zarr_path=default_zarr_path)


if __name__ == "__main__":
    main()
