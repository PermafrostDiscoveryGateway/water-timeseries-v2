"""Tests for committed NRT dashboard fixtures."""

from pathlib import Path

import pandas as pd

NRT_DIR = Path(__file__).parent / "data" / "nrt"
COUNTS_FILE = NRT_DIR / "nrt_monthly_drain_counts.parquet"
BREAKS_FILE = NRT_DIR / "nrt_monthly_drain_breaks.parquet"


class TestNrtTestData:
    """Validate dashboard NRT parquet fixtures."""

    def test_nrt_fixture_files_exist(self):
        assert COUNTS_FILE.exists()
        assert BREAKS_FILE.exists()

    def test_nrt_counts_schema(self):
        counts = pd.read_parquet(COUNTS_FILE)
        assert list(counts.columns) == ["analysis_month", "drained_lake_count"]
        assert len(counts) > 0
        assert counts["drained_lake_count"].ge(0).all()

    def test_nrt_breaks_schema_and_threshold(self):
        breaks = pd.read_parquet(BREAKS_FILE)
        required = {
            "analysis_month",
            "id_geohash",
            "water_residual",
            "water_observed",
            "water_predicted",
            "date",
        }
        assert required.issubset(breaks.columns)
        assert (breaks["water_residual"] < -0.25).all()
        assert set(breaks["analysis_month"]).issubset(set(pd.read_parquet(COUNTS_FILE)["analysis_month"]))

    def test_nrt_counts_match_breaks(self):
        counts = pd.read_parquet(COUNTS_FILE)
        breaks = pd.read_parquet(BREAKS_FILE)
        expected = breaks.groupby("analysis_month").size().rename("expected_count")
        merged = counts.set_index("analysis_month").join(expected, how="left").fillna(0)
        assert (merged["drained_lake_count"] == merged["expected_count"]).all()
