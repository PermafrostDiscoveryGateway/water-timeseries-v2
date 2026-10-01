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


def _write_month(path: Path, month: str, ids: list[str]) -> None:
    """Write one tile's worth of drained-lake rows for *month*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "analysis_month": month,
            "id_geohash": ids,
            "water_residual": [-0.5] * len(ids),
        }
    ).to_parquet(path, index=False)


class TestAggregateNrtDirectory:
    """A month's tiles land in separate subdirectories; aggregation has to span them."""

    @staticmethod
    def _tiled_tree(tmp_path: Path) -> Path:
        # 2026-06 arrives as two tiles in two subdirs, 2026-07 as one.
        _write_month(tmp_path / "2026-06" / "tileA" / "nrt_2026-06_drain_breaks.parquet", "2026-06", ["a1", "a2"])
        _write_month(tmp_path / "2026-06" / "tileB" / "nrt_2026-06_drain_breaks.parquet", "2026-06", ["b1"])
        _write_month(tmp_path / "2026-07" / "tileA" / "nrt_2026-07_drain_breaks.parquet", "2026-07", ["c1", "c2"])
        return tmp_path

    def test_multiple_input_dirs_are_merged(self, tmp_path):
        from water_timeseries.scripts.cli import aggregate_nrt_directory

        root = self._tiled_tree(tmp_path / "nrt")
        out = tmp_path / "out"
        aggregate_nrt_directory(
            [root / "2026-06" / "tileA", root / "2026-06" / "tileB", root / "2026-07" / "tileA"],
            output_dir=out,
        )

        breaks = pd.read_parquet(out / "nrt_monthly_drain_breaks.parquet")
        assert sorted(breaks["id_geohash"]) == ["a1", "a2", "b1", "c1", "c2"]

        counts = pd.read_parquet(out / "nrt_monthly_drain_counts.parquet")
        # One row per month, not one per tile file.
        assert counts.set_index("analysis_month")["drained_lake_count"].to_dict() == {"2026-06": 3, "2026-07": 2}

    def test_recursive_walks_month_subdirs(self, tmp_path):
        from water_timeseries.scripts.cli import aggregate_nrt_directory

        root = self._tiled_tree(tmp_path / "nrt")
        out = tmp_path / "out"
        aggregate_nrt_directory(root, output_dir=out, recursive=True)

        breaks = pd.read_parquet(out / "nrt_monthly_drain_breaks.parquet")
        assert sorted(breaks["id_geohash"]) == ["a1", "a2", "b1", "c1", "c2"]

    def test_overlapping_inputs_are_not_double_counted(self, tmp_path):
        """A parent passed with --recursive alongside one of its own children."""
        from water_timeseries.scripts.cli import aggregate_nrt_directory

        root = self._tiled_tree(tmp_path / "nrt")
        out = tmp_path / "out"
        aggregate_nrt_directory([root, root / "2026-06" / "tileA"], output_dir=out, recursive=True)

        counts = pd.read_parquet(out / "nrt_monthly_drain_counts.parquet")
        assert counts.set_index("analysis_month")["drained_lake_count"].to_dict() == {"2026-06": 3, "2026-07": 2}

    def test_single_dir_still_accepted(self, tmp_path):
        """The old single-Path signature keeps working for existing callers."""
        from water_timeseries.scripts.cli import aggregate_nrt_directory

        flat = tmp_path / "flat"
        _write_month(flat / "nrt_2026-06_drain_breaks.parquet", "2026-06", ["a1"])
        aggregate_nrt_directory(flat)

        assert (flat / "nrt_monthly_drain_breaks.parquet").exists()
        counts = pd.read_parquet(flat / "nrt_monthly_drain_counts.parquet")
        assert counts.set_index("analysis_month")["drained_lake_count"].to_dict() == {"2026-06": 1}
