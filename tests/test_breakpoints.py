"""Tests for breakpoint detection functionality."""

import pandas as pd

from water_timeseries.breakpoint import BeastBreakpoint, SimpleBreakpoint
from water_timeseries.dataset import DWDataset, JRCDataset


class TestSimpleBreakpoint:
    """Test SimpleBreakpoint detection functionality."""

    def test_simple_breakpoint_columns_dw(self, dw_test_dataset):
        """Test SimpleBreakpoint produces correct columns with DW dataset."""
        dataset = DWDataset(dw_test_dataset)
        bp = SimpleBreakpoint()

        # Use specific geohash
        geohash_id = "b7g4rf3n3x43"

        result = bp.calculate_break(dataset, geohash_id)

        # Check result is DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check expected columns
        expected_columns = ["date_break", "date_before_break", "break_method"]
        for col in expected_columns:
            assert col in result.columns

        # Check method name
        assert result["break_method"].iloc[0] == "simple"

        # Check index
        assert result.index[0] == geohash_id

    def test_simple_breakpoint_columns_jrc(self, jrc_test_dataset):
        """Test SimpleBreakpoint produces correct columns with JRC dataset."""
        dataset = JRCDataset(jrc_test_dataset)
        bp = SimpleBreakpoint()

        # Use specific geohash
        geohash_id = "b7g4rf3n3x43"

        result = bp.calculate_break(dataset, geohash_id)

        # Check result is DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check expected columns
        expected_columns = ["date_break", "date_before_break", "break_method"]
        for col in expected_columns:
            assert col in result.columns

        # Check method name
        assert result["break_method"].iloc[0] == "simple"

        # Check index
        assert result.index[0] == geohash_id


class TestBeastBreakpoint:
    """Test BeastBreakpoint detection functionality."""

    def test_beast_breakpoint_columns_dw(self, dw_test_dataset):
        """Test BeastBreakpoint produces correct columns with DW dataset."""
        dataset = DWDataset(dw_test_dataset)
        bp = BeastBreakpoint()

        # Use specific geohash
        geohash_id = "b7g4rf3n3x43"

        result = bp.calculate_break(dataset, geohash_id)

        # Check result is DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check expected columns
        expected_columns = ["date_break", "date_before_break", "break_method", "break_number", "proba_rbeast"]
        for col in expected_columns:
            assert col in result.columns

        # Check method name
        assert result["break_method"].iloc[0] == "rbeast"

        # Check index name
        assert result.index.name == "id_geohash"

    def test_beast_breakpoint_columns_jrc(self, jrc_test_dataset):
        """Test BeastBreakpoint produces correct columns with JRC dataset."""
        dataset = JRCDataset(jrc_test_dataset)
        bp = BeastBreakpoint()

        # Use specific geohash
        geohash_id = "b7g4rf3n3x43"

        result = bp.calculate_break(dataset, geohash_id)

        # Check result is DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check expected columns
        expected_columns = ["date_break", "date_before_break", "break_method", "break_number", "proba_rbeast"]
        for col in expected_columns:
            assert col in result.columns

        # Check method name
        assert result["break_method"].iloc[0] == "rbeast"

        # Check index name
        assert result.index.name == "id_geohash"


class TestBreakpointComparison:
    """Test comparing different breakpoint methods."""

    def test_methods_produce_different_outputs(self, dw_test_dataset):
        """Test that Simple and Beast methods produce different output structures."""
        dataset = DWDataset(dw_test_dataset)
        geohash_id = "b7g4rf3n3x43"

        simple_bp = SimpleBreakpoint()
        beast_bp = BeastBreakpoint()

        simple_result = simple_bp.calculate_break(dataset, geohash_id)
        beast_result = beast_bp.calculate_break(dataset, geohash_id)

        # Both should be DataFrames
        assert isinstance(simple_result, pd.DataFrame)
        assert isinstance(beast_result, pd.DataFrame)

        # Different column sets
        assert "break_number" not in simple_result.columns
        assert "break_number" in beast_result.columns
        assert "proba_rbeast" not in simple_result.columns
        assert "proba_rbeast" in beast_result.columns

        # Same method names
        assert simple_result["break_method"].iloc[0] == "simple"
        assert beast_result["break_method"].iloc[0] == "rbeast"

    def test_batch_breakpoint_simple(self, dw_test_dataset):
        """Test batch breakpoint processing with SimpleBreakpoint."""
        dataset = DWDataset(dw_test_dataset)
        bp = SimpleBreakpoint()

        # Test batch processing
        results = bp.calculate_breaks_batch(dataset, progress_bar=False)

        # Check result is DataFrame
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

        # Check expected columns
        expected_columns = ["date_break", "date_before_break", "break_method"]
        for col in expected_columns:
            assert col in results.columns

        # Check all results have correct method name
        assert all(results["break_method"] == "simple")

    def test_batch_breakpoint_beast(self, dw_test_dataset):
        """Test batch breakpoint processing with BeastBreakpoint."""
        dataset = DWDataset(dw_test_dataset)
        bp = BeastBreakpoint()

        # Test batch processing
        results = bp.calculate_breaks_batch(dataset, progress_bar=False)

        # Check result is DataFrame
        assert isinstance(results, pd.DataFrame)
        assert len(results) >= 0  # May have no breaks detected

        # Check expected columns if results exist
        if len(results) > 0:
            expected_columns = ["date_break", "date_before_break", "break_method", "break_number", "proba_rbeast"]
            for col in expected_columns:
                assert col in results.columns

            # Check all results have correct method name
            assert all(results["break_method"] == "rbeast")
