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
        expected_columns = [
            "date_break",
            "date_before_break",
            "date_after_break",
            "break_method",
            "pre_break_mean",
            "pre_break_median",
            "pre_break_std",
            "pre_break_min",
            "pre_break_max",
            "post_break_mean",
            "post_break_median",
            "post_break_std",
            "post_break_min",
            "post_break_max",
        ]

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
        expected_columns = [
            "date_break",
            "date_before_break",
            "date_after_break",
            "break_method",
            "pre_break_mean",
            "pre_break_median",
            "pre_break_std",
            "pre_break_min",
            "pre_break_max",
            "post_break_mean",
            "post_break_median",
            "post_break_std",
            "post_break_min",
            "post_break_max",
        ]
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
        expected_columns = [
            "date_break",
            "date_before_break",
            "break_method",
            "break_number",
            "proba_rbeast",
            "pre_break_mean",
            "pre_break_median",
            "pre_break_std",
            "pre_break_min",
            "pre_break_max",
            "post_break_mean",
            "post_break_median",
            "post_break_std",
            "post_break_min",
            "post_break_max",
        ]
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
        expected_columns = [
            "date_break",
            "date_before_break",
            "break_method",
            "break_number",
            "proba_rbeast",
            "pre_break_mean",
            "pre_break_median",
            "pre_break_std",
            "pre_break_min",
            "pre_break_max",
            "post_break_mean",
            "post_break_median",
            "post_break_std",
            "post_break_min",
            "post_break_max",
        ]
        for col in expected_columns:
            assert col in result.columns

        # Check method name
        assert result["break_method"].iloc[0] == "rbeast"

        # Check index name
        assert result.index.name == "id_geohash"

    def test_beast_breakpoint_b7uefy0bvcrc_jrc(self, jrc_test_dataset):
        """Test BeastBreakpoint on JRC dataset with known break date.
        
        The test dataset should have a first break at 2018-01-01.
        """
        import pandas as pd
        
        dataset = JRCDataset(jrc_test_dataset)
        bp = BeastBreakpoint()
        
        # Use specific geohash
        geohash_id = "b7uefy0bvcrc"
        
        result = bp.calculate_break(dataset, geohash_id)
        
        # Assert that it has breaks
        assert len(result) > 0, f"Expected breaks for geohash {geohash_id}"
        
        # Assert first_break is 2018-01-01
        first_break = result["date_break"].iloc[0]
        expected_break = pd.Timestamp("2018-01-01")
        assert first_break == expected_break, f"Expected first break at {expected_break}, got {first_break}"

    def test_beast_breakpoint_b7uefy0bvcrc_dw(self, dw_test_dataset):
        """Test BeastBreakpoint on DW dataset with known break date.
        
        The test dataset should have a first break at either 2018-06-01 or 2018-07-01.
        """
        import pandas as pd
        
        dataset = DWDataset(dw_test_dataset)
        bp = BeastBreakpoint()
        
        # Use specific geohash
        geohash_id = "b7uefy0bvcrc"
        
        result = bp.calculate_break(dataset, geohash_id)
        
        # Assert that it has breaks
        assert len(result) > 0, f"Expected breaks for geohash {geohash_id}"
        
        # Assert first_break is either 2018-06-01 or 2018-07-01
        first_break = result["date_break"].iloc[0]
        expected_breaks = [pd.Timestamp("2018-06-01"), pd.Timestamp("2018-07-01")]
        assert first_break in expected_breaks, f"Expected first break at 2018-06-01 or 2018-07-01, got {first_break}"


class TestSimpleBreakpointKnownBreak:
    """Test SimpleBreakpoint detection with known break dates."""

    def test_simple_breakpoint_b7uefy0bvcrc_jrc(self, jrc_test_dataset):
        """Test SimpleBreakpoint on JRC dataset with known break date.
        
        The test dataset should have a first break at 2018-01-01.
        """
        import pandas as pd
        
        dataset = JRCDataset(jrc_test_dataset)
        bp = SimpleBreakpoint()
        
        # Use specific geohash
        geohash_id = "b7uefy0bvcrc"
        
        result = bp.calculate_break(dataset, geohash_id)
        
        # Assert that it has breaks
        assert len(result) > 0, f"Expected breaks for geohash {geohash_id}"
        
        # Assert first_break is 2018-01-01
        first_break = result["date_break"].iloc[0]
        expected_break = pd.Timestamp("2018-01-01")
        assert first_break == expected_break, f"Expected first break at {expected_break}, got {first_break}"

    def test_simple_breakpoint_b7uefy0bvcrc_dw(self, dw_test_dataset):
        """Test SimpleBreakpoint on DW dataset with known break date.
        
        The test dataset should have a first break at either 2018-06-01 or 2018-07-01.
        """
        import pandas as pd
        
        dataset = DWDataset(dw_test_dataset)
        bp = SimpleBreakpoint()
        
        # Use specific geohash
        geohash_id = "b7uefy0bvcrc"
        
        result = bp.calculate_break(dataset, geohash_id)
        
        # Assert that it has breaks
        assert len(result) > 0, f"Expected breaks for geohash {geohash_id}"
        
        # Assert first_break is either 2018-06-01 or 2018-07-01
        first_break = result["date_break"].iloc[0]
        expected_breaks = [pd.Timestamp("2018-06-01"), pd.Timestamp("2018-07-01")]
        assert first_break in expected_breaks, f"Expected first break at 2018-06-01 or 2018-07-01, got {first_break}"

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


class TestBeastBreakpointBatch:
    """Test batch breakpoint calculation with BeastBreakpoint."""

    def test_batch_breakpoint_beast_jrc(self, jrc_test_dataset):
        """Test BeastBreakpoint batch detection functionality."""
        dataset = JRCDataset(jrc_test_dataset)
        bp = BeastBreakpoint()
        breaks = bp.calculate_breaks_batch(dataset)
        assert isinstance(breaks, pd.DataFrame)

    def test_batch_breakpoint_beast_dw(self, dw_test_dataset):
        """Test BeastBreakpoint batch detection functionality."""
        dataset = DWDataset(dw_test_dataset)
        bp = BeastBreakpoint()
        breaks = bp.calculate_breaks_batch(dataset)
        assert isinstance(breaks, pd.DataFrame)

    def test_batch_breakpoint_simple_jrc(self, jrc_test_dataset):
        """Test BeastBreakpoint batch detection functionality."""
        dataset = JRCDataset(jrc_test_dataset)
        bp = SimpleBreakpoint()
        breaks = bp.calculate_breaks_batch(dataset)
        assert isinstance(breaks, pd.DataFrame)

    def test_batch_breakpoint_simple_dw(self, dw_test_dataset):
        """Test BeastBreakpoint batch detection functionality."""
        dataset = DWDataset(dw_test_dataset)
        bp = SimpleBreakpoint()
        breaks = bp.calculate_breaks_batch(dataset)
        assert isinstance(breaks, pd.DataFrame)
