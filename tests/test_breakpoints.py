"""Tests for breakpoint detection functionality."""

import pandas as pd

from water_timeseries.breakpoint import BeastBreakpoint, NRTBreakpoint, SimpleBreakpoint
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
            "date_break_year",
            "date_break_month",
            "water_change_ha",
            "water_change_perc",
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
            "date_break_year",
            "date_break_month",
            "water_change_ha",
            "water_change_perc",
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
            "date_break_year",
            "date_break_month",
            "water_change_ha",
            "water_change_perc",
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
            "date_break_year",
            "date_break_month",
            "water_change_ha",
            "water_change_perc",
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


class TestNrtBreakpointBatch:
    """Test batch breakpoint calculation with NrtBreakpoint."""

    def test_batch_breakpoint_nrt_dw_date_dt(self, dw_test_dataset):
        """Test NrtBreakpoint detection functionality."""
        dataset = DWDataset(dw_test_dataset)
        bp = NRTBreakpoint()
        breaks = bp.calculate_break(dataset, analysis_date=pd.Timestamp("2024-07-01"))
        assert isinstance(breaks, pd.DataFrame)

    def test_batch_breakpoint_nrt_dw_date_str(self, dw_test_dataset):
        """Test NrtBreakpoint detection functionality."""
        dataset = DWDataset(dw_test_dataset)
        bp = NRTBreakpoint()
        breaks = bp.calculate_break(dataset, analysis_date="2024-07-01")
        assert isinstance(breaks, pd.DataFrame)

    def test_batch_breakpoint_nrt_dw_date_str_short(self, dw_test_dataset):
        """Test NrtBreakpoint detection functionality."""
        dataset = DWDataset(dw_test_dataset)
        bp = NRTBreakpoint()
        breaks = bp.calculate_break(dataset, analysis_date="2024-07")
        assert isinstance(breaks, pd.DataFrame)

    def test_batch_breakpoint_nrt_jrc_date_dt(self, jrc_test_dataset):
        """Test NrtBreakpoint detection functionality."""
        dataset = JRCDataset(jrc_test_dataset)
        bp = NRTBreakpoint()
        breaks = bp.calculate_break(dataset, analysis_date=pd.Timestamp("2020-01-01"))
        assert isinstance(breaks, pd.DataFrame)

    def test_batch_breakpoint_nrt_jrc_date_str(self, jrc_test_dataset):
        """Test NrtBreakpoint detection functionality."""
        dataset = JRCDataset(jrc_test_dataset)
        bp = NRTBreakpoint()
        breaks = bp.calculate_break(dataset, analysis_date="2020-01-01")
        assert isinstance(breaks, pd.DataFrame)

    def test_nrt_breakpoint_output_columns_with_absolute_values(self, dw_test_dataset):
        """Test NrtBreakpoint produces correct output columns including absolute values."""
        dataset = DWDataset(dw_test_dataset)
        bp = NRTBreakpoint()

        result = bp.calculate_break(dataset, analysis_date="2024-07")

        # Check result is DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check expected columns match output_columns
        expected_columns = [
            "date",
            "water_observed",
            "water_predicted",
            "water_residual",
            "water_predicted_lower_90",
            "water_predicted_upper_90",
            "water_historical_mean",
            "water_historical_median",
            "water_historical_std",
            "water_historical_min",
            "water_historical_max",
            "drainage_confidence",
            # absolute values
            "water_observed_absolute",
            "water_predicted_absolute",
            "water_residual_absolute",
            "water_predicted_lower_90_absolute",
            "water_predicted_upper_90_absolute",
            "water_historical_mean_absolute",
            "water_historical_median_absolute",
            "water_historical_std_absolute",
            "water_historical_min_absolute",
            "water_historical_max_absolute",
            # confidence interval strings
            "water_predicted_ci",
            "water_predicted_ci_absolute",
        ]

        for col in expected_columns:
            assert col in result.columns, f"Expected column {col} not found in result"

        # Check that output_columns match
        assert result.columns.tolist() == bp.output_columns

    def test_nrt_breakpoint_absolute_values_are_larger_than_normalized(self, dw_test_dataset):
        """Test that absolute values are larger than normalized values (scaling factor > 1)."""
        dataset = DWDataset(dw_test_dataset)
        bp = NRTBreakpoint()

        result = bp.calculate_break(dataset, analysis_date="2024-07")

        # Check that absolute values are in a reasonable range (larger than normalized)
        # The scaling factor should make absolute values larger than 1 for most lakes
        if len(result) > 0:
            # Get scaling factors from dataset
            scaling_factors = dataset.ds.max(dim="date")["area_data"].to_dataframe()

            # Check that water_observed < water_observed_absolute for most entries
            # (unless scaling factor is very small)
            for idx in result.index[:5]:  # Check first 5 entries
                if idx in scaling_factors.index:
                    scaling = scaling_factors.loc[idx, "area_data"]
                    # If scaling factor > 1, absolute should be larger
                    if scaling > 1:
                        assert result.loc[idx, "water_observed_absolute"] >= result.loc[idx, "water_observed"], (
                            f"Absolute value should be >= normalized for {idx}"
                        )


class TestNrtDrainageConfidence:
    """Test the drainage_confidence scale produced by NRTBreakpoint."""

    @staticmethod
    def _score(**overrides) -> pd.DataFrame:
        """One-row frame with normal (non-draining) values, patched by overrides."""
        row = {
            "water_observed": 0.9,
            "water_residual": 0.0,
            "water_predicted_lower_90": 0.8,
            "water_historical_min": 0.3,
        }
        row.update(overrides)
        return NRTBreakpoint()._add_confidence_level(pd.DataFrame([row]))

    def test_no_criterion_met_is_na_not_zero(self):
        """A lake that was scored and is not draining must not land on 0."""
        result = self._score()
        assert pd.isna(result["drainage_confidence"].iloc[0])
        assert (result["drainage_confidence"] == 0).sum() == 0

    def test_criteria_sum_to_confidence(self):
        # Cat 1 only.
        assert self._score(water_residual=-0.5)["drainage_confidence"].iloc[0] == 1
        # Cat 1 + Cat 2.
        assert self._score(water_residual=-0.5, water_observed=0.5)["drainage_confidence"].iloc[0] == 2
        # Cat 1 + Cat 2 + Cat 3.
        assert self._score(water_residual=-0.5, water_observed=0.1)["drainage_confidence"].iloc[0] == 3

    def test_unevaluable_rows_are_invalid(self):
        """Missing criteria inputs score -1, not 0 via NaN comparisons."""
        for missing in ("water_residual", "water_observed", "water_predicted_lower_90", "water_historical_min"):
            result = self._score(**{missing: float("nan")})
            assert result["drainage_confidence"].iloc[0] == -1, missing

    def test_missing_column_is_invalid(self):
        """A frame without a criteria column at all scores -1 rather than raising."""
        df = pd.DataFrame([{"water_observed": 0.9, "water_residual": -0.5}])
        result = NRTBreakpoint()._add_confidence_level(df)
        assert result["drainage_confidence"].iloc[0] == -1

    def test_object_dtype_nan_rows_are_invalid(self):
        """The all-NaN placeholder frames built for lakes with no prediction."""
        df = pd.DataFrame(index=["a", "b"], columns=list(NRTBreakpoint().output_columns))
        result = NRTBreakpoint()._add_confidence_level(df)
        assert result["drainage_confidence"].tolist() == [-1, -1]

    def test_dtype_is_nullable_int(self):
        result = self._score()
        assert str(result["drainage_confidence"].dtype) == "Int64"

    def test_confidence_survives_parquet_roundtrip(self, tmp_path):
        df = pd.concat(
            [self._score(), self._score(water_residual=-0.5), self._score(water_observed=float("nan"))],
            ignore_index=True,
        )
        path = tmp_path / "breaks.parquet"
        df.to_parquet(path, index=False)
        roundtripped = pd.read_parquet(path)["drainage_confidence"]
        assert roundtripped.iloc[1] == 1
        assert roundtripped.iloc[2] == -1
        assert pd.isna(roundtripped.iloc[0])


class TestDrainedMask:
    """Test which NRT breaks rows count as drainage detections."""

    @staticmethod
    def _breaks() -> pd.DataFrame:
        """One row of each shape the breaks table actually mixes."""
        return pd.DataFrame(
            {
                "id_geohash": ["cat1", "cat2_only", "stable", "invalid", "historical_simple", "legacy_nrt"],
                "water_residual": [-0.5, -0.1, 0.02, float("nan"), float("nan"), -0.4],
                "drainage_confidence": pd.array([3, 1, pd.NA, -1, pd.NA, pd.NA], dtype="Int64"),
            }
        )

    def test_keeps_detections_drops_stable_and_invalid(self):
        from water_timeseries.utils.nrt_postprocessing import drained_mask

        result = self._breaks().set_index("id_geohash")
        kept = drained_mask(result)
        assert kept.loc["cat1"]
        # A lake below its prediction interval but with a shallow residual is a
        # real detection; a residual-only filter would drop it.
        assert kept.loc["cat2_only"]
        assert not kept.loc["stable"]
        assert not kept.loc["invalid"]

    def test_keeps_rows_that_were_never_scored(self):
        """Historical simple-method breaks carry no NRT columns and are detections."""
        from water_timeseries.utils.nrt_postprocessing import drained_mask

        kept = drained_mask(self._breaks().set_index("id_geohash"))
        assert kept.loc["historical_simple"]
        assert kept.loc["legacy_nrt"]

    def test_table_without_confidence_column_is_untouched(self):
        from water_timeseries.utils.nrt_postprocessing import drained_mask

        df = pd.DataFrame({"id_geohash": ["a", "b"], "water_change_perc": [-65.0, -87.0]})
        assert drained_mask(df).all()

    def test_empty_table(self):
        from water_timeseries.utils.nrt_postprocessing import drained_mask

        assert drained_mask(pd.DataFrame(columns=["drainage_confidence"])).empty

    def test_agrees_with_calculate_break_output(self, dw_test_dataset):
        """The stable lakes NRT now scores <NA> must not survive the mask."""
        from water_timeseries.utils.nrt_postprocessing import drained_mask

        result = NRTBreakpoint().calculate_break(DWDataset(dw_test_dataset), analysis_date="2024-07")
        kept = drained_mask(result)
        assert kept.sum() == (result["drainage_confidence"] >= 1).sum()
        assert kept.sum() < len(result), "fixture should contain stable lakes to drop"
