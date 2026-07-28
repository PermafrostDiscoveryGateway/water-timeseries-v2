"""Tests for LakeDataset merge functionality."""

import pytest

from water_timeseries.dataset import DWDataset, JRCDataset


class TestDWDatasetMerge:
    """Test DWDataset merge functionality."""

    def test_merge_both_creates_new_dataset(self, dw_test_dataset):
        """Test that merge with how='both' creates a new LakeDataset."""
        # Create two subsets from the test dataset with different dates
        ds1 = dw_test_dataset.isel(date=slice(0, 5))
        ds2 = dw_test_dataset.isel(date=slice(5, 10))

        dw1 = DWDataset(ds1)
        dw2 = DWDataset(ds2)

        merged = dw1.merge(dw2, how="both")

        assert isinstance(merged, DWDataset)
        assert merged.ds is not None

    def test_merge_both_combines_all_dates(self, dw_test_dataset):
        """Test that merge with how='both' combines all dates from both datasets."""
        all_dates = dw_test_dataset.coords["date"].values
        half = len(all_dates) // 2

        ds1 = dw_test_dataset.isel(date=slice(0, half))
        ds2 = dw_test_dataset.isel(date=slice(half, None))

        dw1 = DWDataset(ds1)
        dw2 = DWDataset(ds2)

        merged = dw1.merge(dw2, how="both")

        assert len(merged.ds.coords["date"]) == len(all_dates)

    def test_merge_date_adds_new_dates(self, dw_test_dataset):
        """Test that merge with how='date' adds new dates for same geohash."""
        all_dates = dw_test_dataset.coords["date"].values
        half = len(all_dates) // 2

        # Use first geohash and split dates
        ds1 = dw_test_dataset.isel(id_geohash=0, date=slice(0, half))
        ds2 = dw_test_dataset.isel(id_geohash=0, date=slice(half, None))

        dw1 = DWDataset(ds1)
        dw2 = DWDataset(ds2)

        merged = dw1.merge(dw2, how="date")

        assert len(merged.ds.coords["date"]) == (len(all_dates) - half)
        # Check dates are in order
        dates = merged.ds.coords["date"].values
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))

    def test_merge_date_warns_on_duplicate_dates(self, dw_test_dataset):
        """Test that merge with how='date' warns on duplicate dates."""
        ds1 = dw_test_dataset.isel(id_geohash=0, date=slice(0, 5))
        ds2 = dw_test_dataset.isel(id_geohash=0, date=slice(3, 8))  # Overlapping dates

        dw1 = DWDataset(ds1)
        dw2 = DWDataset(ds2)

        with pytest.warns(UserWarning, match="overlapping dates"):
            dw1.merge(dw2, how="date")

    def test_merge_date_fails_different_geohashes(self, dw_test_dataset):
        """Test that merge with how='date' fails if geohashes differ."""
        # Use two different geohashes
        ds1 = dw_test_dataset.isel(id_geohash=0, date=slice(0, 3))
        ds2 = dw_test_dataset.isel(id_geohash=1, date=slice(3, 6))

        dw1 = DWDataset(ds1)
        dw2 = DWDataset(ds2)

        with pytest.raises(ValueError, match="same id_geohash"):
            dw1.merge(dw2, how="date")

    def test_merge_id_geohash_adds_new_lakes(self, dw_test_dataset):
        """Test that merge with how='id_geohash' adds new lakes with same dates."""
        # Use same dates but different geohashes
        ds1 = dw_test_dataset.isel(id_geohash=0, date=slice(0, 5))
        ds2 = dw_test_dataset.isel(id_geohash=1, date=slice(0, 5))

        dw1 = DWDataset(ds1)
        dw2 = DWDataset(ds2)

        merged = dw1.merge(dw2, how="id_geohash")

        assert len(merged.ds.coords["id_geohash"]) == 2

    def test_merge_id_geohash_warns_on_duplicate_ids(self, dw_test_dataset):
        """Test that merge with how='id_geohash' warns on duplicate ids."""
        ds1 = dw_test_dataset.isel(id_geohash=slice(0, 2), date=slice(0, 3))
        ds2 = dw_test_dataset.isel(id_geohash=slice(1, 3), date=slice(0, 3))  # Overlapping

        dw1 = DWDataset(ds1)
        dw2 = DWDataset(ds2)

        with pytest.warns(UserWarning, match="overlapping"):
            dw1.merge(dw2, how="id_geohash")

    def test_merge_id_geohash_fails_different_dates(self, dw_test_dataset):
        """Test that merge with how='id_geohash' fails if dates differ."""
        ds1 = dw_test_dataset.isel(id_geohash=0, date=slice(0, 3))
        ds2 = dw_test_dataset.isel(id_geohash=1, date=slice(3, 6))

        dw1 = DWDataset(ds1)
        dw2 = DWDataset(ds2)

        with pytest.raises(ValueError, match="same dates"):
            dw1.merge(dw2, how="id_geohash")

    def test_merge_different_types_raises_error(self, dw_test_dataset, jrc_test_dataset):
        """Test that merging different dataset types raises TypeError."""
        dw_ds = DWDataset(dw_test_dataset.isel(date=slice(0, 5)))
        jrc_ds = JRCDataset(jrc_test_dataset.isel(date=slice(0, 5)))

        with pytest.raises(TypeError, match="different types"):
            dw_ds.merge(jrc_ds, how="both")

    def test_merge_different_variables_raises_error(self, dw_test_dataset):
        """Test that merging datasets with different variables raises ValueError."""
        ds_base = dw_test_dataset.isel(id_geohash=0, date=slice(0, 3))

        # Drop one variable to make them different
        ds1 = ds_base.drop_vars("water")
        ds2 = ds_base

        dw1 = DWDataset(ds1)
        dw2 = DWDataset(ds2)

        with pytest.raises(ValueError, match="different variables"):
            dw1.merge(dw2, how="both")

    def test_merge_invalid_how_raises_error(self, dw_test_dataset):
        """Test that invalid 'how' parameter raises ValueError."""
        ds1 = dw_test_dataset.isel(date=slice(0, 5))
        ds2 = dw_test_dataset.isel(date=slice(5, 10))

        dw1 = DWDataset(ds1)
        dw2 = DWDataset(ds2)

        with pytest.raises(ValueError, match="Invalid merge strategy"):
            dw1.merge(dw2, how="invalid")

    def test_merge_preserves_variables(self, dw_test_dataset):
        """Test that merged dataset preserves all variables."""
        ds1 = dw_test_dataset.isel(date=slice(0, 5))
        ds2 = dw_test_dataset.isel(date=slice(5, 10))

        dw1 = DWDataset(ds1)
        dw2 = DWDataset(ds2)

        merged = dw1.merge(dw2, how="both")

        # Check all expected variables are present
        expected_vars = {
            "water",
            "bare",
            "snow_and_ice",
            "trees",
            "grass",
            "flooded_vegetation",
            "crops",
            "shrub_and_scrub",
            "built",
        }
        assert set(merged.ds.data_vars) == expected_vars


class TestJRCDatasetMerge:
    """Test JRCDataset merge functionality."""

    def test_merge_both_creates_new_dataset(self, jrc_test_dataset):
        """Test that merge with how='both' creates a new JRCDataset."""
        ds1 = jrc_test_dataset.isel(date=slice(0, 5))
        ds2 = jrc_test_dataset.isel(date=slice(5, 10))

        jrc1 = JRCDataset(ds1)
        jrc2 = JRCDataset(ds2)

        merged = jrc1.merge(jrc2, how="both")

        assert isinstance(merged, JRCDataset)
        assert merged.ds is not None

    def test_merge_both_combines_all_dates(self, jrc_test_dataset):
        """Test that merge with how='both' combines all dates."""
        all_dates = jrc_test_dataset.coords["date"].values
        half = len(all_dates) // 2

        ds1 = jrc_test_dataset.isel(date=slice(0, half))
        ds2 = jrc_test_dataset.isel(date=slice(half, None))

        jrc1 = JRCDataset(ds1)
        jrc2 = JRCDataset(ds2)

        merged = jrc1.merge(jrc2, how="both")

        assert len(merged.ds.coords["date"]) == len(all_dates)

    def test_merge_date_adds_new_dates(self, jrc_test_dataset):
        """Test that merge with how='date' adds new dates for same geohash."""
        all_dates = jrc_test_dataset.coords["date"].values
        half = len(all_dates) // 2

        ds1 = jrc_test_dataset.isel(id_geohash=0, date=slice(0, half))
        ds2 = jrc_test_dataset.isel(id_geohash=0, date=slice(half, None))

        jrc1 = JRCDataset(ds1)
        jrc2 = JRCDataset(ds2)

        merged = jrc1.merge(jrc2, how="date")

        assert len(merged.ds.coords["date"]) == (len(all_dates) - half)

    def test_merge_date_warns_on_duplicate_dates(self, jrc_test_dataset):
        """Test that merge with how='date' warns on duplicate dates."""
        ds1 = jrc_test_dataset.isel(id_geohash=0, date=slice(0, 5))
        ds2 = jrc_test_dataset.isel(id_geohash=0, date=slice(3, 8))  # Overlapping

        jrc1 = JRCDataset(ds1)
        jrc2 = JRCDataset(ds2)

        with pytest.warns(UserWarning, match="overlapping dates"):
            jrc1.merge(jrc2, how="date")

    def test_merge_date_fails_different_geohashes(self, jrc_test_dataset):
        """Test that merge with how='date' fails if geohashes differ."""
        ds1 = jrc_test_dataset.isel(id_geohash=0, date=slice(0, 3))
        ds2 = jrc_test_dataset.isel(id_geohash=1, date=slice(3, 6))

        jrc1 = JRCDataset(ds1)
        jrc2 = JRCDataset(ds2)

        with pytest.raises(ValueError, match="same id_geohash"):
            jrc1.merge(jrc2, how="date")

    def test_merge_id_geohash_adds_new_lakes(self, jrc_test_dataset):
        """Test that merge with how='id_geohash' adds new lakes with same dates."""
        ds1 = jrc_test_dataset.isel(id_geohash=0, date=slice(0, 5))
        ds2 = jrc_test_dataset.isel(id_geohash=1, date=slice(0, 5))

        jrc1 = JRCDataset(ds1)
        jrc2 = JRCDataset(ds2)

        merged = jrc1.merge(jrc2, how="id_geohash")

        assert len(merged.ds.coords["id_geohash"]) == 2

    def test_merge_id_geohash_warns_on_duplicate_ids(self, jrc_test_dataset):
        """Test that merge with how='id_geohash' warns on duplicate ids."""
        ds1 = jrc_test_dataset.isel(id_geohash=slice(0, 2), date=slice(0, 3))
        ds2 = jrc_test_dataset.isel(id_geohash=slice(1, 3), date=slice(0, 3))  # Overlapping

        jrc1 = JRCDataset(ds1)
        jrc2 = JRCDataset(ds2)

        with pytest.warns(UserWarning, match="overlapping"):
            jrc1.merge(jrc2, how="id_geohash")

    def test_merge_id_geohash_fails_different_dates(self, jrc_test_dataset):
        """Test that merge with how='id_geohash' fails if dates differ."""
        ds1 = jrc_test_dataset.isel(id_geohash=0, date=slice(0, 3))
        ds2 = jrc_test_dataset.isel(id_geohash=1, date=slice(3, 6))

        jrc1 = JRCDataset(ds1)
        jrc2 = JRCDataset(ds2)

        with pytest.raises(ValueError, match="same dates"):
            jrc1.merge(jrc2, how="id_geohash")

    def test_merge_preserves_water_column(self, jrc_test_dataset):
        """Test that JRC dataset preserves water_column attribute."""
        ds = jrc_test_dataset.isel(id_geohash=0, date=slice(0, 3))
        jrc = JRCDataset(ds)

        assert jrc.water_column == "area_water_permanent"

    def test_merge_preserves_variables(self, jrc_test_dataset):
        """Test that merged JRC dataset preserves all variables."""
        ds1 = jrc_test_dataset.isel(date=slice(0, 5))
        ds2 = jrc_test_dataset.isel(date=slice(5, 10))

        jrc1 = JRCDataset(ds1)
        jrc2 = JRCDataset(ds2)

        merged = jrc1.merge(jrc2, how="both")

        expected_vars = {"area_water_permanent", "area_water_seasonal", "area_land"}
        assert set(merged.ds.data_vars) == expected_vars
