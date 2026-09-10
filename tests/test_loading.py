"""Tests for dataset loading and initialization."""

import pytest
import xarray as xr

from water_timeseries.dataset import DWDataset, JRCDataset


def _renamed(dataset: xr.Dataset, new_id: str) -> xr.Dataset:
    """Return a copy of the dataset whose id dimension/coord is renamed to `new_id`."""
    old = "id_geohash"
    return dataset.rename({old: new_id})


class TestDWDatasetLoading:
    """Test DWDataset loading and initialization."""

    def test_dw_dataset_initialization(self, dw_test_dataset):
        """Test that DWDataset initializes correctly."""
        ds = DWDataset(dw_test_dataset)
        assert ds is not None
        assert ds.water_column == "water"

    def test_dw_dataset_data_columns(self, dw_test_dataset):
        """Test that DWDataset has correct data columns."""
        ds = DWDataset(dw_test_dataset)
        expected_columns = [
            "water",
            "bare",
            "snow_and_ice",
            "trees",
            "grass",
            "flooded_vegetation",
            "crops",
            "shrub_and_scrub",
            "built",
        ]
        assert all(col in ds.data_columns for col in expected_columns)

    def test_dw_dataset_has_area_data(self, dw_test_dataset):
        """Test that DWDataset calculates area_data variable."""
        ds = DWDataset(dw_test_dataset)
        assert "area_data" in ds.ds.data_vars

    def test_dw_dataset_has_area_nodata(self, dw_test_dataset):
        """Test that DWDataset calculates area_nodata variable."""
        ds = DWDataset(dw_test_dataset)
        assert "area_nodata" in ds.ds.data_vars

    def test_dw_dataset_preprocessed(self, dw_test_dataset):
        """Test that DWDataset is marked as preprocessed."""
        ds = DWDataset(dw_test_dataset)
        assert ds.preprocessed_ is True


class TestJRCDatasetLoading:
    """Test JRCDataset loading and initialization."""

    def test_jrc_dataset_initialization(self, jrc_test_dataset):
        """Test that JRCDataset initializes correctly."""
        ds = JRCDataset(jrc_test_dataset)
        assert ds is not None
        assert ds.water_column == "area_water_permanent"

    def test_jrc_dataset_data_columns(self, jrc_test_dataset):
        """Test that JRCDataset has correct data columns."""
        ds = JRCDataset(jrc_test_dataset)
        expected_columns = ["area_water_permanent", "area_water_seasonal", "area_land"]
        assert all(col in ds.data_columns for col in expected_columns)

    def test_jrc_dataset_has_area_data(self, jrc_test_dataset):
        """Test that JRCDataset calculates area_data variable."""
        ds = JRCDataset(jrc_test_dataset)
        assert "area_data" in ds.ds.data_vars

    def test_jrc_dataset_preprocessed(self, jrc_test_dataset):
        """Test that JRCDataset is marked as preprocessed."""
        ds = JRCDataset(jrc_test_dataset)
        assert ds.preprocessed_ is True


class TestIdField:
    """Test that the id_field can be any coordinate name, e.g. id_geohash or lake_id."""

    def test_dw_default_id_field_is_id_geohash(self, dw_test_dataset):
        ds = DWDataset(dw_test_dataset)
        assert ds.id_field == "id_geohash"
        assert "id_geohash" in ds.ds.coords
        assert ds.object_ids_

    def test_dw_explicit_id_geohash(self, dw_test_dataset):
        ds = DWDataset(dw_test_dataset, id_field="id_geohash")
        assert ds.id_field == "id_geohash"
        assert ds.object_ids_

    def test_dw_lake_id_field(self, dw_test_dataset):
        """A dataset whose id dimension is named 'lake_id' works via id_field='lake_id'."""
        ds = DWDataset(_renamed(dw_test_dataset, "lake_id"), id_field="lake_id")
        assert ds.id_field == "lake_id"
        assert "lake_id" in ds.ds.coords
        assert list(ds.object_ids_) == list(_renamed(dw_test_dataset, "lake_id").coords["lake_id"].values)

    def test_jrc_default_id_field_is_id_geohash(self, jrc_test_dataset):
        ds = JRCDataset(jrc_test_dataset)
        assert ds.id_field == "id_geohash"
        assert "id_geohash" in ds.ds.coords

    def test_jrc_explicit_id_geohash(self, jrc_test_dataset):
        ds = JRCDataset(jrc_test_dataset, id_field="id_geohash")
        assert ds.id_field == "id_geohash"
        assert ds.object_ids_

    def test_jrc_lake_id_field(self, jrc_test_dataset):
        """A dataset whose id dimension is named 'lake_id' works via id_field='lake_id'."""
        ds = JRCDataset(_renamed(jrc_test_dataset, "lake_id"), id_field="lake_id")
        assert ds.id_field == "lake_id"
        assert "area_data" in ds.ds.data_vars
        assert list(ds.object_ids_) == list(_renamed(jrc_test_dataset, "lake_id").coords["lake_id"].values)

    def test_invalid_id_field_raises(self, dw_test_dataset):
        """An id_field that is not in the dataset must raise a helpful ValueError."""
        with pytest.raises(ValueError) as exc_info:
            DWDataset(dw_test_dataset, id_field="not_a_field")
        msg = str(exc_info.value)
        assert "not_a_field" in msg
        # The message must list what actually IS in the dataset, so the user can
        # pick the right id_field (e.g. id_geohash here, or lake_id for JRC exports).
        assert "id_geohash" in msg
        assert "Available" in msg

    def test_invalid_id_field_raises_jrc(self, jrc_test_dataset):
        with pytest.raises(ValueError) as exc_info:
            JRCDataset(jrc_test_dataset, id_field="does_not_exist")
        assert "does_not_exist" in str(exc_info.value)
        assert "id_geohash" in str(exc_info.value)


class TestObjectIdsAccess:
    """Test that object_ids_ works for both id_geohash and lake_id datasets."""

    def test_dw_object_ids(self, dw_test_dataset):
        ds = DWDataset(dw_test_dataset)
        assert list(ds.object_ids_) == list(dw_test_dataset.coords["id_geohash"].values)
        assert all(isinstance(v, (str, int)) for v in ds.object_ids_)

    def test_dw_object_ids_lake_id(self, dw_test_dataset):
        renamed = _renamed(dw_test_dataset, "lake_id")
        ds = DWDataset(renamed, id_field="lake_id")
        assert list(ds.object_ids_) == list(renamed.coords["lake_id"].values)

    def test_jrc_object_ids(self, jrc_test_dataset):
        ds = JRCDataset(jrc_test_dataset)
        assert list(ds.object_ids_) == list(jrc_test_dataset.coords["id_geohash"].values)

    def test_jrc_object_ids_lake_id(self, jrc_test_dataset):
        renamed = _renamed(jrc_test_dataset, "lake_id")
        ds = JRCDataset(renamed, id_field="lake_id")
        assert list(ds.object_ids_) == list(renamed.coords["lake_id"].values)
