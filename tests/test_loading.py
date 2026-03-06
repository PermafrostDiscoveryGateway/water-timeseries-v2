"""Tests for dataset loading and initialization."""

import pytest
from water_timeseries.dataset import DWDataset, JRCDataset


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
        assert ds.data_columns == expected_columns
    
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
        assert ds.data_columns == expected_columns
    
    def test_jrc_dataset_has_area_data(self, jrc_test_dataset):
        """Test that JRCDataset calculates area_data variable."""
        ds = JRCDataset(jrc_test_dataset)
        assert "area_data" in ds.ds.data_vars
    
    def test_jrc_dataset_preprocessed(self, jrc_test_dataset):
        """Test that JRCDataset is marked as preprocessed."""
        ds = JRCDataset(jrc_test_dataset)
        assert ds.preprocessed_ is True
