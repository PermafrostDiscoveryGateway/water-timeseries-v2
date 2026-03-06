"""Tests for dataset normalization."""

import numpy as np
import pytest
from water_timeseries.dataset import DWDataset, JRCDataset


class TestDWDatasetNormalization:
    """Test DWDataset normalization functionality."""
    
    def test_dw_normalized_available(self, dw_test_dataset):
        """Test that normalized data is available after initialization."""
        ds = DWDataset(dw_test_dataset)
        assert ds.normalized_available_ is True
    
    def test_dw_normalized_has_water(self, dw_test_dataset):
        """Test that normalized dataset contains water variable."""
        ds = DWDataset(dw_test_dataset)
        assert "water" in ds.ds_normalized.data_vars
    
    def test_dw_normalized_values_in_range(self, dw_test_dataset):
        """Test that normalized water values are in valid range [0, 1]."""
        ds = DWDataset(dw_test_dataset)
        water_normalized = ds.ds_normalized["water"].values
        # Allow for small numerical errors
        assert np.nanmin(water_normalized) >= -0.01
        assert np.nanmax(water_normalized) <= 1.01
    
    def test_dw_normalization_reduces_values(self, dw_test_dataset):
        """Test that normalization reduces values compared to original."""
        ds = DWDataset(dw_test_dataset)
        original_mean = float(ds.ds["water"].mean())
        normalized_mean = float(ds.ds_normalized["water"].mean())
        # Normalized mean should be smaller or equal
        assert normalized_mean <= original_mean
    
    def test_dw_normalized_max_near_one(self, dw_test_dataset):
        """Test that maximum normalized value is close to 1."""
        ds = DWDataset(dw_test_dataset)
        water_normalized = ds.ds_normalized["water"].values
        assert np.nanmax(water_normalized) > 0.5  # Should have values near 1


class TestJRCDatasetNormalization:
    """Test JRCDataset normalization functionality."""
    
    def test_jrc_normalized_available(self, jrc_test_dataset):
        """Test that normalized data is available after initialization."""
        ds = JRCDataset(jrc_test_dataset)
        assert ds.normalized_available_ is True
    
    def test_jrc_normalized_has_permanent_water(self, jrc_test_dataset):
        """Test that normalized dataset contains permanent water variable."""
        ds = JRCDataset(jrc_test_dataset)
        assert "area_water_permanent" in ds.ds_normalized.data_vars
    
    def test_jrc_normalized_values_in_range(self, jrc_test_dataset):
        """Test that normalized values are in valid range [0, 1]."""
        ds = JRCDataset(jrc_test_dataset)
        permanent_normalized = ds.ds_normalized["area_water_permanent"].values
        # Allow for small numerical errors
        assert np.nanmin(permanent_normalized) >= -0.01
        assert np.nanmax(permanent_normalized) <= 1.01
    
    def test_jrc_all_vars_normalized(self, jrc_test_dataset):
        """Test that all variables are normalized."""
        ds = JRCDataset(jrc_test_dataset)
        for var in ds.data_columns:
            assert var in ds.ds_normalized.data_vars
            values = ds.ds_normalized[var].values
            assert np.nanmin(values) >= -0.01
            assert np.nanmax(values) <= 1.01
