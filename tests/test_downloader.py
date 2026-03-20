"""
Test suite for EarthEngineDownloader class.

This module contains comprehensive tests for authentication, downloading
Dynamic World data, JRC data, and error handling.
"""

import pathlib
import unittest.mock as mock

import pytest

from water_timeseries.downloader import EarthEngineDownloader, setup_dates_from_options
from water_timeseries.utils.spatial import filter_gdf_by_bbox

# Path to test data
TEST_DATA_DIR = pathlib.Path(__file__).parent / "data"
VECTOR_DATASET = TEST_DATA_DIR / "lake_polygons.parquet"


class TestEarthEngineDownloaderAuthentication:
    """Test authentication and initialization of EarthEngineDownloader."""

    def test_init_with_valid_ee_project(self):
        """Test initialization with a valid ee_project argument."""
        downloader = EarthEngineDownloader(ee_project="my-valid-project")
        assert downloader.ee_project == "my-valid-project"

    def test_init_with_empty_string_ee_project(self):
        """Test that empty string ee_project raises ValueError."""
        with pytest.raises(ValueError, match="ee_project must be provided or set as EE_PROJECT environment variable"):
            EarthEngineDownloader(ee_project="")

    def test_init_with_none_and_env_var_set(self, monkeypatch):
        """Test initialization with None ee_project uses EE_PROJECT env var."""
        monkeypatch.setenv("EE_PROJECT", "env-project-id")
        downloader = EarthEngineDownloader(ee_project=None)
        assert downloader.ee_project == "env-project-id"

    def test_init_with_none_and_no_env_var(self, monkeypatch):
        """Test that None ee_project without env var raises ValueError."""
        monkeypatch.delenv("EE_PROJECT", raising=False)
        with pytest.raises(ValueError, match="ee_project must be provided"):
            EarthEngineDownloader(ee_project=None)

    def test_argument_overwrites_env_var(self, monkeypatch):
        """Test that ee_project argument takes precedence over EE_PROJECT env var."""
        monkeypatch.setenv("EE_PROJECT", "env-project-id")
        downloader = EarthEngineDownloader(ee_project="arg-project-id")
        assert downloader.ee_project == "arg-project-id"

    def test_init_with_none_uses_env_var_even_with_auth(self, monkeypatch):
        """Test that None ee_project works with ee_auth=True when env var is set."""
        monkeypatch.setenv("EE_PROJECT", "env-project-id")
        with mock.patch("geemap.ee_initialize"):
            downloader = EarthEngineDownloader(ee_project=None, ee_auth=True)
            assert downloader.ee_project == "env-project-id"
            assert downloader.ee_auth is True

    def test_invalid_env_var_type(self, monkeypatch):
        """Test that non-string EE_PROJECT env var is ignored."""
        monkeypatch.setenv("EE_PROJECT", "")
        with pytest.raises(ValueError, match="ee_project must be provided"):
            EarthEngineDownloader(ee_project=None)


class TestSpatialFiltering:
    """Test spatial filtering functionality."""

    def test_filter_gdf_by_bboxAlaska(self):
        """Test spatial bbox filtering on test dataset."""
        import geopandas as gpd

        # Load test dataset
        gdf = gpd.read_parquet(VECTOR_DATASET)

        # Verify initial count
        assert len(gdf) == 118, f"Expected 118 features, got {len(gdf)}"

        # Apply bbox filter for Alaska region
        filtered_gdf = filter_gdf_by_bbox(
            gdf,
            bbox_west=-164.2,
            bbox_east=-164,
            bbox_south=66.5,
            bbox_north=66.55,
        )

        # Verify filtered count
        assert len(filtered_gdf) == 17, f"Expected 17 features after filtering, got {len(filtered_gdf)}"

    def test_filter_gdf_by_bbox_all_params(self):
        """Test spatial bbox filtering with all parameters provided."""
        import geopandas as gpd

        gdf = gpd.read_parquet(VECTOR_DATASET)

        # Filter with all bbox parameters
        filtered = filter_gdf_by_bbox(
            gdf,
            bbox_west=-164.2,
            bbox_east=-164,
            bbox_south=66.5,
            bbox_north=66.55,
        )

        assert isinstance(filtered, gpd.GeoDataFrame)
        assert len(filtered) < len(gdf)

    def test_filter_gdf_by_bbox_partial_params(self):
        """Test spatial bbox filtering with only west and east parameters."""
        import geopandas as gpd

        gdf = gpd.read_parquet(VECTOR_DATASET)

        # Filter with only west and east
        filtered = filter_gdf_by_bbox(
            gdf,
            bbox_west=-164.2,
            bbox_east=-164,
        )

        assert len(filtered) < len(gdf)

    def test_filter_gdf_by_bbox_no_params_raises_error(self):
        """Test that filtering without any bbox params raises ValueError."""
        import geopandas as gpd

        gdf = gpd.read_parquet(VECTOR_DATASET)

        with pytest.raises(ValueError, match="At least one bbox parameter must be provided"):
            filter_gdf_by_bbox(gdf)


class TestSetupDatesFromOptions:
    """Test the setup_dates_from_options helper function."""

    def test_date_list_valid_format(self):
        """Test that date_list in YYYY-MM format is converted correctly."""
        dates = setup_dates_from_options(date_list=["2017-06", "2018-07", "2019-08"])
        assert dates == ["2017-06-01", "2018-07-01", "2019-08-01"]

    def test_date_list_single_date(self):
        """Test date_list with a single date."""
        dates = setup_dates_from_options(date_list=["2020-12"])
        assert dates == ["2020-12-01"]

    def test_years_and_months(self):
        """Test that years and months are combined correctly."""
        dates = setup_dates_from_options(years=[2017, 2018], months=[6, 7])
        assert dates == ["2017-06-01", "2017-07-01", "2018-06-01", "2018-07-01"]

    def test_years_and_months_defaults(self):
        """Test that default years and months are applied."""
        dates = setup_dates_from_options()
        # Default years: 2017-2025, default months: 6,7,8,9
        expected_years = list(range(2017, 2026))
        expected_dates = []
        for year in expected_years:
            for month in [6, 7, 8, 9]:
                expected_dates.append(f"{year}-{month:02d}-01")
        assert dates == expected_dates

    def test_date_list_and_years_raises_error(self):
        """Test that providing both date_list and years raises ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            setup_dates_from_options(date_list=["2017-06"], years=[2017])

    def test_date_list_and_months_raises_error(self):
        """Test that providing both date_list and months raises ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            setup_dates_from_options(date_list=["2017-06"], months=[6])

    def test_date_list_and_both_years_months_raises_error(self):
        """Test that providing date_list with years AND months raises ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            setup_dates_from_options(date_list=["2017-06"], years=[2017], months=[6])

    def test_only_years_raises_error(self):
        """Test that providing only years (without months) raises ValueError."""
        with pytest.raises(ValueError, match="both 'years' and 'months'"):
            setup_dates_from_options(years=[2017])

    def test_only_months_raises_error(self):
        """Test that providing only months (without years) raises ValueError."""
        with pytest.raises(ValueError, match="both 'years' and 'months'"):
            setup_dates_from_options(months=[6])

    def test_empty_date_list(self):
        """Test that empty date_list raises ValueError."""
        with pytest.raises(ValueError, match="both 'years' and 'months'"):
            setup_dates_from_options(date_list=[])


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
