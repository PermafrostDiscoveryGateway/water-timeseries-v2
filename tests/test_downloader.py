"""
Test suite for EarthEngineDownloader class.

This module contains comprehensive tests for authentication, downloading
Dynamic World data, JRC data, and error handling.
"""

import pathlib
from unittest import mock

import ee
import geopandas as gpd
import pandas as pd
import pytest
import requests

from water_timeseries.downloader import (
    EST_ACQUISITIONS_PER_MONTH,
    EarthEngineDownloader,
    setup_annual_dates,
    setup_dates_from_options,
)
from water_timeseries.utils.data import dw_bandnames, dw_coverage_stats
from water_timeseries.utils.earthengine import (
    NoDynamicWorldDataError,
    calc_monthly_dw,
    call_with_retry,
    month_slices,
    month_windows,
    season_windows,
    windows_from_date_list,
)
from water_timeseries.utils.spatial import filter_gdf_by_bbox

# Path to test data
TEST_DATA_DIR = pathlib.Path(__file__).parent / "data"
VECTOR_DATASET = TEST_DATA_DIR / "lake_polygons.parquet"


class TestEarthEngineDownloaderAuthentication:
    """Test authentication and initialization of EarthEngineDownloader."""

    def test_init_with_valid_ee_project(self, monkeypatch):
        """Test initialization with a valid ee_project argument."""
        monkeypatch.setenv("EE_PROJECT", "my-valid-project")
        downloader = EarthEngineDownloader()
        assert downloader.ee_project == "my-valid-project"

    def test_init_with_empty_string_ee_project(self, monkeypatch):
        """Test that empty string ee_project raises ValueError."""
        monkeypatch.setenv("EE_PROJECT", "")
        with pytest.raises(ValueError, match="ee_project must be provided or set as EE_PROJECT environment variable"):
            EarthEngineDownloader()

    def test_init_with_none_and_env_var_set(self, monkeypatch):
        """Test initialization with None ee_project uses EE_PROJECT env var."""
        monkeypatch.setenv("EE_PROJECT", "env-project-id")
        downloader = EarthEngineDownloader()
        assert downloader.ee_project == "env-project-id"

    def test_init_with_none_and_no_env_var(self, monkeypatch):
        """Test that None ee_project without env var raises ValueError."""
        monkeypatch.delenv("EE_PROJECT", raising=False)
        with pytest.raises(ValueError, match="ee_project must be provided"):
            EarthEngineDownloader()

    def test_argument_overwrites_env_var(self, monkeypatch):
        """Test that ee_project argument takes precedence over EE_PROJECT env var."""
        monkeypatch.setenv("EE_PROJECT", "env-project-id")
        downloader = EarthEngineDownloader(ee_project="arg-project-id")
        assert downloader.ee_project == "arg-project-id"

    def test_init_with_none_uses_env_var_even_with_auth(self, monkeypatch):
        """Test that None ee_project works with ee_auth=True when env var is set."""
        monkeypatch.setenv("EE_PROJECT", "env-project-id")
        with mock.patch("water_timeseries.utils.earthengine.initialize_earth_engine"):
            downloader = EarthEngineDownloader(ee_auth=True)
            assert downloader.ee_project == "env-project-id"
            assert downloader.ee_auth is True

    def test_invalid_env_var_type(self, monkeypatch):
        """Test that non-string EE_PROJECT env var is ignored."""
        monkeypatch.setenv("EE_PROJECT", "")
        with pytest.raises(ValueError, match="ee_project must be provided"):
            EarthEngineDownloader()


class TestSpatialFiltering:
    """Test spatial filtering functionality."""

    def test_filter_gdf_by_bboxAlaska(self, monkeypatch):
        """Test spatial bbox filtering on test dataset."""
        import geopandas as gpd

        # monkeypatch.setenv("EE_PROJECT", "test-project")

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

    def test_filter_gdf_by_bbox_all_params(self, monkeypatch):
        """Test spatial bbox filtering with all parameters provided."""
        import geopandas as gpd

        # monkeypatch.setenv("EE_PROJECT", "test-project")

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

    def test_filter_gdf_by_bbox_partial_params(self, monkeypatch):
        """Test spatial bbox filtering with only west and east parameters."""
        import geopandas as gpd

        # monkeypatch.setenv("EE_PROJECT", "test-project")

        gdf = gpd.read_parquet(VECTOR_DATASET)

        # Filter with only west and east
        filtered = filter_gdf_by_bbox(
            gdf,
            bbox_west=-164.2,
            bbox_east=-164,
        )

        assert len(filtered) < len(gdf)

    def test_filter_gdf_by_bbox_no_params_raises_error(self, monkeypatch):
        """Test that filtering without any bbox params raises ValueError."""
        import geopandas as gpd

        # monkeypatch.setenv("EE_PROJECT", "test-project")

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


class TestSetupAnnualDates:
    """Test the setup_annual_dates helper function."""

    def test_years_list(self):
        """Test that years list is converted to dates correctly."""
        dates = setup_annual_dates(years=[2017, 2018, 2019])
        assert dates == ["2017-01-01", "2018-01-01", "2019-01-01"]

    def test_default_years(self):
        """Test that default years are applied (2000-2021)."""
        dates = setup_annual_dates()
        expected_dates = [f"{year}-01-01" for year in range(2000, 2022)]
        assert dates == expected_dates


class TestJRCDownloader:
    """Test JRC annual download functionality."""

    def test_jrc_bandnames(self, monkeypatch):
        """Test that JRC band names are correctly defined."""
        # monkeypatch.setenv("EE_PROJECT", "test-project")
        downloader = EarthEngineDownloader()
        expected_bands = [
            "area_nodata",
            "area_land",
            "area_water_seasonal",
            "area_water_permanent",
        ]
        assert downloader.jrc_bandnames == expected_bands

    def test_setup_jrc_reducer(self, monkeypatch):
        """Test that JRC reducer is correctly configured."""
        import geopandas as gpd

        # monkeypatch.setenv("EE_PROJECT", "test-project")
        downloader = EarthEngineDownloader()

        # Load test data
        gdf = gpd.read_parquet(VECTOR_DATASET)

        # Setup reducer
        _fc, reducer_dict = downloader._setup_jrc_reducer(gdf, feature_index_name="id_geohash", scale=30)

        # Verify reducer configuration
        assert "reducer" in reducer_dict
        assert "collection" in reducer_dict
        assert "crs" in reducer_dict
        assert "scale" in reducer_dict
        assert reducer_dict["scale"] == 30
        assert reducer_dict["bands"] == downloader.jrc_bandnames

    def test_download_jrc_annual_no_download_mode(self, monkeypatch):
        """Test that download_jrc_annual works in no_download mode."""
        # monkeypatch.setenv("EE_PROJECT", "test-project")
        downloader = EarthEngineDownloader()

        # Test no_download mode - should return None but log the parameters
        result = downloader.download_jrc_annual(
            vector_dataset=VECTOR_DATASET,
            name_attribute="id_geohash",
            years=[2017, 2018],
            no_download=True,
        )

        # Should return None in no_download mode
        assert result is None

    def test_download_jrc_annual_with_id_list(self, monkeypatch):
        """Test download_jrc_annual with ID filtering."""
        # monkeypatch.setenv("EE_PROJECT", "test-project")
        downloader = EarthEngineDownloader()

        # Test with specific IDs
        result = downloader.download_jrc_annual(
            vector_dataset=VECTOR_DATASET,
            name_attribute="id_geohash",
            years=[2017],
            id_list=["b7g64d3984be", "b7g6j40qk509"],
            no_download=True,
        )

        assert result is None  # no_download mode

    def test_download_jrc_annual_with_bbox(self, monkeypatch):
        """Test download_jrc_annual with bbox filtering."""
        # monkeypatch.setenv("EE_PROJECT", "test-project")
        downloader = EarthEngineDownloader()

        # Test with bbox filter
        result = downloader.download_jrc_annual(
            vector_dataset=VECTOR_DATASET,
            name_attribute="id_geohash",
            years=[2017],
            bbox_west=-164.2,
            bbox_east=-164,
            bbox_south=66.5,
            bbox_north=66.55,
            no_download=True,
        )

        assert result is None  # no_download mode

    def test_download_jrc_annual_invalid_name_attribute(self, monkeypatch):
        """Test that invalid name_attribute raises KeyError."""
        # monkeypatch.setenv("EE_PROJECT", "test-project")
        downloader = EarthEngineDownloader()

        with pytest.raises(KeyError, match="not present in the vector dataset"):
            downloader.download_jrc_annual(
                vector_dataset=VECTOR_DATASET,
                name_attribute="invalid_column",
                years=[2017],
                no_download=True,
            )

    def test_download_jrc_annual_missing_ids(self, monkeypatch):
        """Test that missing IDs in id_list raises ValueError."""
        # monkeypatch.setenv("EE_PROJECT", "test-project")
        downloader = EarthEngineDownloader()

        with pytest.raises(ValueError, match="None of the.*requested IDs found"):
            downloader.download_jrc_annual(
                vector_dataset=VECTOR_DATASET,
                name_attribute="id_geohash",
                years=[2017],
                id_list=["nonexistent_id_1", "nonexistent_id_2"],
                no_download=True,
            )

    def test_download_jrc_annual_years(self, monkeypatch):
        """Test that invalid years are skipped and warnings are logged."""
        # monkeypatch.setenv("EE_PROJECT", "test-project")
        downloader = EarthEngineDownloader()

        downloader.download_jrc_annual(
            vector_dataset=VECTOR_DATASET,
            name_attribute="id_geohash",
            years=[2017, 2018, 2019, 2020, 2021, 2022],
            id_list=["nonexistent_id_1", "nonexistent_id_2"],
            no_download=True,
        )
        assert downloader._validate_years_jrc([2017, 2018, 2019, 2020, 2021, 2022]) == [2017, 2018, 2019, 2020, 2021]

    def test_download_jrc_input_data_both_none(self, monkeypatch):
        """Test that invalid years are skipped and warnings are logged."""
        # monkeypatch.setenv("EE_PROJECT", "test-project")
        downloader = EarthEngineDownloader()
        with pytest.raises(ValueError, match="Either vector_dataset or gdf must be provided"):
            downloader.download_jrc_annual(
                vector_dataset=None,
                gdf=None,
                name_attribute="id_geohash",
                years=[2017, 2018, 2019, 2020, 2021, 2022],
                no_download=True,
            )

    def test_download_jrc_input_data_both_true(self, monkeypatch):
        """Test that invalid years are skipped and warnings are logged."""
        # monkeypatch.setenv("EE_PROJECT", "test-project")
        downloader = EarthEngineDownloader()
        with pytest.raises(ValueError, match="Only one of vector_dataset or gdf should be provided, not both"):
            downloader.download_jrc_annual(
                vector_dataset=VECTOR_DATASET,
                gdf=gpd.GeoDataFrame(),
                name_attribute="id_geohash",
                years=[2017, 2018, 2019, 2020, 2021, 2022],
                no_download=True,
            )


class TestCalcMonthlyDw:
    """Test the calc_monthly_dw function for handling missing data."""

    def test_calc_monthly_dw_no_data_warning(self):
        """Test that calc_monthly_dw returns None and raises a warning for date with no data."""
        import warnings

        import geemap
        import geopandas as gpd

        # Use a location that should have no Dynamic World data (2015 is before DW coverage)
        start_date = "2015-01-01"

        # Load test data and get a polygon
        gdf = gpd.read_parquet(VECTOR_DATASET)
        # Use the first polygon
        polygon = gpd.GeoDataFrame(gdf.iloc[[0]]).to_crs(epsg=4326)
        fc = geemap.gdf_to_ee(polygon)

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calc_monthly_dw(start_date=start_date, polygons=fc)

            # Verify that a warning was raised
            assert len(w) == 1
            assert "No data available" in str(w[0].message)
            assert start_date in str(w[0].message)

            # Verify that None was returned
            assert result is None

    def test_download_dw_input_data_both_none(self, monkeypatch):
        """Test that invalid years are skipped and warnings are logged."""
        # monkeypatch.setenv("EE_PROJECT", "test-project")
        downloader = EarthEngineDownloader()
        with pytest.raises(ValueError, match="Either vector_dataset or gdf must be provided"):
            downloader.download_dw_monthly(
                vector_dataset=None,
                gdf=None,
                name_attribute="id_geohash",
                years=[2017, 2018, 2019, 2020, 2021, 2022],
                months=[6, 7, 8, 9],
                no_download=True,
            )

    def test_download_dw_input_data_both_true(self, monkeypatch):
        """Test that invalid years are skipped and warnings are logged."""
        # monkeypatch.setenv("EE_PROJECT", "test-project")
        downloader = EarthEngineDownloader()
        with pytest.raises(ValueError, match="Only one of vector_dataset or gdf should be provided, not both"):
            downloader.download_dw_monthly(
                vector_dataset=VECTOR_DATASET,
                gdf=gpd.GeoDataFrame(),
                name_attribute="id_geohash",
                years=[2017, 2018, 2019, 2020, 2021, 2022],
                months=[6, 7, 8, 9],
                no_download=True,
            )


class TestCallWithRetry:
    """Test retry behavior for transient versus permanent errors."""

    def test_retries_transient_network_error_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("water_timeseries.utils.earthengine.time.sleep", lambda _s: None)
        fn = mock.Mock(side_effect=[requests.exceptions.ConnectionError("reset"), 42])
        assert call_with_retry(fn, attempts=3) == 42
        assert fn.call_count == 2

    def test_retries_transient_ee_error_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("water_timeseries.utils.earthengine.time.sleep", lambda _s: None)
        fn = mock.Mock(side_effect=[ee.EEException("Computation timed out."), "ok"])
        assert call_with_retry(fn, attempts=3) == "ok"
        assert fn.call_count == 2

    def test_raises_after_exhausting_attempts(self, monkeypatch):
        monkeypatch.setattr("water_timeseries.utils.earthengine.time.sleep", lambda _s: None)
        fn = mock.Mock(side_effect=TimeoutError("slow"))
        with pytest.raises(TimeoutError):
            call_with_retry(fn, attempts=3)
        assert fn.call_count == 3

    def test_permanent_ee_error_is_not_retried(self, monkeypatch):
        monkeypatch.setattr("water_timeseries.utils.earthengine.time.sleep", lambda _s: None)
        fn = mock.Mock(side_effect=ee.EEException("User memory limit exceeded."))
        with pytest.raises(ee.EEException):
            call_with_retry(fn, attempts=3)
        assert fn.call_count == 1


class TestNoDataVersusError:
    """Test that confirmed missing data and request failures are reported differently."""

    @pytest.fixture
    def downloader(self):
        dl = EarthEngineDownloader(ee_project="test-project", ee_auth=False)
        dl._setup_gee_reducer = mock.Mock(return_value=(mock.Mock(), {}))
        return dl

    def test_network_error_propagates_from_chunk(self, downloader):
        with (
            mock.patch(
                "water_timeseries.downloader.calc_monthly_dw",
                side_effect=requests.exceptions.ConnectionError("network down"),
            ),
            pytest.raises(requests.exceptions.ConnectionError),
        ):
            downloader._extract_time_series(dates=["2024-06-01"], gdf_chunk=None, name_attribute="id_geohash")

    def test_no_data_returns_empty_frame_for_chunk(self, downloader):
        with mock.patch("water_timeseries.downloader.calc_monthly_dw", return_value=None):
            df = downloader._extract_time_series(
                dates=["2024-06-01", "2024-07-01"], gdf_chunk=None, name_attribute="id_geohash"
            )
        assert df.empty

    def test_all_chunks_empty_raises_no_data_error(self, downloader):
        gdf = gpd.read_parquet(VECTOR_DATASET).iloc[:4]
        downloader._chunk_gdf = mock.Mock(return_value=[gdf.iloc[:2], gdf.iloc[2:]])
        downloader._extract_time_series = mock.Mock(return_value=pd.DataFrame())
        with pytest.raises(NoDynamicWorldDataError):
            downloader.download_dw_monthly(gdf=gdf, name_attribute="id_geohash", date_list=["2024-06-01"])

    def test_no_data_error_is_backward_compatible_value_error(self, downloader):
        """Callers matching the old ValueError message (e.g. the Argo NRT pipeline) must keep working."""
        gdf = gpd.read_parquet(VECTOR_DATASET).iloc[:4]
        downloader._chunk_gdf = mock.Mock(return_value=[gdf])
        downloader._extract_time_series = mock.Mock(return_value=pd.DataFrame())
        with pytest.raises(ValueError, match="No data was extracted"):
            downloader.download_dw_monthly(gdf=gdf, name_attribute="id_geohash", date_list=["2024-06-01"])

    def test_chunk_error_is_not_reported_as_no_data(self, downloader):
        gdf = gpd.read_parquet(VECTOR_DATASET).iloc[:4]
        downloader._chunk_gdf = mock.Mock(return_value=[gdf.iloc[:2], gdf.iloc[2:]])
        downloader._extract_time_series = mock.Mock(side_effect=ee.EEException("User memory limit exceeded."))
        with pytest.raises(ee.EEException):
            downloader.download_dw_monthly(gdf=gdf, name_attribute="id_geohash", date_list=["2024-06-01"])


class TestSeasonWindows:
    """Pure date-window derivation for the per-acquisition downloader (no Earth Engine)."""

    def test_default_window_bumps_end_by_one_day(self):
        """filterDate is exclusive, so an inclusive 09-30 end must become Oct 1."""
        assert season_windows([2017, 2018]) == [("2017-05-01", "2017-10-01"), ("2018-05-01", "2018-10-01")]

    def test_custom_window(self):
        assert season_windows([2019], start_date="06-15", end_date="09-01") == [("2019-06-15", "2019-09-02")]

    def test_single_day_window(self):
        assert season_windows([2019], start_date="07-01", end_date="07-01") == [("2019-07-01", "2019-07-02")]

    def test_full_iso_date_rejected(self):
        with pytest.raises(ValueError, match="MM-DD"):
            season_windows([2019], start_date="2019-05-01", end_date="09-30")

    def test_wrapping_window_rejected(self):
        with pytest.raises(ValueError, match="wrap"):
            season_windows([2019], start_date="09-30", end_date="05-01")

    def test_month_slices_full_season(self):
        assert month_slices("2017-05-01", "2017-10-01") == [
            ("2017-05-01", "2017-06-01"),
            ("2017-06-01", "2017-07-01"),
            ("2017-07-01", "2017-08-01"),
            ("2017-08-01", "2017-09-01"),
            ("2017-09-01", "2017-10-01"),
        ]

    def test_month_slices_clamp_partial_months(self):
        assert month_slices("2017-05-15", "2017-07-10") == [
            ("2017-05-15", "2017-06-01"),
            ("2017-06-01", "2017-07-01"),
            ("2017-07-01", "2017-07-10"),
        ]

    def test_month_windows_contiguous(self):
        assert month_windows([2020], [6, 7]) == [("2020-06-01", "2020-07-01"), ("2020-07-01", "2020-08-01")]

    def test_month_windows_non_contiguous_stays_separate(self):
        """[6, 9] must not become one span that silently includes July and August."""
        assert month_windows([2020], [6, 9]) == [("2020-06-01", "2020-07-01"), ("2020-09-01", "2020-10-01")]

    def test_month_windows_multiple_years(self):
        assert month_windows([2019, 2020], [9]) == [("2019-09-01", "2019-10-01"), ("2020-09-01", "2020-10-01")]

    def test_month_windows_rejects_out_of_range_month(self):
        with pytest.raises(ValueError, match="between 1 and 12"):
            month_windows([2020], [0])

    def test_month_windows_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            month_windows([2020], [])

    def test_windows_from_date_list(self):
        assert windows_from_date_list(["2020-06", "2020-08"]) == [
            ("2020-06-01", "2020-07-01"),
            ("2020-08-01", "2020-09-01"),
        ]

    def test_windows_from_date_list_rejects_full_date(self):
        with pytest.raises(ValueError, match="YYYY-MM"):
            windows_from_date_list(["2020-06-15"])


LAKE_AREA_HA = 10.0


def _acq_frame(entries) -> pd.DataFrame:
    """Build a frame shaped like ``geemap.ee_to_df`` output.

    ``entries`` is a list of ``(id_geohash, date, coverage)``; class areas are sized so that
    their sum equals ``coverage * LAKE_AREA_HA``, which is what coverage is computed from.
    """
    rows = []
    for lake, day, coverage in entries:
        rec = {"id_geohash": lake, "date": pd.Timestamp(day)}
        for var in dw_bandnames:
            rec[var] = 0.0
        rec["water"] = LAKE_AREA_HA * coverage * 0.5
        rec["bare"] = LAKE_AREA_HA * coverage * 0.5
        rows.append(rec)
    return pd.DataFrame(rows)


class TestDwDaily:
    """Per-acquisition downloader: mocked Earth Engine, no network."""

    @pytest.fixture
    def downloader(self):
        from shapely.geometry import box

        dl = EarthEngineDownloader(ee_project="test-project", ee_auth=False)
        dl._polygon_areas_ha = mock.Mock(return_value={"lakeA": LAKE_AREA_HA, "lakeB": LAKE_AREA_HA})
        dl.gdf = gpd.GeoDataFrame(
            {"id_geohash": ["lakeA", "lakeB"]},
            geometry=[box(0, 0, 0.01, 0.01), box(0.02, 0, 0.03, 0.01)],
            crs="EPSG:4326",
        )
        return dl

    def test_both_inputs_raises(self, downloader):
        with pytest.raises(ValueError, match="Only one of vector_dataset or gdf should be provided"):
            downloader.download_dw_daily(vector_dataset=VECTOR_DATASET, gdf=downloader.gdf, years=[2020])

    def test_no_input_raises(self, downloader):
        with pytest.raises(ValueError, match="Either vector_dataset or gdf must be provided"):
            downloader.download_dw_daily(years=[2020])

    def test_missing_name_attribute_raises(self, downloader):
        with pytest.raises(KeyError, match="not present in the vector dataset"):
            downloader.download_dw_daily(gdf=downloader.gdf, name_attribute="nope", years=[2020])

    def test_no_download_returns_none_without_querying_ee(self, downloader):
        with mock.patch(
            "water_timeseries.downloader.dw_acquisitions", side_effect=AssertionError("must not query EE")
        ):
            result = downloader.download_dw_daily(gdf=downloader.gdf, years=[2020], no_download=True)
        assert result is None

    def test_empty_acquisitions_raises_no_data(self, downloader):
        downloader._extract_acquisitions = mock.Mock(return_value=pd.DataFrame())
        with pytest.raises(NoDynamicWorldDataError, match="No data was extracted"):
            downloader.download_dw_daily(gdf=downloader.gdf, years=[2020])

    def test_connection_error_propagates(self, downloader):
        downloader._extract_acquisitions = mock.Mock(
            side_effect=requests.exceptions.ConnectionError("network down")
        )
        with pytest.raises(requests.exceptions.ConnectionError):
            downloader.download_dw_daily(gdf=downloader.gdf, years=[2020])

    def test_output_schema_and_attrs(self, downloader):
        downloader._extract_acquisitions = mock.Mock(
            return_value=_acq_frame([("lakeA", "2020-06-05", 1.0), ("lakeA", "2020-07-05", 1.0)])
        )
        ds = downloader.download_dw_daily(gdf=downloader.gdf, years=[2020])
        assert set(ds.data_vars) == set(dw_bandnames)
        assert list(ds.dims) == ["id_geohash", "date"]
        assert ds.attrs["temporal_resolution"] == "per_acquisition"
        assert ds.attrs["season_window"] == "05-01..09-30"
        assert ds.attrs["max_nodata"] == pytest.approx(0.02)

    def test_zero_coverage_rows_are_dropped_not_written_as_zero(self, downloader):
        """A fully masked acquisition is 'no observation', never '0 ha of water'."""
        downloader._extract_acquisitions = mock.Mock(
            return_value=_acq_frame(
                [("lakeA", "2020-06-05", 0.0), ("lakeA", "2020-07-05", 1.0), ("lakeB", "2020-07-05", 1.0)]
            )
        )
        ds = downloader.download_dw_daily(gdf=downloader.gdf, years=[2020])
        assert ds["water"].sel(id_geohash="lakeA").notnull().sum() == 1
        assert float(ds["water"].min()) > 0

    def test_duplicate_day_keeps_higher_coverage(self, downloader):
        downloader._extract_acquisitions = mock.Mock(
            return_value=_acq_frame([("lakeA", "2020-06-05", 0.90), ("lakeA", "2020-06-05", 1.0)])
        )
        ds = downloader.download_dw_daily(gdf=downloader.gdf, years=[2020], max_nodata=0.5)
        assert ds.sizes["date"] == 1
        assert float(ds["water"].max()) == pytest.approx(LAKE_AREA_HA * 0.5)  # the full-coverage row

    def test_max_nodata_filters_observations(self, downloader):
        frame = _acq_frame(
            [("lakeA", "2020-06-05", 0.90), ("lakeA", "2020-07-05", 1.0), ("lakeA", "2020-08-05", 0.99)]
        )
        downloader._extract_acquisitions = mock.Mock(return_value=frame)
        strict = downloader.download_dw_daily(gdf=downloader.gdf, years=[2020], max_nodata=0.02)
        loose = downloader.download_dw_daily(gdf=downloader.gdf, years=[2020], max_nodata=0.20)
        assert strict.sizes["date"] == 2  # 1.0 and 0.99 (>= 98 %) kept, 0.90 dropped
        assert loose.sizes["date"] == 3

    def test_all_rows_filtered_raises_no_data(self, downloader):
        downloader._extract_acquisitions = mock.Mock(return_value=_acq_frame([("lakeA", "2020-06-05", 0.5)]))
        with pytest.raises(NoDynamicWorldDataError, match="No data was extracted"):
            downloader.download_dw_daily(gdf=downloader.gdf, years=[2020], max_nodata=0.02)

    def test_invalid_max_nodata_raises(self, downloader):
        with pytest.raises(ValueError, match="max_nodata"):
            downloader.download_dw_daily(gdf=downloader.gdf, years=[2020], max_nodata=1.5)

    def test_months_argument_mirrors_monthly_call(self, downloader):
        """A download_dw_monthly call must work by changing the method name only."""
        downloader._extract_acquisitions = mock.Mock(
            return_value=_acq_frame([("lakeA", "2020-06-05", 1.0)])
        )
        ds = downloader.download_dw_daily(gdf=downloader.gdf, years=[2020], months=[6, 7, 8, 9])
        windows = downloader._extract_acquisitions.call_args.kwargs["windows"]
        assert windows[0] == ("2020-06-01", "2020-07-01")
        assert len(windows) == 4
        assert ds.attrs["season_window"] == "months(6,7,8,9)"

    def test_date_list_argument_mirrors_monthly_call(self, downloader):
        downloader._extract_acquisitions = mock.Mock(
            return_value=_acq_frame([("lakeA", "2020-06-05", 1.0)])
        )
        ds = downloader.download_dw_daily(gdf=downloader.gdf, date_list=["2020-06", "2020-08"])
        windows = downloader._extract_acquisitions.call_args.kwargs["windows"]
        assert windows == [("2020-06-01", "2020-07-01"), ("2020-08-01", "2020-09-01")]
        assert ds.attrs["season_window"].startswith("date_list")

    def test_months_and_window_are_mutually_exclusive(self, downloader):
        with pytest.raises(ValueError, match="not both"):
            downloader.download_dw_daily(gdf=downloader.gdf, years=[2020], months=[6], start_date="06-01")

    def test_months_and_date_list_are_mutually_exclusive(self, downloader):
        with pytest.raises(ValueError, match="not both"):
            downloader.download_dw_daily(gdf=downloader.gdf, months=[6], date_list=["2020-06"])

    def test_chunk_sizing_uses_one_month_of_images(self, downloader):
        """Chunks must be sized by one reduce call's worth of images, not the whole request.

        Sizing by the multi-year total collapses a long request to one polygon per chunk and
        multiplies Earth Engine calls by the number of years.
        """
        downloader._extract_acquisitions = mock.Mock(
            return_value=_acq_frame([("lakeA", "2019-06-05", 1.0)])
        )
        with mock.patch.object(downloader, "_chunk_gdf", wraps=downloader._chunk_gdf) as spy:
            downloader.download_dw_daily(gdf=downloader.gdf, years=[2019, 2020, 2021])
        assert spy.call_args.kwargs["n_dates"] == EST_ACQUISITIONS_PER_MONTH

    def test_gdf_without_crs_raises_a_clear_error(self, downloader):
        """Coverage is relative to the polygon area, so a CRS-less gdf must say so.

        The fixture mocks ``_polygon_areas_ha``, so call the real method through the class.
        """
        no_crs = downloader.gdf.set_crs(None, allow_override=True)
        with pytest.raises(ValueError, match="no CRS"):
            EarthEngineDownloader._polygon_areas_ha(downloader, no_crs, "id_geohash")

    def test_gdf_input_is_accepted_like_vector_dataset(self, downloader):
        """Passing gdf= must work the same way as vector_dataset= (the fixture mocks areas)."""
        downloader._extract_acquisitions = mock.Mock(
            return_value=_acq_frame([("lakeA", "2020-06-05", 1.0)])
        )
        ds = downloader.download_dw_daily(gdf=downloader.gdf, years=[2020], months=[6])
        assert set(ds.data_vars) == set(dw_bandnames)
        assert "accompanying vector dataset" not in ds.attrs  # no path to record

    def test_warns_when_workers_exceed_chunks(self, downloader):
        """Asking for more workers than chunks must say so, with the number to fix it."""

        class Recorder:
            def __init__(self):
                self.warnings = []

            def info(self, message):
                pass

            def warning(self, message):
                self.warnings.append(message)

            def error(self, message):
                pass

        downloader.logger = Recorder()
        downloader._extract_acquisitions = mock.Mock(
            return_value=_acq_frame([("lakeA", "2020-06-05", 1.0)])
        )
        downloader.download_dw_daily(gdf=downloader.gdf, years=[2020], n_parallel=6)

        assert any("will idle" in message for message in downloader.logger.warnings), downloader.logger.warnings
        assert any("max_total_requests=" in message for message in downloader.logger.warnings)

    def test_no_warning_when_workers_fit_chunks(self, downloader):
        class Recorder:
            def __init__(self):
                self.warnings = []

            def info(self, message):
                pass

            def warning(self, message):
                self.warnings.append(message)

            def error(self, message):
                pass

        downloader.logger = Recorder()
        downloader._extract_acquisitions = mock.Mock(
            return_value=_acq_frame([("lakeA", "2020-06-05", 1.0)])
        )
        downloader.download_dw_daily(gdf=downloader.gdf, years=[2020], n_parallel=1)

        assert not any("will idle" in message for message in downloader.logger.warnings)

    def test_result_wraps_in_dwdateset_and_plots(self, downloader):
        """The downloaded object must drop straight into DWDataset and its plotter."""
        from water_timeseries.dataset import DWDataset

        entries = [(lake, day, 1.0) for lake in ("lakeA", "lakeB") for day in pd.date_range("2020-06-01", "2020-09-01", freq="7D")]
        downloader._extract_acquisitions = mock.Mock(return_value=_acq_frame(entries))
        ds = downloader.download_dw_daily(gdf=downloader.gdf, years=[2020])

        dataset = DWDataset(ds)
        assert dataset.object_ids_ == ["lakeA", "lakeB"]
        fig = dataset.plot_timeseries("lakeA")
        assert fig is not None


class TestDwCoverageStats:
    """Per-polygon coverage summary derived from an existing dataset (no geometry)."""

    @pytest.fixture
    def ds(self):
        import numpy as np
        import xarray as xr

        dates = pd.to_datetime(["2020-06-01", "2020-07-01", "2020-08-01"])
        data = {var: (("id_geohash", "date"), np.zeros((2, 3))) for var in dw_bandnames}
        ds = xr.Dataset(data, coords={"id_geohash": ["a", "b"], "date": dates})
        ds["water"] = (("id_geohash", "date"), [[1.0, 2.0, np.nan], [2.0, 2.0, 2.0]])
        return ds

    def test_per_lake_stats(self, ds):
        stats = dw_coverage_stats(ds)
        assert list(stats.index) == ["a", "b"]
        assert stats.loc["a", "n_obs"] == 2
        assert stats.loc["a", "coverage_median"] == pytest.approx(0.75)
        assert stats.loc["a", "first_date"] == pd.Timestamp("2020-06-01")
        assert stats.loc["a", "last_date"] == pd.Timestamp("2020-07-01")
        assert stats.loc["b", "n_obs"] == 3
        assert stats.loc["b", "coverage_median"] == pytest.approx(1.0)
        assert stats.loc["b", "last_date"] == pd.Timestamp("2020-08-01")

    def test_all_nan_lake_reported_as_no_observations(self):
        import numpy as np
        import xarray as xr

        dates = pd.to_datetime(["2020-06-01", "2020-07-01"])
        data = {var: (("id_geohash", "date"), np.zeros((1, 2))) for var in dw_bandnames}
        ds = xr.Dataset(data, coords={"id_geohash": ["empty"], "date": dates})
        ds["water"] = (("id_geohash", "date"), [[np.nan, np.nan]])
        stats = dw_coverage_stats(ds)
        assert stats.loc["empty", "n_obs"] == 0
        assert pd.isna(stats.loc["empty", "coverage_median"])


class TestMonthlyIceFreeAggregation:
    """aggregation='ice_free': scene-level ice filter, median per (polygon, month)."""

    @pytest.fixture
    def downloader(self):
        from shapely.geometry import box

        dl = EarthEngineDownloader(ee_project="test-project", ee_auth=False)
        dl.gdf = gpd.GeoDataFrame(
            {"id_geohash": ["lakeA"]}, geometry=[box(0, 0, 0.01, 0.01)], crs="EPSG:4326"
        )
        return dl

    @staticmethod
    def _scenes(entries) -> pd.DataFrame:
        """Frame shaped like ee_to_df: ``(lake, date, water_ha, snow_ha)`` per acquisition."""
        rows = []
        for lake, day, water, snow in entries:
            rec = {"id_geohash": lake, "date": pd.Timestamp(day)}
            for var in dw_bandnames:
                rec[var] = 0.0
            rec["water"] = water
            rec["snow_and_ice"] = snow
            rows.append(rec)
        return pd.DataFrame(rows)

    def test_icy_scenes_are_discarded_and_survivors_median(self, downloader):
        downloader._extract_acquisitions = mock.Mock(
            return_value=self._scenes(
                [
                    ("lakeA", "2020-06-05", 10.0, 0.0),  # ice-free
                    ("lakeA", "2020-06-12", 20.0, 0.5),  # ice-free (2.4 % of 20.5 ha)
                    ("lakeA", "2020-06-20", 0.0, 100.0),  # fully ice: discarded
                ]
            )
        )
        ds = downloader.download_dw_monthly(
            gdf=downloader.gdf, years=[2020], months=[6], aggregation="ice_free"
        )
        assert ds.sizes["date"] == 1
        assert float(ds["water"].isel(date=0, id_geohash=0)) == pytest.approx(15.0)  # median(10, 20)

    def test_month_without_ice_free_scene_is_absent_not_zero(self, downloader):
        downloader._extract_acquisitions = mock.Mock(
            return_value=self._scenes(
                [
                    ("lakeA", "2020-06-10", 0.0, 50.0),  # June: all ice
                    ("lakeA", "2020-07-10", 12.0, 0.0),  # July: ice-free
                ]
            )
        )
        ds = downloader.download_dw_monthly(
            gdf=downloader.gdf, years=[2020], months=[6, 7], aggregation="ice_free"
        )
        months = pd.to_datetime(ds["date"].values).month.tolist()
        assert months == [7]
        assert float(ds["water"].min()) > 0

    def test_all_scenes_icy_raises(self, downloader):
        downloader._extract_acquisitions = mock.Mock(
            return_value=self._scenes([("lakeA", "2020-06-10", 0.0, 50.0)])
        )
        with pytest.raises(NoDynamicWorldDataError, match="none of the"):
            downloader.download_dw_monthly(
                gdf=downloader.gdf, years=[2020], months=[6], aggregation="ice_free"
            )

    def test_attrs_record_the_method(self, downloader):
        downloader._extract_acquisitions = mock.Mock(
            return_value=self._scenes([("lakeA", "2020-06-10", 12.0, 0.0)])
        )
        ds = downloader.download_dw_monthly(
            gdf=downloader.gdf, years=[2020], months=[6], aggregation="ice_free", max_snow_fraction=0.1
        )
        assert ds.attrs["aggregation"] == "ice_free"
        assert ds.attrs["max_snow_fraction"] == pytest.approx(0.10)

    def test_mode_is_still_the_default_and_unchanged(self, downloader):
        downloader._extract_time_series = mock.Mock(
            return_value=self._scenes([("lakeA", "2020-06-01", 12.0, 0.0)])
        )
        downloader._extract_acquisitions = mock.Mock(side_effect=AssertionError("must not be used"))
        ds = downloader.download_dw_monthly(gdf=downloader.gdf, years=[2020], months=[6])
        assert ds.attrs["aggregation"] == "mode"
        assert "max_snow_fraction" not in ds.attrs

    def test_invalid_aggregation_raises(self, downloader):
        with pytest.raises(ValueError, match="aggregation must be"):
            downloader.download_dw_monthly(gdf=downloader.gdf, years=[2020], months=[6], aggregation="median")

    def test_invalid_max_snow_fraction_raises(self, downloader):
        with pytest.raises(ValueError, match="max_snow_fraction"):
            downloader.download_dw_monthly(
                gdf=downloader.gdf, years=[2020], months=[6], aggregation="ice_free", max_snow_fraction=1.5
            )


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
