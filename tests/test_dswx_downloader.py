"""Tests for the OPERA DSWx-S1 downloaders (mocked Earth Engine, no network)."""

from unittest import mock

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from water_timeseries.downloader import EarthEngineDownloader, NoDSWxDataError
from water_timeseries.utils.data import dswx_bandnames

LAKE_AREA_HA = 20.0


def _dswx_frame(entries) -> pd.DataFrame:
    """Build a frame shaped like ``geemap.ee_to_df`` output for DSWx.

    ``entries`` is a list of ``(id_geohash, date, coverage, open_ha, veg_ha)``; the
    ``area_not_water`` band is filled so that the three classes sum to
    ``coverage * LAKE_AREA_HA``, which is what coverage is computed from.
    """
    rows = []
    for lake, day, coverage, open_ha, veg_ha in entries:
        rows.append(
            {
                "id_geohash": lake,
                "date": pd.Timestamp(day),
                "area_open_water": open_ha,
                "area_inundated_vegetation": veg_ha,
                "area_not_water": LAKE_AREA_HA * coverage - open_ha - veg_ha,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def downloader():
    dl = EarthEngineDownloader(ee_project="test-project", ee_auth=False)
    dl._polygon_areas_ha = mock.Mock(return_value={"lakeA": LAKE_AREA_HA, "lakeB": LAKE_AREA_HA})
    dl.gdf = gpd.GeoDataFrame(
        {"id_geohash": ["lakeA", "lakeB"]},
        geometry=[box(0, 0, 0.01, 0.01), box(0.02, 0, 0.03, 0.01)],
        crs="EPSG:4326",
    )
    return dl


class TestDswxAcquisitionsValidation:
    def test_both_inputs_raises(self, downloader):
        with pytest.raises(ValueError, match="Only one of vector_dataset or gdf should be provided"):
            downloader.download_dswx_acquisitions(vector_dataset="x.parquet", gdf=downloader.gdf, years=[2025])

    def test_no_input_raises(self, downloader):
        with pytest.raises(ValueError, match="Either vector_dataset or gdf must be provided"):
            downloader.download_dswx_acquisitions(years=[2025])

    def test_missing_name_attribute_raises(self, downloader):
        with pytest.raises(KeyError, match="not present in the vector dataset"):
            downloader.download_dswx_acquisitions(gdf=downloader.gdf, name_attribute="nope", years=[2025])

    def test_max_nodata_out_of_range_raises(self, downloader):
        with pytest.raises(ValueError, match="max_nodata must be between 0 and 1"):
            downloader.download_dswx_acquisitions(gdf=downloader.gdf, years=[2025], max_nodata=1.5)

    def test_negative_min_masked_pixels_raises(self, downloader):
        with pytest.raises(ValueError, match="min_masked_pixels must be zero or positive"):
            downloader.download_dswx_acquisitions(gdf=downloader.gdf, years=[2025], min_masked_pixels=-1)

    def test_months_and_within_year_window_conflict(self, downloader):
        with pytest.raises(ValueError, match="not both"):
            downloader.download_dswx_acquisitions(
                gdf=downloader.gdf, years=[2025], months=[6], start_date="05-01"
            )

    def test_months_and_date_list_conflict(self, downloader):
        with pytest.raises(ValueError, match="Provide either 'months' or 'date_list'"):
            downloader.download_dswx_acquisitions(
                gdf=downloader.gdf, years=[2025], months=[6], date_list=["2025-06"]
            )

    def test_no_download_returns_none_without_querying_ee(self, downloader):
        with mock.patch(
            "water_timeseries.downloader.dswx_acquisitions", side_effect=AssertionError("must not query EE")
        ):
            result = downloader.download_dswx_acquisitions(gdf=downloader.gdf, years=[2025], no_download=True)
        assert result is None

    def test_empty_acquisitions_raises_no_data(self, downloader):
        downloader._extract_dswx_acquisitions = mock.Mock(return_value=pd.DataFrame())
        with pytest.raises(NoDSWxDataError, match="No data was extracted"):
            downloader.download_dswx_acquisitions(gdf=downloader.gdf, years=[2025])


class TestDswxYears:
    def test_archive_start_is_the_floor(self, downloader):
        """Nothing exists before 2024-09, so 2019-2023 are skipped rather than downloaded."""
        assert downloader._validate_years_dswx([2019, 2023, 2024, 2025]) == [2024, 2025]

    def test_default_starts_at_archive(self, downloader):
        years = downloader._validate_years_dswx(None)
        assert years[0] == 2024
        assert years == list(range(2024, pd.Timestamp.now().year + 1))

    def test_accepted_types(self, downloader):
        assert downloader._validate_years_dswx(range(2024, 2026)) == [2024, 2025]

    def test_wrong_type_raises(self, downloader):
        with pytest.raises(ValueError, match="Years must be provided as a list"):
            downloader._validate_years_dswx(2025)


class TestDswxDateArguments:
    """The date arguments must behave exactly like the Dynamic World pair."""

    def _windows(self, downloader, **kwargs):
        downloader._extract_dswx_acquisitions = mock.Mock(
            return_value=_dswx_frame([("lakeA", "2025-06-05", 1.0, 6.0, 4.0)])
        )
        downloader.download_dswx_acquisitions(gdf=downloader.gdf, **kwargs)
        return downloader._extract_dswx_acquisitions.call_args.kwargs["windows"]

    def test_months_argument(self, downloader):
        assert self._windows(downloader, years=[2025], months=[6, 7]) == [
            ("2025-06-01", "2025-07-01"),
            ("2025-07-01", "2025-08-01"),
        ]

    def test_date_list_argument_matches_months(self, downloader):
        assert self._windows(downloader, date_list=["2025-06", "2025-07"]) == self._windows(
            downloader, years=[2025], months=[6, 7]
        )

    def test_within_year_window_argument(self, downloader):
        windows = self._windows(downloader, years=[2025], start_date="06-01", end_date="07-31")
        assert windows == [("2025-06-01", "2025-08-01")]

    def test_default_window_is_the_dw_season(self, downloader):
        assert self._windows(downloader, years=[2025]) == [("2025-05-01", "2025-10-01")]


class TestDswxAcquisitionFrames:
    """Tile collapse and the no-data rule, on frames shaped like the EE output."""

    def test_output_schema_and_attrs(self, downloader):
        downloader._extract_dswx_acquisitions = mock.Mock(
            return_value=_dswx_frame(
                [("lakeA", "2025-06-05", 1.0, 6.0, 4.0), ("lakeA", "2025-07-05", 1.0, 7.0, 3.0)]
            )
        )
        ds = downloader.download_dswx_acquisitions(gdf=downloader.gdf, years=[2025])
        assert set(ds.data_vars) == set(dswx_bandnames)
        assert list(ds.dims) == ["id_geohash", "date"]
        assert ds.attrs["temporal_resolution"] == "per_acquisition"
        assert ds.attrs["season_window"] == "05-01..09-30"
        assert ds.attrs["max_nodata"] == pytest.approx(0.02)
        assert ds.attrs["min_masked_pixels"] == pytest.approx(2)
        assert ds.attrs["n_acquisitions"] == 2
        assert ds.attrs["product"].startswith("OPERA DSWx-S1")

    def test_fully_masked_row_is_dropped_not_written_as_zero(self, downloader):
        """A scene with no classified pixel is 'no observation', never '0 ha of water'."""
        downloader._extract_dswx_acquisitions = mock.Mock(
            return_value=_dswx_frame(
                [("lakeA", "2025-06-05", 0.0, 0.0, 0.0), ("lakeA", "2025-07-05", 1.0, 7.0, 3.0)]
            )
        )
        ds = downloader.download_dswx_acquisitions(gdf=downloader.gdf, years=[2025])
        assert ds.sizes["date"] == 1
        assert float(ds["area_open_water"].min()) == pytest.approx(7.0)

    def test_overlapping_tiles_collapse_to_the_best_tile_not_their_sum(self, downloader):
        """Two MGRS tiles cover 60 % and 100 % of the polygon: keep the 100 % one, never add."""
        downloader._extract_dswx_acquisitions = mock.Mock(
            return_value=_dswx_frame(
                [
                    ("lakeA", "2025-06-05", 0.60, 4.0, 2.0),
                    ("lakeA", "2025-06-05", 1.00, 8.0, 4.0),
                ]
            )
        )
        ds = downloader.download_dswx_acquisitions(gdf=downloader.gdf, years=[2025])
        assert ds.sizes["date"] == 1
        assert float(ds["area_open_water"].max()) == pytest.approx(8.0)
        assert float(ds["area_inundated_vegetation"].max()) == pytest.approx(4.0)

    def test_overlapping_tiles_with_equal_coverage_use_the_larger_not_water_tile(self, downloader):
        """Area_data matches to float noise, so the tie-break decides - not the 1e-10 gap."""
        noise = 1e-9
        downloader._extract_dswx_acquisitions = mock.Mock(
            return_value=pd.DataFrame(
                [
                    # classified areas differ by 1e-9 ha; not water 8.0
                    {
                        "id_geohash": "lakeA",
                        "date": pd.Timestamp("2025-06-05"),
                        "area_open_water": 8.0,
                        "area_inundated_vegetation": 4.0,
                        "area_not_water": 8.0,
                    },
                    # the marginally SMALLER classified area, but not water 13.0 -> chosen
                    {
                        "id_geohash": "lakeA",
                        "date": pd.Timestamp("2025-06-05"),
                        "area_open_water": 3.0,
                        "area_inundated_vegetation": 4.0,
                        "area_not_water": 13.0 - noise,
                    },
                ]
            )
        )
        ds = downloader.download_dswx_acquisitions(gdf=downloader.gdf, years=[2025])
        assert ds.sizes["date"] == 1
        assert float(ds["area_open_water"].max()) == pytest.approx(3.0)
        assert float(ds["area_not_water"].max()) == pytest.approx(13.0, abs=1e-6)

    def test_max_nodata_filters_observations(self, downloader):
        frame = _dswx_frame(
            [
                ("lakeA", "2025-06-05", 0.90, 5.0, 4.0),
                ("lakeA", "2025-07-05", 1.00, 7.0, 3.0),
                ("lakeA", "2025-08-05", 0.99, 6.0, 3.98),
            ]
        )
        downloader._extract_dswx_acquisitions = mock.Mock(return_value=frame)
        strict = downloader.download_dswx_acquisitions(gdf=downloader.gdf, years=[2025])
        assert strict.sizes["date"] == 2

        downloader._extract_dswx_acquisitions = mock.Mock(return_value=frame)
        tolerant = downloader.download_dswx_acquisitions(gdf=downloader.gdf, years=[2025], max_nodata=0.2)
        assert tolerant.sizes["date"] == 3

    def test_small_polygon_gets_a_pixel_floor(self, downloader):
        """A 0.5 ha pond is ~6 pixels at 30 m: one missing pixel must not reject the scene."""
        downloader._polygon_areas_ha = mock.Mock(return_value={"lakeA": 0.5})
        frame = pd.DataFrame(
            [
                {
                    "id_geohash": "lakeA",
                    "date": pd.Timestamp("2025-06-05"),
                    "area_open_water": 0.2,
                    "area_inundated_vegetation": 0.2,
                    "area_not_water": 0.0,
                }
            ]
        )
        downloader._extract_dswx_acquisitions = mock.Mock(return_value=frame)

        # 0.4 of 0.5 ha = 80 % coverage: the 2 % rule alone would drop it.
        kept = downloader.download_dswx_acquisitions(gdf=downloader.gdf, years=[2025])
        assert kept.sizes["date"] == 1

        downloader._extract_dswx_acquisitions = mock.Mock(return_value=frame)
        with pytest.raises(NoDSWxDataError):
            downloader.download_dswx_acquisitions(gdf=downloader.gdf, years=[2025], min_masked_pixels=0)

    def test_connection_error_propagates(self, downloader):
        import requests

        downloader._extract_dswx_acquisitions = mock.Mock(
            side_effect=requests.exceptions.ConnectionError("network down")
        )
        with pytest.raises(requests.exceptions.ConnectionError):
            downloader.download_dswx_acquisitions(gdf=downloader.gdf, years=[2025])


class TestDswxMonthly:
    def test_dates_are_month_starts_and_attrs_record_the_composite(self, downloader):
        downloader._extract_dswx_monthly = mock.Mock(
            return_value=_dswx_frame(
                [("lakeA", "2025-06-01", 1.0, 6.0, 4.0), ("lakeA", "2025-07-01", 1.0, 7.0, 3.0)]
            )
        )
        ds = downloader.download_dswx_monthly(gdf=downloader.gdf, years=[2025], months=[6, 7])
        assert set(ds.data_vars) == set(dswx_bandnames)
        assert ds.attrs["temporal_resolution"] == "monthly"
        assert ds.attrs["monthly_composite"] == "mode"
        assert list(pd.to_datetime(ds["date"].values)) == [pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01")]

    def test_dates_passed_to_extraction_are_monthly(self, downloader):
        downloader._extract_dswx_monthly = mock.Mock(
            return_value=_dswx_frame([("lakeA", "2025-06-01", 1.0, 6.0, 4.0)])
        )
        downloader.download_dswx_monthly(gdf=downloader.gdf, date_list=["2025-06", "2025-08"])
        assert downloader._extract_dswx_monthly.call_args.kwargs["dates"] == ["2025-06-01", "2025-08-01"]

    def test_month_without_acquisition_is_absent_not_zero(self, downloader):
        """Only June comes back, so the series has one date - not a July row of zeros."""
        downloader._extract_dswx_monthly = mock.Mock(
            return_value=_dswx_frame([("lakeA", "2025-06-01", 1.0, 6.0, 4.0)])
        )
        ds = downloader.download_dswx_monthly(gdf=downloader.gdf, years=[2025], months=[6, 7])
        assert ds.sizes["date"] == 1
        assert float(ds["area_open_water"].min()) > 0

    def test_requires_months_or_date_list(self, downloader):
        with pytest.raises(ValueError, match="date_list"):
            downloader.download_dswx_monthly(gdf=downloader.gdf, years=[2025])

    def test_no_download_returns_none(self, downloader):
        with mock.patch(
            "water_timeseries.downloader.dswx_month_composite", side_effect=AssertionError("must not query EE")
        ):
            result = downloader.download_dswx_monthly(
                gdf=downloader.gdf, years=[2025], months=[6], no_download=True
            )
        assert result is None
