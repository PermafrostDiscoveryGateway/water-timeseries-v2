"""Tests for OPERA DSWx-HLS: the dataset handler, its registration and the downloaders.

No network: Earth Engine is replaced by frames shaped like ``geemap.ee_to_df`` output.
"""

from unittest import mock

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from water_timeseries.dataset import DSWxDataset, DSWxHLSDataset, DWDataset, create_water_dataset
from water_timeseries.downloader import EarthEngineDownloader, NoDSWxHLSDataError
from water_timeseries.utils.data import (
    DSWX_WATER_TOTAL,
    dswx_hls_bandnames,
    dswx_hls_plot_variables,
    get_water_dataset_type,
)

LAKE_AREA_HA = 20.0


def _cube(open_ha, partial_ha, not_ha, land_ha, dates=None) -> xr.Dataset:
    """A DSWx-HLS-shaped cube: three water classes plus the LAND band."""
    dates = dates if dates is not None else pd.date_range("2025-06-01", periods=len(open_ha), freq="MS")
    return xr.Dataset(
        {
            "area_open_water": (["date", "id_geohash"], [[v] for v in open_ha]),
            "area_partial_water": (["date", "id_geohash"], [[v] for v in partial_ha]),
            "area_not_water": (["date", "id_geohash"], [[v] for v in not_ha]),
            "area_land_cover": (["date", "id_geohash"], [[v] for v in land_ha]),
        },
        coords={"date": dates, "id_geohash": ["lake"]},
    )


def _hls_frame(entries) -> pd.DataFrame:
    """Build a frame shaped like ``geemap.ee_to_df`` output.

    ``entries`` is ``(id, date, coverage, open_ha, partial_ha, land_ha)``; ``area_not_water`` fills
    the rest so the three *water* classes sum to ``coverage * LAKE_AREA_HA``.
    """
    rows = []
    for lake, day, coverage, open_ha, partial_ha, land_ha in entries:
        rows.append(
            {
                "id_geohash": lake,
                "date": pd.Timestamp(day),
                "area_open_water": open_ha,
                "area_partial_water": partial_ha,
                "area_not_water": LAKE_AREA_HA * coverage - open_ha - partial_ha,
                "area_land_cover": land_ha,
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


class TestHLSRegistration:
    def test_sniffer_separates_hls_from_s1(self):
        """Both products carry area_open_water; only HLS has the partial-water class."""
        assert get_water_dataset_type(_cube([1.0], [0.5], [18.0], [2.0])) == "dswx_hls"
        s1 = xr.Dataset(
            {
                "area_open_water": (["date", "id_geohash"], [[1.0]]),
                "area_inundated_vegetation": (["date", "id_geohash"], [[0.5]]),
                "area_not_water": (["date", "id_geohash"], [[18.0]]),
            },
            coords={"date": pd.date_range("2025-06-01", periods=1, freq="MS"), "id_geohash": ["lake"]},
        )
        assert get_water_dataset_type(s1) == "dswx"

    def test_factory_returns_the_right_class(self):
        cube = _cube([1.0], [0.5], [18.0], [2.0])
        assert isinstance(create_water_dataset(cube), DSWxHLSDataset)
        assert isinstance(create_water_dataset(cube, "dswx_hls"), DSWxHLSDataset)
        # and the S1 cube does not become an HLS dataset
        assert DSWxHLSDataset is not DSWxDataset
        assert DWDataset is not DSWxHLSDataset


class TestHLSDataset:
    def test_water_column_is_the_sum_of_both_water_classes(self):
        # both months classify the whole polygon, so masking does not interfere
        dataset = DSWxHLSDataset(_cube([6.0, 4.0], [2.0, 1.0], [10.0, 13.0], [2.0, 3.0]))
        assert dataset.water_column == DSWX_WATER_TOTAL
        values = dataset.ds[DSWX_WATER_TOTAL].values[:, 0].tolist()
        assert values == [pytest.approx(8.0), pytest.approx(5.0)]

    def test_area_data_excludes_land_cover(self):
        """LAND is not the complement of the water classes; adding it would double-count."""
        dataset = DSWxHLSDataset(_cube([6.0], [2.0], [10.0], [15.0]))
        assert float(dataset.ds["area_data"].values[0, 0]) == pytest.approx(18.0)
        assert float(dataset.ds["area_land_cover"].values[0, 0]) == pytest.approx(15.0)

    def test_columns_and_plot_defaults(self):
        dataset = DSWxHLSDataset(_cube([6.0], [2.0], [10.0], [2.0]))
        assert dataset.data_columns == dswx_hls_bandnames
        assert dataset.plot_columns == dswx_hls_plot_variables
        # the land band stays available but is not drawn by default
        assert "area_land_cover" not in dataset.plot_columns

    def test_nodata_share_masks_a_thin_observation(self):
        dataset = DSWxHLSDataset(_cube([10.0, 9.0], [0.0, 0.0], [10.0, 1.0], [0.0, 0.0]))
        # second month classifies only 10 of 20 ha -> >2 % no-data -> masked
        assert dataset.ds[DSWX_WATER_TOTAL].isel(date=1).isnull().all()

    def test_plots_run(self):
        dataset = DSWxHLSDataset(_cube([6.0, 6.5], [2.0, 2.5], [10.0, 9.0], [2.0, 2.0]))
        fig = dataset.plot_timeseries("lake")
        assert fig is not None
        fig_i = dataset.plot_timeseries_interactive("lake")
        assert fig_i is not None


class TestHLSDownloader:
    def test_bandnames_and_attrs(self, downloader):
        downloader._extract_dswx_hls_daily = mock.Mock(
            return_value=_hls_frame([("lakeA", "2025-06-05", 1.0, 6.0, 2.0, 12.0)])
        )
        ds = downloader.download_dswx_hls_daily(gdf=downloader.gdf, years=[2025], months=[6])
        assert set(ds.data_vars) == set(dswx_hls_bandnames)
        assert ds.attrs["temporal_resolution"] == "per_acquisition"
        assert ds.attrs["product"].startswith("OPERA DSWx-HLS")
        assert ds.attrs["n_acquisitions"] == 1

    def test_coverage_ignores_the_land_band(self, downloader):
        """A land-only row must not look covered: coverage is measured on the water classes."""
        downloader._extract_dswx_hls_daily = mock.Mock(
            return_value=_hls_frame([("lakeA", "2025-06-05", 0.05, 1.0, 0.0, 20.0)])
        )
        with pytest.raises(NoDSWxHLSDataError):
            downloader.download_dswx_hls_daily(gdf=downloader.gdf, years=[2025], months=[6])

    def test_fully_masked_row_is_dropped(self, downloader):
        downloader._extract_dswx_hls_daily = mock.Mock(
            return_value=_hls_frame(
                [
                    ("lakeA", "2025-06-05", 0.0, 0.0, 0.0, 0.0),
                    ("lakeA", "2025-07-05", 1.0, 6.0, 2.0, 12.0),
                ]
            )
        )
        ds = downloader.download_dswx_hls_daily(gdf=downloader.gdf, years=[2025], months=[6, 7])
        assert ds.sizes["date"] == 1

    def test_tiles_collapse_to_the_best_covered_one(self, downloader):
        downloader._extract_dswx_hls_daily = mock.Mock(
            return_value=_hls_frame(
                [
                    ("lakeA", "2025-06-05", 0.60, 8.0, 4.0, 8.0),
                    ("lakeA", "2025-06-05", 1.00, 10.0, 6.0, 4.0),
                ]
            )
        )
        ds = downloader.download_dswx_hls_daily(gdf=downloader.gdf, years=[2025], months=[6])
        assert ds.sizes["date"] == 1
        assert float(ds["area_partial_water"].max()) == pytest.approx(6.0)

    def test_months_and_date_list_select_the_same_windows(self, downloader):
        downloader._extract_dswx_hls_daily = mock.Mock(
            return_value=_hls_frame([("lakeA", "2025-06-05", 1.0, 6.0, 2.0, 12.0)])
        )
        downloader.download_dswx_hls_daily(gdf=downloader.gdf, years=[2025], months=[6, 7])
        by_months = downloader._extract_dswx_hls_daily.call_args.kwargs["windows"]
        downloader.download_dswx_hls_daily(gdf=downloader.gdf, date_list=["2025-06", "2025-07"])
        by_date_list = downloader._extract_dswx_hls_daily.call_args.kwargs["windows"]
        assert by_months == by_date_list

    def test_archive_floor_is_2016(self, downloader):
        assert downloader._validate_years_dswx([2015, 2016, 2025], archive_start="2016-03-01") == [2016, 2025]
        assert downloader._validate_years_dswx(None, archive_start="2016-03-01")[0] == 2016

    def test_no_download_makes_no_ee_call(self, downloader):
        with mock.patch(
            "water_timeseries.downloader.dswx_hls_acquisitions", side_effect=AssertionError("must not query EE")
        ):
            assert downloader.download_dswx_hls_daily(gdf=downloader.gdf, years=[2025], no_download=True) is None

    def test_monthly_absent_month_is_absent(self, downloader):
        downloader._extract_dswx_hls_monthly = mock.Mock(
            return_value=_hls_frame([("lakeA", "2025-06-01", 1.0, 6.0, 2.0, 12.0)])
        )
        ds = downloader.download_dswx_hls_monthly(gdf=downloader.gdf, years=[2025], months=[6, 7])
        assert ds.sizes["date"] == 1
        assert ds.attrs["monthly_composite"] == "mode"
        assert ds.attrs["product"].startswith("OPERA DSWx-HLS")

    def test_monthly_requires_months_or_date_list(self, downloader):
        with pytest.raises(ValueError, match="date_list"):
            downloader.download_dswx_hls_monthly(gdf=downloader.gdf, years=[2025])
