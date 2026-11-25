"""Selection by a configurable object-id field (issue #312).

The default stays ``"id_geohash"``: every case here that omits ``id_field`` must
behave exactly as before the change.
"""

import pathlib

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point

from water_timeseries.utils.data import ensure_id_columns, resolve_id_field
from water_timeseries.utils.spatial import select_feature_by_id


@pytest.fixture
def gdf_renamed():
    """Two polygons whose identifier column is called ``lake_id``."""
    return gpd.GeoDataFrame(
        {"lake_id": ["lakeA", "lakeB"], "area": [1.0, 2.0]},
        geometry=[Point(0, 0).buffer(0.01), Point(0, 1).buffer(0.01)],
        crs="EPSG:4326",
    )


@pytest.fixture
def gdf_default():
    """Two polygons with the canonical ``id_geohash`` column."""
    return gpd.GeoDataFrame(
        {"id_geohash": ["b7g4", "b7g5"], "area": [1.0, 2.0]},
        geometry=[Point(0, 0).buffer(0.01), Point(0, 1).buffer(0.01)],
        crs="EPSG:4326",
    )


class TestSelectFeatureById:
    def test_default_id_field_selects_from_canonical_frame(self, gdf_default):
        out = select_feature_by_id(gdf_default, "b7g4")
        assert len(out) == 1
        assert out.iloc[0]["area"] == 1.0
        assert out.index.equals(pd.Index([gdf_default.index[0]]))

    def test_custom_id_field_selects(self, gdf_renamed):
        out = select_feature_by_id(gdf_renamed, "lakeB", id_field="lake_id")
        assert len(out) == 1
        assert out.iloc[0]["area"] == 2.0

    def test_missing_id_field_names_the_columns(self, gdf_renamed):
        with pytest.raises(KeyError, match="lake_id"):
            select_feature_by_id(gdf_renamed, "lakeA")

    def test_unknown_object_raises_value_error(self, gdf_default):
        with pytest.raises(ValueError, match="nope"):
            select_feature_by_id(gdf_default, "nope")


class TestResolveIdField:
    def test_defaults_to_id_geohash(self):
        ds = xr.Dataset(coords={"id_geohash": ["b7g4", "b7g5"], "date": pd.date_range("2020-01-01", periods=2)})
        assert resolve_id_field(ds) == "id_geohash"

    def test_reads_attrs_first(self):
        ds = xr.Dataset(coords={"lake_id": ["a", "b"], "date": pd.date_range("2020-01-01", periods=2)})
        ds.attrs["id_field"] = "lake_id"
        assert resolve_id_field(ds) == "lake_id"

    def test_finds_a_single_known_name_without_attrs(self):
        ds = xr.Dataset(coords={"lake_id": ["a", "b"], "date": pd.date_range("2020-01-01", periods=2)})
        assert resolve_id_field(ds) == "lake_id"

    def test_ambiguous_frame_raises(self):
        ds = xr.Dataset(
            coords={
                "id_geohash": ["a", "b"],
                "lake_id": ["a", "b"],
                "date": pd.date_range("2020-01-01", periods=2),
            }
        )
        with pytest.raises(ValueError, match="ambiguous|id_field"):
            resolve_id_field(ds)

    def test_no_identifier_raises(self):
        ds = xr.Dataset(coords={"date": pd.date_range("2020-01-01", periods=2)})
        with pytest.raises(ValueError, match="id_field|identifier"):
            resolve_id_field(ds)


class TestLoadLakePolygonCached:
    def test_parquet_filter_uses_id_field(self, gdf_renamed, tmp_path):
        from water_timeseries.utils.dashboard import load_lake_polygon_cached

        path = tmp_path / "lakes_lake_id.parquet"
        gdf_renamed.to_parquet(path)

        out = load_lake_polygon_cached(str(path), "lakeA", id_field="lake_id")
        assert len(out) == 1
        assert out.iloc[0]["lake_id"] == "lakeA"

    def test_parquet_filter_default_unchanged(self, gdf_default, tmp_path):
        from water_timeseries.utils.dashboard import load_lake_polygon_cached

        path = tmp_path / "lakes_id_geohash.parquet"
        gdf_default.to_parquet(path)

        out = load_lake_polygon_cached(str(path), "b7g5")
        assert len(out) == 1
        assert out.iloc[0]["id_geohash"] == "b7g5"


DATA = pathlib.Path(__file__).parent / "data" / "lakes_dw_test.nc"


def _renamed(ds, name):
    return ds.rename({"id_geohash": name})


class TestDatasetWiring:
    """The class already knows its id_field; it must hand it to the free functions."""

    def test_id_field_is_recorded_on_attrs(self):
        from water_timeseries.dataset import DWDataset

        obj = DWDataset(_renamed(xr.open_dataset(DATA), "lake_id"), id_field="lake_id")
        assert obj.ds.attrs["id_field"] == "lake_id"

    def test_default_still_records_id_geohash(self):
        from water_timeseries.dataset import DWDataset

        obj = DWDataset(xr.open_dataset(DATA))
        assert obj.ds.attrs["id_field"] == "id_geohash"

    def test_create_timelapse_forwards_id_field(self, monkeypatch, tmp_path):
        from water_timeseries.dataset import DWDataset

        obj = DWDataset(_renamed(xr.open_dataset(DATA), "lake_id"), id_field="lake_id")
        object_id = obj.object_ids_[0]

        captured = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return tmp_path / "spy.gif"

        monkeypatch.setattr("water_timeseries.dataset.create_timelapse", spy)
        gdf = gpd.GeoDataFrame(
            {"lake_id": [object_id]},
            geometry=[Point(0, 0).buffer(1)],
            crs="EPSG:4326",
        )
        obj.create_timelapse(lake_gdf=gdf, lake_id=object_id, gif_outdir=tmp_path)

        assert captured["id_field"] == "lake_id"
        assert captured["id_geohash"] == object_id


class TestEnsureIdColumns:
    """Artefacts carry `id_geohash`; a custom id is carried *beside* it, never instead."""

    def test_noop_when_already_the_contract_name(self):
        df = pd.DataFrame({"id_geohash": ["a"], "x": [1]})
        out = ensure_id_columns(df, "id_geohash")
        assert list(out.columns) == ["id_geohash", "x"]
        assert out.equals(df)

    def test_adds_contract_beside_a_column_id(self):
        df = pd.DataFrame({"lake_id": ["a", "b"], "x": [1, 2]})
        out = ensure_id_columns(df, "lake_id")
        assert list(out.columns) == ["lake_id", "id_geohash", "x"]
        assert out["id_geohash"].tolist() == ["a", "b"]
        assert out["lake_id"].tolist() == ["a", "b"]

    def test_adds_contract_beside_an_index_id(self):
        df = pd.DataFrame({"x": [1, 2]}, index=pd.Index(["a", "b"], name="lake_id"))
        out = ensure_id_columns(df, "lake_id")
        assert out.index.name == "lake_id"
        assert out["id_geohash"].tolist() == ["a", "b"]

    def test_missing_id_raises(self):
        with pytest.raises(KeyError, match="lake_id"):
            ensure_id_columns(pd.DataFrame({"x": [1]}), "lake_id")

    def test_empty_frame_is_returned_untouched(self):
        df = pd.DataFrame(columns=["date_break", "break_method"])
        out = ensure_id_columns(df, "lake_id")
        assert out.empty and list(out.columns) == ["date_break", "break_method"]


class TestBreakOutputKeepsContractColumn:
    """`calculate_break` output: default untouched, custom id carried beside `id_geohash`."""

    @staticmethod
    def _dataset(id_name=None):
        from water_timeseries.dataset import DWDataset

        ds = xr.open_dataset(DATA)
        if id_name:
            ds = _renamed(ds, id_name)
            return DWDataset(ds, id_field=id_name), id_name
        return DWDataset(ds), "id_geohash"

    def test_default_path_adds_nothing(self):
        from water_timeseries.breakpoint import SimpleBreakpoint

        obj, _ = self._dataset()
        out = SimpleBreakpoint().calculate_break(obj, obj.object_ids_[0])
        assert "id_geohash" not in out.columns

    def test_custom_id_carries_both_names(self):
        from water_timeseries.breakpoint import SimpleBreakpoint

        obj, id_name = self._dataset("lake_id")
        object_id = obj.object_ids_[0]
        out = SimpleBreakpoint().calculate_break(obj, object_id)
        assert "id_geohash" in out.columns
        assert out["id_geohash"].tolist() == [object_id]
        assert out.index.name == id_name
