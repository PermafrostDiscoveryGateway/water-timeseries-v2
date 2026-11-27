"""Tests for the tylertoo tiling engine and its dispatch from ``pmtiles_build``."""

import importlib.util
import json

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pytest
import shapely
from shapely.geometry import Polygon

from water_timeseries.utils import pmtiles_build
from water_timeseries.utils.pmtiles_build import tiler_engine
from water_timeseries.utils.pmtiles_build_tylertoo import (
    build_pmtiles_nrt_monthly,
    centroid_parquet,
    find_tylertoo_cli,
    prepare_parquet,
)
from water_timeseries.utils.pmtiles_reader import read_pmtiles_metadata

HAS_TYLERTOO = importlib.util.find_spec("tylertoo") is not None
HAS_CLI = find_tylertoo_cli() is not None


def lakes_fixture(tmp_path, count: int = 3):
    """A tiny GeoParquet with the real schema's column names and CRS."""
    polygons = []
    rows = []
    for index in range(count):
        lon = -160.0 + index * 0.5
        polygons.append(Polygon([(lon, 66.0), (lon + 0.4, 66.0), (lon + 0.4, 66.4), (lon, 66.4)]))
        rows.append(
            {
                "id_geohash": f"lake{index}",
                "Area_start_ha": 10.0 + index,
                "Area_end_ha": 5.0 + index,
                "NetChange_ha": -5.0,
                "NetChange_perc": -50.0,
                "date_break_year": None if index else 2020.0,
                "unused_column": "drop me",
            }
        )
    gdf = gpd.GeoDataFrame(rows, geometry=polygons, crs="EPSG:4326")
    path = tmp_path / "lakes.parquet"
    gdf.to_parquet(path)
    return path


def test_tiler_engine_defaults_to_tylertoo(monkeypatch):
    monkeypatch.delenv("WTS_TILER", raising=False)
    assert tiler_engine() == "tylertoo"


def test_tiler_engine_honours_the_escape_hatch(monkeypatch):
    monkeypatch.setenv("WTS_TILER", "tippecanoe")
    assert tiler_engine() == "tippecanoe"


def test_tiler_engine_rejects_an_unknown_engine(monkeypatch):
    monkeypatch.setenv("WTS_TILER", "mapnik")
    with pytest.raises(ValueError, match="WTS_TILER"):
        tiler_engine()


def test_prepare_parquet_keeps_only_requested_columns_and_filtered_rows(tmp_path):
    source = lakes_fixture(tmp_path, count=4)
    prepared = prepare_parquet(
        source,
        tmp_path / "prepared.parquet",
        columns=["id_geohash", "Area_start_ha", "date_break_year"],
        row_filter=lambda df: df["date_break_year"].notna(),
        sort=False,
    )
    table = pq.read_table(prepared)
    assert table.column_names == ["id_geohash", "Area_start_ha", "date_break_year", "geometry"]
    assert table.num_rows == 1, "row_filter must drop rows before any tile work"
    metadata = table.schema.metadata or {}
    assert b"geo" in metadata, "the geo metadata is what lets tylertoo read the file"
    assert json.loads(metadata[b"geo"])["primary_column"] == "geometry"


def test_prepare_parquet_renders_datetimes_as_strings(tmp_path):
    source = lakes_fixture(tmp_path, count=2)
    gdf = gpd.read_parquet(source)
    gdf["date_break"] = pd.to_datetime(gdf["date_break_year"], errors="coerce")
    gdf.to_parquet(source)

    prepared = prepare_parquet(source, tmp_path / "prepared.parquet", columns=["date_break"], sort=False)
    assert str(pq.read_table(prepared).schema.field("date_break").type) == "string"


def test_centroid_parquet_writes_points_in_lon_lat(tmp_path):
    source = lakes_fixture(tmp_path)
    prepared = prepare_parquet(
        source,
        tmp_path / "prepared.parquet",
        columns=["id_geohash"],
        sort=False,
    )
    points = centroid_parquet(prepared, tmp_path / "centroids.parquet")
    table = pq.read_table(points)
    assert table.num_rows == 3
    geo = json.loads((table.schema.metadata or {})[b"geo"])
    assert geo["columns"]["geometry"]["geometry_types"] == ["Point"]

    first = shapely.from_wkb(table.column("geometry")[0].as_py())
    assert first.geom_type == "Point"
    assert -161 < first.x < -159 and 65 < first.y < 67


@pytest.mark.skipif(not HAS_TYLERTOO, reason="tylertoo is not installed")
def test_build_pmtiles_dispatches_to_tylertoo(tmp_path, monkeypatch):
    if not HAS_CLI:
        pytest.skip("the multi-layer join needs the tylertoo binary (TYLERTOO_CLI)")
    monkeypatch.delenv("WTS_TILER", raising=False)
    source = lakes_fixture(tmp_path, count=6)
    output = tmp_path / "lakes.pmtiles"

    pmtiles_build.build_pmtiles(
        source,
        output,
        property_columns=["id_geohash", "Area_start_ha"],
        point_property_columns=["id_geohash"],
    )

    assert output.exists() and output.stat().st_size > 0
    metadata = read_pmtiles_metadata(output)
    layers = {layer["id"]: (layer["minzoom"], layer["maxzoom"]) for layer in metadata["vector_layers"]}
    assert layers == {
        "lakes": (pmtiles_build.POINT_POLY_SWITCH_ZOOM - 1, pmtiles_build.TILE_MAX_ZOOM),
        "lakes_points": (0, pmtiles_build.POINT_POLY_SWITCH_ZOOM + 1),
    }, "both layers must be present, overlapping around the switch zoom"
    assert "strategies" not in metadata, "tippecanoe's metadata is gone; no code may read it"


def test_nrt_monthly_is_not_migrated_yet():
    with pytest.raises(RuntimeError, match="still uses tippecanoe"):
        build_pmtiles_nrt_monthly()
