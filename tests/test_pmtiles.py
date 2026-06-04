"""Tests for PMTiles build and serve utilities."""

import json
from pathlib import Path

import geopandas as gpd
import pytest

from water_timeseries.utils.pmtiles_build import (
    DEFAULT_TILE_PROPERTIES,
    find_tippecanoe,
    parquet_to_geojsonseq,
)
from water_timeseries.utils.pmtiles_reader import read_pmtiles_header
from water_timeseries.utils.pmtiles_serve import PmtilesServer

TEST_PARQUET = Path(__file__).parent / "data" / "lake_polygons.parquet"


def test_parquet_to_geojsonseq(tmp_path):
    out = tmp_path / "lakes.geojsonl"
    parquet_to_geojsonseq(TEST_PARQUET, out)

    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    gdf = gpd.read_parquet(TEST_PARQUET)
    valid = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    assert len(lines) == len(valid)

    feature = json.loads(lines[0])
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] in ("Polygon", "MultiPolygon")
    for col in DEFAULT_TILE_PROPERTIES:
        assert col in feature["properties"]


@pytest.mark.skipif(find_tippecanoe() is None, reason="tippecanoe not installed")
def test_build_pmtiles_integration(tmp_path):
    from water_timeseries.utils.pmtiles_build import build_pmtiles

    output = tmp_path / "lakes.pmtiles"
    build_pmtiles(TEST_PARQUET, output)
    assert output.exists()
    assert output.stat().st_size > 1000


def test_pmtiles_server_range_requests(tmp_path):
    pmtiles = tmp_path / "test.pmtiles"
    pmtiles.write_bytes(b"0" * 1000)

    with PmtilesServer(pmtiles, port=0) as server:
        import urllib.request

        url = server.url_for("test.pmtiles")
        req = urllib.request.Request(url, headers={"Range": "bytes=0-99"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 206
            assert len(resp.read()) == 100


def test_read_pmtiles_header(tmp_path):
    # Minimal v3 header: magic + version + zeros through bounds at offset 102
    header = bytearray(127)
    header[0:7] = b"PMTiles"
    header[7] = 0x3
    header[100] = 0
    header[101] = 14
    header[118] = 4
    import struct

    struct.pack_into("<iiii", header, 102, -120_000_000, 40_000_000, -80_000_000, 70_000_000)
    struct.pack_into("<ii", header, 119, -100_000_000, 55_000_000)
    pmtiles = tmp_path / "hdr.pmtiles"
    pmtiles.write_bytes(bytes(header) + b"\x00" * 100)

    meta = read_pmtiles_header(pmtiles)
    assert meta["bounds"] == [[-12.0, 4.0], [-8.0, 7.0]]
    assert meta["center"] == [-10.0, 5.5]
    assert meta["min_zoom"] == 0
    assert meta["max_zoom"] == 14


def test_pmtiles_server_large_file_range_without_full_read(tmp_path, monkeypatch):
    """Range requests must not load the entire archive into RAM."""
    pmtiles = tmp_path / "big.pmtiles"
    pmtiles.write_bytes(b"x" * 50_000_000)

    read_calls: list[int] = []

    original_open = open

    def tracking_open(path, mode="rb", *args, **kwargs):
        f = original_open(path, mode, *args, **kwargs)
        if mode == "rb" and Path(path) == pmtiles.resolve():

            class Tracked:
                def __init__(self, inner):
                    self._inner = inner

                def read(self, size=-1):
                    read_calls.append(size if size >= 0 else -1)
                    return self._inner.read(size)

                def seek(self, *a, **kw):
                    return self._inner.seek(*a, **kw)

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    self._inner.close()

                def __getattr__(self, name):
                    return getattr(self._inner, name)

            return Tracked(f)
        return f

    monkeypatch.setattr("builtins.open", tracking_open)

    with PmtilesServer(pmtiles, port=0) as server:
        import urllib.request

        url = server.url_for("big.pmtiles")
        req = urllib.request.Request(url, headers={"Range": "bytes=1000-1999"})
        with urllib.request.urlopen(req) as resp:
            body = resp.read()
        assert len(body) == 1000
        assert sum(1 for c in read_calls if c == -1 or c > 10_000_000) == 0


def test_pmtiles_server_map_page(tmp_path):
    pmtiles = tmp_path / "lakes.pmtiles"
    pmtiles.write_bytes(b"0" * 100)

    with PmtilesServer(pmtiles, port=0) as server:
        import urllib.request

        map_url = server.map_iframe_url({"center": [-164, 66.5], "zoom": 8})
        with urllib.request.urlopen(map_url) as resp:
            html = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "maplibregl" in html
        assert server.url_for("lakes.pmtiles") in html
