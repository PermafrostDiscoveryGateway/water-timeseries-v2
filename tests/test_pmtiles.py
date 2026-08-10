"""Tests for PMTiles build and serve utilities."""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from water_timeseries.utils.pmtiles_build import (
    DEFAULT_TILE_PROPERTIES,
    build_pmtiles_nrt_monthly,
    find_tippecanoe,
    nrt_monthly_tiles_filename,
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


def _write_breaks_fixture(path: Path, geohashes: list[str]) -> Path:
    """Two months of drained lakes: one with confidence, one with only water loss."""
    rows = [
        {"analysis_month": "2026-07", "id_geohash": gid, "drainage_confidence": conf, "water_observed": 0.5}
        for gid, conf in zip(geohashes, [1.0, 2.0, 3.0] * len(geohashes), strict=False)
    ]
    rows += [
        {"analysis_month": "2018-07", "id_geohash": gid, "water_change_perc": -60.0, "water_change_ha": -3.0}
        for gid in geohashes[:2]
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


@pytest.mark.skipif(find_tippecanoe() is None, reason="tippecanoe not installed")
def test_build_pmtiles_nrt_monthly(tmp_path):
    """One archive per month, holding only that month's drained lakes."""
    geohashes = gpd.read_parquet(TEST_PARQUET)["id_geohash"].astype(str).tolist()[:3]
    breaks = _write_breaks_fixture(tmp_path / "breaks.parquet", geohashes)

    outputs = build_pmtiles_nrt_monthly(breaks, TEST_PARQUET, tmp_path / "tiles", keep_geojsonl=True)

    assert set(outputs) == {"2026-07", "2018-07"}
    for month, path in outputs.items():
        assert path.name == nrt_monthly_tiles_filename(month)
        assert path.stat().st_size > 0

    # Only the month's own lakes land in its archive, with that month's values.
    july_2018 = json.loads((tmp_path / "tiles" / "nrt_2018-07_drainage.geojsonl").read_text().splitlines()[0])
    assert july_2018["properties"]["analysis_month"] == "2018-07"
    assert july_2018["properties"]["water_change_perc"] == -60.0
    assert len((tmp_path / "tiles" / "nrt_2018-07_drainage.geojsonl").read_text().strip().splitlines()) == 2

    nrt_lines = (tmp_path / "tiles" / "nrt_2026-07_drainage.geojsonl").read_text().strip().splitlines()
    assert len(nrt_lines) == 3
    assert json.loads(nrt_lines[0])["properties"]["drainage_confidence"] in (1.0, 2.0, 3.0)

    # Centroids are emitted alongside the polygons for the low-zoom layer.
    points = (tmp_path / "tiles" / "nrt_2026-07_drainage_points.geojsonl").read_text().strip().splitlines()
    assert len(points) == 3
    assert json.loads(points[0])["geometry"]["type"] == "Point"


def test_build_pmtiles_nrt_monthly_months_filter(tmp_path):
    """`months` restricts the build; an unknown month is an error, not a silent no-op."""
    geohashes = gpd.read_parquet(TEST_PARQUET)["id_geohash"].astype(str).tolist()[:3]
    breaks = _write_breaks_fixture(tmp_path / "breaks.parquet", geohashes)

    if find_tippecanoe() is not None:
        outputs = build_pmtiles_nrt_monthly(breaks, TEST_PARQUET, tmp_path / "tiles", months=["2018-07"])
        assert set(outputs) == {"2018-07"}

    with pytest.raises(ValueError, match="No rows"):
        build_pmtiles_nrt_monthly(breaks, TEST_PARQUET, tmp_path / "tiles2", months=["1999-01"])


def test_resolve_nrt_monthly_tiles_url(tmp_path, monkeypatch):
    from water_timeseries.map_utils import resolve_nrt_monthly_tiles_url

    monkeypatch.delenv("PMTILES_BASE_URL", raising=False)

    assert resolve_nrt_monthly_tiles_url(None, "2026-07") is None
    # Missing month -> None, which is the caller's signal to use the fallback path.
    assert resolve_nrt_monthly_tiles_url(tmp_path, "2026-07") is None

    (tmp_path / nrt_monthly_tiles_filename("2026-07")).write_bytes(b"0" * 100)
    local_url = resolve_nrt_monthly_tiles_url(tmp_path, "2026-07")
    assert local_url is not None
    assert local_url.endswith("nrt_2026-07_drainage.pmtiles")
    assert local_url.startswith("http://")

    assert (
        resolve_nrt_monthly_tiles_url("https://example.com/tiles/", "2026-06")
        == "https://example.com/tiles/nrt_2026-06_drainage.pmtiles"
    )
    assert (
        resolve_nrt_monthly_tiles_url("gs://bucket/tiles", "2026-06")
        == "https://storage.googleapis.com/bucket/tiles/nrt_2026-06_drainage.pmtiles"
    )


def test_build_pmtiles_map_monthly_tiles_layers():
    """The monthly tileset becomes its own source/layers, with nothing per-lake inlined."""
    from water_timeseries.map_utils import build_pmtiles_map

    tiles_url = "http://localhost:1/nrt_2026-07_drainage.pmtiles"
    m = build_pmtiles_map(
        "http://localhost:1/lakes.pmtiles",
        viz_configuration_name="nrt_drainage",
        nrt_monthly_tiles_url=tiles_url,
    )
    html = m.get_root().render()

    assert f"pmtiles://{tiles_url}" in html
    assert "nrt-drained-fill" in html
    assert "nrt-drained-points" in html
    # No feature-state push. The tooltip prefers the overlay but falls back to
    # the base lakes, so non-drained lakes still hover -- minus the month-
    # specific properties some base tilesets bake from a single NRT run.
    assert "setFeatureState" not in html
    assert '["nrt-drained-fill", "lakes-fill"]' in html
    assert '{"lakes-fill": ["date", "drainage_confidence"]}' in html
    # Nothing per-lake reaches the page: no lake id, no state dict.
    assert "b7uefy0bvcrc" not in html
    assert "stateById" not in html

    # The dict-based fallback is what inlines per-lake state; assert the
    # contrast so a regression back to it is visible.
    fallback = build_pmtiles_map(
        "http://localhost:1/lakes.pmtiles",
        viz_configuration_name="nrt_drainage",
        nrt_confidence_by_id={"b7uefy0bvcrc": 3},
        nrt_tooltip_overrides={"b7uefy0bvcrc": {"date": "2026-07"}},
    )
    fallback_html = fallback.get_root().render()
    assert "setFeatureState" in fallback_html
    assert "b7uefy0bvcrc" in fallback_html


def test_build_pmtiles_map_hide_stable_lakes_hides_base_layers():
    """Hiding stable lakes must switch the base layers off, not just zero their opacity.

    A zero-opacity layer still answers queryRenderedFeatures, which would leave
    hidden lakes producing hover popups.
    """
    from water_timeseries.map_utils import build_pmtiles_map

    def base_layers(hide: bool) -> dict:
        m = build_pmtiles_map(
            "http://localhost:1/lakes.pmtiles",
            viz_configuration_name="nrt_drainage",
            nrt_monthly_tiles_url="http://localhost:1/nrt_2026-07_drainage.pmtiles",
            hide_stable_lakes=hide,
        )
        style = next(child for child in m._children.values() if hasattr(child, "style")).style
        return {layer["id"]: layer for layer in style["layers"]}

    shown = base_layers(False)
    hidden = base_layers(True)

    for layer_id in ("lakes-fill", "lakes-line", "lakes-points"):
        assert "layout" not in shown[layer_id], layer_id
        assert hidden[layer_id]["layout"]["visibility"] == "none", layer_id

    # The month's drained lakes stay visible either way.
    for layer_id in ("nrt-drained-fill", "nrt-drained-line", "nrt-drained-points"):
        assert hidden[layer_id].get("layout", {}).get("visibility") != "none", layer_id


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
