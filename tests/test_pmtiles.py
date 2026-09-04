"""Tests for PMTiles build and serve utilities."""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from water_timeseries.utils.pmtiles_build import (
    DEFAULT_TILE_PROPERTIES,
    POINT_POLY_OVERLAP_ZOOMS,
    POINT_POLY_SWITCH_ZOOM,
    build_pmtiles_nrt_monthly,
    find_tippecanoe,
    nrt_monthly_tiles_filename,
    parquet_to_geojsonseq,
    point_poly_zoom_ranges,
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


@pytest.mark.skipif(find_tippecanoe() is None, reason="tippecanoe not installed")
def test_build_nrt_pmtiles_cli_config_file(tmp_path):
    """``--config-file`` resolves breaks/geometry/output-dir from dashboard config keys."""
    from water_timeseries.scripts.cli import build_nrt_pmtiles

    geohashes = gpd.read_parquet(TEST_PARQUET)["id_geohash"].astype(str).tolist()[:3]
    breaks_dir = tmp_path / "precomputed_nrt"
    breaks_dir.mkdir()
    _write_breaks_fixture(breaks_dir / "nrt_monthly_drain_breaks.parquet", geohashes)

    config_path = tmp_path / "dashboard_config.yaml"
    config_path.write_text(
        f"vector_file: {TEST_PARQUET}\nprecomputed_nrt_dir: {breaks_dir}\nnrt_pmtiles_dir: {tmp_path / 'tiles'}\n"
    )

    build_nrt_pmtiles(config_file=config_path, months="2018-07")

    assert (tmp_path / "tiles" / "nrt_2018-07_drainage.pmtiles").exists()

    # A CLI flag overrides the config file's value for that key.
    build_nrt_pmtiles(config_file=config_path, months="2018-07", output_dir=tmp_path / "tiles_override")
    assert (tmp_path / "tiles_override" / "nrt_2018-07_drainage.pmtiles").exists()


def test_build_nrt_pmtiles_cli_missing_paths_raises(tmp_path):
    """No breaks/geometry/output_dir from CLI or config -> a clear error, not a crash."""
    from water_timeseries.scripts.cli import build_nrt_pmtiles

    with pytest.raises(SystemExit):
        build_nrt_pmtiles()


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


def test_nrt_drained_centroid_polygon_handoff_has_no_gap():
    """Circles must stay visible right up to the zoom the polygons take over at.

    MapLibre hides a layer at zoom >= maxzoom but shows it at zoom >= minzoom,
    so the centroid maxzoom and the polygon minzoom have to be the same number
    to cover every zoom exactly once. Splitting them (e.g. points maxzoom 8 with
    fill minzoom 9) leaves z8 with no drained lakes drawn at all.
    """
    from water_timeseries.map_utils import build_pmtiles_map
    from water_timeseries.utils.pmtiles_build import POINT_POLY_SWITCH_ZOOM

    m = build_pmtiles_map(
        "http://localhost:1/lakes.pmtiles",
        viz_configuration_name="nrt_drainage",
        nrt_monthly_tiles_url="http://localhost:1/nrt_2026-07_drainage.pmtiles",
    )
    html = m.get_root().render()

    # The page renders more than one style block; find the one holding the
    # drained overlay rather than assuming it comes first.
    layers = {}
    marker = '"layers": '
    offset = html.find(marker)
    while offset != -1 and "nrt-drained-points" not in layers:
        try:
            layer_list, _ = json.JSONDecoder().raw_decode(html[offset + len(marker) :])
        except json.JSONDecodeError:
            layer_list = []
        if isinstance(layer_list, list):
            layers = {layer["id"]: layer for layer in layer_list if isinstance(layer, dict) and "id" in layer}
        offset = html.find(marker, offset + 1)
    assert "nrt-drained-points" in layers, "no style block contained the drained overlay layers"

    points = layers["nrt-drained-points"]
    fill = layers["nrt-drained-fill"]
    line = layers["nrt-drained-line"]

    assert points["maxzoom"] == POINT_POLY_SWITCH_ZOOM
    assert fill["minzoom"] == POINT_POLY_SWITCH_ZOOM
    assert line["minzoom"] == POINT_POLY_SWITCH_ZOOM

    # Every zoom must draw the month's drained lakes exactly one way. Guards the
    # gap directly rather than trusting the three numbers above to stay in sync.
    for zoom in range(15):
        as_circles = zoom < points["maxzoom"]
        as_polygons = zoom >= fill["minzoom"]
        assert as_circles != as_polygons, f"z{zoom} draws drained lakes {'twice' if as_circles else 'not at all'}"


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


def test_pmtiles_server_serves_mounted_archives(tmp_path):
    """Two archives from different directories stay reachable at once.

    Switching dashboard modes swaps the tileset, but one process serves every
    browser session, so the previous archive must keep working.
    """
    import urllib.request

    first_dir = tmp_path / "historical"
    second_dir = tmp_path / "nrt"
    first_dir.mkdir()
    second_dir.mkdir()
    # Same filename in both directories: mounts must keep them apart.
    (first_dir / "lakes.pmtiles").write_bytes(b"a" * 100)
    (second_dir / "lakes.pmtiles").write_bytes(b"b" * 200)

    with PmtilesServer(None, port=0) as server:
        first_url = server.pmtiles_url_for(first_dir / "lakes.pmtiles")
        second_url = server.pmtiles_url_for(second_dir / "lakes.pmtiles")
        assert first_url != second_url

        with urllib.request.urlopen(first_url) as resp:
            assert resp.read() == b"a" * 100
        with urllib.request.urlopen(second_url) as resp:
            assert resp.read() == b"b" * 200

        # Re-registering the same archive reuses its mount.
        assert server.pmtiles_url_for(first_dir / "lakes.pmtiles") == first_url


def test_pmtiles_server_map_page_keeps_config_url(tmp_path):
    """A config that carries its own tile URL is not rewritten by the server."""
    import urllib.request

    archive_dir = tmp_path / "tiles"
    archive_dir.mkdir()
    (archive_dir / "nrt.pmtiles").write_bytes(b"0" * 100)
    default = tmp_path / "default.pmtiles"
    default.write_bytes(b"0" * 100)

    with PmtilesServer(default, port=0) as server:
        tile_url = server.pmtiles_url_for(archive_dir / "nrt.pmtiles")
        map_url = server.map_iframe_url({"pmtiles_url": tile_url, "center": [-164, 66.5], "zoom": 8})
        with urllib.request.urlopen(map_url) as resp:
            html = resp.read().decode("utf-8")
        assert tile_url in html
        assert server.url_for("default.pmtiles") not in html


def test_build_stamps_the_same_switch_zoom_it_is_styled_with():
    """The per-feature zoom ranges baked into the tiles must match the style's gates.

    tippecanoe's ``-L`` layer JSON has no minzoom/maxzoom keys, so this split
    only exists because ``_write_features`` stamps it onto each feature. If it
    silently stops being written, the tiles carry both layers at every zoom
    again and the style's gates are all that stand between the user and dots
    drawn on top of polygons.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        poly_path, points_path = parquet_to_geojsonseq(TEST_PARQUET, Path(tmp) / "lakes.geojsonl")

        poly = json.loads(poly_path.read_text(encoding="utf-8").splitlines()[0])
        point = json.loads(points_path.read_text(encoding="utf-8").splitlines()[0])

    poly_zoom, points_zoom = point_poly_zoom_ranges()
    assert poly["tippecanoe"] == {"minzoom": poly_zoom[0], "maxzoom": poly_zoom[1]}
    assert point["tippecanoe"] == {"minzoom": points_zoom[0], "maxzoom": points_zoom[1]}


def test_baked_zoom_band_lets_the_switch_move_without_a_rebuild():
    """Both layers exist for a few zooms around the switch, so retuning it is a style change.

    The style draws each zoom exactly one way (see the handoff tests); this band
    is only about having the data on hand if we decide the dots should give way
    to polygons a level earlier or later.
    """
    poly_zoom, points_zoom = point_poly_zoom_ranges()

    baked_both = set(range(poly_zoom[0], poly_zoom[1] + 1)) & set(range(points_zoom[0], points_zoom[1] + 1))
    movable = set(
        range(POINT_POLY_SWITCH_ZOOM - POINT_POLY_OVERLAP_ZOOMS, POINT_POLY_SWITCH_ZOOM + POINT_POLY_OVERLAP_ZOOMS + 1)
    )
    assert movable <= baked_both, f"switch cannot move to {sorted(movable - baked_both)} without rebuilding"

    # A switch anywhere in the band still has data on both sides of it.
    for switch in sorted(movable):
        assert switch - 1 in range(points_zoom[0], points_zoom[1] + 1), f"no centroids just below z{switch}"
        assert switch in range(poly_zoom[0], poly_zoom[1] + 1), f"no polygons at z{switch}"
