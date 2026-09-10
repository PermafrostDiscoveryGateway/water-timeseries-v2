"""Tests for PMTiles build and serve utilities."""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from water_timeseries.utils.pmtiles_build import (
    DEFAULT_TILE_PROPERTIES,
    NRT_SCORED_LAYER,
    NRT_SCORED_TILE_PROPERTIES,
    POINT_POLY_OVERLAP_ZOOMS,
    POINT_POLY_SWITCH_ZOOM,
    SHARED_GEOMETRY_POINT_PROPERTIES,
    SHARED_GEOMETRY_TILE_PROPERTIES,
    TILE_MAX_ZOOM,
    archive_bakes_low_zoom_centroids,
    build_pmtiles_nrt_monthly,
    build_pmtiles_shared_geometry,
    find_nrt_run_parquets,
    find_tippecanoe,
    nrt_monthly_tiles_filename,
    nrt_scored_rows,
    parquet_to_geojsonseq,
    point_poly_zoom_ranges,
)
from water_timeseries.utils.pmtiles_reader import read_pmtiles_header, read_pmtiles_metadata
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


def _style_layers(html: str, required_layer_id: str) -> dict[str, dict]:
    """Return the MapLibre style layers, keyed by id, from the block holding ``required_layer_id``.

    The page renders more than one style block, so scan for the one that has the
    layer under test rather than assuming it comes first.
    """
    layers: dict[str, dict] = {}
    marker = '"layers": '
    offset = html.find(marker)
    while offset != -1 and required_layer_id not in layers:
        try:
            layer_list, _ = json.JSONDecoder().raw_decode(html[offset + len(marker) :])
        except json.JSONDecodeError:
            layer_list = []
        if isinstance(layer_list, list):
            layers = {layer["id"]: layer for layer in layer_list if isinstance(layer, dict) and "id" in layer}
        offset = html.find(marker, offset + 1)
    assert required_layer_id in layers, f"no style block contained {required_layer_id}"
    return layers


def _assert_single_cover(points: dict, fill: dict, line: dict) -> None:
    """Every zoom must draw a lake exactly one way -- as a dot or as a polygon."""
    assert points["maxzoom"] == POINT_POLY_SWITCH_ZOOM
    assert fill["minzoom"] == POINT_POLY_SWITCH_ZOOM
    assert line["minzoom"] == POINT_POLY_SWITCH_ZOOM

    # Guards the gap directly rather than trusting the three numbers above to
    # stay in sync with each other.
    for zoom in range(TILE_MAX_ZOOM + 1):
        as_circles = zoom < points["maxzoom"]
        as_polygons = zoom >= fill["minzoom"]
        assert as_circles != as_polygons, f"z{zoom} draws lakes {'twice' if as_circles else 'not at all'}"


@pytest.mark.parametrize(
    "viz_configuration_name",
    ["colored_historical", "drainage_year", "nrt_drainage", "generic_water"],
)
def test_base_lake_centroid_polygon_handoff_has_no_gap(viz_configuration_name):
    """Every viz mode draws the base lakes at every zoom, dots below the switch, polygons above.

    The centroid layer used to exist only on the NRT monthly path, so the other
    modes had nothing to draw below the switch zoom -- invisible only for as
    long as the shipped tilesets still (wrongly) baked polygons down to z0.
    """
    from water_timeseries.map_utils import build_pmtiles_map

    m = build_pmtiles_map(
        "http://localhost:1/lakes.pmtiles",
        viz_configuration_name=viz_configuration_name,
        base_has_centroids=True,
    )
    layers = _style_layers(m.get_root().render(), "lakes-points")
    _assert_single_cover(layers["lakes-points"], layers["lakes-fill"], layers["lakes-line"])


@pytest.mark.parametrize(
    "viz_configuration_name",
    ["colored_historical", "drainage_year", "nrt_drainage", "generic_water"],
)
def test_base_polygons_cover_every_zoom_when_the_archive_has_no_centroids(viz_configuration_name):
    """An archive with no usable centroid layer must not have its polygons gated.

    The handoff above assumes centroids exist below the switch. Archives built
    before the per-feature zoom ranges rate-dropped theirs to a couple of dots
    per tile while baking polygons down to z0, so gating the polygons at the
    switch blanked those zooms outright -- which is what the shipped pan-arctic
    base archive did to historical mode.
    """
    from water_timeseries.map_utils import build_pmtiles_map

    m = build_pmtiles_map(
        "http://localhost:1/lakes.pmtiles",
        viz_configuration_name=viz_configuration_name,
        base_has_centroids=False,
        selected_id="b7uefy0bvcrc",
    )
    layers = _style_layers(m.get_root().render(), "lakes-fill")

    assert "lakes-points" not in layers, "circle layer added with no tiles behind it"
    # Every layer reading the base archive's polygons, including the selection
    # highlight, has to reach the zooms the polygons are actually baked at.
    for layer_id in ("lakes-fill", "lakes-line", "lakes-line-selected", "lakes-line-selected-casing"):
        assert "minzoom" not in layers[layer_id], f"{layer_id} still gated at the switch zoom"


@pytest.mark.parametrize(
    "viz_configuration_name",
    ["colored_historical", "drainage_year", "nrt_drainage", "generic_water"],
)
def test_centroids_are_drawn_more_opaque_than_the_polygons_they_replace(viz_configuration_name):
    """A dot painted at the polygon's fill opacity is invisible.

    Opacity that reads as a tint across a whole lake is nothing on a 2px circle:
    drainage_year paints stable lakes at 0.05, which left the zoomed-out map
    looking empty on every basemap but Dark Matter. The circles scale that up
    instead of picking their own number, so each mode keeps its own emphasis,
    and they take a dark ring so they survive a bright or busy basemap too.

    A root curve rather than a multiplier: see the comment in
    ``build_pmtiles_map``. One factor large enough to rescue 0.05 pins every
    other mode at fully opaque, which costs the nrt_drainage base lakes their job
    of staying muted under the drained overlay.
    """
    from water_timeseries.map_utils import CENTROID_OPACITY_EXPONENT, build_pmtiles_map

    assert 0 < CENTROID_OPACITY_EXPONENT < 1, "an exponent >= 1 would dim the dots, not lift them"

    m = build_pmtiles_map(
        "http://localhost:1/lakes.pmtiles",
        viz_configuration_name=viz_configuration_name,
        base_has_centroids=True,
    )
    layers = _style_layers(m.get_root().render(), "lakes-points")
    points_paint = layers["lakes-points"]["paint"]
    fill_opacity = layers["lakes-fill"]["paint"]["fill-opacity"]

    assert points_paint["circle-opacity"] == ["^", fill_opacity, CENTROID_OPACITY_EXPONENT]
    # Modes with a constant opacity can be checked outright: brighter than the
    # polygons, still in range, and never pinned to fully opaque.
    if isinstance(fill_opacity, int | float):
        assert fill_opacity < fill_opacity**CENTROID_OPACITY_EXPONENT < 1
    # The ring fades with the dot, so a deliberately muted lake does not come
    # back as a hard outline.
    assert points_paint["circle-stroke-opacity"] == points_paint["circle-opacity"]
    assert points_paint["circle-stroke-width"]


def test_drained_lakes_draw_over_stable_ones():
    """A drained lake must never be painted over by a stable neighbour.

    Both used to share one layer, coloured by a branch in the paint, so which
    one ended up on top came down to the order features happened to sit in the
    tile -- grey lakes covering the drained ones the map exists to show. They
    are separate layers now, and the whole grey layer is drawn first.
    """
    from water_timeseries.map_utils import BASE_POINT_RADIUS, build_pmtiles_map
    from water_timeseries.utils.map_styles.pmtiles import DRAINED_LAKE_FILTER, STABLE_LAKE_FILTER

    m = build_pmtiles_map(
        "http://localhost:1/lakes.pmtiles",
        viz_configuration_name="drainage_year",
        base_has_centroids=True,
    )
    layers = _style_layers(m.get_root().render(), "lakes-fill")
    order = list(layers)

    for stable, drained in (("lakes-stable-points", "lakes-points"), ("lakes-stable-fill", "lakes-fill")):
        assert order.index(stable) < order.index(drained), f"{stable} paints over {drained}"
        assert layers[stable]["filter"] == STABLE_LAKE_FILTER
        assert layers[drained]["filter"] == DRAINED_LAKE_FILTER

    # Neutral grey, and the same handoff zoom either side of the switch, so a
    # lake does not change treatment as the handoff happens. The grey dots are
    # the ground, so they stay on the smaller base ramp; the drained ones are
    # the figure and take the larger drained size (see below).
    assert layers["lakes-stable-fill"]["paint"]["fill-color"] == "#bdbdbd"
    assert layers["lakes-stable-points"]["paint"]["circle-radius"] == BASE_POINT_RADIUS
    assert layers["lakes-stable-points"]["maxzoom"] == layers["lakes-points"]["maxzoom"]
    assert layers["lakes-stable-fill"].get("minzoom") == layers["lakes-fill"].get("minzoom")


def test_drained_dots_are_the_same_size_in_historical_and_nrt():
    """A drained lake is the figure of both maps, so it is the same dot in both.

    Historical drained lakes used to inherit the base-lake ramp the NRT overlay
    is deliberately bigger than, so the same lake came out a smaller dot in
    historical than under an NRT month -- the two ramps were written out
    separately and drifted.
    """
    from water_timeseries.map_utils import BASE_POINT_RADIUS, DRAINED_POINT_RADIUS, build_pmtiles_map

    historical = build_pmtiles_map(
        "http://localhost:1/lakes.pmtiles",
        viz_configuration_name="drainage_year",
        base_has_centroids=True,
        historical_drained_tiles_url="http://localhost:1/drained.pmtiles",
    )
    nrt = build_pmtiles_map(
        "http://localhost:1/lakes.pmtiles",
        viz_configuration_name="nrt_drainage",
        base_has_centroids=True,
        nrt_monthly_tiles_url="http://localhost:1/nrt_2026-07_drainage.pmtiles",
    )

    historical_drained = _style_layers(historical.get_root().render(), "lakes-points")["lakes-points"]
    nrt_layers = _style_layers(nrt.get_root().render(), "nrt-drained-points")

    assert historical_drained["paint"]["circle-radius"] == DRAINED_POINT_RADIUS
    assert nrt_layers["nrt-drained-points"]["paint"]["circle-radius"] == DRAINED_POINT_RADIUS
    # The lakes underneath either overlay are the ground, and stay smaller.
    assert nrt_layers["lakes-points"]["paint"]["circle-radius"] == BASE_POINT_RADIUS
    assert DRAINED_POINT_RADIUS[-1] > BASE_POINT_RADIUS[-1], "drained dots must not be the smaller mark"


def test_every_centroid_dot_carries_the_same_ring():
    """One ring treatment across every dot, whichever mode drew it.

    The polygon layers outline a lake one step darker than its own fill, which
    keeps touching lakes apart at high zoom but reads as nothing on a dot a few
    pixels across. Historical mode's dots overrode that with a near-black ring;
    the NRT overlay's dots kept inheriting the polygon outline colour, so the
    same lake was ringed two different ways either side of a mode switch.
    """
    from water_timeseries.map_utils import (
        BASE_POINT_STROKE_WIDTH,
        CENTROID_RING_COLOR,
        DRAINED_POINT_STROKE_WIDTH,
        build_pmtiles_map,
    )

    historical = _style_layers(
        build_pmtiles_map(
            "http://localhost:1/lakes.pmtiles",
            viz_configuration_name="drainage_year",
            base_has_centroids=True,
            historical_drained_tiles_url="http://localhost:1/drained.pmtiles",
        )
        .get_root()
        .render(),
        "lakes-points",
    )
    nrt = _style_layers(
        build_pmtiles_map(
            "http://localhost:1/lakes.pmtiles",
            viz_configuration_name="nrt_drainage",
            base_has_centroids=True,
            nrt_monthly_tiles_url="http://localhost:1/nrt_2026-07_drainage.pmtiles",
        )
        .get_root()
        .render(),
        "nrt-drained-points",
    )

    dots = (
        historical["lakes-points"],
        historical["lakes-stable-points"],
        nrt["lakes-points"],
        nrt["nrt-drained-points"],
    )
    for dot in dots:
        paint = dot["paint"]
        assert paint["circle-stroke-color"] == CENTROID_RING_COLOR, dot["id"]
        # A muted dot must not come back as a hard outline, so the ring fades
        # with whatever opacity its own layer paints the dot at.
        assert paint["circle-stroke-opacity"] == paint["circle-opacity"], dot["id"]

    # Ring weight follows the dot size, so a drained dot is a bigger dot rather
    # than a differently-proportioned one.
    for layer_id, width in (
        ("lakes-points", DRAINED_POINT_STROKE_WIDTH),
        ("lakes-stable-points", BASE_POINT_STROKE_WIDTH),
    ):
        assert historical[layer_id]["paint"]["circle-stroke-width"] == width, layer_id
    assert nrt["nrt-drained-points"]["paint"]["circle-stroke-width"] == DRAINED_POINT_STROKE_WIDTH
    assert nrt["lakes-points"]["paint"]["circle-stroke-width"] == BASE_POINT_STROKE_WIDTH


def test_modes_without_a_stable_split_keep_one_set_of_lake_layers():
    """Only drainage_year separates stable from drained; the rest style every lake alike."""
    from water_timeseries.map_utils import build_pmtiles_map

    for viz in ("colored_historical", "generic_water", "nrt_drainage"):
        m = build_pmtiles_map(
            "http://localhost:1/lakes.pmtiles",
            viz_configuration_name=viz,
            base_has_centroids=True,
        )
        layers = _style_layers(m.get_root().render(), "lakes-fill")
        assert "lakes-stable-fill" not in layers, viz
        # No filter either -- these layers carry every lake in the archive.
        assert "filter" not in layers["lakes-fill"], viz


@pytest.mark.skipif(find_tippecanoe() is None, reason="tippecanoe not installed")
def test_historical_drained_overlay_holds_only_lakes_with_a_break(tmp_path):
    """The overlay is the ~0.2% of lakes that drained, and nothing else.

    Small enough to build with no tile budget, which is the whole point: the
    base archive's 4M lakes have to be sampled to fit one, and that sampling
    took drained lakes off the map when zoomed out along with the stable ones.
    """
    from shapely.geometry import box

    from water_timeseries.utils.pmtiles_build import build_pmtiles_historical_drained

    lakes = gpd.GeoDataFrame(
        {
            "id_geohash": ["drained1", "stable1", "drained2", "stable2"],
            "date_break": ["2019-06", None, "2022-07", None],
            "date_break_year": [2019.0, None, 2022.0, None],
            "pre_break_median": [1.0, 2.0, 3.0, 4.0],
            "post_break_median": [0.1, 2.0, 0.3, 4.0],
            "water_change_ha": [-1.0, 0.0, -2.0, 0.0],
            "water_change_perc": [-90.0, 0.0, -90.0, 0.0],
        },
        geometry=[box(x, 70.0, x + 0.01, 70.01) for x in (-150.0, -150.5, -151.0, -151.5)],
        crs="EPSG:4326",
    )
    src = tmp_path / "lakes.parquet"
    lakes.to_parquet(src)

    out = build_pmtiles_historical_drained(src, tmp_path / "drained.pmtiles", keep_geojsonl=True)
    written = [json.loads(line) for line in (tmp_path / "drained.geojsonl").read_text().splitlines()]

    assert {f["properties"]["id_geohash"] for f in written} == {"drained1", "drained2"}
    # Hover reads the polygons off the overlay now, so they carry the full set.
    assert set(written[0]["properties"]) == {c for c in lakes.columns if c != "geometry"}

    metadata = read_pmtiles_metadata(out)
    assert {layer["id"] for layer in metadata["vector_layers"]} == {"drained", "drained_points"}
    # Named to match build_pmtiles_nrt_monthly, which the dashboard layers the
    # same way -- and built with its no-limit args, so nothing is dropped.
    assert not any((s or {}).get("dropped_by_rate") for s in metadata.get("strategies") or [])


def test_drained_overlay_replaces_the_filtered_base_layers():
    """Given the overlay, the coloured layers read it instead of filtering the base."""
    from water_timeseries.map_utils import build_pmtiles_map

    url = "http://localhost:1/drained.pmtiles"
    m = build_pmtiles_map(
        "http://localhost:1/lakes.pmtiles",
        viz_configuration_name="drainage_year",
        base_has_centroids=True,
        historical_drained_tiles_url=url,
    )
    html = m.get_root().render()
    layers = _style_layers(html, "lakes-fill")

    assert f"pmtiles://{url}" in html
    for lid, source_layer in (("lakes-points", "drained_points"), ("lakes-fill", "drained")):
        assert layers[lid]["source"] == "drained_pmtiles"
        assert layers[lid]["source-layer"] == source_layer
        # Every feature in that tileset is drained, so no filter is needed.
        assert "filter" not in layers[lid]

    # The grey lakes still come from the base archive, still underneath.
    assert layers["lakes-stable-points"]["source"] == "lakes_pmtiles"
    order = list(layers)
    assert order.index("lakes-stable-points") < order.index("lakes-points")


def test_centroids_carry_only_the_properties_they_are_drawn_from():
    """Property weight on a centroid costs lakes at low zoom, so it must be earned.

    ``--drop-densest-as-needed`` discards features until a tile fits the byte
    cap, so every property baked onto a centroid is lakes taken off the zoomed
    out map. Hover is gated at the switch zoom and never reads these, so only
    the id and the column the mode colours by belong here -- dropping the other
    five roughly doubled how many lakes survive at z5-z6.
    """
    import tempfile

    from water_timeseries.utils.pmtiles_build import DRAINAGE_YEAR_POINT_PROPERTIES, parquet_to_geojsonseq

    with tempfile.TemporaryDirectory() as tmp:
        poly_path, points_path = parquet_to_geojsonseq(
            TEST_PARQUET,
            Path(tmp) / "lakes.geojsonl",
            property_columns=DEFAULT_TILE_PROPERTIES,
            point_property_columns=("id_geohash",),
        )
        poly = json.loads(poly_path.read_text(encoding="utf-8").splitlines()[0])
        point = json.loads(points_path.read_text(encoding="utf-8").splitlines()[0])

    assert set(point["properties"]) == {"id_geohash"}
    # The polygons keep the full set -- they are what hover reads.
    assert set(poly["properties"]) > set(point["properties"])
    # The id has to be there or the centroid loses its identity for promoteId.
    assert "id_geohash" in DRAINAGE_YEAR_POINT_PROPERTIES


def test_shared_base_polygons_keep_what_a_stable_lake_hovers():
    """A stable lake is in no overlay, so the base tile is all it has to hover.

    The overlays hold drained lakes only. Stripping the shared base to the id
    left 4M stable lakes hovering an empty popup -- the tooltip drops null/"NaT"/
    "nan" values, so a tile with nothing else on it renders no rows at all. The
    area/change columns are the fix: populated for every stable lake, and true of
    the lake rather than of a month, which is what lets one archive serve both
    modes.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        poly_path, points_path = parquet_to_geojsonseq(
            TEST_PARQUET,
            Path(tmp) / "lakes.geojsonl",
            property_columns=SHARED_GEOMETRY_TILE_PROPERTIES,
            point_property_columns=SHARED_GEOMETRY_POINT_PROPERTIES,
        )
        poly = json.loads(poly_path.read_text(encoding="utf-8").splitlines()[0])
        point = json.loads(points_path.read_text(encoding="utf-8").splitlines()[0])

    # What the tooltip JS would actually render: it skips null and the placeholder
    # strings a tileset bakes for missing values.
    def rendered(props):
        return {
            k: v
            for k, v in props.items()
            if v is not None and str(v).strip().lower() not in ("", "nat", "nan", "none", "null")
        }

    shown = rendered(poly["properties"])
    assert len(shown) > 1, f"a stable lake would hover only {shown}"
    assert {"Area_start_ha", "Area_end_ha", "NetChange_ha", "NetChange_perc"} <= set(shown)

    # A datetime column would stringify to "NaT" on all 4M features instead of
    # dropping, so the base deliberately carries the year, not the date.
    assert "date_break" not in SHARED_GEOMETRY_TILE_PROPERTIES
    # date_break_year is what STABLE_LAKE_FILTER tests, and it is null (so
    # dropped) for a stable lake like this one.
    assert "date_break_year" in SHARED_GEOMETRY_TILE_PROPERTIES
    assert "date_break_year" not in shown

    # Centroids stay id-only: hover is gated above the switch zoom and never
    # reads them, while property weight there costs lakes at low zoom.
    assert set(point["properties"]) == {"id_geohash"}

    # None of this may cost the zoom split the styles rely on.
    poly_zoom, points_zoom = point_poly_zoom_ranges()
    assert poly["tippecanoe"] == {"minzoom": poly_zoom[0], "maxzoom": poly_zoom[1]}
    assert point["tippecanoe"] == {"minzoom": points_zoom[0], "maxzoom": points_zoom[1]}


@pytest.mark.skipif(find_tippecanoe() is None, reason="tippecanoe not installed")
def test_shared_geometry_archive_backs_the_centroid_handoff(tmp_path):
    """It must name the layers the styles ask for and bake usable centroids.

    Both modes point ``pmtiles_file`` at this one archive, so if it named its
    layers anything else or lost its centroids to the drop rate, every mode
    would go blank below the switch zoom at once.
    """
    out = build_pmtiles_shared_geometry(TEST_PARQUET, tmp_path / "lakes.pmtiles")

    metadata = read_pmtiles_metadata(out)
    assert {layer["id"] for layer in metadata["vector_layers"]} == {"lakes", "lakes_points"}
    assert archive_bakes_low_zoom_centroids(metadata), "the dot/polygon handoff has nothing to hand off to"


def test_tile_byte_cap_is_raised_above_the_tippecanoe_default():
    """The default 500 KB cap emptied the zoomed-out map; 2 MB keeps z6 complete."""
    from water_timeseries.utils.pmtiles_build import DEFAULT_TIPPECANOE_ARGS

    caps = [f for f in DEFAULT_TIPPECANOE_ARGS if f.startswith("--maximum-tile-bytes")]
    assert len(caps) == 1, "exactly one tile byte cap, or tippecanoe takes the last one silently"
    assert int(caps[0].split("=")[1]) > 500_000


def test_archive_bakes_low_zoom_centroids_reads_the_build_strategies():
    """The predicate answers from what tippecanoe recorded, not from a flag beside the file."""
    healthy = {
        "vector_layers": [{"id": "lakes"}, {"id": "lakes_points"}],
        "strategies": [{} for _ in range(TILE_MAX_ZOOM + 1)],
    }
    assert archive_bakes_low_zoom_centroids(healthy)

    # Thinning to fit the tile size limit is the limit doing its job and still
    # leaves a stipple; thinning by the drop rate is the whole point layer gone.
    as_needed = {**healthy, "strategies": [{"dropped_as_needed": 3_500_000} for _ in range(TILE_MAX_ZOOM + 1)]}
    assert archive_bakes_low_zoom_centroids(as_needed)

    by_rate = {**healthy, "strategies": [{"dropped_by_rate": 4_026_295} for _ in range(TILE_MAX_ZOOM + 1)]}
    assert not archive_bakes_low_zoom_centroids(by_rate)

    # Rate-dropping above the switch says nothing about what is drawn below it.
    high_only = {
        **healthy,
        "strategies": [{} if z < POINT_POLY_SWITCH_ZOOM else {"dropped_by_rate": 10} for z in range(TILE_MAX_ZOOM + 1)],
    }
    assert archive_bakes_low_zoom_centroids(high_only)

    assert not archive_bakes_low_zoom_centroids({**healthy, "vector_layers": [{"id": "lakes"}]})
    assert not archive_bakes_low_zoom_centroids({**healthy, "strategies": None})
    assert not archive_bakes_low_zoom_centroids({})

    # The NRT monthly overlay names its layers differently.
    nrt = {
        "vector_layers": [{"id": "drained"}, {"id": "drained_points"}],
        "strategies": [{} for _ in range(TILE_MAX_ZOOM + 1)],
    }
    assert not archive_bakes_low_zoom_centroids(nrt)
    assert archive_bakes_low_zoom_centroids(nrt, points_layer="drained_points")


def test_shipped_style_archives_are_classified_from_their_own_metadata():
    """Read a real archive end to end: the checked-in fixtures predate the centroid bake."""
    metadata = read_pmtiles_metadata(Path(__file__).parent / "data" / "lakes_test.pmtiles")

    assert {layer["id"] for layer in metadata["vector_layers"]} >= {"lakes", "lakes_points"}
    # A points layer in the metadata is not the same as points in the tiles.
    assert not archive_bakes_low_zoom_centroids(metadata)


def test_hover_gate_starts_where_the_polygons_do():
    """Hover reads polygon properties, so it must open exactly where they are drawn.

    Its ceiling is the map's, not the tileset's: MapLibre overzooms the
    TILE_MAX_ZOOM tile instead of dropping its features, so hover stays useful
    above it.
    """
    import re

    from water_timeseries.map_utils import build_pmtiles_map

    m = build_pmtiles_map("http://localhost:1/lakes.pmtiles", viz_configuration_name="drainage_year")
    html = m.get_root().render()

    gate_min = {int(v) for v in re.findall(r"var minZoom_\w+ = (\d+);", html)}
    gate_max = {int(v) for v in re.findall(r"var maxZoom_\w+ = (\d+);", html)}

    assert gate_min == {POINT_POLY_SWITCH_ZOOM}, f"hover opens at {gate_min}, polygons at {POINT_POLY_SWITCH_ZOOM}"
    assert gate_max == {TILE_MAX_ZOOM + 1}, f"hover stops at {gate_max}, map ceiling is {TILE_MAX_ZOOM + 1}"


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


def test_nrt_drained_centroid_polygon_handoff_has_no_gap():
    """Circles must stay visible right up to the zoom the polygons take over at.

    MapLibre hides a layer at zoom >= maxzoom but shows it at zoom >= minzoom,
    so the centroid maxzoom and the polygon minzoom have to be the same number
    to cover every zoom exactly once. Splitting them (e.g. points maxzoom 8 with
    fill minzoom 9) leaves z8 with no drained lakes drawn at all.
    """
    from water_timeseries.map_utils import build_pmtiles_map

    m = build_pmtiles_map(
        "http://localhost:1/lakes.pmtiles",
        viz_configuration_name="nrt_drainage",
        nrt_monthly_tiles_url="http://localhost:1/nrt_2026-07_drainage.pmtiles",
    )
    layers = _style_layers(m.get_root().render(), "nrt-drained-points")

    points = layers["nrt-drained-points"]
    fill = layers["nrt-drained-fill"]
    line = layers["nrt-drained-line"]

    _assert_single_cover(points, fill, line)


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


def test_build_pmtiles_map_hide_stable_lakes_with_drained_month():
    """The toggle must hide every other lake while a month's drained overlay is on.

    With ``drained_ids`` set, the drainage_year branch is skipped (it requires
    ``not drained_ids``), so there is no separate stable layer to drop: the base
    layers carry every other lake in one flat "Other lakes" blue. The toggle
    leaves the month's drained lakes alone on the map. Switched off, not zeroed,
    so the hidden lakes stop answering queryRenderedFeatures.
    """
    from water_timeseries.map_utils import build_pmtiles_map

    def build(hide: bool):
        m = build_pmtiles_map(
            "http://localhost:1/lakes.pmtiles",
            viz_configuration_name="drainage_year",
            drained_ids=["c2b25p", "c2b25q"],
            hide_stable_lakes=hide,
        )
        style = next(child for child in m._children.values() if hasattr(child, "style")).style
        return m, {layer["id"]: layer for layer in style["layers"]}

    _, shown = build(False)
    hidden_map, hidden = build(True)

    for layer_id in ("lakes-fill", "lakes-line"):
        assert "layout" not in shown[layer_id], layer_id
        assert hidden[layer_id]["layout"]["visibility"] == "none", layer_id

    # The month's drained lakes stay visible either way, and stay hoverable:
    # lakes-fill is the tooltip's usual layer and is switched off here.
    for layer_id in ("lakes-fill-drained", "lakes-line-drained"):
        assert hidden[layer_id].get("layout", {}).get("visibility") != "none", layer_id
    tooltip = next(
        child
        for child in hidden_map.get_root().render().splitlines()
        if "filterLayers_" in child and "lakes-fill" in child
    )
    assert "lakes-fill-drained" in tooltip


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
        # The page gates its circle/polygon layers on the build-side constant,
        # so the server has to hand it over even when the caller's config omits it.
        assert f'"point_poly_switch_zoom": {POINT_POLY_SWITCH_ZOOM}' in html
        # ...and the page only gates on it for archives that bake centroids down
        # there, which a config that says nothing is not claiming.
        assert '"base_has_centroids": false' in html


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


def _write_run_fixture(path: Path, unscored: int = 2) -> tuple[Path, list[str]]:
    """A stand-in for a full NRT run's parquet: a row per lake, most of them scored.

    Mirrors the real shape (``data/DW_NRT/DW_NRT_<month>_run<date>/...``): the
    run covers the whole lake table but predicts only where it had enough of a
    time series, leaving the rest null -- 45.8% scored in the 2026-06 run.
    """
    gdf = gpd.read_parquet(TEST_PARQUET)
    gdf["date"] = pd.Timestamp("2026-07-01")
    gdf["water_observed_absolute"] = 100.0
    gdf["water_predicted_absolute"] = 99.0
    gdf["water_predicted_ci_absolute"] = "98.0 : 100.0"
    gdf["water_residual_absolute"] = 1.0
    gdf["drainage_confidence"] = 0
    if unscored:
        blank = gdf.index[-unscored:]
        gdf.loc[blank, [c for c in NRT_SCORED_TILE_PROPERTIES if c != "id_geohash"]] = None
        gdf.loc[blank, "water_predicted_ci_absolute"] = "nan : nan"
    gdf.to_parquet(path, index=False)
    scored_ids = gdf[gdf["date"].notna()]["id_geohash"].astype(str).tolist()
    return path, scored_ids


def test_nrt_scored_rows_keeps_only_lakes_the_run_predicted_for():
    """`date` is the marker for a scored lake; a "nan : nan" CI is not enough to count."""
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-07-01"), pd.NaT],
            "water_predicted_ci_absolute": ["98.0 : 100.0", "nan : nan"],
        }
    )
    assert nrt_scored_rows(df).tolist() == [True, False]
    # A table with no date column at all scores nothing, rather than erroring.
    assert nrt_scored_rows(pd.DataFrame({"id_geohash": ["a"]})).tolist() == [False]


def test_find_nrt_run_parquets_prefers_repartitioned_and_skips_subsets(tmp_path):
    june = tmp_path / "DW_NRT_2026-06_run2025-06-25"
    june.mkdir()
    (june / "DW_NRT_2026-06_run2025-06-25_allGeoms_v3.parquet").write_bytes(b"0")
    (june / "DW_NRT_2026-06_run2025-06-25_allGeoms_v3_repartitioned.parquet").write_bytes(b"0")
    july = tmp_path / "DW_NRT_2026-07_run2026-07-31"
    july.mkdir()
    # The small local extract must not pass as a month's full run.
    (july / "DW_NRT_2026-07_run2026-07-31_v1_subset.parquet").write_bytes(b"0")
    (tmp_path / "unrelated").mkdir()

    found = find_nrt_run_parquets(tmp_path)
    assert set(found) == {"2026-06"}
    assert found["2026-06"].name.endswith("_repartitioned.parquet")
    assert find_nrt_run_parquets(tmp_path, months=["2026-07"]) == {}
    assert find_nrt_run_parquets(tmp_path / "missing") == {}


@pytest.mark.skipif(find_tippecanoe() is None, reason="tippecanoe not installed")
def test_monthly_archive_carries_every_scored_lake_not_just_the_drained_ones(tmp_path):
    """The month's tileset speaks for every lake the run scored, not only the drained ones."""
    geohashes = gpd.read_parquet(TEST_PARQUET)["id_geohash"].astype(str).tolist()
    run_path, scored_ids = _write_run_fixture(tmp_path / "run.parquet")
    breaks = _write_breaks_fixture(tmp_path / "breaks.parquet", geohashes[:3])

    outputs = build_pmtiles_nrt_monthly(
        breaks,
        TEST_PARQUET,
        tmp_path / "tiles",
        months=["2026-07"],
        run_parquet_by_month={"2026-07": run_path},
        keep_geojsonl=True,
    )

    archive = outputs["2026-07"]
    layer_ids = {layer["id"] for layer in read_pmtiles_metadata(archive)["vector_layers"]}
    assert layer_ids == {"drained", "drained_points", NRT_SCORED_LAYER}

    # Every scored lake is in the scored layer -- 116 of them, against the 3 the
    # month called drained -- and the two the run left null are not.
    scored = [
        json.loads(line)
        for line in (tmp_path / "tiles" / "nrt_2026-07_drainage.scored.geojsonl").read_text().strip().splitlines()
    ]
    assert {f["properties"]["id_geohash"] for f in scored} == set(scored_ids)
    assert len(scored) == len(geohashes) - 2
    assert len(scored) > 3  # far more lakes than the 3 this month called drained
    props = scored[0]["properties"]
    assert props["date"] == "2026-07-01"
    assert props["water_observed_absolute"] == 100.0
    assert props["water_predicted_ci_absolute"] == "98.0 : 100.0"
    assert props["drainage_confidence"] == 0

    # Polygons only: hover is gated to the switch zoom and above, where the base
    # archive's centroids have already handed off.
    assert not (tmp_path / "tiles" / "nrt_2026-07_drainage.scored_points.geojsonl").exists()
    # The join leaves no sidecars behind.
    assert not (tmp_path / "tiles" / "nrt_2026-07_drainage.drained.pmtiles").exists()
    assert not (tmp_path / "tiles" / "nrt_2026-07_drainage.scored.pmtiles").exists()


@pytest.mark.skipif(find_tippecanoe() is None, reason="tippecanoe not installed")
def test_months_without_a_full_run_keep_the_drained_layers_alone(tmp_path):
    """Only a full run scores every lake, so other months are unchanged by this."""
    geohashes = gpd.read_parquet(TEST_PARQUET)["id_geohash"].astype(str).tolist()[:3]
    run_path, _ = _write_run_fixture(tmp_path / "run.parquet")
    breaks = _write_breaks_fixture(tmp_path / "breaks.parquet", geohashes)

    outputs = build_pmtiles_nrt_monthly(
        breaks,
        TEST_PARQUET,
        tmp_path / "tiles",
        run_parquet_by_month={"2026-07": run_path},
    )

    assert {layer["id"] for layer in read_pmtiles_metadata(outputs["2018-07"])["vector_layers"]} == {
        "drained",
        "drained_points",
    }
    assert NRT_SCORED_LAYER in {layer["id"] for layer in read_pmtiles_metadata(outputs["2026-07"])["vector_layers"]}


def test_build_pmtiles_nrt_monthly_rejects_a_missing_run_parquet(tmp_path):
    """Fail before the geometry scan rather than quietly building a scored-less month."""
    geohashes = gpd.read_parquet(TEST_PARQUET)["id_geohash"].astype(str).tolist()[:3]
    breaks = _write_breaks_fixture(tmp_path / "breaks.parquet", geohashes)

    with pytest.raises(FileNotFoundError, match="run parquet"):
        build_pmtiles_nrt_monthly(
            breaks,
            TEST_PARQUET,
            tmp_path / "tiles",
            months=["2026-07"],
            run_parquet_by_month={"2026-07": tmp_path / "nope.parquet"},
        )


def _nrt_map_html(**kwargs) -> str:
    from water_timeseries.map_utils import build_pmtiles_map

    m = build_pmtiles_map(
        "http://localhost:1/lakes.pmtiles",
        viz_configuration_name="nrt_drainage",
        base_has_centroids=True,
        nrt_monthly_tiles_url="http://localhost:1/nrt_2026-06_drainage.pmtiles",
        **kwargs,
    )
    return m.get_root().render()


def test_scored_layer_is_a_hover_target_a_user_never_sees():
    """A non-drained lake hovers the month's prediction without the map looking different."""
    layers = _style_layers(_nrt_map_html(nrt_monthly_has_scored=True), "nrt-scored-fill")
    scored = layers["nrt-scored-fill"]

    assert scored["source-layer"] == NRT_SCORED_LAYER
    # Invisible on purpose: the grey the user sees stays the base layer's single
    # fill, so scored lakes cannot end up darker than unscored ones. A
    # zero-opacity fill is still returned by queryRenderedFeatures (verified
    # against maplibre 2.2.1), which is what makes this work as a hover target.
    assert scored["paint"]["fill-opacity"] == 0
    assert scored.get("layout", {}).get("visibility") != "none"
    assert layers["lakes-fill"]["paint"]["fill-opacity"] > 0
    # Gated with the polygons: below the switch zoom the base archive's
    # centroids are what is drawn, and hover is off there anyway.
    assert scored["minzoom"] == POINT_POLY_SWITCH_ZOOM
    # Polygons only -- a centroid layer here would be ~1.8M features per month
    # that nothing ever reads.
    assert "nrt-scored-points" not in layers


def test_hover_prefers_the_month_over_the_base_tiles():
    """Ordered by how month-specific the values are: drained, then scored, then base."""
    assert '["nrt-drained-fill", "nrt-scored-fill", "lakes-fill"]' in _nrt_map_html(nrt_monthly_has_scored=True)
    # A month with no full run has no scored layer to hover, and must not have a
    # style pointing at one.
    plain = _nrt_map_html()
    assert '["nrt-drained-fill", "lakes-fill"]' in plain
    assert "nrt-scored-fill" not in plain
    assert NRT_SCORED_LAYER not in json.dumps(_style_layers(plain, "nrt-drained-fill"))


def test_hiding_stable_lakes_hides_what_they_hover():
    """A hidden lake must not keep answering hover from the scored layer."""
    for kwargs in ({"hide_stable_lakes": True}, {"hidden_categories": frozenset({"stable"})}):
        layers = _style_layers(_nrt_map_html(nrt_monthly_has_scored=True, **kwargs), "nrt-scored-fill")
        # `visibility: none`, not zero opacity: only the former is dropped from
        # queryRenderedFeatures (see test_scored_layer_is_a_hover_target...).
        assert layers["nrt-scored-fill"]["layout"]["visibility"] == "none"
        assert layers["lakes-fill"]["layout"]["visibility"] == "none"
        # The month's drained lakes are the point of hiding the stable ones.
        assert layers["nrt-drained-fill"].get("layout", {}).get("visibility") != "none"


@pytest.mark.skipif(find_tippecanoe() is None, reason="tippecanoe not installed")
def test_pmtiles_has_layer_reads_it_off_the_archive(tmp_path):
    """Which layers a month has is data, not configuration -- see pmtiles_has_layer."""
    from water_timeseries.map_utils import pmtiles_has_layer

    geohashes = gpd.read_parquet(TEST_PARQUET)["id_geohash"].astype(str).tolist()
    run_path, _ = _write_run_fixture(tmp_path / "run.parquet")
    breaks = _write_breaks_fixture(tmp_path / "breaks.parquet", geohashes[:3])
    outputs = build_pmtiles_nrt_monthly(
        breaks, TEST_PARQUET, tmp_path / "tiles", run_parquet_by_month={"2026-07": run_path}
    )

    assert pmtiles_has_layer(str(outputs["2026-07"]), NRT_SCORED_LAYER) is True
    assert pmtiles_has_layer(str(outputs["2018-07"]), NRT_SCORED_LAYER) is False
    assert pmtiles_has_layer(str(outputs["2018-07"]), "drained") is True
    # An unreadable archive answers "absent" rather than raising: the layer is a
    # hover target, so the cost of guessing wrong that way is a plainer popup.
    assert pmtiles_has_layer(str(tmp_path / "missing.pmtiles"), NRT_SCORED_LAYER) is False
