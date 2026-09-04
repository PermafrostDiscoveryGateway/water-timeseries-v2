"""Build PMTiles archives from lake polygon GeoParquet files."""

from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

# Attributes kept in vector tiles (hover, styling, selection).
DEFAULT_TILE_PROPERTIES: tuple[str, ...] = (
    "id_geohash",
    "Area_start_ha",
    "Area_end_ha",
    "NetChange_ha",
    "NetChange_perc",
)

TIPPECANOE_TEMP_DIR = Path("downloads/tippecanoe_tmp").absolute()
TIPPECANOE_TEMP_DIR.mkdir(exist_ok=True, parents=True)

# Zoom at which every lake tileset switches from centroids to polygons: below it
# a tileset carries only its point layer (`lakes_points` / `drained_points`), at
# it and above only its polygon layer (`lakes` / `drained`). One value for the
# base tiles and the NRT monthly overlay, so a lake hands off from dot to
# polygon at the same zoom whichever layer it is drawn by.
#
# Four things derive from it and have to agree: the baked per-feature zoom
# ranges (`_write_features` here), the style's circle `maxzoom` and polygon
# `minzoom`, and the hover gate (all three in `map_utils.build_pmtiles_map`).
# MapLibre hides a layer at zoom >= maxzoom but shows it at zoom >= minzoom, so
# the same number in both places covers every zoom exactly once -- and a style
# that paints circles above the zoom where point features stop being baked
# paints nothing at all.
POINT_POLY_SWITCH_ZOOM: int = 8

# Highest zoom baked into a tileset. Above it MapLibre overzooms the top tile
# rather than dropping the features, which is why the hover gate deliberately
# does not stop here (see `build_pmtiles_map`).
TILE_MAX_ZOOM: int = 14

# Zoom levels either side of the switch where BOTH layers are baked. Only the
# style decides what is actually drawn; the overlap just means the data is there
# if we move the switch later, so retuning it anywhere in
# [POINT_POLY_SWITCH_ZOOM - N, POINT_POLY_SWITCH_ZOOM + N] is a style change
# rather than a multi-hour rebuild of every archive. Cheap insurance: it adds N
# zooms of polygons below the switch and N of centroids above it, and an
# archive's size is dominated by its top two zooms (z13+z14 were 43% of the
# 2026-07 monthly file), not by these.
POINT_POLY_OVERLAP_ZOOMS: int = 1

# Tippecanoe defaults tuned for global lake polygons (millions of features).
DEFAULT_TIPPECANOE_ARGS: tuple[str, ...] = (
    "--force",
    "--drop-densest-as-needed",
    "--extend-zooms-if-still-dropping",
    "--coalesce-densest-as-needed",
    "--simplification=10",
    "--minimum-zoom=0",
    f"--maximum-zoom={TILE_MAX_ZOOM}",
    # Four times tippecanoe's 500 KB default. `--drop-densest-as-needed` throws
    # lakes away until a tile fits this, and at 500 KB it threw away most of
    # them: a z7 region holding 3,052 lakes kept 60% of them at z6 and 26% at
    # z5, so zooming out visibly emptied the map -- unlike the NRT overlay,
    # which is small enough to run with no limit at all and stays complete at
    # every zoom. At 2 MB the same region keeps 100% at z6 and 77% at z5.
    #
    # Measured, not guessed: 8 MB buys almost nothing over 2 MB (tiles stop
    # growing at ~2 MB, where a different limit binds), and the cost here is a
    # worst-case tile of 2.0 MB, a z5 p95 of 1.1 MB, and ~3% on the archive --
    # the top two zooms dominate its size and are nowhere near this cap.
    "--maximum-tile-bytes=2000000",
    f"--temporary-directory={TIPPECANOE_TEMP_DIR}",
    "-l",
    "lakes",
)

# Properties baked onto the shared base archive's polygons. Historical and NRT
# cover the same lake table with the same geometry, so a base archive per mode is
# the same 4M polygons baked twice over; what the shared one keeps is the subset
# that is true of a lake regardless of mode or month.
#
# Neither mode *styles* from these -- both paint the base lakes a flat grey
# (``get_style_pmtiles_stable_lakes``) -- but they are the last thing a stable lake
# can hover, and for many of them the only thing. Historical mode's overlay holds
# drained lakes only, so a stable lake is in no overlay at all there; NRT's monthly
# archive also carries every lake its run scored (``NRT_SCORED_TILE_PROPERTIES``),
# but a run scores only part of the table (45.8% for 2026-06, 75.2% for 2026-07), so
# the rest still land here. Stripping the base to the id alone left millions of lakes
# with an empty popup. The four area/change columns are populated for 100% of stable
# lakes and describe the lake itself, not a month, which is what makes them safe to
# share; per-month values are not, and stay in the monthly archives (see
# NRT_MONTHLY_TILE_PROPERTIES and NRT_SCORED_TILE_PROPERTIES).
#
# ``date_break_year`` rides along because ``STABLE_LAKE_FILTER`` tests it to tell
# the two layers apart, and it is null for all 4,016,467 stable lakes -- nulls are
# dropped when a tile is baked, so it costs bytes on the ~9,800 drained lakes and
# nothing anywhere else. ``date_break`` is deliberately NOT here: it is a datetime,
# so it stringifies to "NaT" rather than dropping, which is 4M features carrying a
# placeholder the tooltip then has to filter back out.
SHARED_GEOMETRY_TILE_PROPERTIES: tuple[str, ...] = (*DEFAULT_TILE_PROPERTIES, "date_break_year")

# ...and onto its centroids: the id alone. Hover is gated to the switch zoom and
# above (``build_pmtiles_map`` passes ``min_zoom=POINT_POLY_SWITCH_ZOOM``), so it
# never reads a centroid, and every property byte spent down there is a lake
# ``--drop-densest-as-needed`` takes off the zoomed-out map -- see
# DRAINAGE_YEAR_POINT_PROPERTIES below.
SHARED_GEOMETRY_POINT_PROPERTIES: tuple[str, ...] = ("id_geohash",)

# Properties worth baking onto the centroids. They are drawn below
# POINT_POLY_SWITCH_ZOOM, where hover is gated off, so the only ones that earn
# their place are the id (identity, and `promoteId` for feature state) and
# whatever the mode colours by -- and they earn it twice over, because every
# byte of property spent on a centroid is a lake `--drop-densest-as-needed`
# takes off the map at low zoom. Dropping the other five roughly doubled how
# many lakes survive there (z6 32% -> 60%, z5 9% -> 26%) before the tile cap
# above was touched at all.
DRAINAGE_YEAR_POINT_PROPERTIES: tuple[str, ...] = ("id_geohash", "date_break_year")
NRT_POINT_PROPERTIES: tuple[str, ...] = ("id_geohash", "drainage_confidence")

# Properties baked into the per-month NRT drainage tilesets. These carry the
# month's drainage signal *in the tiles*, so the dashboard styles and hovers
# straight from tile properties instead of pushing a per-lake dict into the
# browser on every rerun (see build_pmtiles_nrt_monthly).
NRT_MONTHLY_TILE_PROPERTIES: tuple[str, ...] = (
    "id_geohash",
    "analysis_month",
    "drainage_confidence",
    "water_change_ha",
    "water_change_perc",
    "pre_break_median",
    "post_break_median",
    "water_observed",
    "water_predicted",
    "water_residual",
    "water_predicted_lower_90",
    "water_predicted_upper_90",
)

# The monthly overlay holds tens of thousands of features, not millions, and
# every one of them must survive: a dropped feature is a drained lake missing
# from the map. So no density dropping/coalescing and no tile size limits,
# unlike DEFAULT_TIPPECANOE_ARGS.
NRT_MONTHLY_TIPPECANOE_ARGS: tuple[str, ...] = (
    "--force",
    "--simplification=10",
    "--no-tile-size-limit",
    "--no-feature-limit",
    "-r1",
    f"--temporary-directory={TIPPECANOE_TEMP_DIR}",
)


# The historical drained-lakes overlay, like the NRT monthly one, holds
# thousands of features rather than millions, so it can keep every one of them:
# no density dropping and no tile size limits. That is the whole point of
# splitting it out of the base archive -- see build_pmtiles_historical_drained.
HISTORICAL_DRAINED_TIPPECANOE_ARGS: tuple[str, ...] = NRT_MONTHLY_TIPPECANOE_ARGS


def point_poly_zoom_ranges(
    switch_zoom: int = POINT_POLY_SWITCH_ZOOM,
    max_zoom: int = TILE_MAX_ZOOM,
    overlap: int = POINT_POLY_OVERLAP_ZOOMS,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the ``(poly_zoom, points_zoom)`` ranges to bake for a style switching at ``switch_zoom``.

    The ranges overlap by ``overlap`` levels on each side of the switch (see
    POINT_POLY_OVERLAP_ZOOMS), so the style can move the switch within the band
    without the tiles having to be rebuilt.
    """
    poly_zoom = (max(0, switch_zoom - overlap), max_zoom)
    points_zoom = (0, min(max_zoom, switch_zoom + overlap))
    return poly_zoom, points_zoom


DEFAULT_POLY_ZOOM, DEFAULT_POINTS_ZOOM = point_poly_zoom_ranges()


def archive_bakes_low_zoom_centroids(
    metadata: dict,
    *,
    switch_zoom: int = POINT_POLY_SWITCH_ZOOM,
    points_layer: str = "lakes_points",
) -> bool:
    """Whether an archive's centroid layer actually survives below ``switch_zoom``.

    The style hands off from centroids to polygons at ``POINT_POLY_SWITCH_ZOOM``,
    which only works if the tiles hold centroids below it. Archives built before
    the per-feature zoom ranges (`_write_features`) do not: they baked every
    point at every zoom with the tileset's maxzoom of ``TILE_MAX_ZOOM``, so
    tippecanoe's default drop rate of 2.5 per level thinned them by ~2.5^14 and
    left one or two dots per tile. The shipped pan-arctic base archive is one of
    these -- 4,026,306 centroids in ``tilestats``, ``dropped_by_rate`` of
    4,026,295 at z0 -- which is why the map goes blank below the switch when the
    polygons are gated above it.

    Tippecanoe records that in the metadata it writes, so the answer is readable
    from the archive rather than configured next to it: an archive whose points
    were rate-dropped at any zoom below the switch cannot back the handoff.
    Archives built by the current builder record no ``dropped_by_rate`` at all
    (the per-feature ranges cap the point features at ``switch_zoom + overlap``,
    so the drop schedule has almost nothing to thin); their points thin only by
    ``dropped_as_needed``, which is the tile size limit doing its job and still
    leaves a dense stipple.

    Returns False when the metadata cannot answer -- no points layer, or no
    ``strategies`` (tippecanoe too old to record them). Every archive that
    predates the fix is in that camp, and drawing polygons at every zoom is what
    those archives support.
    """
    layer_ids = {layer.get("id") for layer in metadata.get("vector_layers") or []}
    if points_layer not in layer_ids:
        return False

    strategies = metadata.get("strategies")
    if not isinstance(strategies, list) or len(strategies) < switch_zoom:
        return False

    return not any((strategies[zoom] or {}).get("dropped_by_rate") for zoom in range(switch_zoom))


def find_tippecanoe() -> str | None:
    """Return path to tippecanoe executable, or None if not installed."""
    return shutil.which("tippecanoe")


def _sanitize_properties(gdf: gpd.GeoDataFrame, columns: Sequence[str]) -> gpd.GeoDataFrame:
    """Keep only tile-safe property columns with JSON-serializable values."""
    keep = [c for c in columns if c in gdf.columns]
    out = gdf[keep + [gdf.geometry.name]].copy()

    for col in keep:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype(str)
        elif out[col].dtype == object:
            out[col] = out[col].apply(
                lambda v: (
                    None
                    if v is None or (isinstance(v, float) and pd.isna(v))
                    else (v.isoformat() if hasattr(v, "isoformat") else v)
                )
            )

    for col in keep:
        if pd.api.types.is_numeric_dtype(out[col]):
            # Round numeric float/double properties to 2 decimal places
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    return out


def parquet_to_geojsonseq(
    parquet_path: Path | str,
    output_path: Path | str,
    property_columns: Sequence[str] = DEFAULT_TILE_PROPERTIES,
    geometry_column: str = "geometry",
    generate_points: bool = True,
    point_property_columns: Sequence[str] | None = None,
    row_filter: Callable[[pd.DataFrame], pd.Series[bool]] | None = None,
) -> tuple[Path, Path | None]:
    """Export a GeoParquet file to newline-delimited GeoJSON for tippecanoe.

    Reads in chunks to prevent memory issues. Can optionally generate a second
    file containing point centroids for low-zoom density visualization.

    Args:
        parquet_path: Input GeoParquet path.
        output_path: Output ``.geojsonl`` / ``.ndjson`` path.
        property_columns: Feature properties to include in tiles.
        geometry_column: Geometry column name.
        generate_points: Whether to also create a points geojsonl file.

    Returns:
        Tuple of (polygon_geojsonl_path, point_geojsonl_path or None)
    """
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points_path = None
    if generate_points:
        points_path = output_path.with_name(f"{output_path.stem}_points.geojsonl")

    pq_file = pq.ParquetFile(parquet_path)

    # Open files for writing
    fh_poly = output_path.open("w", encoding="utf-8")
    fh_points = points_path.open("w", encoding="utf-8") if generate_points else None

    try:
        for i in range(pq_file.num_row_groups):
            table = pq_file.read_row_group(i)
            # Use GeoPandas to interpret the geometry
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = table.to_pandas()
                # Filter before decoding WKB: parsing 4M geometries to keep the
                # 9,839 that have a break is most of the runtime of a subset
                # build, and none of it is needed.
                if row_filter is not None:
                    df = df[row_filter(df)]
                    if df.empty:
                        continue
                if geometry_column in df.columns and len(df) > 0 and isinstance(df[geometry_column].iloc[0], bytes):
                    df[geometry_column] = gpd.GeoSeries.from_wkb(df[geometry_column])
                gdf = gpd.GeoDataFrame(df)

            if geometry_column in gdf.columns:
                gdf = gdf.set_geometry(geometry_column)
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            else:
                gdf = gdf.to_crs(epsg=4326)

            gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
            gdf = _sanitize_properties(gdf, property_columns)
            _write_features(
                gdf,
                property_columns,
                fh_poly,
                fh_points,
                point_property_columns=point_property_columns,
            )
    finally:
        fh_poly.close()
        if fh_points:
            fh_points.close()

    return output_path, points_path


def _write_features(
    gdf: gpd.GeoDataFrame,
    property_columns: Sequence[str],
    fh_poly,
    fh_points,
    *,
    point_property_columns: Sequence[str] | None = None,
    poly_zoom: tuple[int, int] = DEFAULT_POLY_ZOOM,
    points_zoom: tuple[int, int] = DEFAULT_POINTS_ZOOM,
) -> None:
    """Append ``gdf`` to open GeoJSONL handles as polygon (and centroid) features.

    ``poly_zoom``/``points_zoom`` come from ``point_poly_zoom_ranges`` and
    overlap around the switch zoom on purpose; the style, not the data, decides
    where the handoff happens. They are stamped onto each feature as its
    ``tippecanoe.minzoom``/``maxzoom`` (the documented per-feature zoom-range
    keys, see ``man tippecanoe``). This is NOT the same as the ``minzoom``/
    ``maxzoom`` keys in an ``-L`` layer spec passed to ``_run_tippecanoe`` —
    those aren't part of tippecanoe's ``-L`` JSON schema (only ``file``,
    ``layer``, ``description``, ``format`` are) and are silently ignored, so
    setting them there does not restrict a layer's zoom range at all. Confirmed
    by decoding output tiles: with only the (ignored) ``-L`` keys set, the
    "drained" polygon layer showed up at z0 anyway. Per-feature is the fix.
    """
    point_columns = property_columns if point_property_columns is None else point_property_columns

    for _, row in gdf.iterrows():
        props = {c: row[c] for c in property_columns if c in gdf.columns}
        for key, val in list(props.items()):
            if isinstance(val, float) and pd.isna(val):
                props[key] = None
            elif hasattr(val, "item"):
                props[key] = val.item()
        point_props = props if point_property_columns is None else {c: props[c] for c in point_columns if c in props}

        # Write polygon feature
        geom_poly = row.geometry.__geo_interface__
        feat_poly = {
            "type": "Feature",
            "tippecanoe": {"minzoom": poly_zoom[0], "maxzoom": poly_zoom[1]},
            "properties": props,
            "geometry": geom_poly,
        }
        fh_poly.write(json.dumps(feat_poly, separators=(",", ":")) + "\n")

        # Write point feature
        if fh_points:
            geom_pt = row.geometry.centroid.__geo_interface__
            feat_pt = {
                "type": "Feature",
                "tippecanoe": {"minzoom": points_zoom[0], "maxzoom": points_zoom[1]},
                "properties": point_props,
                "geometry": geom_pt,
            }
            fh_points.write(json.dumps(feat_pt, separators=(",", ":")) + "\n")


def _run_tippecanoe(
    output_path: Path,
    layers: Sequence[dict],
    base_flags: Sequence[str],
    tippecanoe_bin: str | None = None,
    delete_tempdir: bool = True,
) -> Path:
    """Run tippecanoe for the given ``-L`` layer specs, writing ``output_path``."""
    tippecanoe_bin = tippecanoe_bin or find_tippecanoe()
    if not tippecanoe_bin:
        raise RuntimeError("tippecanoe is not installed or not on PATH. Install it with: brew install tippecanoe")

    # An earlier build may have removed the shared temp directory on its way
    # out; tippecanoe fails outright if --temporary-directory does not exist.
    TIPPECANOE_TEMP_DIR.mkdir(exist_ok=True, parents=True)

    print(f"Running tippecanoe to build PMTiles at {output_path}...")
    args: list[str] = [tippecanoe_bin, "-o", str(output_path), *base_flags]
    for layer in layers:
        args.extend(["-L", json.dumps(layer)])

    print("Executing command: " + " ".join(args))
    subprocess.run(args, check=True)

    if delete_tempdir:
        shutil.rmtree(TIPPECANOE_TEMP_DIR, ignore_errors=True)

    return output_path


def build_pmtiles(
    parquet_path: Path | str,
    output_path: Path | str,
    *,
    property_columns: Sequence[str] = DEFAULT_TILE_PROPERTIES,
    point_property_columns: Sequence[str] | None = None,
    layer_names: tuple[str, str] = ("lakes", "lakes_points"),
    row_filter: Callable[[pd.DataFrame], pd.Series[bool]] | None = None,
    tippecanoe_args: Sequence[str] | None = None,
    tippecanoe_bin: str | None = None,
    keep_geojsonl: bool = False,
    delete_tempdir: bool = True,
) -> Path:
    """Convert a lake GeoParquet file to a single ``.pmtiles`` archive.

    Requires `tippecanoe <https://github.com/felt/tippecanoe>`_ (v2.17+ for
    direct PMTiles output). Install via Homebrew: ``brew install tippecanoe``.

    Args:
        parquet_path: Input GeoParquet.
        output_path: Output ``.pmtiles`` path.
        property_columns: Columns embedded in tile features.
        point_property_columns: Columns embedded in the centroid features, if the
            centroids need fewer than the polygons. Defaults to the same set.
        layer_names: ``(polygon_layer, point_layer)`` names inside the archive.
        row_filter: Optional mask over each row group, to build a tileset from a
            subset of the parquet without materializing it first.
        tippecanoe_args: Extra CLI flags (merged with sensible defaults).
        tippecanoe_bin: Path to tippecanoe binary (auto-detected if None).
        keep_geojsonl: If True, keep intermediate GeoJSONL next to output.

    Returns:
        Path to the created PMTiles file.
    """
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fail before the (slow) GeoJSON export rather than after it.
    tippecanoe_bin = tippecanoe_bin or find_tippecanoe()
    if not tippecanoe_bin:
        raise RuntimeError("tippecanoe is not installed or not on PATH. Install it with: brew install tippecanoe")

    geojsonl_path = output_path.with_suffix(".geojsonl")
    print(f"Generating GeoJSON sequences for {parquet_path}...")
    poly_path, point_path = parquet_to_geojsonseq(
        parquet_path,
        geojsonl_path,
        property_columns=property_columns,
        point_property_columns=point_property_columns,
        row_filter=row_filter,
    )

    if tippecanoe_args:
        base_flags = list(tippecanoe_args)
    else:
        base_flags = [
            f
            for f in DEFAULT_TIPPECANOE_ARGS
            if not f.startswith(("--minimum-zoom", "--maximum-zoom")) and f not in ("-l", "lakes")
        ]

    # The zoom split (see point_poly_zoom_ranges) is enforced by the
    # per-feature "tippecanoe" property that parquet_to_geojsonseq ->
    # _write_features stamps
    # onto each feature, not by minzoom/maxzoom keys here — tippecanoe's -L JSON
    # doesn't have those (see _write_features).
    layers = [{"file": str(poly_path), "layer": layer_names[0]}]
    if point_path:
        layers.append({"file": str(point_path), "layer": layer_names[1]})

    _run_tippecanoe(
        output_path,
        layers,
        base_flags,
        tippecanoe_bin=tippecanoe_bin,
        delete_tempdir=delete_tempdir,
    )

    if not keep_geojsonl:
        poly_path.unlink(missing_ok=True)
        if point_path:
            point_path.unlink(missing_ok=True)

    return output_path


def nrt_monthly_tiles_filename(month: str) -> str:
    """Return the tileset filename the dashboard looks for, e.g. ``nrt_2026-07_drainage.pmtiles``."""
    return f"nrt_{month}_drainage.pmtiles"


def _collect_geometries(
    geometry_parquet: Path | str,
    wanted_ids: set[str],
    geometry_column: str = "geometry",
    id_column: str = "id_geohash",
    batch_size: int = 100_000,
) -> dict[str, bytes]:
    """Return ``{id_geohash: wkb}`` for ``wanted_ids``, streaming the source once.

    The geometry source is the full lake table (millions of rows, GBs of WKB),
    so it is read in batches of two columns and filtered as it goes; only the
    matched geometries are held in memory.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    pq_file = pq.ParquetFile(geometry_parquet)
    value_set = pa.array(sorted(wanted_ids), type=pa.string())
    found: dict[str, bytes] = {}

    for batch in pq_file.iter_batches(batch_size=batch_size, columns=[id_column, geometry_column]):
        ids = batch.column(id_column)
        mask = pc.is_in(ids, value_set=value_set)
        if not pc.any(mask).as_py():
            continue
        matched = batch.filter(mask)
        for gid, wkb in zip(
            matched.column(id_column).to_pylist(),
            matched.column(geometry_column).to_pylist(),
            strict=True,
        ):
            if gid is not None and wkb is not None:
                found[gid] = wkb

    return found


def build_pmtiles_nrt_monthly(
    breaks_parquet: Path | str,
    geometry_parquet: Path | str,
    output_dir: Path | str,
    *,
    months: Sequence[str] | None = None,
    property_columns: Sequence[str] = NRT_MONTHLY_TILE_PROPERTIES,
    tippecanoe_args: Sequence[str] | None = None,
    tippecanoe_bin: str | None = None,
    keep_geojsonl: bool = False,
    month_column: str = "analysis_month",
    id_column: str = "id_geohash",
    poly_max_zoom: int = TILE_MAX_ZOOM,
) -> dict[str, Path]:
    """Build one small drained-lakes-only PMTiles archive per NRT analysis month.

    Each archive holds only the lakes that drained in that month (~1-2% of the
    full lake table) with the month's drainage signal baked in as tile
    properties. The dashboard layers the month's archive over the static base
    tiles, so switching months is a source-URL swap: no per-lake data is
    inlined into the page and no ``setFeatureState`` push is needed.

    Args:
        breaks_parquet: Aggregated NRT breaks table (``nrt_monthly_drain_breaks.parquet``);
            supplies which lakes drained per month and their per-month values.
        geometry_parquet: Lake table carrying ``id_geohash`` + geometry (the
            ``*_with_allgeoms_*`` parquet). Read once for all months.
        output_dir: Directory to write ``nrt_<month>_drainage.pmtiles`` into.
        months: Months to build (``YYYY-MM``). Defaults to every month in the
            breaks table.
        property_columns: Columns to bake into tile properties (missing ones are skipped).
        keep_geojsonl: Keep the intermediate GeoJSONL files next to the output.
        poly_max_zoom: Highest zoom to bake polygon geometry at. Lowering this is
            by far the biggest size lever, because the top zooms dominate the
            archive: for 2026-07, z13+z14 alone were 43% of the file. Dropping
            it to 12 roughly halves the polygon layer (measured 96.6 -> 48.6 MB)
            and costs only coordinate quantization above z12, not detail —
            MapLibre overzooms the z12 tiles, and z12 already carries
            full-resolution geometry (a sample tile held 919 vertices at
            ``poly_max_zoom=12`` vs 920 at 14). Kept at 14 by default so
            rebuilds are byte-comparable with existing archives.

    Returns:
        ``{month: pmtiles_path}`` for the months actually built.
    """
    breaks_parquet = Path(breaks_parquet)
    geometry_parquet = Path(geometry_parquet)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fail before the expensive geometry scan rather than after it.
    tippecanoe_bin = tippecanoe_bin or find_tippecanoe()
    if not tippecanoe_bin:
        raise RuntimeError("tippecanoe is not installed or not on PATH. Install it with: brew install tippecanoe")

    breaks = pd.read_parquet(breaks_parquet)
    if month_column not in breaks.columns:
        raise ValueError(f"{breaks_parquet} has no '{month_column}' column")

    breaks = breaks[breaks[month_column].notna()].copy()
    breaks[month_column] = breaks[month_column].astype(str)
    if months is not None:
        breaks = breaks[breaks[month_column].isin(set(months))]
    if breaks.empty:
        raise ValueError(f"No rows in {breaks_parquet} for months={months}")

    # One lake can drain in several months; dedupe within a month so a month's
    # tileset has exactly one feature per lake.
    breaks = breaks.drop_duplicates(subset=[month_column, id_column], keep="last")

    wanted_ids = set(breaks[id_column].astype(str))
    print(f"Collecting geometries for {len(wanted_ids)} lakes from {geometry_parquet}...")
    geom_by_id = _collect_geometries(geometry_parquet, wanted_ids, id_column=id_column)
    print(f"Found geometries for {len(geom_by_id)}/{len(wanted_ids)} lakes")
    if not geom_by_id:
        raise ValueError(f"No geometries in {geometry_parquet} matched ids from {breaks_parquet}")

    poly_zoom, points_zoom = point_poly_zoom_ranges(max_zoom=poly_max_zoom)

    keep_columns = [c for c in property_columns if c in breaks.columns]
    outputs: dict[str, Path] = {}

    for month, month_rows in breaks.groupby(month_column, sort=True):
        month_rows = month_rows[month_rows[id_column].astype(str).isin(geom_by_id)]
        if month_rows.empty:
            print(f"[{month}] no lakes with geometry, skipping")
            continue

        geoms = gpd.GeoSeries.from_wkb(
            [geom_by_id[gid] for gid in month_rows[id_column].astype(str)],
            crs="EPSG:4326",
        )
        gdf = gpd.GeoDataFrame(month_rows[keep_columns].reset_index(drop=True), geometry=geoms.reset_index(drop=True))
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        gdf = _sanitize_properties(gdf, keep_columns)

        output_path = output_dir / nrt_monthly_tiles_filename(str(month))
        poly_path = output_path.with_suffix(".geojsonl")
        points_path = poly_path.with_name(f"{poly_path.stem}_points.geojsonl")
        print(f"[{month}] writing {len(gdf)} features to {poly_path.name}...")
        with poly_path.open("w", encoding="utf-8") as fh_poly, points_path.open("w", encoding="utf-8") as fh_points:
            _write_features(
                gdf,
                keep_columns,
                fh_poly,
                fh_points,
                poly_zoom=poly_zoom,
                points_zoom=points_zoom,
            )

        # Polygons from around POINT_POLY_SWITCH_ZOOM up, centroids from
        # around it down, so drained lakes stay visible when zoomed out (where
        # the polygons are sub-pixel) without needing per-lake browser markers.
        # The split is enforced by the per-feature "tippecanoe" property written
        # above, not by these dict keys (tippecanoe's -L JSON has no
        # minzoom/maxzoom of its own; see _write_features).
        layers = [
            {"file": str(poly_path), "layer": "drained"},
            {"file": str(points_path), "layer": "drained_points"},
        ]
        _run_tippecanoe(
            output_path,
            layers,
            list(tippecanoe_args) if tippecanoe_args else list(NRT_MONTHLY_TIPPECANOE_ARGS),
            tippecanoe_bin=tippecanoe_bin,
            delete_tempdir=False,
        )

        if not keep_geojsonl:
            poly_path.unlink(missing_ok=True)
            points_path.unlink(missing_ok=True)

        size_mb = output_path.stat().st_size / 1e6
        print(f"[{month}] wrote {output_path.name} ({size_mb:.1f} MB)")
        outputs[str(month)] = output_path

    return outputs


def build_pmtiles_drainage_year(
    parquet_path: Path | str,
    output_path: Path | str,
    **kwargs,
) -> Path:
    """Build PMTiles with drainage year styling properties."""
    columns = (
        "id_geohash",
        "date_break",
        "date_break_year",
        "pre_break_median",
        "post_break_median",
        "water_change_ha",
        "water_change_perc",
    )
    return build_pmtiles(
        parquet_path,
        output_path,
        property_columns=columns,
        point_property_columns=DRAINAGE_YEAR_POINT_PROPERTIES,
        **kwargs,
    )


def build_pmtiles_shared_geometry(
    parquet_path: Path | str,
    output_path: Path | str,
    **kwargs,
) -> Path:
    """Build the one base archive both viz modes render their grey lakes from.

    Historical and NRT describe the same lake table -- same ids, same geometry --
    so building a base archive per mode bakes those 4M polygons twice for no
    gain: the only difference is mode-specific properties that nothing reads.
    The base layers are flat grey in both modes, and a drained lake hovers its
    overlay (``build_pmtiles_historical_drained`` for the years,
    ``build_pmtiles_nrt_monthly`` for the months). What the base still has to
    carry is what a *stable* lake hovers, since those are in no overlay: the
    mode-agnostic area/change columns, per ``SHARED_GEOMETRY_TILE_PROPERTIES``.
    Centroids keep the id alone (``SHARED_GEOMETRY_POINT_PROPERTIES``).

    Layer names stay ``lakes``/``lakes_points``, the defaults the styles ask
    for, so the centroid/polygon handoff at ``POINT_POLY_SWITCH_ZOOM`` is
    unchanged: point both modes' ``pmtiles_file`` at the result.
    """
    return build_pmtiles(
        parquet_path,
        output_path,
        property_columns=SHARED_GEOMETRY_TILE_PROPERTIES,
        point_property_columns=SHARED_GEOMETRY_POINT_PROPERTIES,
        **kwargs,
    )


def historical_drained_tiles_path(pmtiles_path: Path | str) -> Path:
    """Return the drained-overlay path that goes with a base archive.

    ``<base>.pmtiles`` -> ``<base>_drained.pmtiles``, so the two travel together
    and the dashboard can find one from the other.
    """
    pmtiles_path = Path(pmtiles_path)
    return pmtiles_path.with_name(f"{pmtiles_path.stem}_drained{pmtiles_path.suffix}")


def build_pmtiles_historical_drained(
    parquet_path: Path | str,
    output_path: Path | str,
    break_year_column: str = "date_break_year",
    **kwargs,
) -> Path:
    """Build the drained-lakes overlay for historical mode: every lake with a break.

    The base archive has to drop most of its lakes at low zoom to keep tiles
    under the byte cap -- 4M centroids do not fit in a z4 tile at any cap -- and
    the lakes with a break go with them, even though they are the ones the map
    exists to show. There are only ~9,800 of them in the whole record, a quarter
    of a single NRT month, so they get the same treatment the NRT monthly
    overlay gets: their own small tileset with no dropping and no size limits,
    drawn over the thinned grey base. Complete at every zoom, as a result.

    Layer names match the NRT overlay (``drained`` / ``drained_points``) because
    the style treats the two the same way.
    """
    columns = (
        "id_geohash",
        "date_break",
        "date_break_year",
        "pre_break_median",
        "post_break_median",
        "water_change_ha",
        "water_change_perc",
    )
    return build_pmtiles(
        parquet_path,
        output_path,
        property_columns=columns,
        point_property_columns=DRAINAGE_YEAR_POINT_PROPERTIES,
        layer_names=("drained", "drained_points"),
        row_filter=lambda df: df[break_year_column].notna(),
        tippecanoe_args=HISTORICAL_DRAINED_TIPPECANOE_ARGS,
        **kwargs,
    )


def build_pmtiles_nrt_drainage(
    parquet_path: Path | str,
    output_path: Path | str,
    **kwargs,
) -> Path:
    """Build PMTiles with drainage year styling properties."""
    # columns = (
    #     "id_geohash",
    #     "date",
    #     "water_observed",
    #     "water_predicted",
    #     "water_residual",
    #     "drainage_confidence",
    # )

    columns_absolute = (
        "id_geohash",
        "date",
        "water_observed_absolute",
        "water_predicted_absolute",
        "water_predicted_ci_absolute",
        "water_residual_absolute",
        "drainage_confidence",
    )

    return build_pmtiles(
        parquet_path,
        output_path,
        property_columns=columns_absolute,
        point_property_columns=NRT_POINT_PROPERTIES,
        **kwargs,
    )
