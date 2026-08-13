"""Build PMTiles archives from lake polygon GeoParquet files."""

from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from collections.abc import Sequence
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
# Tippecanoe defaults tuned for global lake polygons (millions of features).
DEFAULT_TIPPECANOE_ARGS: tuple[str, ...] = (
    "--force",
    "--drop-densest-as-needed",
    "--extend-zooms-if-still-dropping",
    "--coalesce-densest-as-needed",
    "--simplification=10",
    "--minimum-zoom=0",
    "--maximum-zoom=14",
    f"--temporary-directory={TIPPECANOE_TEMP_DIR}",
    "-l",
    "lakes",
)

# Properties baked into the per-month NRT drainage tilesets. These carry the
# month's drainage signal *in the tiles*, so the dashboard styles and hovers
# straight from tile properties instead of pushing a per-lake dict into the
# browser on every rerun (see build_pmtiles_nrt_monthly).
NRT_MONTHLY_TILE_PROPERTIES: tuple[str, ...] = (
    "id_geohash",
    # "date",
    "analysis_month",  # check if this is necessary
    "water_observed_absolute",
    "water_predicted_absolute",
    "water_predicted_ci_absolute",
    "water_residual_absolute",
    "water_change_ha",
    "water_change_perc",
    "drainage_confidence",
    # "pre_break_median",
    # "post_break_median",
    # "water_observed",
    # "water_predicted",
    # "water_residual",
    # "water_predicted_lower_90",
    # "water_predicted_upper_90",
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
            _write_features(gdf, property_columns, fh_poly, fh_points)
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
    poly_zoom: tuple[int, int] = (6, 14),
    points_zoom: tuple[int, int] = (0, 5),
) -> None:
    """Append ``gdf`` to open GeoJSONL handles as polygon (and centroid) features.

    ``poly_zoom``/``points_zoom`` are stamped onto each feature as its
    ``tippecanoe.minzoom``/``maxzoom`` (the documented per-feature zoom-range
    keys, see ``man tippecanoe``). This is NOT the same as the ``minzoom``/
    ``maxzoom`` keys in an ``-L`` layer spec passed to ``_run_tippecanoe`` —
    those aren't part of tippecanoe's ``-L`` JSON schema (only ``file``,
    ``layer``, ``description``, ``format`` are) and are silently ignored, so
    setting them there does not restrict a layer's zoom range at all. Confirmed
    by decoding output tiles: with only the (ignored) ``-L`` keys set, the
    "drained" polygon layer showed up at z0 anyway. Per-feature is the fix.
    """
    for _, row in gdf.iterrows():
        props = {c: row[c] for c in property_columns if c in gdf.columns}
        for key, val in list(props.items()):
            if isinstance(val, float) and pd.isna(val):
                props[key] = None
            elif hasattr(val, "item"):
                props[key] = val.item()

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
                "properties": props,
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
    poly_path, point_path = parquet_to_geojsonseq(parquet_path, geojsonl_path, property_columns=property_columns)

    if tippecanoe_args:
        base_flags = list(tippecanoe_args)
    else:
        base_flags = [
            f
            for f in DEFAULT_TIPPECANOE_ARGS
            if not f.startswith(("--minimum-zoom", "--maximum-zoom")) and f not in ("-l", "lakes")
        ]

    # Zoom split (polygons z6-14, centroids z0-5) is enforced by the per-feature
    # "tippecanoe" property that parquet_to_geojsonseq -> _write_features stamps
    # onto each feature, not by minzoom/maxzoom keys here — tippecanoe's -L JSON
    # doesn't have those (see _write_features).
    layers = [{"file": str(poly_path), "layer": "lakes"}]
    if point_path:
        layers.append({"file": str(point_path), "layer": "lakes_points"})

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
    poly_min_zoom=8,
    poly_max_zoom: int = 14,
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

    def _infer_month_from_date(df):
        if "date" in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df["date"]):
                df["analysis_month"] = df["date"].dt.strftime("%Y-%m")
        else:
            raise ValueError
        return df

    breaks = pd.read_parquet(breaks_parquet)
    if month_column not in breaks.columns:
        try:
            breaks = _infer_month_from_date(breaks)
        except ValueError:
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
                poly_zoom=(poly_min_zoom, poly_max_zoom),
                points_zoom=(0, 9),
            )

        # Polygons above z6 and centroids below it, mirroring the base tileset,
        # so drained lakes stay visible when zoomed out (where the polygons are
        # sub-pixel) without needing per-lake browser markers. The zoom split
        # is enforced by the per-feature "tippecanoe" property written above,
        # not by these dict keys (tippecanoe's -L JSON has no minzoom/maxzoom
        # of its own; see _write_features).
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
        **kwargs,
    )
