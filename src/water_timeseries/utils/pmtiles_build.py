"""Build PMTiles archives from lake polygon GeoParquet files."""

from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Optional, Sequence

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


def find_tippecanoe() -> Optional[str]:
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
) -> tuple[Path, Optional[Path]]:
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
                if geometry_column in df.columns:
                    if len(df) > 0 and isinstance(df[geometry_column].iloc[0], bytes):
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

            for _, row in gdf.iterrows():
                props = {c: row[c] for c in property_columns if c in gdf.columns}
                for key, val in list(props.items()):
                    if isinstance(val, float) and pd.isna(val):
                        props[key] = None
                    elif hasattr(val, "item"):
                        props[key] = val.item()

                # Write polygon feature
                geom_poly = row.geometry.__geo_interface__
                feat_poly = {"type": "Feature", "properties": props, "geometry": geom_poly}
                fh_poly.write(json.dumps(feat_poly, separators=(",", ":")) + "\n")

                # Write point feature
                if fh_points:
                    geom_pt = row.geometry.centroid.__geo_interface__
                    feat_pt = {"type": "Feature", "properties": props, "geometry": geom_pt}
                    fh_points.write(json.dumps(feat_pt, separators=(",", ":")) + "\n")
    finally:
        fh_poly.close()
        if fh_points:
            fh_points.close()

    return output_path, points_path


def build_pmtiles(
    parquet_path: Path | str,
    output_path: Path | str,
    *,
    property_columns: Sequence[str] = DEFAULT_TILE_PROPERTIES,
    tippecanoe_args: Optional[Sequence[str]] = None,
    tippecanoe_bin: Optional[str] = None,
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

    tippecanoe_bin = tippecanoe_bin or find_tippecanoe()
    if not tippecanoe_bin:
        raise RuntimeError("tippecanoe is not installed or not on PATH. Install it with: brew install tippecanoe")

    geojsonl_path = output_path.with_suffix(".geojsonl")
    print(f"Generating GeoJSON sequences for {parquet_path}...")
    poly_path, point_path = parquet_to_geojsonseq(parquet_path, geojsonl_path, property_columns=property_columns)

    print(f"Running tippecanoe to build PMTiles at {output_path}...")
    args: list[str] = [tippecanoe_bin, "-o", str(output_path)]

    base_flags = []
    if tippecanoe_args:
        base_flags = list(tippecanoe_args)
    else:
        for f in DEFAULT_TIPPECANOE_ARGS:
            if f.startswith("--minimum-zoom") or f.startswith("--maximum-zoom") or f == "-l" or f == "lakes":
                continue
            base_flags.append(f)

    args.extend(base_flags)

    poly_layer = {"file": str(poly_path), "layer": "lakes", "minzoom": 6, "maxzoom": 14}
    args.extend(["-L", json.dumps(poly_layer)])

    if point_path:
        point_layer = {"file": str(point_path), "layer": "lakes_points", "minzoom": 0, "maxzoom": 5}
        args.extend(["-L", json.dumps(point_layer)])

    print("Executing command: " + " ".join(args))
    subprocess.run(args, check=True)

    if not keep_geojsonl:
        poly_path.unlink(missing_ok=True)
        if point_path:
            point_path.unlink(missing_ok=True)

    # cleanup and del tmp dir
    if delete_tempdir:
        shutil.rmtree(TIPPECANOE_TEMP_DIR)

    return output_path


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
