"""Build PMTiles archives from lake polygon GeoParquet files."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import pandas as pd

from water_timeseries.utils.io import load_vector_dataset

# Attributes kept in vector tiles (hover, styling, selection).
DEFAULT_TILE_PROPERTIES: tuple[str, ...] = (
    "id_geohash",
    "Area_start_ha",
    "Area_end_ha",
    "NetChange_ha",
    "NetChange_perc",
)

# Tippecanoe defaults tuned for global lake polygons (millions of features).
DEFAULT_TIPPECANOE_ARGS: tuple[str, ...] = (
    "--force",
    "--drop-densest-as-needed",
    "--extend-zooms-if-still-dropping",
    "--coalesce-densest-as-needed",
    "--simplification=10",
    "--minimum-zoom=0",
    "--maximum-zoom=14",
    "--no-clipping",
    "--no-duplication",
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
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def parquet_to_geojsonseq(
    parquet_path: Path | str,
    output_path: Path | str,
    property_columns: Sequence[str] = DEFAULT_TILE_PROPERTIES,
    geometry_column: str = "geometry",
) -> Path:
    """Export a GeoParquet file to newline-delimited GeoJSON for tippecanoe.

    Args:
        parquet_path: Input GeoParquet path.
        output_path: Output ``.geojsonl`` / ``.ndjson`` path.
        property_columns: Feature properties to include in tiles.
        geometry_column: Geometry column name.

    Returns:
        Path to the written GeoJSON sequence file.
    """
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf = load_vector_dataset(parquet_path)
    if geometry_column in gdf.columns:
        gdf = gdf.set_geometry(geometry_column)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = _sanitize_properties(gdf, property_columns)

    with output_path.open("w", encoding="utf-8") as fh:
        for _, row in gdf.iterrows():
            geom = row.geometry.__geo_interface__
            props = {c: row[c] for c in property_columns if c in gdf.columns}
            for key, val in list(props.items()):
                if isinstance(val, float) and pd.isna(val):
                    props[key] = None
                elif hasattr(val, "item"):
                    props[key] = val.item()
            feature = {"type": "Feature", "properties": props, "geometry": geom}
            fh.write(json.dumps(feature, separators=(",", ":")))
            fh.write("\n")

    return output_path


def build_pmtiles(
    parquet_path: Path | str,
    output_path: Path | str,
    *,
    property_columns: Sequence[str] = DEFAULT_TILE_PROPERTIES,
    tippecanoe_args: Optional[Sequence[str]] = None,
    tippecanoe_bin: Optional[str] = None,
    keep_geojsonl: bool = False,
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
        raise RuntimeError(
            "tippecanoe is not installed or not on PATH. "
            "Install it with: brew install tippecanoe"
        )

    geojsonl_path = output_path.with_suffix(".geojsonl")
    parquet_to_geojsonseq(parquet_path, geojsonl_path, property_columns=property_columns)

    args: list[str] = [
        tippecanoe_bin,
        *DEFAULT_TIPPECANOE_ARGS,
        "-o",
        str(output_path),
        str(geojsonl_path),
    ]
    if tippecanoe_args:
        args = [tippecanoe_bin, *tippecanoe_args, "-o", str(output_path), str(geojsonl_path)]

    subprocess.run(args, check=True)

    if not keep_geojsonl:
        geojsonl_path.unlink(missing_ok=True)

    return output_path


def build_pmtiles_for_nrt(
    parquet_path: Path | str,
    output_path: Path | str,
    **kwargs,
) -> Path:
    """Build PMTiles with NRT drainage styling properties."""
    nrt_columns = (
        "id_geohash",
        "water_residual",
        "water_observed",
        "water_predicted",
        "water_historical_median",
        "drainage_confidence",
    )
    return build_pmtiles(
        parquet_path,
        output_path,
        property_columns=nrt_columns,
        **kwargs,
    )
