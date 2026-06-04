#!/usr/bin/env python3
"""One-time preprocessing pipeline for the viewport-filtered Streamlit dashboard.

Reads lake polygons (GeoJSON Lines, GeoParquet, etc.), applies Hilbert-curve spatial
sorting, and exports optimized Parquet for DuckDB spatial queries. Optionally exports
time series from a Zarr dataset into ``lake_timeseries.parquet``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


def _xy_to_hilbert_index(xi: int, yi: int, order: int) -> int:
    """Convert grid (x, y) to Hilbert index using inverse of d2xy."""

    def rot(n: int, x_pt: int, y_pt: int, rx: int, ry: int) -> tuple[int, int]:
        if ry == 0:
            if rx == 1:
                x_pt = n - 1 - x_pt
                y_pt = n - 1 - y_pt
            x_pt, y_pt = y_pt, x_pt
        return x_pt, y_pt

    def hilbert_xy2d(n: int, x_pt: int, y_pt: int) -> int:
        d = 0
        s = n // 2
        while s > 0:
            rx = 1 if (x_pt & s) > 0 else 0
            ry = 1 if (y_pt & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            x_pt, y_pt = rot(s, x_pt, y_pt, rx, ry)
            s //= 2
        return d

    return hilbert_xy2d(1 << order, xi, yi)


def hilbert_sort_geodataframe(gdf: gpd.GeoDataFrame, order: int = 16) -> gpd.GeoDataFrame:
    """Sort geometries by Hilbert index of their centroids on a fixed Arctic grid."""
    gdf = gdf.to_crs(epsg=4326)
    projected = gdf.to_crs(epsg=3572)
    centroids = projected.geometry.centroid
    lon = centroids.x.to_numpy()
    lat = centroids.y.to_numpy()

    lon_min, lat_min, lon_max, lat_max = -180.0, 55.0, 180.0, 90.0
    grid_max = (1 << order) - 1
    xi = np.clip(
        ((lon - lon_min) / (lon_max - lon_min) * grid_max).astype(np.int64),
        0,
        grid_max,
    )
    yi = np.clip(
        ((lat - lat_min) / (lat_max - lat_min) * grid_max).astype(np.int64),
        0,
        grid_max,
    )

    keys = np.array([_xy_to_hilbert_index(int(x), int(y), order) for x, y in zip(xi, yi)])
    return gdf.iloc[np.argsort(keys)].reset_index(drop=True)


def load_lakes_vector(path: Path) -> gpd.GeoDataFrame:
    """Load lake polygons from common vector formats."""
    suffix = path.suffix.lower()
    if suffix == ".geojsonl":
        gdf = gpd.read_file(path, engine="pyogrio")
    else:
        gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def export_optimized_lakes(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    id_column: str = "id_geohash",
) -> None:
    """Hilbert-sort and write GeoParquet for DuckDB spatial slicing."""
    if id_column in gdf.columns:
        gdf = gdf.copy()
        gdf["lake_id"] = gdf[id_column].astype(str)
    elif "lake_id" not in gdf.columns:
        raise ValueError(f"Expected column '{id_column}' or 'lake_id' in lake polygons.")

    keep_cols = ["lake_id", "geometry"]
    for col in ("Area_start_ha", "Area_end_ha", "NetChange_ha", "NetChange_perc"):
        if col in gdf.columns:
            keep_cols.append(col)

    gdf = gdf[keep_cols]
    gdf = hilbert_sort_geodataframe(gdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path, index=False)
    print(f"Wrote {len(gdf):,} lakes to {output_path}")


def _export_timeseries_chunks(
    ds: xr.Dataset,
    output_path: Path,
    water_column: str = "water",
    chunks: int = 500,
) -> None:
    if water_column not in ds:
        raise ValueError(f"Column '{water_column}' not in dataset. Vars: {list(ds.data_vars)}")

    ids = ds["id_geohash"].values
    frames = []
    for i in tqdm(range(0, len(ids), chunks), desc="Exporting time series"):
        chunk_ids = ids[i : i + chunks]
        sub = ds.sel(id_geohash=chunk_ids)
        df = sub[water_column].to_dataframe(name="water").reset_index()
        df = df.rename(columns={"id_geohash": "lake_id"})
        frames.append(df[["lake_id", "date", "water"]])

    out = pd.concat(frames, ignore_index=True)
    out["lake_id"] = out["lake_id"].astype(str)
    out = out.sort_values(["lake_id", "date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"Wrote {len(out):,} rows to {output_path}")


def export_timeseries_from_zarr(
    zarr_path: Path,
    output_path: Path,
    water_column: str = "water",
    chunks: int = 500,
) -> None:
    """Export long-format time series sorted by lake_id and date."""
    ds = xr.open_zarr(zarr_path)
    _export_timeseries_chunks(ds, output_path, water_column=water_column, chunks=chunks)


def export_timeseries_from_netcdf(
    nc_path: Path,
    output_path: Path,
    water_column: str = "water",
    chunks: int = 500,
) -> None:
    """Export long-format time series from NetCDF (chunked along id_geohash)."""
    ds = xr.open_dataset(nc_path, chunks={"id_geohash": chunks})
    _export_timeseries_chunks(ds, output_path, water_column=water_column, chunks=chunks)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Preprocess lakes and time series for the viewport dashboard.")
    data_dir = repo_root / "data"
    parser.add_argument(
        "--lakes-input",
        type=Path,
        default=data_dir / "Nitze_etal_Lakes_filtered_full_set_V2d.parquet",
        help="Path to lake polygons (geojsonl, geojson, geoparquet, etc.).",
    )
    parser.add_argument(
        "--lakes-output",
        type=Path,
        default=data_dir / "optimized_lakes.parquet",
        help="Output path for Hilbert-sorted lake polygons.",
    )
    parser.add_argument(
        "--timeseries-zarr",
        type=Path,
        default=None,
        help="Optional Zarr dataset to export as lake_timeseries.parquet.",
    )
    parser.add_argument(
        "--timeseries-nc",
        type=Path,
        default=None,
        help="Optional NetCDF dataset to export as lake_timeseries.parquet.",
    )
    parser.add_argument(
        "--timeseries-output",
        type=Path,
        default=data_dir / "lake_timeseries.parquet",
        help="Output path for long-format time series.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="id_geohash",
        help="Lake identifier column in the vector input.",
    )
    parser.add_argument(
        "--water-column",
        type=str,
        default="water",
        help="Water variable name in the Zarr dataset.",
    )
    parser.add_argument(
        "--skip-lakes",
        action="store_true",
        help="Skip polygon preprocessing (only export time series).",
    )
    parser.add_argument(
        "--skip-timeseries",
        action="store_true",
        help="Skip time series export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_lakes:
        if not args.lakes_input.exists():
            raise FileNotFoundError(f"Lakes input not found: {args.lakes_input}")
        print(f"Loading lakes from {args.lakes_input} ...")
        gdf = load_lakes_vector(args.lakes_input)
        export_optimized_lakes(gdf, args.lakes_output, id_column=args.id_column)

    if not args.skip_timeseries:
        if args.timeseries_zarr is not None:
            if not args.timeseries_zarr.exists():
                raise FileNotFoundError(f"Zarr dataset not found: {args.timeseries_zarr}")
            export_timeseries_from_zarr(
                args.timeseries_zarr,
                args.timeseries_output,
                water_column=args.water_column,
            )
        elif args.timeseries_nc is not None:
            if not args.timeseries_nc.exists():
                raise FileNotFoundError(f"NetCDF dataset not found: {args.timeseries_nc}")
            export_timeseries_from_netcdf(
                args.timeseries_nc,
                args.timeseries_output,
                water_column=args.water_column,
            )
        else:
            print("No --timeseries-zarr or --timeseries-nc provided; skipping time series export.")


if __name__ == "__main__":
    main()
