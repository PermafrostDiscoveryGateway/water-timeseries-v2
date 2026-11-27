"""Build PMTiles archives with tylertoo instead of tippecanoe.

tylertoo (https://github.com/geoparquet-io/tylertoo) reads GeoParquet directly and
tiles in Rust, so the GeoJSONL bridge that ``pmtiles_build`` writes for tippecanoe
(~15 min of a ~35 min base-archive build) disappears entirely. This module keeps the
same public builders, the same layer names and the same zoom handoff, so the
dashboard cannot tell which engine produced an archive.

How the tippecanoe concepts map here:

* one ``-L`` layer -> one input GeoParquet, tiled on its own
* per-feature ``tippecanoe.minzoom/maxzoom`` -> a **zoom range per layer**, joined
  with ``pyramid``; layers may share zooms, which is what preserves the
  ``POINT_POLY_OVERLAP_ZOOMS`` band around the point/polygon switch
* ``--drop-densest-as-needed`` -> ``tile_size_limit`` (the size valve)
* ``--no-tile-size-limit -r1`` -> ``tile_size_limit=0`` and ``drop_rate=1``
* ``tile-join`` -> ``pyramid`` over two pre-tiled archives (per-layer budgets kept)
* ``tippecanoe-decode`` -> ``tylertoo decode``

Two facts about the released 0.8.0 wheel shaped this module: it exposes the Python
API only (``overview``, ``export_pmtiles``, ``convert``, ``validate``) with no
console script, and the multi-layer join exists only as the CLI. Hence
``find_tylertoo_cli`` and the ``TYLERTOO_CLI`` escape hatch; the tiling itself never
needs the binary.

Inputs are Hilbert-sorted with ``gpio sort hilbert`` (geoparquet-io is already a
dependency) because tylertoo reads rows in order and warns on unsorted input. The
sort runs on a column-pruned extract, so it also shrinks the ~3 GB source down to
what the tiles carry.

The NRT monthly build stays on tippecanoe for now: its two layers need opposite
thinning budgets, and that join has to be validated feature by feature before the
drained layer can be trusted to keep every lake at every zoom.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely

from water_timeseries.utils.pmtiles_build import (
    DEFAULT_POINTS_ZOOM,
    DEFAULT_POLY_ZOOM,
    DEFAULT_TILE_PROPERTIES,
    DRAINAGE_YEAR_POINT_PROPERTIES,
    SHARED_GEOMETRY_POINT_PROPERTIES,
    SHARED_GEOMETRY_TILE_PROPERTIES,
)

GEOMETRY_COLUMN = "geometry"
ID_COLUMN = "id_geohash"

# Four times tippecanoe's 500 KB default, for the same measured reason the polygon
# layer was tuned to it: at 500 KB low zooms lose most of a dense region's lakes.
TILE_BYTE_CAP = 2_000_000

# The rate the dense layers were tuned with (tippecanoe's default). tylertoo's own
# default is 1.65, which keeps more features at middle zooms than the published
# archives carry.
DENSE_DROP_RATE = 2.5


def _tylertoo():
    """Import the tylertoo API or fail with the install hint."""
    try:
        import tylertoo
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "tylertoo is not installed. Install it with `pip install tylertoo` "
            "(or `uv sync --extra pmtiles`) and retry, or set WTS_TILER=tippecanoe."
        ) from exc
    return tylertoo


def find_tylertoo_cli() -> str | None:
    """Return the tylertoo binary used for the multi-layer join, or None.

    The PyPI wheel ships no console script, so on a wheel-only install this is None
    and the join needs the release binary (``TYLERTOO_CLI=/path/to/tylertoo``).
    """
    return os.environ.get("TYLERTOO_CLI") or shutil.which("tylertoo")


def find_gpio() -> str | None:
    """Return the geoparquet-io CLI used to Hilbert-sort the input, or None."""
    return os.environ.get("GPIO_CLI") or shutil.which("gpio")


def _sanitize(table: pa.Table, columns: Sequence[str]) -> pa.Table:
    """Keep tile-safe columns: datetimes as strings, numeric floats rounded to 2 dp.

    Nulls stay nulls -- both engines drop them when baking a tile, which the
    steady-state styling in ``map_styles.pmtiles`` relies on (``date_break_year``).
    """
    keep = [c for c in columns if c in table.column_names]
    out = table.select(keep + [GEOMETRY_COLUMN])
    for col in keep:
        index = out.column_names.index(col)
        field_type = out.schema.field(col).type
        if pa.types.is_timestamp(field_type) or pa.types.is_date(field_type):
            out = out.set_column(index, col, pc.cast(out.column(col), pa.string()))
        elif pa.types.is_floating(field_type):
            out = out.set_column(index, col, pc.round(pc.cast(out.column(col), pa.float64()), ndigits=2))
    return out


def prepare_parquet(
    parquet_path: Path | str,
    output_path: Path | str,
    *,
    columns: Sequence[str],
    row_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
    sort: bool = True,
) -> Path:
    """Write a column-pruned (and Hilbert-sorted) copy of ``parquet_path``.

    Replaces the GeoJSONL pass: same row-group streaming and the same ``row_filter``
    hook, but the output stays GeoParquet and keeps the source's ``geo`` metadata, so
    the tiler reads only the columns the tiles carry.

    Args:
        parquet_path: Input lake GeoParquet.
        output_path: Destination parquet.
        columns: Property columns to keep (geometry is always kept).
        row_filter: Optional mask over each row group, applied before any WKB work.
        sort: Hilbert-sort the result with ``gpio sort hilbert`` when available.

    Returns:
        Path to the prepared parquet (possibly the ``*_sorted`` sibling).
    """
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = pq.ParquetFile(parquet_path)
    geo_metadata = (source.schema_arrow.metadata or {}).get(b"geo")
    read_columns = [c for c in columns if c in source.schema_arrow.names]
    read_columns.append(GEOMETRY_COLUMN)

    writer: pq.ParquetWriter | None = None
    try:
        for index in range(source.num_row_groups):
            table = source.read_row_group(index, columns=read_columns)
            if row_filter is not None:
                mask = row_filter(table.to_pandas())
                table = table.filter(pa.array(np.asarray(mask, dtype=bool)))
                if table.num_rows == 0:
                    continue
            table = _sanitize(table, columns)
            if geo_metadata is not None:
                table = table.replace_schema_metadata({b"geo": geo_metadata})
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError(f"{parquet_path} produced no rows for columns={list(columns)}")

    if not sort:
        return output_path

    gpio = find_gpio()
    if gpio is None:
        print(
            "gpio (geoparquet-io) is not on PATH: feeding tylertoo unsorted input. "
            "It will warn; the output stays correct but coarse-zoom selection is worse."
        )
        return output_path

    sorted_path = output_path.with_name(f"{output_path.stem}_sorted{output_path.suffix}")
    subprocess.run(
        [gpio, "sort", "hilbert", str(output_path), str(sorted_path), "--row-group-size-mb", "128"],
        check=True,
    )
    return sorted_path


def centroid_parquet(
    sorted_parquet: Path | str,
    output_path: Path | str,
    *,
    id_column: str = ID_COLUMN,
    columns: Sequence[str] = (ID_COLUMN,),
) -> Path:
    """Write the low-zoom dot layer's input: the id and a centroid point per lake.

    Centroids live in their own layer partly so they can carry fewer properties:
    measured on 20 k lakes, an id-only point layer kept 26 % of the lakes at z6 where
    the six-property version kept 2 % (flags not matched, so read it as direction).
    """
    sorted_parquet = Path(sorted_parquet)
    output_path = Path(output_path)

    source = pq.ParquetFile(sorted_parquet)
    geo_metadata = (source.schema_arrow.metadata or {}).get(b"geo")
    keep = [c for c in columns if c in source.schema_arrow.names]
    if id_column not in keep:
        keep.insert(0, id_column)

    writer: pq.ParquetWriter | None = None
    try:
        for batch in source.iter_batches(batch_size=100_000, columns=[*keep, GEOMETRY_COLUMN]):
            points = shapely.centroid(shapely.from_wkb(batch.column(GEOMETRY_COLUMN).to_pylist()))
            data = {col: batch.column(col) for col in keep}
            data[GEOMETRY_COLUMN] = pa.array(shapely.to_wkb(points), type=pa.binary())
            table = pa.table(data)
            if geo_metadata is not None:
                geo = json.loads(geo_metadata)
                geo["columns"][GEOMETRY_COLUMN]["geometry_types"] = ["Point"]
                table = table.replace_schema_metadata({b"geo": json.dumps(geo).encode()})
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError(f"{sorted_parquet} produced no centroids")
    return output_path


def _layer_archive(
    source_parquet: Path | str,
    output_path: Path | str,
    *,
    layer: str,
    zoom: tuple[int, int],
    byte_cap: int,
    drop_rate: float,
    work_dir: Path,
    keep_intermediate: bool,
) -> Path:
    """Tile one input into its own PMTiles archive: overview, then export."""
    output_path = Path(output_path)
    overview_path = work_dir / f"{output_path.stem}.overview.parquet"

    # `partitioning` is legal here because no representation bands are used: the
    # point/polygon split lives in separate layers, not inside one. It writes the
    # overview about 2.8x smaller than `duplicating` (measured 10.2 vs 28.4 MB per
    # 20 k lakes) at the cost of a slower export.
    api = _tylertoo()
    api.overview(
        str(source_parquet),
        str(overview_path),
        min_zoom=zoom[0],
        max_zoom=zoom[1],
        mode="partitioning",
        drop_rate=drop_rate,
        streaming=True,
    )
    api.export_pmtiles(
        str(overview_path),
        str(output_path),
        layer_name=layer,
        tile_size_limit=byte_cap,
    )
    if not keep_intermediate:
        overview_path.unlink(missing_ok=True)
    return output_path


def _pyramid(bands: Sequence[str], output_path: Path, *, work_dir: Path) -> Path:
    """Join per-layer archives into one, keeping every layer at the zooms they share.

    This is the ``tile-join -pk`` replacement: each band was tiled with its own budget,
    so nothing is re-thinned here. The CLI is required because 0.8.0's Python API has
    no ``pyramid``.
    """
    cli = find_tylertoo_cli()
    if not cli:
        raise RuntimeError(
            "the multi-layer join needs the tylertoo binary, which the PyPI wheel does not "
            "ship at 0.8.0. Download it from the release page and set TYLERTOO_CLI, or set "
            "WTS_TILER=tippecanoe."
        )

    args: list[str] = [cli, "pyramid", "--work-dir", str(work_dir), "-f"]
    for band in bands:
        args.extend(["--band", band])
    args.append(str(output_path))
    print("Executing command: " + " ".join(args))
    subprocess.run(args, check=True)
    return output_path


def build_pmtiles(
    parquet_path: Path | str,
    output_path: Path | str,
    *,
    property_columns: Sequence[str] = DEFAULT_TILE_PROPERTIES,
    point_property_columns: Sequence[str] | None = None,
    layer_names: tuple[str, str] = ("lakes", "lakes_points"),
    row_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
    keep_every_feature: bool = False,
    byte_cap: int = TILE_BYTE_CAP,
    drop_rate: float = DENSE_DROP_RATE,
    poly_zoom: tuple[int, int] = DEFAULT_POLY_ZOOM,
    points_zoom: tuple[int, int] = DEFAULT_POINTS_ZOOM,
    id_column: str = ID_COLUMN,
    keep_intermediate: bool = False,
) -> Path:
    """Convert a lake GeoParquet file to a single ``.pmtiles`` archive with tylertoo.

    Same contract as ``pmtiles_build.build_pmtiles``: a polygon layer (from
    ``poly_zoom`` up) and a centroid layer (to ``points_zoom``), overlapping around
    ``POINT_POLY_SWITCH_ZOOM`` so the style can move the switch without a rebuild.

    Args:
        parquet_path: Input GeoParquet.
        output_path: Output PMTiles path.
        property_columns: Columns embedded in the polygon features.
        point_property_columns: Columns embedded in the centroid features; defaults
            to the polygon set. The base archive passes the id alone.
        layer_names: ``(polygon_layer, point_layer)``.
        row_filter: Mask over each source row group, so an overlay can be built from a
            subset without materializing the whole table first.
        keep_every_feature: Disable the size cap and the density budget
            (tippecanoe's ``--no-tile-size-limit --no-feature-limit -r1``).
        byte_cap: Per-tile size limit for both layers.
        drop_rate: Density-budget decay; ``2.5`` matches the published archives.
        poly_zoom: Zoom range for the polygon layer.
        points_zoom: Zoom range for the centroid layer.
        id_column: Column carrying the lake id.
        keep_intermediate: Keep the prepared parquets and overviews beside the output.

    Returns:
        Path to the created PMTiles file.
    """
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    point_columns = tuple(property_columns if point_property_columns is None else point_property_columns)
    work_dir = output_path.parent / f"{output_path.stem}.tylertoo"
    work_dir.mkdir(exist_ok=True)

    if keep_every_feature:
        byte_cap, drop_rate = 0, 1.0

    print(f"Preparing {len(property_columns)} property column(s) from {parquet_path}...")
    polys = prepare_parquet(
        parquet_path,
        work_dir / "lakes.parquet",
        columns=property_columns,
        row_filter=row_filter,
        sort=True,
    )
    points = centroid_parquet(
        polys,
        work_dir / "lakes_points.parquet",
        id_column=id_column,
        columns=point_columns,
    )

    poly_archive = _layer_archive(
        polys,
        work_dir / f"{layer_names[0]}.pmtiles",
        layer=layer_names[0],
        zoom=poly_zoom,
        byte_cap=byte_cap,
        drop_rate=drop_rate,
        work_dir=work_dir,
        keep_intermediate=keep_intermediate,
    )
    point_archive = _layer_archive(
        points,
        work_dir / f"{layer_names[1]}.pmtiles",
        layer=layer_names[1],
        zoom=points_zoom,
        byte_cap=byte_cap,
        drop_rate=drop_rate,
        work_dir=work_dir,
        keep_intermediate=keep_intermediate,
    )

    _pyramid(
        [
            f"{points_zoom[0]}-{points_zoom[1]}:{point_archive}:{layer_names[1]}",
            f"{poly_zoom[0]}-{poly_zoom[1]}:{poly_archive}:{layer_names[0]}",
        ],
        output_path,
        work_dir=work_dir,
    )

    size_mb = output_path.stat().st_size / 1e6
    print(f"Wrote {output_path} ({size_mb:.1f} MB) with tylertoo")
    if not keep_intermediate:
        # The per-layer archives and overviews are inputs to the join, not outputs;
        # leaving them beside the dashboard archive would accumulate hundreds of MB
        # per build (the tippecanoe path deletes its temp dir the same way).
        shutil.rmtree(work_dir, ignore_errors=True)
    return output_path


def build_pmtiles_shared_geometry(parquet_path: Path | str, output_path: Path | str, **kwargs) -> Path:
    """Build the shared base archive both dashboard modes render their grey lakes from."""
    return build_pmtiles(
        parquet_path,
        output_path,
        property_columns=SHARED_GEOMETRY_TILE_PROPERTIES,
        point_property_columns=SHARED_GEOMETRY_POINT_PROPERTIES,
        **kwargs,
    )


def build_pmtiles_historical_drained(
    parquet_path: Path | str,
    output_path: Path | str,
    break_year_column: str = "date_break_year",
    **kwargs,
) -> Path:
    """Build the historical drained-lakes overlay: every lake with a break, every zoom.

    The overlay exists because the base archive thins its lakes at low zoom; this one
    keeps every feature, so ``keep_every_feature`` disables both the cap and the
    density budget (the equivalent of ``NRT_MONTHLY_TIPPECANOE_ARGS``).
    """
    columns = (
        "id_geohash",
        "date_break",
        "date_break_year",
        "date_break_month",
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
        keep_every_feature=True,
        **kwargs,
    )


def build_pmtiles_nrt_monthly(*args, **kwargs):
    """Not migrated yet: the two-budget join has to be validated first."""
    raise RuntimeError(
        "the NRT monthly build still uses tippecanoe (two layers with opposite thinning "
        "budgets). Run build-nrt-pmtiles with WTS_TILER=tippecanoe."
    )


__all__ = [
    "DENSE_DROP_RATE",
    "GEOMETRY_COLUMN",
    "ID_COLUMN",
    "TILE_BYTE_CAP",
    "build_pmtiles",
    "build_pmtiles_historical_drained",
    "build_pmtiles_nrt_monthly",
    "build_pmtiles_shared_geometry",
    "centroid_parquet",
    "find_gpio",
    "find_tylertoo_cli",
    "prepare_parquet",
]
