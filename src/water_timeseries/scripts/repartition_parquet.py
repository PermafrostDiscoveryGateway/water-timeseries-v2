"""One-time re-partitioning of a large vector parquet for fast per-lake reads.

The NRT lake-polygon parquet (~2.6 GB, 4M rows in 4 row groups of ~1M rows)
cannot be pruned by ``id_geohash`` filters: a single-lake read has to scan
~1M rows. Rewriting the file **sorted by ``id_geohash`` with small row
groups** lets parquet predicate pushdown (min/max row-group statistics) skip
everything but the one small row group containing the lake — this is what
the dashboard's ``load_lake_polygon_cached`` and the spatial-click fallback
rely on for speed.

Memory notes
------------
The whole table is loaded into RAM for the sort, so run this on a machine
with roughly 2-3x the file size in available memory. It is a one-time
offline migration, not part of the serving path.

Example
-------
.. code-block:: bash

    uv run water-timeseries repartition-parquet \\
        downloads/..._allGeoms_v3.parquet \\
        downloads/..._allGeoms_v3_repartitioned.parquet \\
        --row-group-size 2000

Afterwards point ``vector_file`` in ``downloads/dashboard_nrt.yaml`` at the
repartitioned file.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


def _widen_binary_columns(table: pa.Table) -> pa.Table:
    """Cast ``binary``/``string`` columns to their 64-bit ``large_*`` variants.

    ``Table.sort_by`` reorders rows with a ``take`` that concatenates each
    column into a single array. A 32-bit ``binary`` column (e.g. WKB geometry)
    whose combined size exceeds 2 GB overflows the 32-bit offset and raises
    ``ArrowInvalid: offset overflow``. The ``large_binary``/``large_string``
    types use 64-bit offsets and avoid this. Casting is chunk-wise (no
    concatenation), so it does not itself overflow. Schema metadata (including
    the GeoParquet ``geo`` key) is preserved.
    """
    fields = []
    columns = []
    changed = False
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if pa.types.is_binary(field.type):
            col = col.cast(pa.large_binary())
            field = field.with_type(pa.large_binary())
            changed = True
        elif pa.types.is_string(field.type):
            col = col.cast(pa.large_string())
            field = field.with_type(pa.large_string())
            changed = True
        fields.append(field)
        columns.append(col)
    if not changed:
        return table
    schema = pa.schema(fields, metadata=table.schema.metadata)
    return pa.Table.from_arrays(columns, schema=schema)


def repartition_parquet(
    input_file: str | Path,
    output_file: str | Path,
    sort_column: str = "id_geohash",
    row_group_size: int = 2000,
) -> Path:
    """Rewrite a parquet file sorted by *sort_column* with small row groups.

    Schema metadata (including the GeoParquet ``geo`` key) is preserved, so
    the output stays readable with ``geopandas.read_parquet``.

    Args:
        input_file: Source parquet file.
        output_file: Destination path (must differ from input_file).
        sort_column: Column to sort by; per-row-group min/max statistics on
            this column enable row-group pruning on equality/range filters.
        row_group_size: Rows per row group in the output. Small groups mean
            a per-lake read touches only a few thousand rows.

    Returns:
        The resolved output path.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    if input_file.resolve() == output_file.resolve():
        raise ValueError("output_file must differ from input_file (no in-place rewrite)")

    logger.info(f"Reading {input_file} ...")
    table = pq.read_table(input_file)
    logger.info(f"Loaded {table.num_rows} rows, {table.nbytes / 1e9:.2f} GB in memory")

    if sort_column not in table.column_names:
        raise ValueError(f"Sort column {sort_column!r} not found in {input_file} (columns: {table.column_names})")

    # Widen binary/string columns so the sort's internal concatenation does not
    # overflow 32-bit offsets on large columns (e.g. multi-GB WKB geometry).
    table = _widen_binary_columns(table)

    logger.info(f"Sorting by {sort_column!r} ...")
    table = table.sort_by([(sort_column, "ascending")])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {output_file} with row_group_size={row_group_size} ...")
    pq.write_table(table, output_file, row_group_size=row_group_size)

    meta = pq.read_metadata(output_file)
    logger.info(f"Done: {meta.num_rows} rows in {meta.num_row_groups} row groups at {output_file}")
    return output_file


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--sort-column", default="id_geohash")
    parser.add_argument("--row-group-size", type=int, default=2000)
    args = parser.parse_args()
    repartition_parquet(args.input_file, args.output_file, args.sort_column, args.row_group_size)


if __name__ == "__main__":
    main()
