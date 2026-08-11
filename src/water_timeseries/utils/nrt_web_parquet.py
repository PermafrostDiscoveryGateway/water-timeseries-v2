"""Build the web-readable NRT drainage parquet consumed directly by the browser.

This is the parquet-side counterpart of :mod:`water_timeseries.utils.pmtiles_build`'s
per-month drainage tilesets. Instead of baking each month's drained lakes into
its own vector tileset, the whole breaks table is rewritten once into a single
compact parquet laid out so a browser can fetch *one month's rows* out of it
with HTTP range requests, decode them with hyparquet, and drive the map's
``feature-state`` paint from the result.

The layout that makes that possible:

* only the columns the map actually reads are kept, so a month's rows are
  small on the wire;
* rows are sorted by ``analysis_month`` (then ``id_geohash``, which groups
  geographically-adjacent geohashes together and compresses well);
* each month gets its own row group, so a month is a contiguous byte range
  the reader can request without touching the rest of the file;
* a sidecar JSON manifest records each month's row range and byte range, so
  the page knows what to ask for without first scanning the footer.

Snappy is used deliberately: it is the one codec hyparquet decompresses with
no extra bundle.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

#: Columns kept in the web parquet. ``id_geohash`` addresses the tile feature;
#: ``drainage_confidence`` drives the paint; the rest populate the tooltip for
#: months whose lakes are not baked into a tileset.
NRT_WEB_PARQUET_COLUMNS: tuple[str, ...] = (
    "id_geohash",
    "drainage_confidence",
    "water_change_perc",
    "water_change_ha",
    "pre_break_median",
    "post_break_median",
)

NRT_WEB_PARQUET_FILENAME = "nrt_monthly_drain_breaks_web.parquet"
NRT_WEB_MANIFEST_FILENAME = "nrt_monthly_drain_breaks_web.manifest.json"


def _month_row_groups(metadata: pq.FileMetaData, month_sizes: list[tuple[str, int]]) -> dict[str, dict]:
    """Map each month to its row range, row groups and byte range in the file."""
    manifest: dict[str, dict] = {}
    row_cursor = 0
    group_cursor = 0
    for month, n_rows in month_sizes:
        # One row group per month, but stay tolerant of pyarrow splitting a
        # very large month across several groups.
        groups: list[int] = []
        rows_seen = 0
        while rows_seen < n_rows and group_cursor < metadata.num_row_groups:
            rg = metadata.row_group(group_cursor)
            groups.append(group_cursor)
            rows_seen += rg.num_rows
            group_cursor += 1

        chunks = [metadata.row_group(g).column(c) for g in groups for c in range(metadata.row_group(g).num_columns)]
        # ``file_offset`` is left at 0 by current parquet writers, so derive the
        # chunk start from its first page instead.
        starts = [
            col.dictionary_page_offset
            if col.dictionary_page_offset and col.dictionary_page_offset > 0
            else col.data_page_offset
            for col in chunks
        ]
        byte_start = min(starts)
        byte_end = max(start + col.total_compressed_size for start, col in zip(starts, chunks))
        manifest[month] = {
            "row_start": row_cursor,
            "row_end": row_cursor + rows_seen,
            "num_rows": rows_seen,
            "row_groups": groups,
            "byte_start": int(byte_start),
            "byte_end": int(byte_end),
        }
        row_cursor += rows_seen
    return manifest


def build_nrt_web_parquet(
    breaks_parquet: Path | str,
    output_dir: Path | str,
    *,
    months: Sequence[str] | None = None,
    columns: Sequence[str] = NRT_WEB_PARQUET_COLUMNS,
    month_column: str = "analysis_month",
    id_column: str = "id_geohash",
) -> tuple[Path, Path]:
    """Rewrite *breaks_parquet* into a browser-readable parquet + manifest.

    Args:
        breaks_parquet: Aggregated NRT breaks table (``nrt_monthly_drain_breaks.parquet``).
        output_dir: Directory to write the parquet and its manifest into.
        months: Restrict to these months (``YYYY-MM``). Defaults to all of them.
        columns: Columns to keep (missing ones are skipped).

    Returns:
        ``(parquet_path, manifest_path)``.
    """
    breaks_parquet = Path(breaks_parquet)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(breaks_parquet)
    if month_column not in df.columns:
        raise ValueError(f"{breaks_parquet} has no '{month_column}' column")

    df = df[df[month_column].notna()].copy()
    df[month_column] = df[month_column].astype(str)
    if months is not None:
        df = df[df[month_column].isin(set(months))]
    if df.empty:
        raise ValueError(f"No rows in {breaks_parquet} for months={months}")

    # One feature per lake per month, matching the tileset builder.
    df = df.drop_duplicates(subset=[month_column, id_column], keep="last")

    keep = [c for c in columns if c in df.columns]
    # Sorting by id within a month groups shared geohash prefixes, which snappy
    # exploits; sorting by month is what makes a month one contiguous range.
    df = df.sort_values([month_column, id_column], kind="stable")

    # float32 halves the numeric payload; the map rounds to 2 decimals anyway.
    out = df[keep].copy()
    for col in out.columns:
        if col == id_column:
            continue
        if pd.api.types.is_float_dtype(out[col]) or pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].astype("float32")

    month_sizes = [(m, int(n)) for m, n in df.groupby(month_column, sort=True).size().items()]
    table = pa.Table.from_pandas(out, preserve_index=False)

    parquet_path = output_dir / NRT_WEB_PARQUET_FILENAME
    # The id column is ~99% of the bytes on the wire (12-char geohashes, one per
    # drained lake). Dictionary-encoding it is pure overhead when every value is
    # distinct, so it gets DELTA_LENGTH_BYTE_ARRAY instead (~17% smaller); the
    # remaining columns are a handful of distinct values each and stay
    # dictionary-encoded.
    #
    # DELTA_BYTE_ARRAY would be better still (~36%) by sharing the long common
    # prefixes of sorted geohashes, and gzip would roughly halve the file, but
    # hyparquet 1.28 supports neither without the separate hyparquet-compressors
    # bundle. Snappy is the codec it decompresses on its own.
    dictionary_columns = [c for c in out.columns if c != id_column]
    writer = pq.ParquetWriter(
        parquet_path,
        table.schema,
        compression="snappy",
        use_dictionary=dictionary_columns,
        column_encoding={id_column: "DELTA_LENGTH_BYTE_ARRAY"},
    )
    try:
        offset = 0
        for _month, n_rows in month_sizes:
            writer.write_table(table.slice(offset, n_rows), row_group_size=n_rows)
            offset += n_rows
    finally:
        writer.close()

    metadata = pq.ParquetFile(parquet_path).metadata
    months_manifest = _month_row_groups(metadata, month_sizes)

    # hyparquet's parquetMetadataAsync speculatively fetches the last 512 KB to
    # find the footer. Telling it the real footer size instead turns that into a
    # ~30 KB request, which matters when a month's data is only ~800 KB.
    file_size = parquet_path.stat().st_size
    data_end = max((info["byte_end"] for info in months_manifest.values()), default=0)
    footer_fetch_size = file_size - data_end + 1024

    manifest = {
        "parquet": parquet_path.name,
        "columns": keep,
        "id_column": id_column,
        "num_rows": int(metadata.num_rows),
        "file_size": int(file_size),
        "footer_fetch_size": int(footer_fetch_size),
        "months": months_manifest,
    }
    manifest_path = output_dir / NRT_WEB_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    size_mb = parquet_path.stat().st_size / 1e6
    print(
        f"wrote {parquet_path.name} ({size_mb:.2f} MB, {metadata.num_rows} rows, {metadata.num_row_groups} row groups)"
    )
    for month, info in manifest["months"].items():
        span_kb = (info["byte_end"] - info["byte_start"]) / 1e3
        print(f"  {month}: {info['num_rows']:>6} rows, {span_kb:>8.1f} KB")
    return parquet_path, manifest_path
