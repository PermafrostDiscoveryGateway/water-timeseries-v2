"""Merge real ``drainage_confidence`` from a full GCS NRT run into the breaks table.

A full NRT pipeline run on GCS (``DW_NRT_<month>_run<date>_allGeoms_v*.parquet``
under ``gs://pdg-storage-default/workflows_optimization/dashboard_nrt/``) already
carries a real per-lake ``drainage_confidence`` column (0-3), computed upstream
by whatever pipeline stamps the ``run<date>`` suffix. Nothing downstream reads
that file directly, though: ``build-nrt-pmtiles`` and the dashboard's runtime
fallback path both only ever read ``nrt_monthly_drain_breaks.parquet``. So a
month's confidence has to be extracted from the GCS file and merged into that
breaks table before it can reach the map — this is far cheaper than the
alternative of computing confidence from scratch with the ~30h/month ARIMA
batch (``breakpoint-analysis-nrt``).

The extraction filters on ``water_residual < drain_threshold`` (the same
threshold ``precompute_nrt_monthly`` uses to classify drained lakes), which
reproduces the breaks table's own schema.

Example
-------
.. code-block:: bash

    uv run water-timeseries merge-nrt-confidence 2026-06 \\
        "DW_NRT_2026-06_run2025-06-25_allGeoms_v*.parquet" \\
        --breaks-file data/DW_historicalbp_simple_merged_breaks_nogeoms_v4/nrt_monthly_drain_breaks.parquet
"""

from __future__ import annotations

import shutil
from pathlib import Path

import gcsfs
import pandas as pd
import pyarrow.dataset as ds
from loguru import logger

DEFAULT_GCS_PREFIX = "pdg-storage-default/workflows_optimization/dashboard_nrt"

# Matches the schema precompute_nrt_monthly writes for drained lakes.
TARGET_COLS: tuple[str, ...] = (
    "id_geohash",
    "date",
    "water_observed",
    "water_predicted",
    "water_residual",
    "water_predicted_lower_90",
    "water_predicted_upper_90",
    "water_historical_mean",
    "water_historical_median",
    "water_historical_std",
    "water_historical_min",
    "water_historical_max",
    "drainage_confidence",
)


def _extract_month_confidence(
    month: str,
    gcs_glob: str,
    gcs_prefix: str,
    drain_threshold: float,
) -> pd.DataFrame:
    """Pull drained-lake rows with confidence out of a GCS full-run parquet.

    Uses column-pruned, filtered reads via ``gcsfs`` so only the matching rows
    and needed columns are pulled over the network — no full download.
    """
    fs = gcsfs.GCSFileSystem()
    pattern = f"{gcs_prefix.rstrip('/')}/{gcs_glob}"
    logger.info(f"Reading {pattern} from GCS...")
    # pyarrow does not expand globs -- it treats the pattern as a literal object
    # name and raises a bare FileNotFoundError. Resolve it through gcsfs first,
    # which also catches a typo'd run name here rather than deep in pyarrow.
    paths = sorted(fs.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No GCS objects match {pattern}")
    logger.info(f"Matched {len(paths)} object(s): {', '.join(paths)}")
    dset = ds.dataset(paths, format="parquet", filesystem=fs)

    table = dset.to_table(
        columns=list(TARGET_COLS),
        filter=ds.field("water_residual") < drain_threshold,
    )
    df = table.to_pandas()
    df.insert(0, "analysis_month", month)
    logger.info(f"{month}: extracted {len(df)} drained lakes with confidence")
    return df


def _backfill_analysis_month(breaks: pd.DataFrame) -> pd.DataFrame:
    """Fill ``analysis_month`` on historical-schema rows from date_break_year/month.

    Older breakpoint rows predate the ``analysis_month`` column and only carry
    ``date_break_year``/``date_break_month``. Only touches rows that are
    missing ``analysis_month`` but have those columns populated.
    """
    if "date_break_year" not in breaks.columns or "date_break_month" not in breaks.columns:
        return breaks

    needs_backfill = breaks["analysis_month"].isna() & breaks["date_break_year"].notna()
    if not needs_backfill.any():
        return breaks

    breaks = breaks.copy()
    breaks.loc[needs_backfill, "analysis_month"] = (
        breaks.loc[needs_backfill, "date_break_year"].astype(int).astype(str)
        + "-"
        + breaks.loc[needs_backfill, "date_break_month"].astype(int).astype(str).str.zfill(2)
    )
    logger.info(f"Backfilled analysis_month on {needs_backfill.sum()} historical rows")
    return breaks


def merge_nrt_confidence(
    breaks_file: str | Path,
    month: str,
    gcs_glob: str,
    gcs_prefix: str = DEFAULT_GCS_PREFIX,
    drain_threshold: float = -0.25,
) -> Path:
    """Extract a month's confidence from GCS and merge it into ``breaks_file``.

    Backs up the existing ``breaks_file`` to ``<breaks_file>.bak`` before
    overwriting. Re-running for the same month first drops that month's
    existing rows, so this is safe to re-run (e.g. after a new GCS run lands).

    Args:
        breaks_file: Path to ``nrt_monthly_drain_breaks.parquet`` (the file
            the dashboard and ``build-nrt-pmtiles`` both read).
        month: Analysis month as ``YYYY-MM``, e.g. ``"2026-06"``.
        gcs_glob: Filename glob for the GCS run, e.g.
            ``"DW_NRT_2026-06_run2025-06-25_allGeoms_v*.parquet"``.
        gcs_prefix: GCS bucket/path prefix the glob is resolved against.
        drain_threshold: ``water_residual`` cutoff below which a lake counts
            as drained (must match ``precompute_nrt_monthly``'s threshold).

    Returns:
        The resolved ``breaks_file`` path.
    """
    breaks_file = Path(breaks_file)
    if not breaks_file.exists():
        raise FileNotFoundError(f"breaks_file not found: {breaks_file}")

    new_rows = _extract_month_confidence(month, gcs_glob, gcs_prefix, drain_threshold)
    if new_rows.empty:
        raise ValueError(f"No rows extracted for {month} from {gcs_prefix}/{gcs_glob} (threshold {drain_threshold})")

    breaks = pd.read_parquet(breaks_file)
    breaks = _backfill_analysis_month(breaks)

    dropped = int((breaks["analysis_month"] == month).sum())
    if dropped:
        logger.info(f"Dropping {dropped} existing rows for {month} before merge")
    breaks = breaks[breaks["analysis_month"] != month]

    merged = pd.concat([breaks, new_rows], ignore_index=True)

    backup_path = breaks_file.with_suffix(breaks_file.suffix + ".bak")
    shutil.copy(breaks_file, backup_path)
    logger.info(f"Backed up {breaks_file} -> {backup_path}")

    merged.to_parquet(breaks_file, index=False)
    logger.info(f"Wrote {len(merged)} total rows ({len(new_rows)} for {month}) to {breaks_file}")
    return breaks_file


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("month", help="Analysis month as YYYY-MM, e.g. 2026-06")
    parser.add_argument("gcs_glob", help="Filename glob for the GCS run parquet(s)")
    parser.add_argument("--breaks-file", type=Path, required=True)
    parser.add_argument("--gcs-prefix", default=DEFAULT_GCS_PREFIX)
    parser.add_argument("--drain-threshold", type=float, default=-0.25)
    args = parser.parse_args()
    merge_nrt_confidence(
        args.breaks_file,
        args.month,
        args.gcs_glob,
        gcs_prefix=args.gcs_prefix,
        drain_threshold=args.drain_threshold,
    )


if __name__ == "__main__":
    main()
