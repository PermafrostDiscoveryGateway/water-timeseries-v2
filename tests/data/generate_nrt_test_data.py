"""Generate dashboard NRT fixtures from ``lakes_dw_test.zarr``.

Writes two parquet files under ``tests/data/nrt/``:

* ``nrt_monthly_drain_counts.parquet`` – per-month drained-lake counts
* ``nrt_monthly_drain_breaks.parquet`` – per-lake drained rows (water_residual < -0.25)

Run from the repo root::

    uv run python tests/data/generate_nrt_test_data.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from water_timeseries.scripts.precompute_nrt_monthly import precompute_nrt_monthly

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = REPO_ROOT / "tests" / "data" / "lakes_dw_test.zarr"
OUTPUT_DIR = REPO_ROOT / "tests" / "data" / "nrt"
COUNTS_FILE = OUTPUT_DIR / "nrt_monthly_drain_counts.parquet"
BREAKS_FILE = OUTPUT_DIR / "nrt_monthly_drain_breaks.parquet"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _analysis_months(dataset_file: Path) -> list[str]:
    """Return sorted YYYY-MM strings present in the dataset."""
    ds = xr.open_zarr(dataset_file)
    dates = pd.to_datetime(ds["date"].values)
    return sorted({d.strftime("%Y-%m") for d in dates})


def generate(
    dataset_file: Path = DATASET_FILE,
    output_dir: Path = OUTPUT_DIR,
    lake_chunk_size: int = 10,
    n_jobs: int = 2,
    drain_threshold: float = -0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run NRT for every month in the test dataset and write dashboard fixtures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    months = _analysis_months(dataset_file)
    logger.info("Processing %d analysis months from %s", len(months), dataset_file)

    month_breaks: list[pd.DataFrame] = []
    counts_rows: list[dict[str, object]] = []

    for month in months:
        month_file = output_dir / f"nrt_{month}_drain_breaks.parquet"
        breaks_df = precompute_nrt_monthly(
            dataset_file=dataset_file,
            output_file=month_file,
            analysis_date=month,
            drain_threshold=drain_threshold,
            lake_chunk_size=lake_chunk_size,
            n_jobs=n_jobs,
        )
        counts_rows.append({"analysis_month": month, "drained_lake_count": len(breaks_df)})
        if not breaks_df.empty:
            month_breaks.append(breaks_df)
        month_file.unlink(missing_ok=True)

    counts_df = pd.DataFrame(counts_rows)
    breaks_df = pd.concat(month_breaks, ignore_index=True) if month_breaks else pd.DataFrame()

    counts_df.to_parquet(COUNTS_FILE, index=False)
    breaks_df.to_parquet(BREAKS_FILE, index=False)

    logger.info(
        "Wrote %s (%d months) and %s (%d drained rows)",
        COUNTS_FILE.relative_to(REPO_ROOT),
        len(counts_df),
        BREAKS_FILE.relative_to(REPO_ROOT),
        len(breaks_df),
    )
    return counts_df, breaks_df


if __name__ == "__main__":
    generate()
