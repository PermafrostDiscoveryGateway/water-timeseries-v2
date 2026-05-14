"""Pre-compute NRT monthly drained-lake results from a DW dataset file.

This script iterates over every available analysis month in a Dynamic World
dataset, runs :class:`~water_timeseries.breakpoint.NRTBreakpoint` for each
month, and writes two parquet files that the dashboard can load directly
instead of computing on the fly.

Outputs
-------
``<output_dir>/nrt_monthly_drain_counts.parquet``
    One row per month with columns ``analysis_month`` (``YYYY-MM`` string)
    and ``drained_lake_count`` (int or NaN if the month failed).

``<output_dir>/nrt_monthly_drain_breaks.parquet``
    Full per-lake NRT results for every month, with an added
    ``analysis_month`` column.  Only rows where
    ``water_residual < drain_threshold`` are retained (i.e. drained lakes).

Memory notes
------------
For large datasets (100k+ lakes) the default joblib parallelism inside
``NRTBreakpoint`` serialises the entire historical xarray dataset once per
worker, which can exhaust RAM.  Use ``lake_chunk_size`` to process lakes in
batches (each chunk owns a small xarray slice) and ``n_jobs`` to limit the
number of parallel ARIMA workers per chunk.

Example
-------
.. code-block:: bash

    # NetCDF (.ncin / .nc) — memory-safe defaults
    uv run water-timeseries nrt-precompute \\
        downloads/lakes_dw_V2d.nc \\
        --output-dir precomputed/nrt

    # Tune memory vs speed
    uv run water-timeseries nrt-precompute \\
        downloads/lakes_dw_V2d.nc \\
        --output-dir precomputed/nrt \\
        --lake-chunk-size 2000 \\
        --n-jobs 4
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from water_timeseries.breakpoint import NRTBreakpoint
from water_timeseries.dataset import DWDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_dataset(dataset_file: str | Path, id_chunk: int = 2000) -> xr.Dataset:
    """Open a dataset file with dask chunking so all operations stay lazy.

    - ``.zarr`` → :func:`xarray.open_zarr` (already dask-backed)
    - anything else (e.g. ``.ncin``, ``.nc``) → :func:`xarray.open_dataset`
      with ``chunks={"id_geohash": id_chunk}``

    Chunking along ``id_geohash`` means reductions like ``.max(dim="date")``
    only load ``id_chunk`` lakes at a time, keeping peak RAM small.
    """
    path = Path(dataset_file)
    if path.suffix.lower() == ".zarr" or path.is_dir():
        logger.info("Opening dataset as Zarr: %s", path)
        return xr.open_zarr(str(path))
    else:
        logger.info("Opening dataset as NetCDF with dask chunks (id_geohash=%d): %s", id_chunk, path)
        return xr.open_dataset(str(path), chunks={"id_geohash": id_chunk})


def _detect_water_column(raw_ds: xr.Dataset) -> str:
    """Return the water column name for a raw DW or JRC dataset."""
    if "water" in raw_ds.data_vars:
        return "water"
    if "area_water_permanent" in raw_ds.data_vars:
        return "area_water_permanent"
    raise ValueError(f"Cannot detect water column; available vars: {list(raw_ds.data_vars)}")


def _get_available_analysis_dates(raw_ds: xr.Dataset, water_col: str) -> list[pd.Timestamp]:
    """Return dates that have at least one non-NaN water observation.

    Uses dask so only one chunk of lakes is in memory at a time.
    """
    valid_counts = raw_ds[water_col].notnull().sum(dim="id_geohash").compute()
    available = valid_counts["date"].values[np.asarray(valid_counts.values) > 0]
    if len(available) == 0:
        return []
    return sorted(pd.to_datetime(available).tolist())


def _run_nrt_for_month(
    raw_ds: xr.Dataset,
    water_col: str,
    month_ts: pd.Timestamp,
    data_aggregation_period: str,
    lake_chunk_size: int,
    n_jobs: int,
) -> Optional[pd.DataFrame]:
    """Run NRT breakpoint detection for one analysis month.

    Each chunk of ``lake_chunk_size`` lakes is materialised from the dask-backed
    ``raw_ds`` with ``.compute()``, wrapped in a ``DWDataset``, and processed.
    This keeps peak RAM proportional to ``lake_chunk_size`` rather than the
    full dataset size.

    Returns a combined DataFrame for the month, or an empty DataFrame.
    """
    nrt = NRTBreakpoint()

    # Determine valid lake IDs for this month (lazy, one dask chunk at a time)
    month_water = raw_ds[water_col].sel(date=month_ts).compute()
    valid_ids = month_water.dropna(dim="id_geohash").id_geohash.values.tolist()
    del month_water  # free immediately

    if not valid_ids:
        logger.info("%s: no valid lakes", month_ts.strftime("%Y-%m"))
        return pd.DataFrame()

    all_ids = np.array(valid_ids)
    chunk_results: list[pd.DataFrame] = []

    # Patch os.cpu_count so NRTBreakpoint's internal Parallel uses our n_jobs
    original_cpu_count = os.cpu_count

    def _patched_cpu_count():
        return n_jobs

    os.cpu_count = _patched_cpu_count
    try:
        for start in range(0, len(all_ids), lake_chunk_size):
            chunk_ids = all_ids[start : start + lake_chunk_size].tolist()

            # Materialise only these 2000 lakes — peak RAM ∝ chunk_size, not total lakes
            chunk_raw = raw_ds.sel(id_geohash=chunk_ids).compute()
            chunk_dw = DWDataset(chunk_raw)

            try:
                result = nrt.calculate_break(
                    chunk_dw,
                    analysis_date=month_ts,
                    data_aggregation_period=data_aggregation_period,
                )
                if result is not None and not result.empty:
                    chunk_results.append(result)
            except Exception as exc:
                logger.warning(
                    "%s chunk [%d:%d] failed: %s",
                    month_ts.strftime("%Y-%m"),
                    start,
                    start + lake_chunk_size,
                    exc,
                )
            finally:
                del chunk_raw, chunk_dw  # release this chunk before the next
    finally:
        os.cpu_count = original_cpu_count

    if not chunk_results:
        return pd.DataFrame()
    return pd.concat(chunk_results)


# ---------------------------------------------------------------------------
# Core pre-computation
# ---------------------------------------------------------------------------

def precompute_nrt_monthly(
    dataset_file: str | Path,
    output_dir: str | Path,
    drain_threshold: float = -0.25,
    data_aggregation_period: str = "all",
    resume: bool = True,
    lake_chunk_size: int = 5000,
    n_jobs: int = 4,
    lake_ids: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-compute NRT monthly drained-lake results.

    Parameters
    ----------
    dataset_file:
        Path to the DW dataset (``.ncin`` / ``.nc`` NetCDF or ``.zarr``).
    output_dir:
        Directory where the two output parquet files are written.
    drain_threshold:
        ``water_residual`` threshold below which a lake is classified as
        drained (default ``-0.25``).
    data_aggregation_period:
        Passed directly to :meth:`NRTBreakpoint.calculate_break`
        (default ``"all"``).
    resume:
        If ``True`` (default), skip months that are already present in
        ``nrt_monthly_drain_breaks.parquet`` so the run can be resumed
        after an interruption.
    lake_chunk_size:
        Number of lakes to process per chunk within each analysis month.
        Smaller values use less RAM at the cost of slightly more overhead.
        Default is 5000.
    n_jobs:
        Number of parallel ARIMA workers per chunk.  Reduce if RAM is tight
        (each worker serialises the chunk's historical xarray slice).
        Default is 4.
    lake_ids:
        Optional list of ``id_geohash`` values to restrict processing to a
        subset of lakes (e.g. the demo visualization lakes).  When ``None``
        (default) all lakes in the dataset are processed.

    Returns
    -------
    counts_df : pd.DataFrame
        Summary table (``analysis_month``, ``drained_lake_count``).
    breaks_df : pd.DataFrame
        Full per-lake NRT breaks for every month (drained lakes only).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts_path = output_dir / "nrt_monthly_drain_counts.parquet"
    breaks_path = output_dir / "nrt_monthly_drain_breaks.parquet"

    # Open lazily with dask chunks — no full-dataset DWDataset is ever created,
    # so peak RAM stays proportional to lake_chunk_size, not total lake count.
    logger.info("Opening dataset (dask-chunked) from %s", dataset_file)
    raw_ds = _open_dataset(dataset_file, id_chunk=lake_chunk_size)

    # Optionally restrict to a specific set of lake IDs (e.g. demo lakes)
    if lake_ids is not None:
        available_in_ds = set(raw_ds.id_geohash.values.tolist())
        filtered = [lid for lid in lake_ids if lid in available_in_ds]
        missing = len(lake_ids) - len(filtered)
        if missing:
            logger.warning("%d of %d requested lake IDs not found in dataset", missing, len(lake_ids))
        logger.info("Filtering dataset to %d requested lake IDs (of %d total)", len(filtered), len(available_in_ds))
        raw_ds = raw_ds.sel(id_geohash=filtered)

    water_col = _detect_water_column(raw_ds)
    n_lakes = len(raw_ds.id_geohash.values)
    logger.info(
        "Dataset opened: %d lakes, water column = %s, chunk_size = %d, n_jobs = %d",
        n_lakes, water_col, lake_chunk_size, n_jobs,
    )

    available_dates = _get_available_analysis_dates(raw_ds, water_col)
    if not available_dates:
        raise RuntimeError("No valid analysis dates found in the dataset.")
    logger.info(
        "Found %d analysis months spanning %s – %s",
        len(available_dates),
        available_dates[0].strftime("%Y-%m"),
        available_dates[-1].strftime("%Y-%m"),
    )

    # Optionally resume: find which months have already been processed
    already_done: set[str] = set()
    existing_breaks: list[pd.DataFrame] = []
    if resume and breaks_path.exists():
        existing_df = pd.read_parquet(breaks_path)
        if "analysis_month" in existing_df.columns:
            already_done = set(existing_df["analysis_month"].unique())
            existing_breaks.append(existing_df)
            logger.info("Resuming – skipping %d already-processed months", len(already_done))
    if resume and counts_path.exists():
        existing_counts_df = pd.read_parquet(counts_path)
        for _, row in existing_counts_df.iterrows():
            already_done.add(row["analysis_month"])

    count_rows: list[dict] = []
    break_rows: list[pd.DataFrame] = []

    for month_ts in tqdm(available_dates, desc="NRT monthly pre-compute"):
        month_str = month_ts.strftime("%Y-%m")

        if month_str in already_done:
            logger.debug("Skipping %s (already done)", month_str)
            continue

        try:
            month_breaks = _run_nrt_for_month(
                raw_ds=raw_ds,
                water_col=water_col,
                month_ts=month_ts,
                data_aggregation_period=data_aggregation_period,
                lake_chunk_size=lake_chunk_size,
                n_jobs=n_jobs,
            )
        except Exception as exc:
            logger.warning("Failed for %s: %s", month_str, exc)
            count_rows.append({"analysis_month": month_str, "drained_lake_count": None})
            continue

        if month_breaks is None or month_breaks.empty:
            drained_count = 0
            drained_df = pd.DataFrame()
        else:
            drained_df = month_breaks.query("water_residual < @drain_threshold").copy()
            drained_count = len(drained_df)

        count_rows.append({"analysis_month": month_str, "drained_lake_count": drained_count})

        if not drained_df.empty:
            if drained_df.index.name == "id_geohash":
                drained_df = drained_df.reset_index(names="id_geohash")
            else:
                drained_df = drained_df.reset_index(drop=False)
            drained_df.insert(0, "analysis_month", month_str)
            break_rows.append(drained_df)

        logger.info("%s: %d drained lakes (of %d valid)", month_str, drained_count,
                    len(month_breaks) if month_breaks is not None else 0)

        # Write incrementally after each month so progress is never lost
        _flush(count_rows, break_rows, existing_breaks, counts_path, breaks_path, resume)

    # Final consolidated write
    return _flush(count_rows, break_rows, existing_breaks, counts_path, breaks_path, resume)


def _flush(
    count_rows: list[dict],
    break_rows: list[pd.DataFrame],
    existing_breaks: list[pd.DataFrame],
    counts_path: Path,
    breaks_path: Path,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge new results with previously saved data and write to parquet."""
    # Counts
    new_counts = pd.DataFrame(count_rows)
    if resume and counts_path.exists():
        existing_counts = pd.read_parquet(counts_path)
        counts_df = (
            pd.concat([existing_counts, new_counts], ignore_index=True)
            .drop_duplicates(subset=["analysis_month"], keep="last")
            .sort_values("analysis_month")
            .reset_index(drop=True)
        )
    else:
        counts_df = new_counts.sort_values("analysis_month").reset_index(drop=True) if not new_counts.empty else new_counts

    if not counts_df.empty:
        counts_df.to_parquet(counts_path, index=False)

    # Breaks
    all_break_dfs = existing_breaks + break_rows
    breaks_df = pd.concat(all_break_dfs, ignore_index=True) if all_break_dfs else pd.DataFrame()

    if not breaks_df.empty:
        breaks_df = breaks_df.sort_values(["analysis_month", "id_geohash"]).reset_index(drop=True)
        breaks_df.to_parquet(breaks_path, index=False)

    return counts_df, breaks_df
