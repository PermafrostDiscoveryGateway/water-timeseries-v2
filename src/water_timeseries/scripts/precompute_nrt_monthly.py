"""Pre-compute NRT results for a single analysis month from a DW dataset file.

This script runs :class:`~water_timeseries.breakpoint.NRTBreakpoint` for a
specific analysis month and writes the drained-lake results to a single parquet
file that the dashboard can load directly instead of computing on the fly.

Output
------
``<output_file>``
    Per-lake NRT results for the requested month.  Only rows where
    ``water_residual < drain_threshold`` are retained (i.e. drained lakes).
    An ``analysis_month`` column (``YYYY-MM`` string) is included.

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

    # Process a single month
    uv run water-timeseries breakpoint-analysis-nrt \\
        downloads/lakes_dw_V2d.nc \\
        --analysis-date 2024-01 \\
        --output-file precomputed/nrt/nrt_2024-01_drain_breaks.parquet

    # Tune memory vs speed
    uv run water-timeseries breakpoint-analysis-nrt \\
        downloads/lakes_dw_V2d.nc \\
        --analysis-date 2024-01 \\
        --output-file precomputed/nrt/nrt_2024-01_drain_breaks.parquet \\
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

            # Materialise only these lakes — peak RAM ∝ chunk_size, not total lakes
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
    output_file: str | Path,
    analysis_date: str,
    drain_threshold: float = -0.25,
    data_aggregation_period: str = "all",
    lake_chunk_size: int = 5000,
    n_jobs: int = 4,
    lake_ids: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Pre-compute NRT drained-lake results for a single analysis month.

    Parameters
    ----------
    dataset_file:
        Path to the DW dataset (``.ncin`` / ``.nc`` NetCDF or ``.zarr``).
    output_file:
        Destination parquet file for the per-lake NRT break results.  The
        parent directory is created automatically if it does not exist.
    analysis_date:
        Month to analyse, as a ``YYYY-MM`` string (e.g. ``"2024-01"``).
        Must correspond to a date present in *dataset_file*.
    drain_threshold:
        ``water_residual`` threshold below which a lake is classified as
        drained (default ``-0.25``).
    data_aggregation_period:
        Passed directly to :meth:`NRTBreakpoint.calculate_break`
        (default ``"all"``).
    lake_chunk_size:
        Number of lakes to process per chunk.  Smaller values use less RAM
        at the cost of slightly more overhead.  Default is 5000.
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
    breaks_df : pd.DataFrame
        Per-lake NRT break results for the requested month (drained lakes
        only), with an ``analysis_month`` column.  Written to *output_file*.
    """
    try:
        month_ts = pd.Timestamp(analysis_date)
    except Exception as exc:
        raise ValueError(f"Invalid analysis_date {analysis_date!r}. Expected 'YYYY-MM' format.") from exc

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Opening dataset (dask-chunked) from %s", dataset_file)
    raw_ds = _open_dataset(dataset_file, id_chunk=lake_chunk_size)

    if lake_ids is not None:
        available_in_ds = set(raw_ds.id_geohash.values.tolist())
        filtered = [lid for lid in lake_ids if lid in available_in_ds]
        missing = len(lake_ids) - len(filtered)
        if missing:
            logger.warning("%d of %d requested lake IDs not found in dataset", missing, len(lake_ids))
        logger.info(
            "Filtering dataset to %d requested lake IDs (of %d total)",
            len(filtered),
            len(available_in_ds),
        )
        raw_ds = raw_ds.sel(id_geohash=filtered)

    water_col = _detect_water_column(raw_ds)
    n_lakes = len(raw_ds.id_geohash.values)
    logger.info(
        "Dataset opened: %d lakes, water column = %s, chunk_size = %d, n_jobs = %d",
        n_lakes,
        water_col,
        lake_chunk_size,
        n_jobs,
    )

    # Validate that the requested date exists in the dataset
    available_dates_vals = raw_ds["date"].values
    available_dates = sorted(pd.to_datetime(available_dates_vals).tolist())
    if not available_dates:
        raise RuntimeError("No dates found in the dataset.")

    month_str = month_ts.strftime("%Y-%m")
    matching = [d for d in available_dates if d.strftime("%Y-%m") == month_str]
    if not matching:
        available_strs = sorted({d.strftime("%Y-%m") for d in available_dates})
        raise ValueError(
            f"analysis_date {month_str!r} not found in dataset. "
            f"Available months: {available_strs[0]} – {available_strs[-1]}"
        )
    month_ts = matching[0]

    logger.info("Running NRT analysis for %s", month_str)
    month_breaks = _run_nrt_for_month(
        raw_ds=raw_ds,
        water_col=water_col,
        month_ts=month_ts,
        data_aggregation_period=data_aggregation_period,
        lake_chunk_size=lake_chunk_size,
        n_jobs=n_jobs,
    )

    if month_breaks is None or month_breaks.empty:
        drained_df = pd.DataFrame()
        drained_count = 0
    else:
        drained_df = month_breaks.query("water_residual < @drain_threshold").copy()
        drained_count = len(drained_df)

    if not drained_df.empty:
        if drained_df.index.name == "id_geohash":
            drained_df = drained_df.reset_index(names="id_geohash")
        else:
            drained_df = drained_df.reset_index(drop=False)
        drained_df.insert(0, "analysis_month", month_str)
        drained_df = drained_df.sort_values("id_geohash").reset_index(drop=True)
        drained_df.to_parquet(output_file, index=False)
        logger.info(
            "%s: %d drained lakes written to %s",
            month_str,
            drained_count,
            output_file,
        )
    else:
        logger.info("%s: no drained lakes found (threshold %.3f)", month_str, drain_threshold)
        drained_df.to_parquet(output_file, index=False)

    logger.info(
        "%s: %d drained lakes (of %d valid)",
        month_str,
        drained_count,
        len(month_breaks) if month_breaks is not None else 0,
    )
    return drained_df
