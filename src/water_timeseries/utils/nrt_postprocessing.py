import geopandas as gpd
import pandas as pd
import xarray as xr

# Columns the three drainage criteria in ``NRTBreakpoint._add_confidence_level``
# are computed from. A row missing any of them cannot be scored at all: every
# comparison against NaN is False, so an unevaluable lake would otherwise score
# the same as a lake that was checked and found perfectly normal.
CONFIDENCE_INPUT_COLUMNS: tuple[str, ...] = (
    "water_residual",
    "water_observed",
    "water_predicted_lower_90",
    "water_historical_min",
)

# ``drainage_confidence`` for a lake whose criteria could not be evaluated --
# no observation for the analysis month, an ARIMA fit that was skipped for too
# short a history, or missing historical stats. Distinct from <NA>, which means
# the lake *was* evaluated and is not draining.
CONFIDENCE_INVALID: int = -1

# ``water_residual`` below which a lake counts as drained. The same number as
# criterion 1 of the confidence score, and the threshold ``precompute_nrt_monthly``
# and ``merge_nrt_confidence`` have always classified drained lakes with.
DRAIN_THRESHOLD: float = -0.25


def drained_mask(df: pd.DataFrame, drain_threshold: float = DRAIN_THRESHOLD) -> pd.Series:
    """Mask of the rows in an NRT breaks table that are actual drainage detections.

    The breaks table is not self-evidently a table of drained lakes. Since the
    ``drain_threshold`` default became ``None`` (so that no data is thrown away
    at the source), ``precompute_nrt_monthly`` writes a row for *every* lake its
    run scored, and everything downstream -- the per-month tileset, the "N
    drained" count, the map overlay -- treats "has a row this month" as "drained
    this month". This is the filter that makes that true again.

    ``nrt_monthly_drain_breaks.parquet`` mixes rows of three shapes, so the rule
    is to drop only what can be positively identified as *not* a detection:

    * **NRT-scored** (the 2026 months): ``drainage_confidence`` 1-3 is a real
      detection and is kept; ``<NA>`` means evaluated and not draining, and
      ``CONFIDENCE_INVALID`` means it could not be evaluated -- neither is a
      drainage detection, so both are dropped. Confidence, not residual, is the
      test: a lake can meet criterion 2 or 3 (below the prediction interval, or
      below its historical minimum) without meeting criterion 1, and a residual
      filter would silently drop it.
    * **Historical** (2016-2025): breaks found by the *simple* method and
      backfilled into this table (see ``merge_nrt_confidence``). They carry no
      NRT columns at all -- no residual, no confidence, just ``date_break`` and
      ``water_change_perc`` -- and every one of them is a detection. A null
      confidence with a null residual is therefore kept: nothing scored it, so
      nothing found it stable.
    * **NRT rows written before the confidence column existed**: null
      confidence but a real residual, already filtered by residual when they
      were written. Kept when the residual clears the threshold.

    Args:
        df: An NRT breaks table (or one month of one).
        drain_threshold: Residual threshold for rows carrying no confidence.

    Returns:
        Boolean Series aligned to ``df.index``.
    """
    if df.empty:
        return pd.Series(False, index=df.index, dtype=bool)

    if "drainage_confidence" not in df.columns:
        # Nothing was scored, so nothing can be shown to be stable.
        return pd.Series(True, index=df.index, dtype=bool)

    confidence = pd.to_numeric(df["drainage_confidence"], errors="coerce")
    detected = (confidence >= 1).fillna(False)

    if "water_residual" in df.columns:
        residual = pd.to_numeric(df["water_residual"], errors="coerce")
        # `~(residual >= threshold)` rather than `residual < threshold`, so a
        # null residual (an unscored row) stays in: only a residual that is
        # present and above the threshold proves a lake is not draining.
        unscored = confidence.isna() & ~(residual >= drain_threshold)
    else:
        unscored = confidence.isna()

    return (detected | unscored).fillna(False)


cols = [
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
]
rename_dict = {
    "water_observed": "water_observed_absolute",
    "water_predicted": "water_predicted_absolute",
    "water_residual": "water_residual_absolute",
    "water_predicted_lower_90": "water_predicted_lower_90_absolute",
    "water_predicted_upper_90": "water_predicted_upper_90_absolute",
    "water_historical_mean": "water_historical_mean_absolute",
    "water_historical_median": "water_historical_median_absolute",
    "water_historical_std": "water_historical_std_absolute",
    "water_historical_min": "water_historical_min_absolute",
    "water_historical_max": "water_historical_max_absolute",
}


def add_confidence_interval_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates and appends formatted string representations ('lower : upper')
    for both relative and absolute confidence intervals to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'water_predicted_lower_90',
            'water_predicted_upper_90', and their absolute counterparts.

    Returns:
        pd.DataFrame: A copy of the DataFrame with 'water_predicted_ci' and
            'water_predicted_ci_absolute' columns added.
    """
    df = df.copy()

    # Format relative CI string
    df["water_predicted_ci"] = (
        df["water_predicted_lower_90"].round(2).astype(str)
        + " : "
        + df["water_predicted_upper_90"].round(2).astype(str)
    )

    # Format absolute CI string
    df["water_predicted_ci_absolute"] = (
        df["water_predicted_lower_90_absolute"].round(2).astype(str)
        + " : "
        + df["water_predicted_upper_90_absolute"].round(2).astype(str)
    )

    return df


def recalculate_absolute_and_prepare(
    nrt_dataset: pd.DataFrame,
    dw_dataset_raw: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    input_cols: list[str],
    all_geoms: bool = True,
    add_ci_range: bool = True,
) -> gpd.GeoDataFrame:
    """
    Recalculates relative predictions into absolute values using a scaling factor derived
    from a raw dataset, joins the results with spatial geometries, and formats the final payload.

    Args:
        nrt_dataset (pd.DataFrame): The near-real-time dataset containing relative predictions.
            The index must consist of `id_geohash` values.
        dw_dataset_raw (xr.Dataset): The raw data xarray Dataset used to compute scaling factors,
            containing an `id_geohash` dimension.
        gdf (gpd.GeoDataFrame): The spatial GeoDataFrame containing the base geometries and
            an `id_geohash` column to map data to regions.
        input_cols (list[str]): List of column names in `nrt_dataset` representing the relative
            metrics to be scaled and included.
        all_geoms (bool, optional): Determines the join type with the geometry DataFrame.
            If True, performs a 'left' join (keeps all geometries). If False, performs a
            'right' join (keeps only geometries present in the dataset). Defaults to True.
        add_ci_range (bool, optional): If True, computes and appends formatted string representations
            for the relative and absolute confidence intervals ('lower : upper'). Defaults to True.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame sorted by geohash containing spatial boundaries, original
            metrics, recalculated absolute metrics, and metadata columns (`date`, `drainage_confidence`).
    """
    # Protect original input list from mutation leaks across pipeline runs
    input_cols = input_cols.copy()
    input_cols_abs = list(rename_dict.values())

    # find ids in raw ds
    ds_raw_subset = dw_dataset_raw.sel(id_geohash=nrt_dataset.index.values)
    # get max area data which is the scaling factor
    scaling_factor = ds_raw_subset.to_array(dim="variable").sum(dim="variable").max(axis=1).to_pandas()
    # create df to calculate abs values
    nrt_dataset_abs = nrt_dataset[input_cols].multiply(scaling_factor, axis=0).rename(columns=rename_dict)

    # join with geoms
    if all_geoms:
        joined = (
            gdf[["id_geohash", "geometry"]].set_index("id_geohash").join(nrt_dataset.join(nrt_dataset_abs), how="left")
        )
    else:
        joined = (
            gdf[["id_geohash", "geometry"]].set_index("id_geohash").join(nrt_dataset.join(nrt_dataset_abs), how="right")
        )

    if add_ci_range:
        # Call helper function to add the new string columns
        joined = add_confidence_interval_strings(joined)

        # Keep the append track updates right here in the main function scope
        input_cols.append("water_predicted_ci")
        input_cols_abs.append("water_predicted_ci_absolute")

    # setup final cols setup
    final_cols = ["id_geohash", "date"] + input_cols + input_cols_abs + ["drainage_confidence", "geometry"]
    # setup final output
    final_with_geoms = joined.sort_index().reset_index(drop=False).rename(columns={"index": "id_geohash"})[final_cols]

    return final_with_geoms
