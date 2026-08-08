from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


def calculate_water_area_after(df_water, break_date_after, water_column: str, stats=None):
    if stats is None:
        stats = ["mean", "median", "std", "min", "max"]
    after = df_water.loc[break_date_after:][water_column].agg(stats)
    cols_out = [f"post_break_{col}" for col in after.index]
    after.index = cols_out
    return after


def calculate_water_area_before(df_water, break_date, water_column: str, stats=None):
    if stats is None:
        stats = ["mean", "median", "std", "min", "max"]
    before = df_water.loc[:break_date][water_column].agg(stats)
    cols_out = [f"pre_break_{col}" for col in before.index]
    before.index = cols_out
    return before


def get_water_dataset_type(input_ds) -> str:
    """Determine the water dataset type based on the presence of specific variables in the dataset."""
    if "area_water_permanent" in input_ds.data_vars:
        water_dataset_type = "jrc"
    elif "water" in input_ds.data_vars:
        water_dataset_type = "dynamic_world"
    else:
        raise ValueError("Unknown water dataset type")

    return water_dataset_type


def calculate_temporal_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate temporal statistics for a given DataFrame."""
    df = df.copy()
    df["pre_break_median"] = df["pre_break_median"].where(df["pre_break_median"] != 0, np.nan)
    df["post_break_median"] = df["post_break_median"].where(df["post_break_median"] != 0, np.nan)
    # df.dropna(subset=["pre_break_median", "post_break_median"], inplace=True)
    breaks = pd.to_datetime(df["date_break"])
    df["date_break_year"] = breaks.dt.year
    df["date_break_month"] = breaks.dt.month
    # change area ha
    df["water_change_ha"] = df["post_break_median"] - df["pre_break_median"]
    # change area perc
    df["water_change_perc"] = df["water_change_ha"].div(df["pre_break_median"].replace(0, np.nan)) * 100
    return df


def annotate_xr_dataset_jrc(ds: xr.Dataset, input_vector_file: Path | str | None = None) -> xr.Dataset:
    """
    Annotates an xarray Dataset with units, description, author, and contact information.

    Parameters:
    ds (xarray.Dataset): The dataset to be annotated.
    input_vector_file (str|Path, optional): The path to the accompanying vector dataset. Defaults to None.

    Returns:
    xarray.Dataset: The annotated dataset.

    """
    # variable annotations
    for var in list(ds.data_vars):
        ds[var].attrs["units"] = "ha"

    # dataset annotations
    ds.attrs["description"] = (
        'This datasets provides the annual area of permanent water, seasonal water, land, and no data for selected lake polygons. The areas were calculated from the JRC annual surface water dataset through Google Earth Engine. Lake polygons were calculated by Ingmar Nitze through the Permafrost Discovery Gateway Project. "id_geohash" is the lake_id, which needs be joined to the accompanying polygon vector dataset'
    )
    if input_vector_file is not None:
        input_vector_file = Path(input_vector_file)
        ds.attrs["accompanying vector dataset"] = input_vector_file.name
    ds.attrs["source"] = "https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2"
    ds.attrs["author"] = "Ingmar Nitze (Alfred Wegener Institute), Todd Nicholson(NCSA, U Illinois)"
    ds.attrs["contact"] = "ingmar.nitze@awi.de"

    return ds


def annotate_xr_dataset_dw(ds: xr.Dataset, input_vector_file: Path | str | None = None) -> xr.Dataset:
    """
    Annotates an xarray Dataset with units, description, author, and contact information.

    Parameters:
    ds (xarray.Dataset): The dataset to be annotated.
    input_vector_file (str|Path, optional): The path to the accompanying vector dataset. Defaults to None.

    Returns:
    xarray.Dataset: The annotated dataset.

    """
    # variable annotations
    for var in list(ds.data_vars):
        ds[var].attrs["units"] = "ha"

    # dataset annotations
    ds.attrs["description"] = (
        'This datasets provides the monthly area of the dynamic world classes (water, trees, grass, flooded_vegetation, crops, shrub_and_scrub, built, bare, snow_and_ice) for selected lake polygons. The areas were calculated from the Dynamic World V1 dataset through Google Earth Engine. Lake polygons were calculated by Ingmar Nitze through the Permafrost Discovery Gateway Project. "id_geohash" is the lake_id, which needs be joined to the accompanying polygon vector dataset'
    )
    if input_vector_file is not None:
        input_vector_file = Path(input_vector_file)
        ds.attrs["accompanying vector dataset"] = input_vector_file.name
    ds.attrs["source"] = "https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2"
    ds.attrs["author"] = (
        "Ingmar Nitze (Alfred Wegener Institute), Kayla Hardie (Google), Chen Wang (NCSA, U Illinois), Todd Nicholson(NCSA, U Illinois)"
    )
    ds.attrs["contact"] = "ingmar.nitze@awi.de"
    return ds


dw_bandnames = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

jrc_bandnames = [
    "area_nodata",
    "area_land",
    "area_water_seasonal",
    "area_water_permanent",
]


def find_data_gaps(dataset: xr.Dataset, variable: str = "water", id_geohash_subset=None):
    """
    Identifies lakes with significant data gaps.

    Parameters:
    - dataset: The xarray Dataset (ds)
    - variable: The primary variable to check for missingness (e.g., 'water')
    - id_geohash_subset: An optional slice or range for the 'id_geohash' dimension.
                           Example: slice(0, 100) or a list of indices [5, 10, 20]
    """
    print(f"Analyzing gaps using variable: {variable}...")

    # Logic to handle subsetting
    if id_geohash_subset is not None:
        print(f"Applying subset for 'id_geohash': {id_geohash_subset}")
        ds_to_analyze = dataset.isel(id_geohash=id_geohash_subset)
    else:
        print("No subset provided. Analyzing the full dataset.")
        ds_to_analyze = dataset

    # 1. Identify missing values (NaNs) and sum them along the time dimension
    # Dask will handle this calculation in parallel chunks.
    missing_counts = ds_to_analyze[variable].isnull().sum(dim="date")

    # 2. Calculate completion percentage directly in Xarray
    total_months = ds_to_analyze.dims["date"]
    completion_pct = ((total_months - missing_counts) / total_months) * 100

    # 3. Convert to a clean DataFrame
    # We use .values.flatten() to ensure we get the raw data into the dataframe correctly
    df = pd.DataFrame(
        {
            "id_geohash": ds_to_analyze.id_geohash.values,
            "missing_count": missing_counts.values.flatten(),
            "completion_pct": completion_pct.values.flatten(),
        }
    )

    # 4. Filter for large gaps (e.g., less than 50% data present)
    large_gaps = df[df["completion_pct"] < 50].sort_values(by="completion_pct")

    print(f"Found {len(large_gaps)} lakes with less than 50% data coverage.")
    return large_gaps


def load_and_merge_parquets(file_list: list[Path], logger=None) -> pd.DataFrame:
    """Loads multiple parquet files and concatenates them into one DataFrame."""
    if logger:
        logger.info(f"Found {len(file_list)} files to merge.")
    df_list = []
    required_column = "id_geohash"

    index_count = 0
    col_count = 0

    for f in tqdm(file_list, desc="Loading Parquet Files"):
        try:
            df = pd.read_parquet(f)
            if required_column not in df.columns:
                if df.index.name == required_column:
                    df = df.reset_index()
                    index_count += 1
                else:
                    if logger:
                        logger.warning(
                            f"Skipping {f.name}: No '{required_column}' found in columns or as an index name."
                        )
                    continue
            else:
                col_count += 1

            if not df.empty:
                df_list.append(df)
            else:
                if logger:
                    logger.warning(f"Skipping {f.name}: File is empty.")

        except Exception as e: #noqa:BLE001
            if logger:
                logger.error(f"Failed to read {f.name} (possibly corrupt/inaccessible): {e}")

    if not df_list:
        raise FileNotFoundError("No valid parquet files containing 'id_geohash' were loaded.")

    if logger:
        logger.info(
            f"Merge Summary: {index_count} files had '{required_column}' as Index, {col_count} files had it as a Column."
        )
    return pd.concat(df_list, ignore_index=True)
