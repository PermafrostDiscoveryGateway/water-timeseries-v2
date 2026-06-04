from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_ID_FIELD_CANDIDATES = ("lake_id", "id_geohash")


def infer_id_field(ds: xr.Dataset) -> str:
    """Return the coordinate name used for per-lake time series IDs."""
    for name in _ID_FIELD_CANDIDATES:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(
        f"No lake ID coordinate found. Expected one of {_ID_FIELD_CANDIDATES}. "
        f"Coords/dims: {list(ds.coords)} / {list(ds.dims)}"
    )


def calculate_water_area_after(
    df_water, break_date_after, water_column: str, stats=["mean", "median", "std", "min", "max"]
):
    after = df_water.loc[break_date_after:][water_column].agg(stats)
    cols_out = [f"post_break_{col}" for col in after.index]
    after.index = cols_out
    return after


def calculate_water_area_before(df_water, break_date, water_column: str, stats=["mean", "median", "std", "min", "max"]):
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


def annotate_xr_dataset_jrc(ds: xr.Dataset, input_vector_file: Path | str = None) -> xr.Dataset:
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


def annotate_xr_dataset_dw(ds: xr.Dataset, input_vector_file: Path | str = None) -> xr.Dataset:
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
