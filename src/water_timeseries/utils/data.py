import numpy as np
import pandas as pd


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
    df["water_change_ha"] = df["pre_break_median"] - df["post_break_median"]
    # change area perc
    df["water_change_perc"] = df["water_change_ha"].div(df["pre_break_median"].replace(0, np.nan)) * 100
    return df
