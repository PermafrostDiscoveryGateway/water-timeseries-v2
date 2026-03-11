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
