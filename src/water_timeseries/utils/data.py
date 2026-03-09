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
