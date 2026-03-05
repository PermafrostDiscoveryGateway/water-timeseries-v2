import numpy as np
import pandas as pd
import Rbeast as rb
from tqdm import tqdm

from water_timeseries.dataset import LakeDataset


class BreakpointMethod:
    def __init__(self, method_name):
        self.method_name = method_name

    def get_first_break_date(self, df: pd.DataFrame, column: str = "water") -> tuple:
        return (None, None)

    def calculate_break(self, dataset):
        pass

    def calculate_breaks_batch(self, dataset, progress_bar=False):
        # Batch processing of breakpoints for all objects in the dataset
        dataset.ds_normalized.load()
        dataset.ds.load()
        results = []
        if progress_bar:
            progress = tqdm(dataset.ds_normalized.id_geohash.values)
        else:
            progress = dataset.ds_normalized.id_geohash.values
        for object_id in progress:
            result = self.calculate_break(dataset, object_id)
            results.append(result)
        return pd.concat(results)


class SimpleBreakpoint(BreakpointMethod):
    def __init__(self, kwargs_break: dict):
        super().__init__(method_name="simple")
        self.kwargs_break = kwargs_break
        self.breakpoint_columns = ["date_break", "date_before_break", "break_method"]

    def get_first_break_date(self, df: pd.DataFrame, column: str = "water") -> tuple:
        """Find the first break date and the immediately preceding index value.

        Args:
            df (pd.DataFrame): DataFrame with a datetime-like index and a water column.
            column (str, optional): Column name to evaluate. Defaults to "water".

        Returns:
            tuple: (first_break_date (pd.Timestamp or None), previous_date (pd.Timestamp or None))
        """
        df = df.drop(columns=["id_geohash"]).dropna()

        # Calculate rolling difference
        if self.kwargs_break["method"] == "max":
            rolling_diff = df - df.rolling(window=self.kwargs_break["window"]).max()
        elif self.kwargs_break["method"] == "mean":
            rolling_diff = df - df.rolling(window=self.kwargs_break["window"]).mean()
        elif self.kwargs_break["method"] == "median":
            rolling_diff = df - df.rolling(window=self.kwargs_break["window"]).median()
        else:
            raise ValueError("Please assign correct rolling value [max, mean, median]")

        # Create a boolean mask for values less than the threshold
        mask = rolling_diff[column] < self.kwargs_break["threshold"]

        # Shift the mask to check for consecutive values
        consecutive_mask = mask & mask.shift(-1)

        # Get the first index where there are at least two consecutive True values
        first_break_date = (
            rolling_diff[consecutive_mask].index.min()
            if not rolling_diff[consecutive_mask].empty
            else None
        )

        # Determine the preceding index value (previous_date) if available
        previous_date = None
        if first_break_date is not None:
            try:
                pos = df.index.get_loc(first_break_date)
                # get_loc may return a slice or integer; handle integer positions
                if isinstance(pos, slice):
                    pos = pos.start if pos.start is not None else 0
                if pos > 0:
                    previous_date = df.index[pos - 1]
            except Exception:
                previous_date = None

        return first_break_date, previous_date

    def calculate_break(self, dataset: LakeDataset, object_id: str) -> pd.DataFrame:
        ds = dataset.ds_normalized
        df_normed = ds.sel(id_geohash=object_id).to_pandas()
        first_break, previous_date = self.get_first_break_date(
            df=df_normed, column=dataset.water_column
        )
        df_out = pd.DataFrame(
            {
                self.breakpoint_columns[0]: [first_break],
                self.breakpoint_columns[1]: [previous_date],
                self.breakpoint_columns[2]: [self.method_name],
            },
            index=[object_id],
        )

        return df_out


class BeastBreakpoint(BreakpointMethod):
    def __init__(
        self,
        kwargs_break: dict = dict(trendMaxOrder=0, trendMinSepDist=1),
        break_threshold: float = 0.5,
    ):
        super().__init__(method_name="rbeast")
        self.kwargs_break = kwargs_break
        self.break_threshold = break_threshold
        self.breakpoint_columns = [
            "date_break",
            "date_before_break",
            "break_method",
            "break_number",
            "proba_rbeast",
        ]

    def calculate_break(self, dataset: LakeDataset, object_id: str) -> pd.DataFrame:
        # Example implementation for BeastBreakpoint
        # In a real application, this would use the rbeast library or similar
        ds = dataset.ds_normalized
        df = ds.sel(id_geohash=object_id).to_pandas()
        df["date"] = df.index
        data = df[dataset.water_column]

        # Run BEAST (simple: no season). Use priors tuned for sudden drops
        # and allowing short segments (small minimum separation between CPs).
        o = rb.beast(data, season="none", quiet=True, prior=self.kwargs_break)

        cp_prob = o.trend.cpOccPr
        # print(len(cp_prob))

        # get break indices
        break_indices = np.where(cp_prob > self.break_threshold)[0]
        # # get previous date
        break_indices_before = np.array(break_indices) - 1
        # return df
        break_dates_before = df.iloc[break_indices_before]["date"].to_list()

        # ensure we're working with copies to avoid pandas SettingWithCopyWarning
        df = df.copy()
        df["proba_rbeast"] = cp_prob
        # print(break_indices)
        break_df = df.iloc[break_indices].copy()

        # safely add the previous-date column
        break_df.loc[:, "date_before_break"] = break_dates_before

        # sort by probability descending, then add sequential break numbers
        break_df = break_df.sort_values("proba_rbeast", ascending=False).copy()
        break_df["break_number"] = range(1, len(break_df) + 1)

        break_df_out = break_df.rename(columns={"date": "date_break"}).set_index(
            "id_geohash"
        )
        break_df_out["break_method"] = self.method_name
        return break_df_out[self.breakpoint_columns]
