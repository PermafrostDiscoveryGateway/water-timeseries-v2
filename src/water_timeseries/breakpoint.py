"""Breakpoint detection strategies for water‑timeseries.

This module provides a hierarchy of breakpoint detection methods for analyzing
water‑timeseries data. It includes:

* ``BreakpointMethod`` – abstract base class with shared helpers.
* ``SimpleBreakpoint`` – a fast rolling‑window statistical detector.
* ``BeastBreakpoint`` – a Bayesian RBEAST‑based detector.
* ``NRTBreakpoint`` – a Near‑Real‑Time breakpoint detector with custom logic.

Each concrete class implements ``calculate_break`` for a single lake and
inherits ``calculate_breaks_batch`` from the base class. The classes can be
used directly in Python code *or* indirectly through the ``water-timeseries``
CLI.

Example
-------
>>> from water_timeseries.breakpoint import SimpleBreakpoint
>>> breakpoint = SimpleBreakpoint(kwargs_break=dict(window=3, method="median", threshold=-0.25))
>>> # breakpoint.calculate_break(dataset)  # Returns DataFrame with breakpoint info
"""

import logging
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import Rbeast as rb
import xarray as xr
from joblib import Parallel, delayed
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from tqdm import tqdm

from water_timeseries.dataset import LakeDataset
from water_timeseries.utils.data import (
    calculate_temporal_stats,
    calculate_water_area_after,
    calculate_water_area_before,
)

warnings.filterwarnings("ignore")


class BreakpointMethod:
    """Base class for breakpoint detection methods.

    Parameters
    ----------
    method_name : str
        Short identifier stored in the ``break_method`` column of the output
        DataFrames. Sub‑classes pass values such as ``"simple"`` or ``"rbeast"``.
    """

    def __init__(self, method_name: str):
        self.method_name = method_name

    def get_first_break_date(self, df: pd.DataFrame, column: str = "water") -> tuple:
        """Placeholder implementation for the abstract base.

        Concrete subclasses override this method.  The default returns a
        ``(None, None, None)`` tuple so that calling code can safely handle the lack of
        a breakpoint.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a datetime-like index and a water column.
        column : str, optional
            Column name to evaluate. Defaults to "water".

        Returns
        -------
        tuple
            (first_break_date, previous_date, after_date) - All values are None
            for the default implementation.
        """
        return (None, None, None)

    def calculate_break(
            self,
            dataset: LakeDataset,
            analysis_date: str | pd.Timestamp = None,
            data_aggregation_period: str = "all",
            object_id: str | Optional[str] = None,
            keep_nans: bool | Optional[bool] = False,
    ) -> pd.DataFrame:
        """Base implementation - subclasses should override.

        Parameters
        ----------
        dataset : LakeDataset
            Dataset containing lake water‑area data.
        analysis_date : str or pd.Timestamp, optional
            The date for which to perform the breakpoint analysis.
        data_aggregation_period : str, optional
            The period of data to consider for the analysis (e.g., "all", "monthly")
        object_id : str, optional
            Unique identifier (geohash) for the lake object.
        keep_nans : bool, optional
            Set True if you want to return historical water stats.

        Returns
        -------
        pd.DataFrame
            DataFrame containing breakpoint information.
        """
        raise NotImplementedError("Subclasses must implement calculate_break with this signature")

    def calculate_breaks_batch(self, dataset: LakeDataset, progress_bar: bool = False) -> pd.DataFrame:
        """Run ``calculate_break`` for every lake in *dataset*.

        Parameters
        ----------
        dataset : LakeDataset
            Dataset providing both raw and normalized water‑area arrays.
        progress_bar : bool, optional
            Show a ``tqdm`` progress bar when ``True``. Default is ``False``.

        Returns
        -------
        pd.DataFrame
            Concatenated results from all lakes in the dataset.
        """
        # Batch processing of breakpoints for all objects in the dataset
        dataset.ds.load()
        dataset.ds_normalized.load()
        results = []
        if progress_bar:
            progress = tqdm(dataset.ds_normalized.id_geohash.values)
        else:
            progress = dataset.ds_normalized.id_geohash.values
        for object_id in progress:
            # Call with object_id as keyword argument to match the new signature
            result = self.calculate_break(dataset, object_id=object_id)
            results.append(result)
        return pd.concat(results)


class SimpleBreakpoint(BreakpointMethod):
    """Fast rolling‑window statistical breakpoint detector.

    This method detects breakpoints by comparing current water values against
    rolling window statistics (mean, median, or max). A breakpoint is identified
    when values fall below a threshold in both a primary and secondary window
    for consecutive time points, which helps reduce false positives.

    Parameters
    ----------
    kwargs_break : dict, optional
        Configuration dictionary with the following keys:
        - ``window`` : int, default 3
            Size of the primary rolling window.
        - ``method`` : str, default "median"
            Rolling statistic to use: "mean", "median", or "max".
        - ``threshold`` : float, default -0.25
            Threshold for detecting a break (values below this indicate a break).

    Attributes
    ----------
    breakpoint_columns : list
        List of column names in the output DataFrame.
    """

    def __init__(self, kwargs_break: dict = dict(window=3, method="median", threshold=-0.25)):
        super().__init__(method_name="simple")
        self.kwargs_break = kwargs_break
        self.breakpoint_columns = ["date_break", "date_before_break", "date_after_break", "break_method"]

    def get_first_break_date(self, df: pd.DataFrame, column: str = "water") -> tuple:
        """Find the first break date and the immediately preceding index value."""
        df = df.drop(columns=["id_geohash"]).dropna()

        primary_window = self.kwargs_break["window"]
        secondary_window = primary_window + 2

        if self.kwargs_break["method"] == "max":
            primary_rolling = df.rolling(window=primary_window).max()
            secondary_rolling = df.rolling(window=secondary_window).max()
        elif self.kwargs_break["method"] == "mean":
            primary_rolling = df.rolling(window=primary_window).mean()
            secondary_rolling = df.rolling(window=secondary_window).mean()
        elif self.kwargs_break["method"] == "median":
            primary_rolling = df.rolling(window=primary_window).median()
            secondary_rolling = df.rolling(window=secondary_window).median()
        else:
            raise ValueError("Please assign correct rolling value: 'max', 'mean', or 'median'")

        rolling_diff_primary = df - primary_rolling
        rolling_diff_secondary = df - secondary_rolling

        mask_primary = rolling_diff_primary[column] < self.kwargs_break["threshold"]
        mask_secondary = rolling_diff_secondary[column] < self.kwargs_break["threshold"]
        consecutive_mask = mask_primary & mask_secondary.shift(-1)

        first_break_date = (
            rolling_diff_primary[consecutive_mask].index.min()
            if not rolling_diff_primary[consecutive_mask].empty
            else None
        )

        previous_date = None
        after_date = None
        if first_break_date is not None:
            try:
                pos = df.index.get_loc(first_break_date)
                if isinstance(pos, slice):
                    pos = pos.start if pos.start is not None else 0
                if pos > 0:
                    previous_date = df.index[pos - 1]
                    after_date = df.index[pos + 1] if pos + 1 < len(df) else None
            except Exception:
                previous_date = None
                after_date = None

        return first_break_date, previous_date, after_date

    def calculate_break(
            self,
            dataset: LakeDataset,
            analysis_date: str | pd.Timestamp = None,
            data_aggregation_period: str = "all",
            object_id: str | Optional[str] = None,
            keep_nans: bool | Optional[bool] = False,
    ) -> pd.DataFrame:
        """Calculate breakpoints for a single lake object."""
        if object_id is None:
            raise ValueError("SimpleBreakpoint requires object_id parameter")

        ds = dataset.ds_normalized
        df_normed = ds.sel(id_geohash=object_id).to_pandas()
        first_break, previous_date, after_date = self.get_first_break_date(df=df_normed, column=dataset.water_column)
        if first_break is None:
            return pd.DataFrame(columns=self.breakpoint_columns)
        df_out = pd.DataFrame(
            {
                self.breakpoint_columns[0]: [first_break],
                self.breakpoint_columns[1]: [previous_date],
                self.breakpoint_columns[2]: [after_date],
                self.breakpoint_columns[3]: [self.method_name],
            },
            index=[object_id],
        )

        break_list = []
        df_water = dataset.ds.sel(id_geohash=object_id).to_dataframe()
        for i, row in df_out.iterrows():
            id_geohash = row.name
            df_breaks = pd.concat(
                [
                    row,
                    calculate_water_area_before(
                        df_water, break_date=row["date_break"], water_column=dataset.water_column
                    ),
                    calculate_water_area_after(
                        df_water, break_date_after=row["date_after_break"], water_column=dataset.water_column
                    ),
                ]
            )
            df_breaks.name = id_geohash
            break_list.append(df_breaks)

        break_df = pd.concat(break_list, axis=1).T
        break_df = calculate_temporal_stats(break_df)
        return break_df


class BeastBreakpoint(BreakpointMethod):
    """Bayesian RBEAST-based breakpoint detector."""

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
            "date_after_break",
            "break_method",
            "break_number",
            "proba_rbeast",
        ]

    def calculate_break(
            self,
            dataset: LakeDataset,
            analysis_date: str | pd.Timestamp = None,
            data_aggregation_period: str = "all",
            object_id: str | Optional[str] = None,
            keep_nans: bool | Optional[bool] = False,
    ) -> pd.DataFrame:
        """Calculate breakpoints for a single lake object using RBEAST."""
        if object_id is None:
            raise ValueError("BeastBreakpoint requires object_id parameter")

        ds = dataset.ds_normalized
        df = ds.sel(id_geohash=object_id).to_pandas()
        df["date"] = df.index
        data = df[dataset.water_column]

        o = rb.beast(data, season="none", quiet=True, prior=self.kwargs_break)
        cp_prob = o.trend.cpOccPr
        break_indices = np.where(cp_prob > self.break_threshold)[0]

        if break_indices.size == 0:
            return pd.DataFrame(columns=self.breakpoint_columns)

        break_indices_before = np.array(break_indices) - 1
        break_indices_after = np.array(break_indices) + 1
        break_dates_before = df.iloc[break_indices_before]["date"].to_list()
        break_dates_after = df.iloc[break_indices_after]["date"].to_list()

        df = df.copy()
        df["proba_rbeast"] = cp_prob
        break_df = df.iloc[break_indices].copy()
        break_df.loc[:, "date_before_break"] = break_dates_before
        break_df.loc[:, "date_after_break"] = break_dates_after
        break_df = break_df.sort_values("proba_rbeast", ascending=False).copy()
        break_df["break_number"] = range(1, len(break_df) + 1)

        break_df_out = break_df.rename(columns={"date": "date_break"}).set_index("id_geohash")
        break_df_out["break_method"] = self.method_name
        df_out = break_df_out[self.breakpoint_columns]

        break_list = []
        df_water = dataset.ds.sel(id_geohash=object_id).to_dataframe()
        for i, row in df_out.iterrows():
            id_geohash = row.name
            df_breaks = pd.concat(
                [
                    row,
                    calculate_water_area_before(
                        df_water, break_date=row["date_break"], water_column=dataset.water_column
                    ),
                    calculate_water_area_after(
                        df_water, break_date_after=row["date_after_break"], water_column=dataset.water_column
                    ),
                ]
            )
            df_breaks.name = id_geohash
            break_list.append(df_breaks)
        break_df = pd.concat(break_list, axis=1).T
        break_df.index.name = "id_geohash"
        break_df = calculate_temporal_stats(break_df)
        return break_df


class NRTBreakpoint(BreakpointMethod):
    """Near‑Real‑Time (NRT) breakpoint detector."""

    def __init__(self, kwargs_break: dict = dict()):
        super().__init__(method_name="nrt")
        self.kwargs_break = kwargs_break
        self.breakpoint_columns = ["date_break", "date_before_break", "date_after_break", "break_method"]
        self.output_columns = [
            "date",
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
            "drainage_confidence",
        ]
        self.output_columns_base = [
            "date",
            "water_observed",
            "water_predicted",
            "water_residual",
            "water_predicted_lower_90",
            "water_predicted_upper_90",
            "drainage_confidence",
        ]

    def predict_nrt_arima(
            self, ds_in: xr.Dataset, id_geohash: str, min_length: int = 3, water_column: str = "water"
    ) -> pd.Series:
        df_in = (
            ds_in.sel(id_geohash=id_geohash)
            .to_dataframe()
            .drop(columns=["id_geohash"])[water_column]
            .reset_index(drop=True)
            .dropna()
        )

        if len(df_in) < min_length:
            print(f"Time-series has less than {min_length} observations. Skip processing for {id_geohash}")
            return None

        model = AutoARIMA(
            stepwise=True,
            suppress_warnings=True,
            trace=False,
            error_action="ignore",
            seasonal=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df_in)
            fh = ForecastingHorizon([1], is_relative=True)
            y_pred = model.predict(fh=fh)
            y_pred_int = model.predict_interval(fh=fh, coverage=0.90)

            result = pd.Series(
                {
                    "water_predicted": y_pred.iloc[0],
                    "water_predicted_lower_90": y_pred_int.iloc[0, 0],
                    "water_predicted_upper_90": y_pred_int.iloc[0, 1],
                },
                name=id_geohash,
            )
            return result

    def _validate_analysis_date(self, analysis_date: str | pd.Timestamp) -> pd.Timestamp:
        if isinstance(analysis_date, str):
            try:
                return pd.to_datetime(analysis_date, format="%Y-%m")
            except (ValueError, TypeError):
                return pd.to_datetime(analysis_date)
        elif isinstance(analysis_date, pd.Timestamp):
            return analysis_date
        else:
            return pd.to_datetime(analysis_date)

    def _filter_valid_ids(self, ds_analysis: xr.Dataset, ds_historical: xr.Dataset) -> tuple:
        valid_ids = ds_analysis.dropna(dim="id_geohash", how="all").id_geohash.values
        all_ids = ds_analysis.id_geohash.values
        nan_ids = all_ids[~pd.Series(all_ids).isin(pd.Series(valid_ids))]
        total_ids = len(all_ids)
        valid_count = len(valid_ids)
        filtered_count = total_ids - valid_count
        logging.info(f"Filtered {filtered_count} id_geohash(es) with NaN data, kept {valid_count} valid id_geohash(es)")
        ds_analysis_filtered = ds_analysis.sel(id_geohash=valid_ids)
        ds_historical_filtered = ds_historical.sel(id_geohash=valid_ids)
        return ds_analysis_filtered, ds_historical_filtered, valid_ids, nan_ids

    def _get_ds_stats(self, dataset: xr.Dataset, filter_month: int = None, water_column: str = "water") -> pd.DataFrame:
        if filter_month is not None:
            dataset = dataset.where(dataset.date.dt.month == filter_month, drop=True)
        out_df = dataset.to_dataframe()[water_column].groupby("id_geohash").agg(["mean", "median", "std", "min", "max"])
        return out_df

    def _add_confidence_level(self, break_output_df: pd.DataFrame) -> pd.DataFrame:
        cat1 = break_output_df["water_residual"] < -0.25
        cat2 = break_output_df["water_observed"] < break_output_df["water_predicted_lower_90"]
        cat3 = break_output_df["water_observed"] < break_output_df["water_historical_min"]
        drain_confidence = pd.concat([cat1, cat2, cat3], axis=1).sum(axis=1)
        break_output_df["drainage_confidence"] = drain_confidence.astype(int)
        return break_output_df

    def calculate_break(
            self,
            dataset: LakeDataset,
            analysis_date: str | pd.Timestamp = None,
            data_aggregation_period: str = "all",
            object_id: str | Optional[str] = None,
            keep_nans: bool | Optional[bool] = False,
    ) -> pd.DataFrame:
        """Calculate breakpoints using NRT logic."""

        # ===== VERIFICATION LINE =====
        print("=" * 60)
        print("RUNNING WITH FIXED NRT BREAKPOINT - VERSION WITH ALIGNMENT FIX")
        print("=" * 60)

        if analysis_date is None:
            raise ValueError("NRTBreakpoint requires analysis_date parameter")

        analysis_date = self._validate_analysis_date(analysis_date)
        print(analysis_date)
        print(analysis_date.strftime("%Y-%m"))

        if analysis_date not in dataset.dates_:
            raise ValueError(f"Analysis date {analysis_date.strftime('%Y-%m')} is not available in the dataset.")

        data = dataset.ds_normalized

        if object_id is not None:
            if isinstance(object_id, str):
                object_id = [object_id]
            object_id = [obj for obj in object_id if obj in dataset.object_ids_]
            data = data.sel(id_geohash=object_id)

        ds_analysis = data.sel(date=analysis_date)
        ds_historical = data.where(data["date"] < analysis_date, drop=True)

        if data_aggregation_period == "monthly":
            print("Filtering to monthly data for analysis date month:", analysis_date.month)
            ds_historical = ds_historical.where(ds_historical.date.dt.month == analysis_date.month, drop=True)

        ds_analysis_filtered, ds_historical_filtered, valid_ids, nan_ids = self._filter_valid_ids(
            ds_analysis, ds_historical
        )

        if len(valid_ids) == 0:
            if keep_nans:
                return pd.DataFrame(index=nan_ids, columns=self.output_columns)
            else:
                return pd.DataFrame(columns=self.output_columns)

        cpu_count = os.cpu_count() or 1
        n_jobs = max(1, min(cpu_count, len(valid_ids)))
        predictions = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self.predict_nrt_arima)(
                ds_in=ds_historical_filtered, id_geohash=idx, water_column=dataset.water_column
            )
            for idx in tqdm(valid_ids, desc="NRT breakpoints")
        )

        # Create prediction DataFrame with ALL valid_ids (fill missing with NaN)
        predictions = [pred for pred in predictions if pred is not None]

        if predictions:
            prediction_df = pd.DataFrame(predictions)
            prediction_df = prediction_df.reset_index()
            prediction_df.rename(columns={'index': 'id_geohash'}, inplace=True)

            # Add any missing valid_ids as NaN rows
            existing_ids = set(prediction_df['id_geohash'].values)
            missing_ids = set(valid_ids) - existing_ids
            for missing_id in missing_ids:
                missing_row = pd.DataFrame({
                    'id_geohash': [missing_id],
                    'water_predicted': [np.nan],
                    'water_predicted_lower_90': [np.nan],
                    'water_predicted_upper_90': [np.nan]
                })
                prediction_df = pd.concat([prediction_df, missing_row], ignore_index=True)
        else:
            prediction_df = pd.DataFrame({
                'id_geohash': valid_ids,
                'water_predicted': np.nan,
                'water_predicted_lower_90': np.nan,
                'water_predicted_upper_90': np.nan
            })

        # Get water data as flat DataFrame
        water_df = ds_analysis_filtered[dataset.water_column].to_dataframe().reset_index()

        if 'index' in water_df.columns:
            water_df = water_df.rename(columns={'index': 'id_geohash'})

        # Ensure we have id_geohash as a column
        if 'id_geohash' not in water_df.columns:
            water_df = water_df.reset_index()
            water_df.rename(columns={'index': 'id_geohash'}, inplace=True)

        # Merge - now both should have same number of unique ids
        df_output = water_df.merge(prediction_df, on='id_geohash', how='left', suffixes=('', '_pred'))
        df_output = df_output.round(4)

        # Rename water column
        water_col_name = dataset.water_column
        if water_col_name in df_output.columns:
            df_output.rename(columns={water_col_name: "water_observed"}, inplace=True)

        # Handle date column
        if 'date' in df_output.columns:
            df_output['date'] = pd.to_datetime(df_output['date'])
        elif 'date_x' in df_output.columns:
            df_output['date'] = pd.to_datetime(df_output['date_x'])
            df_output.drop(columns=['date_x'], inplace=True)

        # Ensure water_predicted exists
        if 'water_predicted' not in df_output.columns:
            df_output['water_predicted'] = np.nan

        # Remove duplicates
        if 'id_geohash' in df_output.columns and 'date' in df_output.columns:
            before_count = len(df_output)
            df_output = df_output.drop_duplicates(subset=['id_geohash', 'date'])
            if before_count != len(df_output):
                print(f"Removed {before_count - len(df_output)} duplicate rows")

        # Reset index for clean alignment
        df_output = df_output.reset_index(drop=True)

        # Now calculate residuals - both columns should have same length
        print(f"Calculating residuals for {len(df_output)} rows")
        print(f"water_observed non-null: {df_output['water_observed'].notna().sum()}")
        print(f"water_predicted non-null: {df_output['water_predicted'].notna().sum()}")

        # Use pandas subtraction (index is simple integer range)
        df_output["water_residual"] = df_output["water_observed"] - df_output["water_predicted"]

        # Get historical stats
        df_historical_stats = self._get_ds_stats(ds_historical_filtered, water_column=dataset.water_column).round(4)
        df_historical_stats.columns = "water_historical_" + df_historical_stats.columns.astype(str)
        df_historical_stats = df_historical_stats.reset_index()

        # Merge historical stats
        df_output = df_output.merge(df_historical_stats, on='id_geohash', how='left')
        df_output = df_output.round(4)

        # Add confidence
        df_output = self._add_confidence_level(df_output)

        # Handle NaN ids if requested
        if keep_nans and len(nan_ids) > 0:
            df_output_nan = pd.DataFrame({'id_geohash': nan_ids})
            if len(nan_ids) > 0:
                df_historical_stats_nans = self._get_ds_stats(
                    ds_historical.sel(id_geohash=nan_ids), water_column=dataset.water_column
                ).round(4)
                if not df_historical_stats_nans.empty:
                    df_historical_stats_nans.columns = "water_historical_" + df_historical_stats_nans.columns.astype(
                        str)
                    df_historical_stats_nans = df_historical_stats_nans.reset_index()
                    df_output_nan = df_output_nan.merge(df_historical_stats_nans, on='id_geohash', how='left')

            for col in self.output_columns:
                if col not in df_output_nan.columns and col != 'date':
                    df_output_nan[col] = np.nan

            df_output = pd.concat([df_output, df_output_nan], ignore_index=True)

        # Select output columns
        result_columns = [col for col in self.output_columns if col in df_output.columns]
        df_output = df_output[result_columns]

        # Set index at the very end (optional)
        if 'id_geohash' in df_output.columns and 'date' in df_output.columns:
            df_output = df_output.set_index(['id_geohash', 'date'])

        return df_output