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

    def calculate_break(self, dataset: LakeDataset) -> pd.DataFrame:
        """Calculate breakpoints for a single object.

        Sub‑classes must implement the actual detection algorithm and return a
        ``pandas.DataFrame`` containing at least the columns defined in
        ``self.breakpoint_columns``.

        Parameters
        ----------
        dataset : LakeDataset
            Dataset containing lake water‑area data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing breakpoint information.
        """
        pass

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
        # dataset.ds_normalized.load()
        dataset.ds.load()
        dataset.ds_normalized.load()
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
        """Find the first break date and the immediately preceding index value.

        The detection uses a dual‑window approach: a primary rolling window
        (kwargs_break['window']) and a secondary window that is window+2.
        A break is detected when the current value falls below BOTH window
        calculations for consecutive points, reducing false positives.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a datetime‑like index and a water column.
        column : str, optional
            Column name to evaluate. Defaults to "water".

        Returns
        -------
        tuple
            (first_break_date, previous_date, after_date) where each element is
            a pandas Timestamp or None if no break was found.
        """
        df = df.drop(columns=["id_geohash"]).dropna()

        # Determine the rolling window sizes
        primary_window = self.kwargs_break["window"]
        secondary_window = primary_window + 2

        # Calculate rolling statistics based on the specified method
        # (mean, median, or max of the rolling window)
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

        # Calculate the difference from the rolling reference
        # This measures how much the current value deviates from the window baseline
        rolling_diff_primary = df - primary_rolling
        rolling_diff_secondary = df - secondary_rolling

        # Create masks for values that fall below the threshold
        # Both windows must indicate a break for a point to be considered
        mask_primary = rolling_diff_primary[column] < self.kwargs_break["threshold"]
        mask_secondary = rolling_diff_secondary[column] < self.kwargs_break["threshold"]

        # Require consecutive confirmation: current point meets condition
        # AND the next point also meets it (using shifted mask)
        consecutive_mask = mask_primary & mask_secondary.shift(-1)

        # Find the first break date where both conditions are met
        first_break_date = (
            rolling_diff_primary[consecutive_mask].index.min()
            if not rolling_diff_primary[consecutive_mask].empty
            else None
        )

        # Determine the preceding and following dates if a break was found
        previous_date = None
        after_date = None
        if first_break_date is not None:
            try:
                pos = df.index.get_loc(first_break_date)
                # get_loc may return a slice or integer; handle integer positions
                if isinstance(pos, slice):
                    pos = pos.start if pos.start is not None else 0
                if pos > 0:
                    previous_date = df.index[pos - 1]
                    after_date = df.index[pos + 1] if pos + 1 < len(df) else None
            except Exception:
                previous_date = None
                after_date = None

        return first_break_date, previous_date, after_date

    def calculate_break(self, dataset: LakeDataset, object_id: str) -> pd.DataFrame:
        """Calculate breakpoints for a single lake object.

        Parameters
        ----------
        dataset : LakeDataset
            Dataset containing lake water‑area data.
        object_id : str
            Unique identifier (geohash) for the lake object.

        Returns
        -------
        pd.DataFrame
            DataFrame containing breakpoint information with columns defined in
            ``self.breakpoint_columns`` plus calculated temporal statistics.
        """
        # dataset._normalize_ds()
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
        # TODO: can this be optimized to process the entire dataframe?
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
        # calculate additional stats
        break_df = calculate_temporal_stats(break_df)

        return break_df


class BeastBreakpoint(BreakpointMethod):
    """Bayesian RBEAST-based breakpoint detector.

    This method uses the RBEAST library to detect breakpoints in water‑timeseries
    data using Bayesian change‑point detection. It identifies points where the
    statistical properties of the time series change significantly.

    Parameters
    ----------
    kwargs_break : dict, optional
        Configuration dictionary for RBEAST priors. Common keys include:
        - ``trendMaxOrder`` : int, default 0
            Maximum order of the trend component.
        - ``trendMinSepDist`` : int, default 1
            Minimum separation distance between change points.
    break_threshold : float, optional
        Probability threshold for detecting a break point. Default is 0.5.

    Attributes
    ----------
    breakpoint_columns : list
        List of column names in the output DataFrame.
    """

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

    def calculate_break(self, dataset: LakeDataset, object_id: str) -> pd.DataFrame:
        """Calculate breakpoints for a single lake object using RBEAST.

        Parameters
        ----------
        dataset : LakeDataset
            Dataset containing lake water‑area data.
        object_id : str
            Unique identifier (geohash) for the lake object.

        Returns
        -------
        pd.DataFrame
            DataFrame containing breakpoint information with columns defined in
            ``self.breakpoint_columns`` plus calculated temporal statistics.
        """
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

        if break_indices.size == 0:
            return pd.DataFrame(columns=self.breakpoint_columns)

        # get previous date
        break_indices_before = np.array(break_indices) - 1
        # get after date
        break_indices_after = np.array(break_indices) + 1
        # return df
        break_dates_before = df.iloc[break_indices_before]["date"].to_list()
        break_dates_after = df.iloc[break_indices_after]["date"].to_list()

        # ensure we're working with copies to avoid pandas SettingWithCopyWarning
        df = df.copy()
        df["proba_rbeast"] = cp_prob
        # print(break_indices)
        break_df = df.iloc[break_indices].copy()

        # safely add the previous-date column
        break_df.loc[:, "date_before_break"] = break_dates_before
        break_df.loc[:, "date_after_break"] = break_dates_after

        # sort by probability descending, then add sequential break numbers
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
    """Near‑Real‑Time (NRT) breakpoint detector.

    This method implements custom logic for detecting breakpoints in water‑timeseries
    data. It follows the same interface as other breakpoint methods but uses internal
    logic that is distinct from the SimpleBreakpoint and BeastBreakpoint classes.

    The NRT method uses AutoARIMA to predict the expected water extent and compares
    it against the observed value. It also calculates historical statistics and
    assigns a drainage confidence level based on three criteria.

    Parameters
    ----------
    kwargs_break : dict, optional
        Configuration dictionary for NRT-specific parameters. Default is an empty dict.

    Attributes
    ----------
    breakpoint_columns : list
        List of column names in the output DataFrame.
    output_columns : list
        List of column names in the output DataFrame, including normalized values
        (0-1 scale) and their absolute equivalents (scaled by max area).
    output_columns_base : list
        Subset of output columns for handling NaN entries.

    Notes
    -----
    The output includes both normalized values (0-1 range) and absolute values.
    Absolute values are computed by multiplying normalized values with the scaling
    factor (max area per id_geohash): ``absolute = normalized * max_area_data``.

    Examples
    --------
    >>> from water_timeseries.breakpoint import NRTBreakpoint
    >>> from water_timeseries.dataset import DWDataset
    >>> bp = NRTBreakpoint()
    >>> dataset = DWDataset(xr.open_dataset("data.zarr"))
    >>> result = bp.calculate_break(dataset, analysis_date="2024-07")
    """

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
            # absolute values
            "water_observed_absolute",
            "water_predicted_absolute",
            "water_residual_absolute",
            "water_predicted_lower_90_absolute",
            "water_predicted_upper_90_absolute",
            "water_historical_mean_absolute",
            "water_historical_median_absolute",
            "water_historical_std_absolute",
            "water_historical_min_absolute",
            "water_historical_max_absolute",
        ]
        self.output_columns_base = [
            "date",
            "water_observed",
            "water_predicted",
            "water_residual",
            "water_predicted_lower_90",
            "water_predicted_upper_90",
            "drainage_confidence",
            # absolute values
            "water_observed_absolute",
            "water_predicted_absolute",
            "water_residual_absolute",
            "water_predicted_lower_90_absolute",
            "water_predicted_upper_90_absolute",
        ]

    def predict_nrt_arima(
        self, ds_in: xr.Dataset, id_geohash: str, min_length: int = 3, water_column: str = "water"
    ) -> pd.Series:
        """_summary_

        Args:
            ds_in (xr.Dataset): _description_
            id_geohash (str): _description_
            min_length (int): Minimum length of the time series.
            water_column (str): Name of the water column in the dataset.

        Returns:
            pd.Series: _description_
        """

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

        # Step 3: Fit model
        model = AutoARIMA(
            stepwise=True,
            suppress_warnings=True,
            trace=False,
            error_action="ignore",
            seasonal=False,
        )

        # TODO. catch wild warnings to declutter console
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Fit
            model.fit(df_in)

            # Predict
            fh = ForecastingHorizon([1], is_relative=True)
            y_pred = model.predict(fh=fh)
            y_pred_int = model.predict_interval(fh=fh, coverage=0.90)

            result = pd.Series(
                {
                    "water_predicted": y_pred.iloc[0],
                    "water_predicted_lower_90": y_pred_int.iloc[0, 0],  # first row, first interval column
                    "water_predicted_upper_90": y_pred_int.iloc[0, 1],  # first row, second interval column
                },
                name=id_geohash,
            )

            return result

    def _validate_analysis_date(self, analysis_date: str | pd.Timestamp) -> pd.Timestamp:
        """Validate and format analysis_date to datetime object.

        Parameters
        ----------
        analysis_date : str or pd.Timestamp
            The date to validate and format. Can be a string in %Y-%m format or a datetime object.

        Returns
        -------
        pd.Timestamp
            Formatted datetime object.
        """
        if isinstance(analysis_date, str):
            try:
                analysis_date_ts = pd.to_datetime(analysis_date, format="%Y-%m")
                return analysis_date_ts
            except (ValueError, TypeError):
                analysis_date_ts = pd.to_datetime(analysis_date)
                return analysis_date_ts
        elif isinstance(analysis_date, pd.Timestamp):
            return analysis_date
        else:
            return pd.to_datetime(analysis_date)

    def _filter_valid_ids(self, ds_analysis: xr.Dataset, ds_historical: xr.Dataset) -> tuple:
        """Filter datasets to only include valid id_geohash values with non-NaN data.

        Parameters
        ----------
        ds_analysis : xr.Dataset
            Dataset containing analysis date data.
        ds_historical : xr.Dataset
            Dataset containing historical data.

        Returns
        -------
        tuple
            Filtered (ds_analysis, ds_historical) datasets.
        """
        # Get valid id_geohash values that have non-NaN data in ds_analysis
        valid_ids = ds_analysis.dropna(dim="id_geohash", how="all").id_geohash.values

        # get invalid ids with nan values
        all_ids = ds_analysis.id_geohash.values
        nan_ids = all_ids[~pd.Series(all_ids).isin(pd.Series(valid_ids))]

        # Count total ids and valid ids for logging
        total_ids = len(all_ids)
        valid_count = len(valid_ids)
        filtered_count = total_ids - valid_count

        # Log the filtering results
        logging.info(f"Filtered {filtered_count} id_geohash(es) with NaN data, kept {valid_count} valid id_geohash(es)")

        # Filter both datasets to only include valid ids
        ds_analysis_filtered = ds_analysis.sel(id_geohash=valid_ids)
        ds_historical_filtered = ds_historical.sel(id_geohash=valid_ids)

        return ds_analysis_filtered, ds_historical_filtered, valid_ids, nan_ids

    def _get_ds_stats(self, dataset: xr.Dataset, filter_month: int = None, water_column: str = "water") -> pd.DataFrame:
        """Calculate statistics for the given dataset."""
        if filter_month is not None:
            dataset = dataset.where(dataset.date.dt.month == filter_month, drop=True)
        out_df = dataset.to_dataframe()[water_column].groupby("id_geohash").agg(["mean", "median", "std", "min", "max"])
        return out_df

    def _add_confidence_level(self, break_output_df: pd.DataFrame) -> pd.DataFrame:
        """Add a drainage confidence level to the breakpoint output DataFrame.

        The confidence level is computed by evaluating three criteria that
        indicate abnormal water drainage:

        * **Cat 1** – Residual below threshold: ``water_residual < -0.25``
        * **Cat 2** – Observed water below prediction interval:
        ``water_observed < water_predicted_lower_90``
        * **Cat 3** – Observed water below historical minimum:
        ``water_observed < water_historical_min``

        Each satisfied criterion contributes 1 to the score. The final
        ``drainage_confidence`` column contains:

        | Score | Meaning   |
        |-------|-----------|
        | 1     | Low       |
        | 2     | Medium    |
        | 3     | High      |

        Parameters
        ----------
        break_output_df : pd.DataFrame
            DataFrame with at least the following columns: ``water_residual``,
            ``water_observed``, ``water_predicted_lower_90``, and
            ``water_historical_min``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with an additional ``drainage_confidence`` column.
        """

        cat1 = break_output_df["water_residual"] < -0.25  # observed water area min. 25 less than expected
        cat2 = (
            break_output_df["water_observed"] < break_output_df["water_predicted_lower_90"]
        )  # water area less than lower 90% confidence
        cat3 = (
            break_output_df["water_observed"] < break_output_df["water_historical_min"]
        )  # minimum observed water extent ever

        # sum all 3 criteria and output confidence (1:low, 2: medium, 3: high)
        drain_confidence = pd.concat([cat1, cat2, cat3], axis=1).sum(axis=1)
        break_output_df["drainage_confidence"] = drain_confidence

        # force int dtype
        break_output_df["drainage_confidence"] = break_output_df["drainage_confidence"].astype(int)

        return break_output_df

    def calculate_break(
        self,
        dataset: LakeDataset,
        analysis_date: str | pd.Timestamp,
        data_aggregation_period: str = "all",
        object_id: str | Optional[str] = None,
        keep_nans: bool | Optional[bool] = False,
    ) -> pd.DataFrame:
        """Calculate breakpoints for a single lake object using NRT logic.

        This method implements the NRT-specific breakpoint detection and returns
        a DataFrame with breakpoint information following the same structure as
        other breakpoint methods.

        Parameters
        ----------
        dataset : LakeDataset
            Dataset containing lake water‑area data.
        object_id : str | Optional[str]
            Unique identifier (geohash) for the lake object.
        analysis_date : str or pd.Timestamp
            The date for which to perform the NRT breakpoint analysis.
        data_aggregation_period : str, optional
            The period of data to consider for the analysis (e.g., "all", "monthly")
        process_nans : bool, optional
            Set True if you want to return historical water stats
        Returns
        -------
        pd.DataFrame
            DataFrame containing breakpoint information with columns defined in
            ``self.breakpoint_columns`` plus calculated temporal statistics.
        """

        analysis_date = self._validate_analysis_date(analysis_date)
        print(analysis_date)
        print(analysis_date.strftime("%Y-%m"))

        # Check if analysis_date in dataset.dates_ (convert to YYYY-MM format for comparison)
        if analysis_date not in dataset.dates_:
            raise ValueError(f"Analysis date {analysis_date.strftime('%Y-%m')} is not available in the dataset.")

        # select dataset - default normalized data
        data = dataset.ds_normalized

        if object_id is not None:
            if isinstance(object_id, str):
                object_id = [object_id]
            object_id = [obj for obj in object_id if obj in dataset.object_ids_]
            data = data.sel(id_geohash=object_id)

        # split data into historical and analysis datasets based on analysis_date
        ds_analysis = data.sel(date=analysis_date)
        ds_historical = data.where(data["date"] < analysis_date, drop=True)

        if data_aggregation_period == "monthly":
            print("Filtering to monthly data for analysis date month:", analysis_date.month)
            ds_historical = ds_historical.where(ds_historical.date.dt.month == analysis_date.month, drop=True)

        # filter to dates where analysis date has some data
        ds_analysis_filtered, ds_historical_filtered, valid_ids, nan_ids = self._filter_valid_ids(
            ds_analysis, ds_historical
        )

        if len(valid_ids) == 0:
            if keep_nans:
                return pd.DataFrame(index=nan_ids, columns=self.output_columns)
            else:
                return pd.DataFrame(columns=self.output_columns)

        # loop over each lake and predict next value using ARIMA, then compare to observed value in ds_analysis_filtered
        # predictions = [self.predict_nrt_arima(ds_in=ds_historical_filtered, id_geohash=idx) for idx in tqdm(valid_ids, desc='NRT breakpoints')]
        cpu_count = os.cpu_count() or 1
        n_jobs = max(1, min(cpu_count, len(valid_ids)))
        predictions = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self.predict_nrt_arima)(
                ds_in=ds_historical_filtered, id_geohash=idx, water_column=dataset.water_column
            )
            for idx in tqdm(valid_ids, desc="NRT breakpoints")
        )
        # remove None values
        # if not process_nans:
        predictions = [prediction for prediction in predictions if prediction is not None]
        prediction_df = pd.DataFrame(predictions)
        if prediction_df.empty:
            prediction_df = pd.DataFrame(
                index=ds_analysis_filtered.id_geohash.values,
                columns=self.output_columns,
            )

        # merge output into a single dataframe
        df_output = ds_analysis_filtered[dataset.water_column].to_dataframe().join(prediction_df).round(4)
        # rename observed water column for clarity
        df_output.rename(columns={dataset.water_column: "water_observed"}, inplace=True)

        # calculate residuals
        df_output["water_residual"] = df_output["water_observed"] - df_output["water_predicted"]

        df_historical_stats = self._get_ds_stats(ds_historical_filtered, water_column=dataset.water_column).round(4)
        df_historical_stats.columns = "water_historical_" + df_historical_stats.columns.astype(str)

        df_output = df_output.join(df_historical_stats, how="left").round(4)

        # add confidence level to output
        df_output = self._add_confidence_level(df_output)

        # compute absolute values by multiplying normalized values with scaling factor
        # scaling factor is the max area per id_geohash: ds.max(dim="date")["area_data"]
        scaling_factors = dataset.ds.max(dim="date")["area_data"].to_dataframe()
        scaling_factors = scaling_factors.loc[df_output.index]

        # columns to convert to absolute values
        water_cols_absolute = [
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
        for col in water_cols_absolute:
            if col in df_output.columns:
                df_output[f"{col}_absolute"] = df_output[col] * scaling_factors["area_data"]

        # if keep_nans is selected: calculate historical stats for these and append to calculated data
        if keep_nans:
            # base columns only (without absolute values, as predictions are not available)
            base_columns_no_abs = [col for col in self.output_columns_base if not col.endswith("_absolute")]
            prediction_df_nan = pd.DataFrame(
                index=nan_ids,
                columns=base_columns_no_abs,
            )
            df_historical_stats_nans = self._get_ds_stats(
                ds_historical.sel(id_geohash=nan_ids), water_column=dataset.water_column
            ).round(4)
            df_historical_stats_nans.columns = "water_historical_" + df_historical_stats_nans.columns.astype(str)
            df_output_nan = prediction_df_nan.join(df_historical_stats_nans, how="left").round(4)

            # compute absolute values for nan entries
            scaling_factors_nan = dataset.ds.max(dim="date")["area_data"].to_dataframe()
            scaling_factors_nan = scaling_factors_nan.loc[df_output_nan.index]
            for col in water_cols_absolute:
                if col in df_output_nan.columns:
                    df_output_nan[f"{col}_absolute"] = df_output_nan[col] * scaling_factors_nan["area_data"]

            df_output = pd.concat([df_output, df_output_nan]).sort_index()

        return df_output[self.output_columns]
