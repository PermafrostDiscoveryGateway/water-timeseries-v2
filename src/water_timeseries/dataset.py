"""Dataset processing classes for water timeseries analysis.

This module provides classes for processing and normalizing satellite-derived
land cover and water classification data. It includes specialized handlers for
different data sources and processing pipelines.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr

from water_timeseries.utils.data import dw_bandnames, jrc_bandnames
from water_timeseries.utils.earthengine import create_timelapse
from water_timeseries.utils.plotting import (
    plot_water_time_series_dw,
    plot_water_time_series_jrc,
    prepare_data_for_plot_dw,
)
from water_timeseries.utils.plotting_dynamic import (
    plot_water_time_series_dw_interactive,
    plot_water_time_series_jrc_interactive,
)

if TYPE_CHECKING:
    from water_timeseries.breakpoint import BreakpointMethod


__all__ = ["DWDataset", "JRCDataset", "LakeDataset"]


# =============================================================================
# LakeDataset (base class) — handles dataset-level loading and validation
# =============================================================================


def _ensure_id_field_exists(ds: xr.Dataset, id_field: str, name: str) -> None:
    """Validate that the identifier field exists in the input Dataset.

    Args:
        ds: Input xarray Dataset to search.
        id_field: The name of the coordinate/dimension/data variable to look for.
        name: Name of the class performing the check (used in error messages).

    Raises:
        ValueError: If no coordinate, dimension, or data variable matches `id_field`.
    """
    if id_field not in ds.coords and id_field not in ds.dims and id_field not in ds.data_vars:
        available_dims = list(ds.dims)
        available_coords = list(ds.coords)
        available_vars = list(ds.data_vars)
        raise ValueError(
            f"{name} does not have a coordinate/dimension/data variable '{id_field}' "
            f"in the input dataset. "
            f"Available dimensions: {available_dims!r}, "
            f"coordinates: {available_coords!r}, "
            f"data variables: {available_vars!r}.",
        )


class LakeDataset:
    """Base class for processing lake and water body datasets.

    Handles common operations for dataset preprocessing, normalization, and masking.
    Provides a framework that can be extended for different data sources.

    Attributes:
        ds: The input xarray Dataset containing raw data.
        ds_normalized: Normalized version of the dataset (0-1 scale).
        preprocessed_: Whether preprocessing has been completed.
        normalized_available_: Whether normalized data is available.
        water_column: Name of the water/water extent column.
        data_columns: Names of all data columns in the dataset.
        ds_ismasked_: Whether the original dataset has been masked.
        ds_normalized_ismasked_: Whether the normalized dataset has been masked.
        id_field: The name of the coordinate/dimension that holds lake identifiers.
            Defaults to "id_geohash" in subclasses.
        mask_data: Whether invalid data should be masked (default: False).

    Example:
        >>> lake_data = LakeDataset(xr.Dataset(...))
        >>> normalized = lake_data.ds_normalized
    """

    def __init__(
        self,
        ds: xr.Dataset,
        id_field: str = "id_geohash",
        mask_data: bool = True,
    ) -> None:
        """Initialize the LakeDataset.

        Args:
            ds: Input xarray Dataset with land cover or water classification data.
                If `ds` is a DataArray, it will be converted to a Dataset via
                `ds.to_dataset()`. Pass an existing (in-memory or on-disk) Dataset
                object instead of a lazy file path to avoid opening and scanning the
                NetCDF file during initialization.
            id_field: The field name for the lake identifier coordinate dimension.
                Defaults to "id_geohash". Override with a different string if your
                exported dataset uses another column name (e.g. "lake_id", "object_id").
            mask_data: Whether invalid data should be masked (default: True).

        Raises:
            TypeError: If `ds` is not an xr.Dataset, xr.DataArray, or pathlib.Path.
            ValueError: If the identifier field `id_field` does not exist in the dataset.

        Notes:
            - The identifier field is validated lazily on first access to `object_ids_`,
              which means instantiation cost scales with the number of unique lakes.
              Pass an in-memory Dataset object instead of a file path to avoid this
              cost when you know the identifier exists ahead of time.
        """
        self.ds: xr.Dataset

        if isinstance(ds, xr.Dataset):
            self.ds = ds
        elif isinstance(ds, xr.DataArray):
            self.ds = ds.to_dataset()
        elif isinstance(ds, Path):
            self.ds = xr.open_dataset(str(ds))
        else:
            raise TypeError(f"Invalid input dataset type: {type(ds)}.")

        self._id_field_validated_: bool = False
        self.preprocessed_: bool = False
        self.normalized_available_: bool = False
        self.water_column: str | None = None
        self.data_columns: list[str] | None = None
        self.ds_ismasked_: bool = False
        self.ds_normalized_ismasked_: bool = False
        self.mask_data: bool = mask_data
        self.id_field: str = id_field
        self._preprocess()
        self._normalize_ds()
        self._mask_invalid()

    def _resolve_object_id(self, object_id, id_geohash=None) -> str | int:
        """Resolve the object identifier from the new `object_id` kwarg.

        Accepts either the new `object_id` keyword or the legacy `id_geohash`
        keyword (for backward compatibility). Both forms point to the same value:
        an entry from :attr:`object_ids_` — a coordinate value on the dataset's
        identifier dimension (whatever name `self.id_field` happens to carry).

        Only one of the two keywords may be provided. When `id_geohash` is used
        a :class:`FutureWarning` is emitted to encourage migration to
        `object_id`, which is stable regardless of the column name.

        Args:
            object_id: value passed via the new keyword (may be ``None``).
            id_geohash: value passed via the legacy keyword (may be ``None``).

        Returns:
            The resolved object identifier (str or int, matching the caller's value).

        Raises:
            TypeError: If both `object_id` and `id_geohash` are provided.
            ValueError: If neither is provided (the identifier is required).
        """
        if object_id is not None and id_geohash is not None:
            raise TypeError(
                "Please provide exactly one of `object_id` or `id_geohash`. "
                "`object_id` is the preferred, forward-compatible name."
            )
        if id_geohash is not None:
            warnings.warn(
                "The `id_geohash` keyword is deprecated. Please use `object_id` instead. "
                "`id_geohash` will be removed in a future release.",
                FutureWarning,
                stacklevel=3,
            )
            return id_geohash
        if object_id is None:
            raise ValueError("At least one of `object_id` or `id_geohash` must be provided.")
        return object_id

    def _check_id_field(self) -> None:
        """Check that the identifier field exists in this instance's ds.

        Validates only once per instance to avoid repeated checks across calls.
        """
        if not self._id_field_validated_:
            _ensure_id_field_exists(
                ds=self.ds,
                id_field=self.id_field,
                name=type(self).__name__,
            )
            self._id_field_validated_ = True

    @property
    def object_ids_(self) -> list[str]:
        """Get all valid lake/object IDs from the dataset.

        Validates that `id_field` exists on first access, then simply returns the
        coordinate values (which are guaranteed to exist by the validation above).

        Returns:
            A list of coordinate scalar values from the identifier dimension. Each
            value is a string (the "lake name" / geohash code) but may also be
            a timestamp or numeric string depending on how the dataset was exported.
        """
        self._check_id_field()
        return list(self.ds.coords[self.id_field].values)

    @property
    def dates_(self) -> list[str | pd.Timestamp]:
        """Get all valid dates from the dataset.

        Returns:
            A list of coordinate scalar values from the 'date' dimension. Each value
            is a string (e.g., "2018-04-26" or any other date format used on-disk)
            that can be parsed by `pd.to_datetime`.
        """
        return list(self.ds.coords["date"].values)

    def _preprocess(self):
        """Preprocess the dataset.

        This method should be overridden in subclasses to implement data-source-specific
        preprocessing steps such as calculating composite indicators or adding derived fields.
        """

    def _normalize_ds(self) -> None:
        """Normalize the dataset by dividing by maximum values.

        Scales all data to 0-1 range based on the maximum area value per time series,
        ensuring comparability across different spatial extents. Normalized values are
        non-negative; an NA value is set where water was already absent in all available
        observations (all zeros before normalization). Negative normalized values would
        indicate invalid data in this dataset.

        Raises:
            RuntimeError: If called before `id_field` has been validated.
        """
        self._check_id_field()

        self.ds_normalized = self.ds.copy()
        # Compute global max of area_data per object, then divide
        ds_max = self.ds.max(dim="date", skipna=True)
        self.ds_normalized = self.ds / ds_max["area_data"]
        self.normalized_available_ = True

    def _mask_invalid(self) -> None:
        """Mask invalid data based on quality criteria.

        This method should be overridden in subclasses to implement data-source-specific
        masking logic based on their quality thresholds and constraints.
        """

    def _first_breakpoint_date_from_any_source(
        self,
        object_id: str,
        breakpoints: BreakpointMethod | pd.Timestamp | str | list[pd.Timestamp] | list[str] | None,
    ) -> pd.Timestamp | None:
        """Extract the first breakpoint date from any supplied source.

        Returns a single `pd.Timestamp` or None if no valid breakpoint is found.

        Args:
            object_id: The object identifier used as the search key for BreakpointMethod objects.
            breakpoints: A BreakpointMethod instance, a single date, a list of dates,
                or None.

        Returns:
            pd.Timestamp or None: The first valid breakpoint date found, or None if
                no breakpoint is detected and no date was explicitly provided.
        """
        if breakpoints is None:
            return None

        # Check first: does this instance have a BreakpointMethod attached?  If it does,
        # use that (it carries the object_id via its dataset metadata) rather than
        # falling back to a per-object computation based solely on the raw timeseries.
        if self._is_breakpoint_method(breakpoints):
            breaks = breakpoints.calculate_break(self, object_id=object_id)
            if breaks is not None and len(breaks) > 0:
                return pd.Timestamp(breaks["date_break"].iloc[0])

        # Fall back to parsing the date string(s) as provided.
        return self._get_first_breakpoint(object_id, breakpoints)

    @staticmethod
    def _is_breakpoint_method(obj: Any) -> bool:
        """Check if an object is a BreakpointMethod instance or subclass.

        Notes:
            This method avoids importing `BreakpointMethod` to avoid circular
            dependencies since the module hierarchy includes a separate folder for
            the breakpoint detection subpackage (e.g., `pybreakpoints`).  If you need
            to check for an arbitrary subclass, use `inspect.getmro(obj)`.
        """
        bp_class_name = "BreakpointMethod"
        current_class = obj.__class__

        while current_class is not None:
            if current_class.__name__ == bp_class_name:
                return True
            current_class = current_class.__bases__[0] if current_class.__bases__ else None

        return False

    def _get_first_breakpoint(
        self,
        object_id: str,
        breakpoints: pd.Timestamp | str | list[pd.Timestamp] | list[str] | None,
    ) -> pd.Timestamp | None:
        """Extract the first breakpoint date from various input types.

        Args:
            object_id: The object identifier (used only in error messages).
            breakpoints: A single date string or Timestamp, a list of such dates,
                or None.

        Returns:
            pd.Timestamp or None: The first valid breakpoint date, or None if none
                could be parsed from `breakpoints`.

        Raises:
            ValueError: If `breakpoints` is not a string, list of strings,
                Timestamp, list of Timestamps, or None.
        """
        if breakpoints is None:
            return None

        dates = breakpoints if isinstance(breakpoints, list) else [breakpoints]

        for date in dates:
            if isinstance(date, str):
                return pd.Timestamp(date)
            elif isinstance(date, pd.Timestamp):
                return date

        raise ValueError(
            f"Unable to parse the following breakpoint date(s) as a string or Timestamp:\n"
            f"{breakpoints!r}. Acceptable formats include a single date string "
            f"(e.g. '2018-04-26'), a pd.Timestamp, a list of date strings, or "
            f"an instance of BreakpointMethod which will be detected automatically."
        )

    # =============================================================================
    # Merge (shared between DWDataset and JRCDataset)
    # =============================================================================

    def merge(
        self,
        other: LakeDataset,
        how: str = "both",
    ) -> LakeDataset:
        """Merge this LakeDataset with another LakeDataset.

        Combines the .ds attributes of both datasets. Both datasets must have the same
        variables. The merge strategy is determined by the `how` parameter.
        Both datasets must be of the same type (e.g., both DWDataset or both JRCDataset).

        Args:
            other (LakeDataset): Another LakeDataset instance to merge with.
            how (str): Merge strategy. Options:
                - "both": Merge along both dimensions (date and id). Combines all
                  data from both datasets, keeping all unique dates and ids.
                - "date": Merge along the "date" dimension only. Both datasets must have
                  the same id values, but can have different dates. New dates are
                  appended to the existing time series.
                - "id_geohash": Merge along the id dimension only. Both datasets
                  must have the same dates, but can have different ids. New
                  ids (lakes) are added with their time series.

        Returns:
            LakeDataset: A new LakeDataset with merged .ds data.

        Raises:
            TypeError: If the datasets are of different types.
            ValueError: If the merge strategy is invalid or datasets are incompatible.

        Example:
            >>> merged = dataset1.merge(dataset2, how="both")
            >>> merged = dataset1.merge(dataset2, how="date")  # Add new dates
            >>> merged = dataset1.merge(dataset2, how="id_geohash")  # Add new lakes
        """
        self._validate_merge(other, how)

        if how == "both":
            merged_ds = self._merge_both(self.ds, other.ds)
        elif how == "date":
            merged_ds = self._merge_by_date(self.ds, other.ds)
        else:  # how == "id_geohash"
            merged_ds = self._merge_by_id(self.ds, other.ds)

        merged = self.__class__(merged_ds, id_field=self.id_field)  # type: ignore[call-arg]
        merged.mask_data = self.mask_data
        return merged

    def _validate_merge(self, other: LakeDataset, how: str) -> None:
        """Validate datasets before merging."""

        if how not in {"both", "date", "id_geohash"}:
            raise ValueError(f"Invalid merge strategy '{how}'. Must be 'both', 'date', or 'id_geohash'.")

        if type(self) is not type(other):
            raise TypeError(
                f"Cannot merge {type(self).__name__} with {type(other).__name__}. "
                f"Both datasets must be of the same type; different types cannot be merged."
            )

        if set(self.ds.data_vars) != set(other.ds.data_vars):
            raise ValueError("Datasets have different variables.")

    def _merge_both(self, ds1: xr.Dataset, ds2: xr.Dataset) -> xr.Dataset:
        """Merge along both dimensions."""

        return xr.merge([ds1, ds2])

    def _merge_by_date(self, ds1: xr.Dataset, ds2: xr.Dataset) -> xr.Dataset:
        """Merge along date dimension (same ids, new dates)."""

        def _ids(d: xr.Dataset) -> set:
            return set(np.atleast_1d(d.coords[self.id_field].values).astype(str))

        if _ids(ds1) != _ids(ds2):
            raise ValueError(f"For merge how='date', both datasets must have the same {self.id_field} values.")

        # Check for duplicate dates
        dates1 = set(ds1.coords["date"].values)
        dates2 = set(ds2.coords["date"].values)
        duplicate_dates = dates1 & dates2
        if duplicate_dates:
            warnings.warn(
                f"Datasets have {len(duplicate_dates)} overlapping dates. "
                f"Data from the second dataset will overwrite the first for these dates.",
                UserWarning,
            )

        merged = xr.concat([ds1, ds2], dim="date")
        return merged.sortby("date")

    def _merge_by_id(self, ds1: xr.Dataset, ds2: xr.Dataset) -> xr.Dataset:
        """Merge along id dimension (same dates, new ids)."""

        if set(ds1.coords["date"].values) != set(ds2.coords["date"].values):
            raise ValueError("For merge how='id_geohash', both datasets must have the same dates.")

        # Check for duplicate ids
        ids1 = set(np.atleast_1d(ds1.coords[self.id_field].values).astype(str))
        ids2 = set(np.atleast_1d(ds2.coords[self.id_field].values).astype(str))
        duplicate_ids = ids1 & ids2
        if duplicate_ids:
            warnings.warn(
                f"Datasets have {len(duplicate_ids)} overlapping {self.id_field} values. "
                f"Data from the second dataset will overwrite the first for these values.",
                UserWarning,
            )

        return xr.concat([ds1, ds2], dim=self.id_field)

    # =============================================================================
    # DWDataset subclass
    # =============================================================================


class DWDataset(LakeDataset):
    """Handler for Dynamic World land cover classification data.

    Processes Dynamic World land cover classes including water, bare soil, snow/ice,
    trees, grass, flooded vegetation, crops, shrub/scrub, and built-up areas.

    Attributes:
        water_column (str): Fixed as "water" for DW data.
        data_columns (list): All 9 DW land cover class variables.
            ['water', 'bare', 'snow_and_ice', 'trees', 'grass',
             'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built'].

    Example:
        >>> dw_data = DWDataset(xr.open_dataset("dynamic_world.nc"))
        >>> water_time_series = dw_data.ds_normalized["water"]
        >>> print(dw_data.data_columns)
        ['water', 'bare', 'snow_and_ice', 'trees', 'grass', ...]
    """

    def __init__(
        self,
        ds: xr.Dataset,
        id_field: str = "id_geohash",
        mask_data: bool = True,
    ) -> None:
        """Initialize DWDataset with Dynamic World data.

        Args:
            ds: Input xarray Dataset with at least the 9 DW class variables.
                If `ds` is a DataArray, convert it first via `ds.to_dataset()`.
                Prefer passing an in-memory or already-opened Dataset over a file path
                to avoid opening and scanning a large NetCDF file during initialization.
            id_field: The field name for the lake identifier coordinate dimension.
                Defaults to "id_geohash". Override with another string if your exported
                dataset uses a different column name (e.g., "lake_id", "object_id").
            mask_data: Whether invalid data should be masked during preprocessing.
                Default is True; set False to disable masking.

        Raises:
            TypeError: If `ds` is not an xr.Dataset, xr.DataArray, or pathlib.Path.
            ValueError: If `id_field` does not exist in the dataset and is not a valid
                       Dynamic World variable name.

        Notes:
            The required 9 DW variables are: water, bare, snow_and_ice, trees,
            grass, flooded_vegetation, crops, shrub_and_scrub, and built. If your
            exported dataset omits any of these, normalization will fail on the missing
            variable (e.g., if it is not found as a coordinate/dimension/data variable).

        See Also:
            JRCDataset: for JRC water classification data handling using lake_id by default.
        """
        super().__init__(ds, id_field=id_field, mask_data=mask_data)
        self.water_column = "water"
        self.data_columns = dw_bandnames

    # ---- Preprocessing & normalization ----

    def _preprocess(self) -> None:
        """Preprocess Dynamic World data.

        Calculates total area as the sum of all land cover classes and computes
        the no-data area as the difference from maximum area across time.

        Notes:
            - If a required band is missing (e.g., a coordinate named "bare"
              does not exist), a NameError is raised before normalization can proceed.
            - For masked datasets, use the unmasked `.ds` attribute instead of
              `.ds_normalized`.
        """
        super()._preprocess()

        ds = self.ds

        # Aggregate all DW land cover classes into "area_data", so that the
        # normalized sum is guaranteed to be <= 1 (since "no data" is excluded
        # from the per-object maximum used for normalization):
        ds["area_data"] = (
            ds["bare"]
            + ds["water"]
            + ds["snow_and_ice"]
            + ds["trees"]
            + ds["grass"]
            + ds["flooded_vegetation"]
            + ds["crops"]
            + ds["shrub_and_scrub"]
            + ds["built"]
        )

        # Compute no-data area per time series; round to 4 decimal places for
        # comparability:
        max_area = ds["area_data"].max(dim="date", skipna=True)
        ds["area_nodata"] = (max_area - ds["area_data"]).round(4)

        self.preprocessed_ = True
        self.ds = ds

    def _mask_invalid(self) -> None:
        """Mask invalid data based on quality criteria.

        Removes observations where:
          - The no-data area exceeds 0 (i.e., all DW classes have zero area), or
          - Snow and ice coverage exceeds 5% (indicating a poor classification).

        After masking, both `.ds` and `.ds_normalized` represent in-memory subsets of the
        original NetCDF file, so that any later access to the masked coordinates is O(1).
        """

        if self.mask_data is False:
            return

        ds = self.ds_normalized
        mask_nodata = ds["area_nodata"] <= 0
        mask_snow = ds["snow_and_ice"] <= 0.05
        mask = mask_nodata & mask_snow

        self.ds = self.ds.where(mask)
        self.ds_normalized = self.ds_normalized.where(mask)

        self.ds_ismasked_ = True
        self.ds_normalized_ismasked_ = True

    # ---- Plotting ----

    def plot_timeseries(
        self,
        object_id: str | int | None = None,
        breakpoints: BreakpointMethod | pd.Timestamp | str | list[pd.Timestamp] | list[str] | None = None,
        plot_variables: list[str | None] | None = None,
        save_path: str | Path | None = None,
        *,
        id_geohash: str | int | None = None,
    ) -> plt.Figure:
        """Plot the time series for a specific lake using matplotlib.

        Creates a static matplotlib figure showing the Dynamic World land cover
        time series for a single lake, with an optional vertical line indicating
        a breakpoint or specific date. The figure includes a 'Breakpoint' entry
        in the legend when a breakpoint is provided.

        Args:
            object_id: The lake identifier as stored in the dataset's
                identifier dimension. Use a value from :attr:`object_ids_`.
            breakpoints: Breakpoint detection method, single date, or list of dates.
                If a BreakpointMethod object is passed, its first detected breakpoint
                date is used. A string or Timestamp is treated as the target date(s).
                If None, no vertical line is drawn.
            plot_variables: Which variables to show. Acceptable choices are
                'water', 'bare', 'vegetation'. If None (default), all variables
                are shown. A string name maps to a list via `self.data_columns`.
            save_path: Optional path to save the image as PNG/PDF/SVG.
            id_geohash: DEPRECATED alias for `object_id`. Use `object_id` instead;
                `id_geohash` is emitted as a :class:`FutureWarning`.

        Returns:
            matplotlib.Figure with a single axes and a legend. To access the axes:
                fig = dataset.plot_timeseries("1132035748")
                ax = fig.axes[0]

            The x-axis is ordered chronologically; the y-axis is in units of the
            area data variable (typically "area_data"). To convert to a 0–1 fraction
            by area, multiply each value by `df["area_data"].max()`.

        Notes:
            - `plot_variables` can be a single string or a list of strings.
            - Backward compatibility: `id_geohash=...` is still accepted and
              triggers a deprecation warning.

        See Also:
            plot_timeseries_interactive: For an interactive Plotly-based version.
        """
        object_id = self._resolve_object_id(object_id=object_id, id_geohash=id_geohash)
        df = (self.ds.sel({self.id_field: object_id}).load().to_dataframe()).dropna()
        df_plot = prepare_data_for_plot_dw(df, group_vegetation=True)

        bp = self._first_breakpoint_date_from_any_source(object_id, breakpoints)

        figure = plot_water_time_series_dw(  # type: ignore
            df=df_plot,  # xarray Dataset
            plot_variables=plot_variables,
            normalization_factor=df["area_data"].max(),
            first_break=bp,
            object_id=object_id,
            save_path=save_path,
        )

        return figure

    def plot_timeseries_interactive(
        self,
        object_id: str | int | None = None,
        breakpoints: BreakpointMethod | pd.Timestamp | str | list[pd.Timestamp] | list[str] | None = None,
        plot_variables: list[str | None] | None = None,
        save_path: str | Path | None = None,
        *,
        id_geohash: str | int | None = None,
    ) -> go.Figure:
        """Plot the interactive time series for a specific lake using Plotly.

        Creates an interactive Plotly figure showing the Dynamic World land cover
        time series for a single lake. Hovering over a point shows its normalized
        value and the date (e.g., "2018-04-26"). An optional vertical line is
        drawn at the breakpoint date when provided.

        Args:
            object_id: The lake identifier to plot (a value from :attr:`object_ids_`).
                May be a string or an integer, depending on how the dataset stores
                identifiers.
            breakpoints: Breakpoint detection method, single date, or list of dates.
                If a BreakpointMethod object is passed, its first detected breakpoint
                date is used. A string or Timestamp is treated as the target date(s).
            plot_variables: Which variables to show. Defaults to all available DW
                categories (via `self.data_columns`).
            save_path: Optional path to save the figure as an HTML file.
            id_geohash: DEPRECATED alias for `object_id`. Use `object_id` instead;
                `id_geohash` emits a :class:`FutureWarning`.

        Returns:
            Plotly Figure with the timeseries trace(s), a breakpoint line if provided,
            and a legend entry for 'Breakpoint' if applicable.

        Notes:
            See also `plot_timeseries` for a static matplotlib-based alternative that
            is suitable for embedding within a Streamlit app.
            Backward compatibility: `id_geohash=...` is still accepted and
            triggers a deprecation warning.

        Example:
            >>> fig = dataset.plot_timeseries_interactive("1132035748")
            >>> fig.show()  # Open in browser or save as HTML
        """
        object_id = self._resolve_object_id(object_id=object_id, id_geohash=id_geohash)
        # Select rows for the given lake identifier using self.id_field.
        # This works whether self.id_field == "id_geohash", "lake_id", or any other name.
        df = (self.ds.sel({self.id_field: object_id}).load().to_dataframe()).dropna()
        df_plot = prepare_data_for_plot_dw(df, group_vegetation=True)

        bp = self._first_breakpoint_date_from_any_source(object_id, breakpoints)

        figure = plot_water_time_series_dw_interactive(  # type: ignore
            df=df_plot,
            plot_variables=plot_variables,
            normalization_factor=df["area_data"].max(),
            first_break=bp,
            lake_id=object_id,
            save_path=save_path,
        )

        return figure

    def create_timelapse(
        self,
        lake_gdf: gpd.GeoDataFrame,
        lake_id: str,
        timelapse_source: str = "sentinel2",
        gif_outdir: str | Path = "gifs",
        buffer: float = 100,
        start_year: int = 2001,
        end_year: int = 2025,
        start_date: str = "07-01",
        end_date: str = "08-31",
        frames_per_second: int = 1,
        dimensions: int = 512,
        overwrite_exists: bool = False,
    ) -> Path | None:
        """Create a timelapse GIF for a specific lake using Sentinel-2 imagery.

        Generates an animated GIF showing Sentinel-2 surface reflectance over time
        for a given lake. Only valid (non-masked) observations are included; cloudy
        or masked timesteps are skipped automatically. The animation captures the
        summer period (July–August) each year to maximize cloud-free observations.

        Args:
            lake_gdf: GeoDataFrame containing lake geometries with an 'id_geohash'
                column. Should be written to disk (e.g., GPKG or GeoJSON) before
                loading into the ETL pipeline due to memory constraints for large
                numbers of lakes.
            lake_id: The geohash identifier for the specific lake to visualize.
            timelapse_source: Image source for timelapse imagery ('sentinel2' or
                'landsat'). Default is 'sentinel2'.
            gif_outdir: Output directory for the GIF file (default: 'gifs').
            buffer: Buffer distance in meters to expand the lake bounding box so that
                the land/water mask can classify pixels adjacent to the shoreline.
                Use values larger than the typical Sentinel-2 cloud extent (e.g. ~150 km)
                to show a wide-area seasonal progression of the water body, or smaller
                values to highlight shoreline change. The default value is sufficient for
                most use cases.
            start_year: Start year for the timelapse (default: 2001).
            end_year: End year for the timelapse (default: 2025).
            start_date: Start date within each year (MM-DD format, default: '07-01').
            end_date: End date within each year (MM-DD format, default: '08-31').
            frames_per_second: Animation speed in frames per second (default: 1).
            dimensions: Pixel dimensions for the output GIF (512 or 1024) (default: 512).
            overwrite_exists: If False (default), skip download and return the existing
                Path if the output file already exists. If True, always re-download and
                regenerate the timelapse GIF (slower).

        Returns:
            Path | None: The absolute path to the generated GIF file, or None if the
                output file already exists and `overwrite_exists` is False. Raises
                an OSError if the download was skipped but the output file was not
                present at the expected location.

        Example:
            >>> gdf = gpd.GeoDataFrame(
            ...     geometry=[Point(-149.39608,59.5765).buffer(67)] * 2,
            ...     crs="EPSG:4326",
            ...     data={"id_geohash": ["-bq7b79", "-bq7b7p"]},
            ...     index=["1132035785", "1132035748"],
            ...     columns=[geometry, id_geohash],
            ... ).set_crs("EPSG:4326")
            >>> gdf.to_file("yukechi.geojson", driver="GeoJSON")
            >>> gifsdir = Path("gifs")
            >>> timelapse_gif = dataset.yukon_sentinel2.create_timelapse(
            ...     lake_geometry=gdf.geometry,
            ...     id_geohash="1132035748",
            ...     gif_outdir=gifsdir,
            ... )  # uses default Sentinel-2 source
            >>> timelapse_landsat = dataset.yukon_sentinel2.create_timelapse(
            ...     lake_geometry=gdf.geometry,
            ...     id_geohash="1132035748",
            ...     gif_outdir=gifsdir,
            ...     timelapse_source="landsat",
            ... )  # uses Landsat instead
        """
        return create_timelapse(
            input_lake_gdf=lake_gdf,
            id_geohash=lake_id,
            timelapse_source=timelapse_source,
            gif_outdir=gif_outdir,
            buffer=buffer,
            start_year=start_year,
            end_year=end_year,
            start_date=start_date,
            end_date=end_date,
            frames_per_second=frames_per_second,
            dimensions=dimensions,
            overwrite_exists=overwrite_exists,
        )


# =============================================================================
# JRCDataset subclass
# =============================================================================


class JRCDataset(LakeDataset):
    """Handler for JRC (Joint Research Centre) water classification data.

    Processes JRC water occurrence data with separate classes for permanent water,
    seasonal water, and land cover.

    Attributes:
        water_column (str): Fixed as "area_water_permanent" for JRC data.
        data_columns (list): ['area_water_permanent', 'area_water_seasonal', 'area_land'].

    Example:
        >>> jrc_data = JRCDataset(xr.open_dataset("jrc_water.nc"))
        >>> permanent_water = jrc_data.ds_normalized["area_water_permanent"]
        >>> seasonal_water = jrc_data.ds_normalized["area_water_seasonal"]
    """

    def __init__(self, ds: xr.Dataset, id_field: str = "id_geohash", mask_data: bool = True) -> None:
        """Initialize JRCDataset with JRC water classification data.

        Args:
            ds: Input xarray Dataset with at least the 3 JRC class variables.
                If `ds` is a DataArray, convert it first via `ds.to_dataset()`.
                Prefer passing an in-memory or already-opened Dataset over a file path
                to avoid opening and scanning a large NetCDF file during initialization.
            id_field: The field name for the lake identifier coordinate dimension.
                Defaults to "id_geohash" to maintain consistency with DWDataset and other
                sources. Override with another string (e.g., "lake_id") if your exported
                JRC dataset uses a different column name for the lake identifier.
            mask_data: Whether invalid data should be masked during preprocessing.
                Default is True; set False to disable masking.

        Raises:
            TypeError: If `ds` is not an xr.Dataset, xr.DataArray, or pathlib.Path.
            ValueError: If `id_field` does not exist in the dataset and is not a valid
                       JRC variable name.

        Notes:
            The required 3 JRC variables are: area_water_permanent,
            area_water_seasonal, and area_land. If your exported dataset omits any
            of these, normalization or further processing will fail.

        Example usage:
            # Using the default identifier column (backward compatible):
            >>> jrc = JRCDataset(ds_jrc)

            # With a custom identifier column name (e.g., from manual export):
            >>> jrc = JRCDataset(ds_jrc, id_field="id_geohash")
        """
        super().__init__(ds, id_field=id_field, mask_data=mask_data)
        self.water_column = "area_water_permanent"
        self.data_columns = jrc_bandnames

    # ---- Preprocessing & normalization ----

    def _preprocess(self) -> None:
        """Preprocess JRC water data.

        Calculates total area as the sum of permanent water, seasonal water, and land.
        """
        super()._preprocess()

        ds = self.ds
        ds["area_data"] = ds["area_land"] + ds["area_water_permanent"] + ds["area_water_seasonal"]

        max_area = ds["area_data"].max(dim="date", skipna=True)
        ds["area_nodata"] = (max_area - ds["area_data"]).round(4)

        self.preprocessed_ = True
        self.ds = ds

    def _mask_invalid(self) -> None:
        """Mask invalid data based on data quality.

        Removes observations where the no-data area exceeds quality thresholds.
        """

        if self.mask_data is False:
            return

        ds = self.ds_normalized
        mask = ds["area_nodata"] <= 0
        self.ds = self.ds.where(mask)
        self.ds_normalized = self.ds_normalized.where(mask)

        self.ds_ismasked_ = True
        self.ds_normalized_ismasked_ = True

    # ---- Plotting ----

    def plot_timeseries(
        self,
        object_id: str | int | None = None,
        breakpoints: BreakpointMethod | pd.Timestamp | str | list[pd.Timestamp] | list[str] | None = None,
        plot_variables: list[str | None] | None = None,
        save_path: str | Path | None = None,
        *,
        id_geohash: str | int | None = None,
    ) -> plt.Figure:
        """Plot the time series for a specific lake using matplotlib.

        Creates a static matplotlib figure showing the JRC water classification
        time series for a single lake, with an optional vertical line indicating
        a breakpoint or specific date. The figure includes a 'Breakpoint' entry
        in the legend when a breakpoint is provided.

        Args:
            object_id: The object identifier (a value from :attr:`object_ids_`).
            breakpoints: Breakpoint detection method, single date, or list of dates.
                If a BreakpointMethod object is passed, its first detected breakpoint
                date is used. A string or Timestamp is treated as the target date(s).
                If None, no vertical line is drawn.
            plot_variables: Which variables to show. Acceptable choices are
                'area_water_permanent', 'area_water_seasonal', 'area_land'.
                To select multiple variables, wrap them in a list (e.g.,
                `plot_variables=["area_water_permanent", "area_water_seasonal"]`).
                If None (default), all available JRC categories are shown.
            save_path: Optional path to save the image as PNG/PDF/SVG.
            id_geohash: DEPRECATED alias for `object_id`. Use `object_id` instead;
                `id_geohash` emits a :class:`FutureWarning`.

        Returns:
            matplotlib.Figure with a single axes and a legend. See also
                `plot_timeseries_interactive` for an interactive Plotly-based version.

        Example:
            >>> fig = dataset.jrc.plot_timeseries("1132035748")          # static
            >>> fig = dataset.jrc.plot_timeseries(
            ...     "1132035748", breakpoints=some_breakpoint_method,
            ...     plot_variables=["area_water_permanent"],
            ... )

        Notes:
            - The `plot_variables` default value is determined at runtime after
              preprocessing; it is the list of keys present in `self.data_columns`.
              Therefore, even though it is not a formal argument, its effective default
              value is dynamically computed. This keeps the callable signature minimal,
              and enables plotting all available JRC variables without specifying them
              manually when desired.
            - Backward compatibility: `id_geohash=...` is still accepted and
              triggers a deprecation warning.

        See Also:
            plot_timeseries_interactive: For an interactive Plotly-based version.
        """
        object_id = self._resolve_object_id(object_id=object_id, id_geohash=id_geohash)
        df = (self.ds.sel({self.id_field: object_id}).load().to_dataframe()).dropna().reset_index(drop=False)
        normalization_factor = df["area_data"].max()

        bp = self._first_breakpoint_date_from_any_source(object_id, breakpoints)

        # Default plot variables for JRC if not explicitly provided:
        if plot_variables is None:
            plot_variables = ["area_water_permanent", "area_water_seasonal", "area_land"]

        fig = plot_water_time_series_jrc(  # type: ignore
            df=df,
            first_break=bp,
            plot_variables=plot_variables,
            normalization_factor=normalization_factor,
            object_id=object_id,
            save_path=save_path,
        )

        return fig

    def plot_timeseries_interactive(
        self,
        object_id: str | int | None = None,
        breakpoints: BreakpointMethod | pd.Timestamp | str | list[pd.Timestamp] | list[str] | None = None,
        plot_variables: list[str | None] | None = None,
        save_path: str | Path | None = None,
        *,
        id_geohash: str | int | None = None,
    ) -> go.Figure:
        """Plot the interactive time series for a specific lake using Plotly.

        Creates an interactive Plotly figure showing the JRC water classification
        time series for a single lake. Hovering over a point shows its normalized
        value and the date (e.g., "2018-04-26"). An optional vertical line is
        drawn at the breakpoint date when provided.

        Args:
            object_id: The object identifier to plot (a value from :attr:`object_ids_`).
            breakpoints: Breakpoint detection method, single date, or list of dates.
                If a BreakpointMethod object is passed, its first detected breakpoint
                date is used. A string or Timestamp is treated as the target date(s).
            plot_variables: Which variables to show. Defaults to all available JRC
                categories (via `self.data_columns`).
            save_path: Optional path to save the figure as an HTML file.
            id_geohash: DEPRECATED alias for `object_id`. Use `object_id` instead;
                `id_geohash` emits a :class:`FutureWarning`.

        Returns:
            Plotly Figure with the timeseries trace(s), a breakpoint line if provided,
            and a legend entry for 'Breakpoint' if applicable.

        Notes:
            See also `plot_timeseries` for a static matplotlib-based alternative that
            is suitable for embedding within a Streamlit app.
            Backward compatibility: `id_geohash=...` is still accepted and
            triggers a deprecation warning.

        Example:
            >>> fig = dataset.jrc.plot_timeseries_interactive("1132035748")
        """
        object_id = self._resolve_object_id(object_id=object_id, id_geohash=id_geohash)
        df = (self.ds.sel({self.id_field: object_id}).load().to_dataframe()).dropna().reset_index(drop=False)
        normalization_factor = df["area_data"].max()

        bp = self._first_breakpoint_date_from_any_source(object_id, breakpoints)

        # Default plot variables for JRC if not explicitly provided:
        if plot_variables is None:
            plot_variables = ["area_water_permanent", "area_water_seasonal", "area_land"]

        fig = plot_water_time_series_jrc_interactive(  # type: ignore
            df=df,
            first_break=bp,
            plot_variables=plot_variables,
            normalization_factor=normalization_factor,
            lake_id=object_id,
            save_path=save_path,
        )

        return fig
