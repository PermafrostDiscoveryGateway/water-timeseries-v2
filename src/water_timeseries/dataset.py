"""Dataset processing classes for water timeseries analysis.

This module provides classes for processing and normalizing satellite-derived
land cover and water classification data. It includes specialized handlers for
different data sources and processing pipelines.
"""

import warnings
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

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


class LakeDataset:
    """Base class for processing lake and water body datasets.

    Handles common operations for dataset preprocessing, normalization, and masking.
    Provides a framework that can be extended for different data sources.

    Attributes:
        ds (xr.Dataset): The input xarray Dataset containing raw data.
        ds_normalized (xr.Dataset): Normalized version of the dataset (0-1 scale).
        preprocessed_ (bool): Whether preprocessing has been completed.
        normalized_available_ (bool): Whether normalized data is available.
        water_column (str): Name of the water/water extent column.
        data_columns (list): Names of all data columns in the dataset.
        ds_ismasked_ (bool): Whether the original dataset has been masked.
        ds_normalized_ismasked_ (bool): Whether the normalized dataset has been masked.

    Example:
        >>> lake_data = LakeDataset(xr.Dataset(...))
        >>> normalized = lake_data.ds_normalized
    """

    def __init__(self, ds, id_field: str = "id_geohash"):
        """Initialize the LakeDataset.

        Args:
            ds (xr.Dataset): Input xarray Dataset with land cover or water classification data.
            id_field (str): Name of the coordinate field that identifies individual time series (default: "id_geohash").
        """
        self.ds = ds
        self.preprocessed_ = False
        self.normalized_available_ = False
        self.water_column = None
        self.data_columns = None
        self.ds_ismasked_ = False
        self.ds_normalized_ismasked_ = False
        self.id_field = id_field
        self._preprocess()
        self._normalize_ds()
        self._mask_invalid()

    @property
    def object_ids_(self) -> list:
        """Get all valid object IDs from the dataset.

        Returns:
            list: List of all object IDs from the id_field coordinate.
        """
        return list(self.ds.coords[self.id_field].values)

    @property
    def dates_(self) -> list:
        """Get all valid dates from the dataset.

        Returns:
            list: List of all dates from the 'date' coordinate.
        """
        return list(self.ds.coords["date"].values)

    def _preprocess(self):
        """Preprocess the dataset.

        This method should be overridden in subclasses to implement data-source-specific
        preprocessing steps such as calculating composite indicators or adding derived fields.
        """
        pass

    def _normalize_ds(self):
        """Normalize the dataset by dividing by maximum values.

        Scales all data to 0-1 range based on the maximum area value per time series.
        This ensures comparability across different spatial extents.
        """
        self.ds_normalized = self.ds.copy()
        self.ds_normalized = self.ds / self.ds.max(dim="date")["area_data"]
        self.normalized_available_ = True

    def _mask_invalid(self):
        """Mask invalid data based on quality criteria.

        This method should be overridden in subclasses to implement data-source-specific
        masking logic based on their quality thresholds and constraints.
        """
        pass

    def create_timelapse(
        self,
        lake_gdf: gpd.GeoDataFrame,
        id_geohash: str,
        timelapse_source: str = "sentinel2",
        gif_outdir: str | Path = "gifs",
        buffer: float = 100,
        start_year: int = 2016,
        end_year: int = 2025,
        start_date: str = "07-01",
        end_date: str = "08-31",
        frames_per_second: int = 1,
        dimensions: int = 512,
        overwrite_exists: bool = False,
    ) -> Path | None:
        """
        Create a timelapse GIF for a specific lake.

        This method generates an animated GIF showing satellite imagery
        over a date range for a lake identified by its geohash. The timelapse captures
        the summer period (July-August) each year to maximize cloud-free observations.

        Args:
            lake_gdf: GeoDataFrame containing lake geometries with an 'id_geohash' column.
            id_geohash: The geohash identifier for the specific lake to visualize.
            timelapse_source: Image source for timelapse imagery ('sentinel2' or 'landsat').
            gif_outdir: Output directory for the GIF file (default: 'gifs').
            buffer: Buffer distance in meters to expand the lake bounding box (default: 100).
            start_year: Start year for the timelapse (default: 2016).
            end_year: End year for the timelapse (default: 2025).
            start_date: Start date within each year (MM-DD format, default: '07-01').
            end_date: End date within each year (MM-DD format, default: '08-31').
            frames_per_second: Animation speed (default: 1).
            dimensions: Pixel dimensions for the output GIF (default: 512).
            overwrite_exists: If False (default), skip download if output file already exists.
                              If True, always re-download and overwrite existing file.

        Returns:
            Path | None: Path to the generated GIF file, or None if skipped due to existing file.
        """
        return create_timelapse(
            input_lake_gdf=lake_gdf,
            id_geohash=id_geohash,
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

    def plot_timeseries(self, id_geohash: str, breakpoints: None) -> plt.Figure:
        """Plot the time series for a specific geohash.

        Args:
            id_geohash (str): The geohash identifier for the location.
            breakpoints (BreakpointMethod, optional): Breakpoint detection method to use.
        """
        pass

    def calculate_changes(self, break_df: pd.DataFrame, id_geohash: str) -> pd.DataFrame:

        pass

    def merge(
        self,
        other: "LakeDataset",
        how: str = "both",
    ) -> "LakeDataset":
        """Merge this LakeDataset with another LakeDataset.

        Combines the .ds attributes of both datasets. Both datasets must have the same
        variables. The merge strategy is determined by the `how` parameter.
        Both datasets must be of the same type (e.g., both DWDataset or both JRCDataset).

        Args:
            other (LakeDataset): Another LakeDataset instance to merge with.
            how (str): Merge strategy. Options:
                - "both": Merge along both dimensions (date and id_geohash). Combines all
                  data from both datasets, keeping all unique dates and id_geohashes.
                - "date": Merge along the "date" dimension only. Both datasets must have
                  the same id_geohash values, but can have different dates. New dates are
                  appended to the existing time series.
                - "id_geohash": Merge along the "id_geohash" dimension only. Both datasets
                  must have the same dates, but can have different id_geohashes. New
                  id_geohashes (lakes) are added with their time series.

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

        merged = self.__class__(merged_ds)
        merged.id_field = self.id_field
        return merged

    def _validate_merge(self, other: "LakeDataset", how: str):
        """Validate datasets before merging."""

        if how not in {"both", "date", "id_geohash"}:
            raise ValueError(f"Invalid merge strategy '{how}'. Must be 'both', 'date', or 'id_geohash'.")

        if type(self) is not type(other):
            raise TypeError(
                f"Cannot merge {type(self).__name__} with {type(other).__name__}. Both datasets must be the same type."
            )

        if set(self.ds.data_vars) != set(other.ds.data_vars):
            raise ValueError("Datasets have different variables.")

    def _merge_both(self, ds1, ds2):
        """Merge along both dimensions."""

        return xr.merge([ds1, ds2])

    def _merge_by_date(self, ds1, ds2):
        """Merge along date dimension (same id_geohash, new dates)."""

        if set(ds1.coords[self.id_field].values) != set(ds2.coords[self.id_field].values):
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

    def _merge_by_id(self, ds1, ds2):
        """Merge along id_geohash dimension (same dates, new id_geohashes)."""

        if set(ds1.coords["date"].values) != set(ds2.coords["date"].values):
            raise ValueError("For merge how='id_geohash', both datasets must have the same dates.")

        # Check for duplicate id_geohashes
        ids1 = set(ds1.coords[self.id_field].values)
        ids2 = set(ds2.coords[self.id_field].values)
        duplicate_ids = ids1 & ids2
        if duplicate_ids:
            warnings.warn(
                f"Datasets have {len(duplicate_ids)} overlapping {self.id_field} values. "
                f"Data from the second dataset will overwrite the first for these values.",
                UserWarning,
            )

        return xr.concat([ds1, ds2], dim=self.id_field)


class DWDataset(LakeDataset):
    """Handler for Dynamic World land cover classification data.

    Processes Dynamic World land cover classes including water, bare soil, snow/ice,
    trees, grass, flooded vegetation, crops, shrub/scrub, and built areas.

    Attributes:
        water_column (str): Fixed as "water" for DW data.
        data_columns (list): All 9 DW land cover classes.

    Example:
        >>> dw_data = DWDataset(xr.open_dataset("dynamic_world.nc"))
        >>> water_time_series = dw_data.ds_normalized["water"]
        >>> print(dw_data.data_columns)
        ['water', 'bare', 'snow_and_ice', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built']
    """

    def __init__(self, ds):
        """Initialize DWDataset with Dynamic World data.

        Args:
            ds (xr.Dataset): Input xarray Dataset with at least the 9 DW class variables.
        """
        super().__init__(ds)
        self.water_column = "water"
        self.data_columns = [
            "water",
            "bare",
            "snow_and_ice",
            "trees",
            "grass",
            "flooded_vegetation",
            "crops",
            "shrub_and_scrub",
            "built",
        ]

    def _preprocess(self):
        """Preprocess Dynamic World data.

        Calculates total area as the sum of all land cover classes and computes
        the no-data area as the difference from maximum area across time.
        """
        super()._preprocess()
        ds = self.ds
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

        max_area = ds["area_data"].max(dim="date", skipna=True)
        ds["area_nodata"] = (max_area - ds["area_data"]).round(4)

        self.preprocessed_ = True
        self.ds = ds

    def _mask_invalid(self):
        """Mask invalid data based on quality criteria.

        Removes observations where data quality is poor (high no-data area) or
        where snow/ice coverage is excessive (more than 5%), which indicates poor classification.
        """
        ds = self.ds_normalized
        # Mask where no-data area > 0
        mask_nodata = ds["area_nodata"] <= 0
        # Mask where snow/ice > 5% (indicates poor classification)
        mask_snow = ds["snow_and_ice"] <= 0.05
        # Combine masks
        mask = mask_nodata & mask_snow

        self.ds = self.ds.where(mask)
        self.ds_normalized = self.ds_normalized.where(mask)

        self.ds_ismasked_ = True
        self.ds_normalized_ismasked_ = True

    # create_timelapse is inherited from LakeDataset

    def plot_timeseries(self, id_geohash: str, breakpoints=None, save_path: Optional[str | Path] = None) -> plt.Figure:
        """Plot the time series for a specific geohash.

        Args:
            id_geohash (str): The geohash identifier for the location.
            breakpoints (BreakpointMethod, optional): Breakpoint detection method to use.
            save_path (str | Path, optional): Path to save the plot as an image file.

        Returns:
            plt.Figure: The matplotlib figure object.
        """
        # self._normalize_ds()
        df = self.ds.sel(id_geohash=id_geohash).load().to_dataframe().dropna()
        df_plot = prepare_data_for_plot_dw(df, group_vegetation=True)
        normalization_factor = df["area_data"].max()

        if breakpoints is not None:
            breaks = breakpoints.calculate_break(self, object_id=id_geohash)
            if breaks is not None:
                if len(breaks) > 0:
                    bp = breaks["date_break"].iloc[0]
                else:
                    bp = None
        else:
            bp = None

        figure = plot_water_time_series_dw(
            df_plot,
            first_break=bp,
            normalization_factor=normalization_factor,
            lake_id=id_geohash,
            save_path=save_path,
        )

        return figure

    def plot_timeseries_interactive(
        self,
        id_geohash: str,
        breakpoints=None,
        save_path: Optional[str | Path] = None,
    ):
        """Plot the interactive time series for a specific geohash using Plotly.

        Args:
            id_geohash (str): The geohash identifier for the location.
            breakpoints (BreakpointMethod, optional): Breakpoint detection method to use.
            save_path (str | Path, optional): Path to save the plot as HTML file.

        Returns:
            plotly.graph_objects.Figure: Interactive Plotly figure.
        """
        df = self.ds.sel(id_geohash=id_geohash).load().to_dataframe().dropna()
        df_plot = prepare_data_for_plot_dw(df, group_vegetation=True)
        normalization_factor = df["area_data"].max()

        if breakpoints is not None:
            breaks = breakpoints.calculate_break(self, object_id=id_geohash)
            if breaks is not None:
                if len(breaks) > 0:
                    bp = breaks["date_break"].iloc[0]
                else:
                    bp = None
        else:
            bp = None

        figure = plot_water_time_series_dw_interactive(
            df_plot,
            first_break=bp,
            normalization_factor=normalization_factor,
            lake_id=id_geohash,
            save_path=save_path,
        )

        return figure


class JRCDataset(LakeDataset):
    """Handler for JRC (Joint Research Centre) water classification data.

    Processes JRC water occurrence data with separate classes for permanent water,
    seasonal water, and land.

    Attributes:
        water_column (str): Fixed as "area_water_permanent" for JRC data.
        data_columns (list): ['area_water_permanent', 'area_water_seasonal', 'area_land'].

    Example:
        >>> jrc_data = JRCDataset(xr.open_dataset("jrc_water.nc"))
        >>> permanent_water = jrc_data.ds_normalized["area_water_permanent"]
        >>> seasonal_water = jrc_data.ds_normalized["area_water_seasonal"]
    """

    def __init__(self, ds):
        """Initialize JRCDataset with JRC water classification data.

        Args:
            ds (xr.Dataset): Input xarray Dataset with JRC water classification variables.
        """
        super().__init__(ds)
        self.water_column = "area_water_permanent"
        self.data_columns = ["area_water_permanent", "area_water_seasonal", "area_land"]

    def _preprocess(self):
        """Preprocess JRC water data.

        Calculates total area as the sum of permanent water, seasonal water, and land.
        """
        ds = self.ds
        ds["area_data"] = ds["area_land"] + ds["area_water_permanent"] + ds["area_water_seasonal"]

        max_area = ds["area_data"].max(dim="date", skipna=True)
        ds["area_nodata"] = (max_area - ds["area_data"]).round(4)

        self.preprocessed_ = True
        self.ds = ds

    def _mask_invalid(self):
        """Mask invalid data based on data quality.

        Removes observations where the no-data area exceeds quality thresholds.
        """
        ds = self.ds_normalized
        mask = ds["area_nodata"] <= 0
        self.ds = self.ds.where(mask)
        self.ds_normalized = self.ds_normalized.where(mask)

        self.ds_ismasked_ = True
        self.ds_normalized_ismasked_ = True

    def create_timelapse(
        self,
        lake_gdf: gpd.GeoDataFrame,
        id_geohash: str,
        timelapse_source: str = "landsat",
        gif_outdir: str | Path = "gifs",
        buffer: float = 100,
        start_year: int = 2000,
        end_year: int = 2025,
        start_date: str = "07-01",
        end_date: str = "08-31",
        frames_per_second: int = 1,
        dimensions: int = 512,
        overwrite_exists: bool = False,
    ) -> Path | None:
        """
        Create a timelapse GIF for a specific lake.

        This method generates an animated GIF showing satellite imagery
        over a date range for a lake identified by its geohash. The timelapse captures
        the summer period (July-August) each year to maximize cloud-free observations.

        Default timelapse_source is 'landsat' for JRC data.

        Args:
            lake_gdf: GeoDataFrame containing lake geometries with an 'id_geohash' column.
            id_geohash: The geohash identifier for the specific lake to visualize.
            timelapse_source: Image source for timelapse imagery ('sentinel2' or 'landsat').
            gif_outdir: Output directory for the GIF file (default: 'gifs').
            buffer: Buffer distance in meters to expand the lake bounding box (default: 100).
            start_year: Start year for the timelapse (default: 2000).
            end_year: End year for the timelapse (default: 2025).
            start_date: Start date within each year (MM-DD format, default: '07-01').
            end_date: End date within each year (MM-DD format, default: '08-31').
            frames_per_second: Animation speed (default: 1).
            dimensions: Pixel dimensions for the output GIF (default: 512).
            overwrite_exists: If False (default), skip download if output file already exists.
                              If True, always re-download and overwrite existing file.

        Returns:
            Path | None: Path to the generated GIF file, or None if skipped due to existing file.
        """
        return create_timelapse(
            input_lake_gdf=lake_gdf,
            id_geohash=id_geohash,
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

    def plot_timeseries(self, id_geohash: str, breakpoints=None, save_path: Optional[str | Path] = None) -> plt.Figure:
        """Plot the time series for a specific geohash.

        Args:
            id_geohash (str): The geohash identifier for the location.
            breakpoints (BreakpointMethod, optional): Breakpoint detection method to use.
            save_path (str | Path, optional): Path to save the plot as an image file.

        Returns:
            plt.Figure: The matplotlib figure object.
        """
        df = self.ds.sel(id_geohash=id_geohash).load().to_dataframe().dropna().reset_index(drop=False)
        normalization_factor = df["area_data"].max()

        # TODO: breaks are not visualized correctly
        if breakpoints is not None:
            breaks = breakpoints.calculate_break(self, object_id=id_geohash)
            if breaks is not None:
                bp = breaks["date_break"].iloc[0]
        else:
            bp = None

        fig = plot_water_time_series_jrc(
            df,
            first_break=bp,
            plot_variables=["area_water_permanent", "area_water_seasonal", "area_land"],
            normalization_factor=normalization_factor,
            lake_id=id_geohash,
            save_path=save_path,
        )

        # return figure
        return fig

    def plot_timeseries_interactive(
        self,
        id_geohash: str,
        breakpoints=None,
        save_path: Optional[str | Path] = None,
    ):
        """Plot the interactive time series for a specific geohash using Plotly.

        Args:
            id_geohash (str): The geohash identifier for the location.
            breakpoints (BreakpointMethod, optional): Breakpoint detection method to use (not used currently).
            save_path (str | Path, optional): Path to save the plot as HTML file.

        Returns:
            plotly.graph_objects.Figure: Interactive Plotly figure.
        """
        df = self.ds.sel(id_geohash=id_geohash).load().to_dataframe().dropna().reset_index(drop=False)
        normalization_factor = df["area_data"].max()

        # Breakpoint processing disabled for now
        bp = None

        fig = plot_water_time_series_jrc_interactive(
            df,
            first_break=bp,
            plot_variables=["area_water_permanent", "area_water_seasonal", "area_land"],
            normalization_factor=normalization_factor,
            lake_id=id_geohash,
            save_path=save_path,
        )

        return fig
