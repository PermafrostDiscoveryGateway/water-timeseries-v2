"""Dataset processing classes for water timeseries analysis.

This module provides classes for processing and normalizing satellite-derived
land cover and water classification data. It includes specialized handlers for
different data sources and processing pipelines.
"""

import matplotlib.pyplot as plt

from .utils.plotting import (
    plot_water_time_series_dw,
    plot_water_time_series_jrc,
    prepare_data_for_plot_dw,
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

    def __init__(self, ds):
        """Initialize the LakeDataset.

        Args:
            ds (xr.Dataset): Input xarray Dataset with land cover or water classification data.
        """
        self.ds = ds
        # self.ds_normalized = None
        self.preprocessed_ = False
        self.normalized_available_ = False
        self.water_column = None
        self.data_columns = None
        self.ds_ismasked_ = False
        self.ds_normalized_ismasked_ = False
        self._preprocess()
        self._normalize_ds()
        self._mask_invalid()

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
        # ds_normed = self.ds / self.ds.max(dim="date")["area_data"]
        self.ds_normalized = self.ds.copy()
        self.ds_normalized = self.ds / self.ds.max(dim="date")["area_data"]
        self.normalized_available_ = True

    def _mask_invalid(self):
        """Mask invalid data based on quality criteria.

        This method should be overridden in subclasses to implement data-source-specific
        masking logic based on their quality thresholds and constraints.
        """
        pass

    def plot_timeseries(self, id_geohash: str, breakpoints: None) -> plt.Figure:
        """Plot the time series for a specific geohash.

        Args:
            id_geohash (str): The geohash identifier for the location.
            breakpoints (BreakpointMethod, optional): Breakpoint detection method to use.
        """
        pass


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
        where snow/ice coverage is excessive, which indicates poor classification.
        """
        ds = self.ds_normalized
        mask = (ds["area_nodata"] <= 0) & (ds["snow_and_ice"] < 0.05)
        self.ds = self.ds.where(mask)
        self.ds_normalized = self.ds_normalized.where(mask)

        self.ds_ismasked_ = True
        self.ds_normalized_ismasked_ = True

    def plot_timeseries(self, id_geohash: str, breakpoints=None) -> plt.Figure:
        """Plot the time series for a specific geohash.

        Args:
            id_geohash (str): The geohash identifier for the location.
            breakpoints (BreakpointMethod, optional): Breakpoint detection method to use.
        """
        self._normalize_ds()
        df = self.ds.sel(id_geohash=id_geohash).load().to_dataframe().dropna()
        df_plot = prepare_data_for_plot_dw(df, group_vegetation=True)
        normalization_factor = df["area_data"].max()

        if breakpoints is not None:
            breaks = breakpoints.calculate_break(self, object_id=id_geohash)
            if breaks is not None:
                bp = breaks["date_break"].iloc[0]
        else:
            bp = None

        figure = plot_water_time_series_dw(
            df_plot,
            first_break=bp,
            normalization_factor=normalization_factor,
            lake_id=id_geohash,
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

    def plot_timeseries(self, id_geohash: str, breakpoints=None) -> plt.Figure:
        """Plot the time series for a specific geohash.

        Args:
            id_geohash (str): The geohash identifier for the location.
            breakpoints (BreakpointMethod, optional): Breakpoint detection method to use.
        """
        self._normalize_ds()
        df = self.ds.sel(id_geohash=id_geohash).load().to_dataframe().dropna().reset_index(drop=False)
        normalization_factor = df["area_data"].max()

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
        )

        # return figure
        return fig
