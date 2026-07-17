import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import ee
import eemont  # noqa: F401
import geemap
import geopandas as gpd
import google.auth
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import xarray as xr
from loguru import logger
from shapely.geometry import box
from xee import helpers


def initialize_earth_engine(project: str | None = None, token_name: str = "EARTHENGINE_TOKEN") -> None:
    """Initialize the Earth Engine Python client.

    Supports two authentication modes:
    - Service account: via GOOGLE_APPLICATION_CREDENTIALS env var (non-interactive, for servers)
    - OAuth token: via persisted credentials or geemap token_name (interactive, for local use)
    """
    if project is None:
        project = os.environ.get("EE_PROJECT") or None
    if project == "":
        project = None
    
    key_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or None
    
    # Service account mode takes priority
    if key_file is not None:
        try:
            credentials, project_id = google.auth.default(
                scopes=["https://www.googleapis.com/auth/earthengine", 
                        "https://www.googleapis.com/auth/cloud-platform"]
            )
            ee.Initialize(credentials, project=project)
            logger.info("Successfully authenticated with Earth Engine Service Account!")
            return
        except (google.auth.exceptions.GoogleAuthError, ValueError) as e:
            logger.error(f"Failed to authenticate with service account: {e}")
            raise
    
    # OAuth/persisted credentials mode
    # geemap >= 0.37 moved ``ee_initialize`` to ``geemap.coreutils``; older releases
    # expose it on the top-level ``geemap`` module.
    try:
        from geemap.coreutils import ee_initialize as _geemap_ee_initialize
        _geemap_ee_initialize(token_name=token_name, project=project)
        return
    except (ImportError, AttributeError):
        pass
    
    if hasattr(geemap, "ee_initialize"):
        geemap.ee_initialize(token_name=token_name, project=project)
        return
    
    # Fall back to persisted credentials
    ee.Initialize(project=project)


def get_bbox(gdf, to_crs=4326, return_ee=True):
    """
    Create a bounding-box geometry from a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): input geopandas dataframe.
        to_crs (int|None): EPSG code to reproject bbox to (default 4326). If None, keep original CRS.
        return_ee (bool): whether to also return an ee.Geometry (requires EE initialized and lon/lat CRS).

    Returns:
        dict with keys:
          - 'shapely': shapely.geometry.Polygon bbox in gdf.crs
          - 'gdf': GeoDataFrame with one geometry (bbox) in CRS `to_crs` (or original if to_crs is None)
          - 'ee': ee.Geometry (or None if return_ee is False)
    """
    # get minx, miny, maxx, maxy
    minx, miny, maxx, maxy = gdf.total_bounds

    # shapely geometry in original CRS
    bbox_shapely = box(minx, miny, maxx, maxy)

    # GeoDataFrame with same CRS as input
    bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox_shapely]}, crs=gdf.crs)

    # reproject if requested
    if to_crs is not None:
        bbox_gdf = bbox_gdf.to_crs(epsg=to_crs)

    ee_geom = None
    if return_ee:
        # ensure geometry is in lon/lat (EPSG:4326) for EE
        if bbox_gdf.crs.to_epsg() != 4326:
            bbox_for_ee = bbox_gdf.to_crs(epsg=4326)
        else:
            bbox_for_ee = bbox_gdf
        geojson = bbox_for_ee.geometry.iloc[0].__geo_interface__
        ee_geom = ee.Geometry(geojson)

    return {"shapely": bbox_shapely, "gdf": bbox_gdf, "ee": ee_geom}


def drop_z_from_gdf(gdf, inplace=False):
    """
    Return a GeoDataFrame with 3D geometries (POLYGON Z / MULTIPOLYGON Z) converted to 2D.
    If inplace=True modifies and returns the same GeoDataFrame object.
    """
    from shapely.ops import transform

    def _to_2d(geom):
        if geom is None:
            return None
        return transform(lambda x, y, z=None: (x, y), geom)

    target = gdf if inplace else gdf.copy()
    target["geometry"] = target["geometry"].apply(_to_2d)
    return target


def create_no_data_image():
    """Creates an image that represents having no data, this is later used for filtering."""
    return ee.Image().rename(["no_data"])


def calc_monthly_dw(
    start_date: str,
    polygons: ee.FeatureCollection,
    crs: str = "EPSG:3572",
    scale: float = 10,
) -> ee.Image | None:
    """
    Generate a monthly Dynamic World composite.

    Generates a monthly Dynamic World composite and returns the mode of the
    land cover classification for each pixel.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format. The mask will be
            calculated for start_date to start_date + 1 month.
        polygons (ee.FeatureCollection): The region of interest to filter images.
        crs (str, optional): The coordinate reference system. Defaults to 'EPSG:3572'.
        scale (float, optional): The pixel scale in meters. Defaults to 10.

    Returns:
        ee.Image | None: The Dynamic World composite image, or None if no data
            is available for the given time period and location.
    """
    import warnings

    # Cast startDate back to an ee.Date, type erasure happens when mapping on a list.
    start_date = ee.Date(start_date)
    end_date = start_date.advance(1, "month")
    dw_filtered_image_collection = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(polygons).filterDate(start_date, end_date)
    )

    try:
        # Check if collection is empty on the client side before processing
        size = dw_filtered_image_collection.size().getInfo()
        if size == 0:
            warnings.warn(f"No data available for start_date: {start_date.getInfo()}")
            return None

        image = (
            dw_filtered_image_collection.select("label")
            .reduce(ee.Reducer.mode())
            .set("system:time_start", start_date.millis())
            .setDefaultProjection(crs=crs, scale=scale)
        )
    except ee.EEException as e:
        # Check if this is a band matching error indicating no data
        if "did not match any bands" in str(e) or "no_data" in str(e):
            warnings.warn(f"No data available for start_date: {start_date.getInfo()}")
            return None
        raise
    except Exception:
        warnings.warn(f"No data available for start_date: {start_date.getInfo()}")
        return None

    return ee.Image(image)


def calc_dw_aggregate(
    polygons: ee.FeatureCollection,
    start_date: str = None,
    end_date: str = None,
    year: int = None,
    month: int = None,
    crs: str = "EPSG:3572",
    scale: float = 10,
    timestamp_date: str = None,
) -> ee.Image:
    """
    Generate a Dynamic World composite for a date range or year/month.

    Generates a Dynamic World composite and returns the mode of the
    land cover classification. Accepts either a date range (start_date/end_date)
    or a specific year/month.

    Args:
        polygons (ee.FeatureCollection): The region of interest to filter images.
        start_date (str, optional): Start date in 'YYYY-MM-DD' format.
        end_date (str, optional): End date in 'YYYY-MM-DD' format.
        year (int, optional): Specific year to filter.
        month (int, optional): Specific month to filter.
        crs (str, optional): The coordinate reference system. Defaults to 'EPSG:3572'.
        scale (float, optional): The pixel scale in meters. Defaults to 10.
        timestamp_date (str, optional): Date string for the image timestamp.

    Returns:
        ee.Image: The Dynamic World composite image, or a no-data image if
            no images are available for the specified period.

    Raises:
        ValueError: If neither (start_date and end_date) nor (year and month)
            are provided.
    """
    # Cast startDate back to an ee.Date, type erasure happens when mapping on a list.
    if (start_date and end_date) is not None:
        start_date = ee.Date(start_date)
        end_date = ee.Date(end_date)
        dw_filtered_image_collection = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(polygons).filterDate(start_date, end_date)
        )
    elif (year and month) is not None:
        year_ee = ee.Filter.calendarRange(year, year, "year")
        month_ee = ee.Filter.calendarRange(month, month, "month")
        dw_filtered_image_collection = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(polygons).filter(year_ee).filter(month_ee)
        )
    else:
        raise ValueError("Please add values for either start_date and end_date or year and month!")

    if timestamp_date is None:
        if start_date:
            timestamp_date = start_date
        else:
            timestamp_date = f"{year}-{month}"

    image = ee.Algorithms.If(
        dw_filtered_image_collection.size().eq(0),
        create_no_data_image(),
        dw_filtered_image_collection.select("label")
        .reduce(ee.Reducer.mode())
        .set("system:time_start", ee.Date(timestamp_date).millis())
        .setDefaultProjection(crs=crs, scale=scale),
    )
    return ee.Image(image)


def calc_dw_aggregate_v2(
    start_date: str,
    end_date: str,
    polygons: ee.FeatureCollection,
    crs: str = "EPSG:3572",
    scale: float = 10,
    timestamp_date: str = None,
) -> ee.Image | None:
    """
    Generate a Dynamic World composite for a date range.

    Generates a Dynamic World composite reduced by mode and returns an ee.Image,
    or None if no images are available for the period.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        polygons (ee.FeatureCollection): The region of interest to filter images.
        crs (str, optional): The coordinate reference system. Defaults to 'EPSG:3572'.
        scale (float, optional): The pixel scale in meters. Defaults to 10.
        timestamp_date (str, optional): Date string for the image timestamp.
            Defaults to start_date if not provided.

    Returns:
        ee.Image | None: The Dynamic World composite image, or None if no
            images are available for the given time period and location.
    """
    # Cast startDate back to an ee.Date, type erasure happens when mapping on a list.
    start_date = ee.Date(start_date)
    end_date = ee.Date(end_date)
    dw_filtered_image_collection = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(polygons).filterDate(start_date, end_date)
    )

    if timestamp_date is None:
        timestamp_date = start_date

    # Query collection size on the client; return None if empty
    size = dw_filtered_image_collection.size().getInfo()
    if size == 0:
        return None

    image = (
        dw_filtered_image_collection.select("label")
        .reduce(ee.Reducer.mode())
        .set("system:time_start", ee.Date(timestamp_date).millis())
        .setDefaultProjection(crs=crs, scale=scale)
    )
    return ee.Image(image)


def create_dw_classes_mask(image):
    """
    Creates a mask for all classes in the given image.

    image: ee.Image, the input Dynamic World image.

    Returns:
        ee.Image, the input image with additional bands for each class, where each band
        contains a binary mask indicating whether the pixel belongs to that class.
    """
    class_dictionary = {
        0: "water",
        1: "trees",
        2: "grass",
        3: "flooded_vegetation",
        4: "crops",
        5: "shrub_and_scrub",
        6: "built",
        7: "bare",
        8: "snow_and_ice",
    }

    # Loop through each class ID and create a mask
    label_mode = image.select(["label_mode"])
    for class_id, class_name in class_dictionary.items():
        masked_image = label_mode.eq(class_id).rename(class_name).multiply(ee.Image.pixelArea()).multiply(1e-4)
        image = image.addBands(masked_image)

    return ee.Image(image)


def create_jrc_classes_mask(image):
    """
    Creates area masks for all JRC water classes from the given image.

    The JRC YearlyHistory collection has a 'waterClass' band with values:
    - 0: No data
    - 1: Land
    - 2: Seasonal water
    - 3: Permanent water

    This function calculates the pixel area in hectares for each class.

    image: ee.Image, the input JRC image with 'waterClass' band.

    Returns:
        ee.Image, the input image with additional bands for each class area in hectares:
            - area_nodata
            - area_land
            - area_water_seasonal
            - area_water_permanent
    """
    band_names = [
        "area_nodata",
        "area_land",
        "area_water_seasonal",
        "area_water_permanent",
    ]

    # Loop through each class ID and calculate pixel area in hectares
    # pixelArea() returns m², multiply by 1e-4 to get hectares
    water_class = image.select("waterClass")
    for i, band_name in enumerate(band_names):
        area_mask = water_class.eq(i).multiply(ee.Image.pixelArea()).multiply(1e-4).rename(band_name)
        image = image.addBands(area_mask)

    return image


def calc_annual_jrc(
    year: int,
    polygons: ee.FeatureCollection,
    crs: str = "EPSG:4326",
    scale: float = 30,
) -> ee.Image:
    """
    Generates an annual JRC (Joint Research Centre) water classification composite.

    Args:
        year: The year to extract data for.
        polygons: ee.FeatureCollection, the polygons to extract data for.
        crs: str, optional. The coordinate reference system (default: EPSG:4326).
        scale: float, optional. The pixel scale in meters (default: 30).

    Returns:
        ee.Image: An image with JRC water classification bands for the specified year.
    """
    start_date = ee.Date(f"{year}-01-01")
    end_date = ee.Date(f"{year + 1}-01-01")

    jrc_collection = (
        ee.ImageCollection("JRC/GSW1_4/YearlyHistory").filterBounds(polygons).filterDate(start_date, end_date)
    )

    # Use ee.Algorithms.If to handle empty collections on the server side
    image = ee.Algorithms.If(
        jrc_collection.size().eq(0),
        create_no_data_image(),
        jrc_collection.first().set("system:time_start", start_date.millis()).setDefaultProjection(crs=crs, scale=scale),
    )

    return ee.Image(image)


def make_date_window(date, window, mode="each", fmt="%Y-%m-%d"):
    """
    Create start/end dates around a central date.

    Args:
        date (str|datetime): central date (e.g. '2025-07-01').
        window (int): number of days. Interpretation depends on mode.
        mode (str): 'each' (default) -> window days on each side:
                    start = date - window, end = date + window.
                    'total' -> total window length centered on date:
                    start = date - floor(window/2), end = start + window.
        fmt (str): output date string format.

    Returns:
        dict with keys:
          - 'start_dt', 'end_dt' (datetime objects)
          - 'start_date', 'end_date' (formatted strings)
    """
    if isinstance(date, str):
        center = datetime.fromisoformat(date)
    elif isinstance(date, datetime):
        center = date
    else:
        raise TypeError("date must be str or datetime")

    if mode == "each":
        start_dt = center - timedelta(days=window)
        end_dt = center + timedelta(days=window)
    elif mode == "total":
        half = window // 2
        start_dt = center - timedelta(days=half)
        end_dt = start_dt + timedelta(days=window)
    else:
        raise ValueError("mode must be 'each' or 'total'")

    return {
        "start_dt": start_dt,
        "end_dt": end_dt,
        "start_date": start_dt.strftime(fmt),
        "end_date": end_dt.strftime(fmt),
    }


def weekly_dates(start="2025-06-04", step=7, count=None, end_date=None, fmt="%Y-%m-%d"):
    """
    Return list of dates (strings) starting at `start`, every `step` days.
    Provide either `count` (number of items) or `end_date` (inclusive).
    """
    from datetime import datetime, timedelta

    def to_date(d):
        return datetime.fromisoformat(d).date() if isinstance(d, str) else d

    start_dt = to_date(start)
    end_dt = to_date(end_date) if end_date is not None else None

    if count is None and end_dt is None:
        raise ValueError("Provide either count or end_date")

    dates = []
    cur = start_dt
    if count is not None:
        for _ in range(count):
            dates.append(cur.strftime(fmt))
            cur += timedelta(days=step)
    else:
        while cur <= end_dt:
            dates.append(cur.strftime(fmt))
            cur += timedelta(days=step)

    return dates


def monthly(start="2025-06-04", step=7, count=None, end_date=None, fmt="%Y-%m-%d"):
    """
    Return list of dates (strings) starting at `start`, every `step` months.
    Provide either `count` (number of items) or `end_date` (inclusive).
    """
    from datetime import datetime, timedelta

    def to_date(d):
        return datetime.fromisoformat(d).date() if isinstance(d, str) else d

    start_dt = to_date(start)
    end_dt = to_date(end_date) if end_date is not None else None

    if count is None and end_dt is None:
        raise ValueError("Provide either count or end_date")

    dates = []
    cur = start_dt
    if count is not None:
        for _ in range(count):
            dates.append(cur.strftime(fmt))
            cur += timedelta(days=step)
    else:
        while cur <= end_dt:
            dates.append(cur.strftime(fmt))
            cur += timedelta(days=step)

    return dates


def setup_monthly_dates(years: List[int], months: List[int]) -> List[str]:
    """Generate a list of monthly dates from the given years and months.

    Creates dates starting from the first day of each specified month.

    Args:
        years: List of years (e.g., [2017, 2018]).
        months: List of months as integers (1-12).

    Returns:
        List of formatted dates in 'YYYY-MM-DD' format (e.g., '2017-06-01').

    Example:
        >>> setup_monthly_dates([2023], [1, 2])
        ['2023-01-01', '2023-02-01']
    """
    dates = []
    for year in years:
        for month in months:
            dates.append(f"{year}-{month:02d}-01")
    return dates


def setup_dates_from_options(
    years: Optional[List[int]] = None,
    months: Optional[List[int]] = None,
    date_list: Optional[List[str]] = None,
) -> List[str]:
    """Validate and generate a list of dates from either date_list OR (years AND months).

    This function enforces mutual exclusivity between date_list and (years, months).

    Args:
        years: List of years (e.g., [2017, 2018]). Must be provided together with
            `months` if `date_list` is not used.
        months: List of months as integers (1-12). Must be provided together with
            `years` if `date_list` is not used.
        date_list: Optional list of dates in 'YYYY-MM' format (e.g., ['2017-06', '2018-07']).
            If provided, `years` and `months` are ignored.

    Returns:
        List of formatted dates in 'YYYY-MM-DD' format.

    Raises:
        ValueError: If neither (years and months) nor date_list is provided,
            or if both are provided.

    Example:
        >>> setup_dates_from_options(date_list=['2017-06', '2018-07'])
        ['2017-06-01', '2018-07-01']
        >>> setup_dates_from_options(years=[2017], months=[6, 7])
        ['2017-06-01', '2017-07-01']
    """

    # Validate date parameters: either date_list OR (years AND months)
    if date_list is not None and (years is not None or months is not None):
        raise ValueError(
            "Invalid date parameters: either provide 'date_list' OR ('years' AND 'months'), "
            "but not both. These options are mutually exclusive."
        )

    if date_list is None and (years is None or months is None):
        raise ValueError(
            "Invalid date parameters: either provide 'date_list' or both 'years' and 'months'. "
            f"Received: years={years}, months={months}, date_list={date_list}"
        )

    # Generate dates based on the input
    if date_list is not None:
        # Convert YYYY-MM format to YYYY-MM-DD format (first day of month)
        return [f"{d}-01" for d in date_list]
    else:
        # Use years and months (with defaults if not provided)
        years = years if years is not None else [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
        months = months if months is not None else [6, 7, 8, 9]
        return setup_monthly_dates(years=years, months=months)


def setup_annual_dates(years: Optional[List[int]] = None, date_list: Optional[List[str]] = None) -> List[str]:
    """Generate a list of annual dates from the given years or date_list.

    Creates dates starting from January 1st of each specified year.

    Args:
        years: List of years (e.g., [2017, 2018]). Default: [2000-2021].
        date_list: Optional list of years as strings (e.g., ['2017', '2018']).
            If provided, `years` is ignored.

    Returns:
        List of formatted dates in 'YYYY-01-01' format (e.g., '2017-01-01').
    """
    if date_list is not None and years is not None:
        raise ValueError("Invalid date parameters: either provide 'date_list' OR 'years', but not both.")

    if date_list is None and years is None:
        # Default years for JRC data (2000-2021)
        years = list(range(2000, 2022))

    if date_list is not None:
        return [f"{year}-01-01" for year in date_list]
    else:
        return [f"{year}-01-01" for year in years]


def calculate_data_area(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the total area data and no-data values from an xarray Dataset.

    This function computes a new variable `area_data` by summing the values
    of the `bare`, `water`, `snow_and_ice`, and `other` data variables.
    It also calculates a new variable `area_nodata`, which represents the
    difference between the maximum value of `area_data` across the 'date'
    dimension and the current value of `area_data`, rounded to four decimal places.

    Parameters:
    -----------
    ds : xarray.Dataset
        An xarray Dataset containing the following data variables:
        - 'bare': Area covered by bare ground.
        - 'water': Area covered by water.
        - 'snow_and_ice': Area covered by snow and ice.
        - 'other': Area covered by other land cover types.

    Returns:
    --------
    xarray.Dataset
        The input dataset with two additional data variables:
        - 'area_data': Total area calculated as the sum of 'bare', 'water',
          'snow_and_ice', and 'other'.
        - 'area_nodata': The no-data values calculated based on the maximum
          value of 'area_data' across dates.

    Example:
    --------
    >>> ds = xr.open_dataset('path_to_your_file.nc')
    >>> ds_with_area = calculate_data_area(ds)

    Notes:
    ------
    Ensure that the input dataset contains all required variables before
    calling this function to avoid KeyErrors.
    """

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
    ds["area_nodata"] = (ds["area_data"].max(dim="date") - ds["area_data"]).round(4)

    return ds


def create_plot_per_site(
    df: pd.DataFrame,
    site: str,
    name_field: str = "Name",
    ylabel: str = "area [ha]",
    plot_flooded_vegetation: bool = True,
    plot_ice: bool = False,
):
    fig, ax = plt.subplots(figsize=(10, 4))
    if plot_flooded_vegetation:
        df.query(f'{name_field} == "{site}" and area_nodata == 0').plot(
            x="date", y="flooded_vegetation", ax=ax, c="#31a354", marker=".", title=site
        )
    if plot_ice:
        df.query(f'{name_field} == "{site}" and area_nodata == 0').plot(
            x="date",
            y="snow_and_ice",
            ax=ax,
            c="#666666",
            marker=".",
            alpha=0.7,
            zorder=0,
        )
    df.query(f'{name_field} == "{site}" and area_nodata == 0').plot(x="date", y="water", ax=ax, c="#2c7fb8", marker=".")
    ax.tick_params(axis="x", rotation=45)
    ax.grid()
    ax.set_ylabel(ylabel)

    return fig


def chunk_list(seq, size=4):
    """Split `seq` into sublists with up to `size` items each."""
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def normalize_values(df: pd.DataFrame, name_field: str) -> pd.DataFrame:
    non_numeric_cols = [name_field, "date", "reducer"]
    if "year" in df.columns:
        non_numeric_cols.append("year")
    if "month" in df.columns:
        non_numeric_cols.append("month")
    df_normed = df.drop(columns=non_numeric_cols).divide(df[["area_data", "area_nodata"]].sum(axis=1), axis=0)
    return df[non_numeric_cols].join(df_normed)


def create_timelapse(
    input_lake_gdf: gpd.GeoDataFrame,
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
    Create a Sentinel-2 timelapse GIF for a specific lake.

    This function generates an animated GIF showing Sentinel-2 satellite imagery
    over a date range for a lake identified by its geohash. The timelapse captures
    the summer period (July-August) each year to maximize cloud-free observations.

    Args:
        input_lake_gdf: GeoDataFrame containing lake geometries with an 'id_geohash' column.
        id_geohash: The geohash identifier for the specific lake to visualize.
        timelapse_source: Image source for timelapse imagery ('sentinel2' or 'landsat')
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
    # Ensure output directory exists
    gif_outdir = Path(gif_outdir)
    gif_outdir.mkdir(exist_ok=True, parents=True)

    # Construct output filename based on data source
    if timelapse_source == "landsat":
        outfile = gif_outdir / f"{id_geohash}_LS.gif"
    else:
        outfile = gif_outdir / f"{id_geohash}_S2.gif"

    # Check if output file already exists
    if outfile.exists() and not overwrite_exists:
        print(f"[INFO] Output file already exists: {outfile}")
        print("[INFO] Skipping download (use overwrite_exists=True to re-download)")
        return None

    # Filter to the specific lake feature by geohash
    feature = input_lake_gdf[input_lake_gdf["id_geohash"] == id_geohash]

    if feature.empty:
        raise ValueError(f"No feature found with id_geohash: {id_geohash}")

    # Create a bounding box around the lake with a buffer:
    # 1. Project to EPSG:3995 (World Mercator) for accurate meter-based buffering
    # 2. Apply buffer in meters
    # 3. Convert back to WGS84 (EPSG:4326) for geemap compatibility
    bbox = feature.to_crs(3995).buffer(buffer).to_crs(4326).bounds

    # Convert bbox to Earth Engine FeatureCollection via geemap utilities
    # Note: geemap.bbox_to_gdf creates a GeoDataFrame, then gdf_to_ee converts to EE
    fc = geemap.gdf_to_ee(geemap.bbox_to_gdf(bbox.iloc[0]))
    fc_lake = geemap.gdf_to_ee(feature)

    # Common kwargs shared between sentinel2_timelapse and landsat_timelapse
    timelapse_kwargs = {
        "roi": fc,
        "start_year": start_year,
        "end_year": end_year,
        "start_date": start_date,
        "end_date": end_date,
        "out_gif": str(outfile),
        "frames_per_second": frames_per_second,
        "dimensions": dimensions,
        "title": id_geohash,
        "text_sequence": list(range(start_year, end_year + 1)),
        "overlay_data": fc_lake,
        "overlay_color": "#eeeeee",
    }

    if timelapse_source == "sentinel2":
        # Generate the Sentinel-2 timelapse GIF
        # Uses summer months (Jul-Aug) to maximize cloud-free observations
        geemap.sentinel2_timelapse(**timelapse_kwargs)

    elif timelapse_source == "landsat":
        geemap.landsat_timelapse(**timelapse_kwargs)

    return outfile


def fix_xee_grid_utm(grid: dict) -> dict:
    """
    Fix the UTM grid transformation parameters for xee compatibility.

    This function corrects the Y-scale transformation value in the grid's
    crs_transform tuple to ensure proper pixel alignment when working with
    UTM-projected data in Earth Engine.

    Args:
        grid (dict): A grid dictionary containing 'crs_transform' tuple with
            transformation parameters [origin_x, pixel_size_x, 0, origin_y,
            pixel_size_y, 0].

    Returns:
        dict: The input grid dictionary with corrected crs_transform,
              specifically the pixel Y-size set to -10.

    Example:
        >>> grid = {"crs_transform": (0, 10, 0, 0, 10, 0)}
        >>> fixed_grid = fix_xee_grid_utm(grid)
        >>> fixed_grid["crs_transform"]
        (0, 10, 0, 0, -10, 0)
    """
    transform = list(grid["crs_transform"])
    transform[4] = -10
    grid["crs_transform"] = tuple(transform)
    return grid


def visualize_s2_first_and_last(ds: xr.Dataset, style: str = "rgb") -> plt.Figure:
    """
    Visualize the first and last Sentinel-2 acquisition from an xarray Dataset.

    This function creates a two-panel figure showing Sentinel-2 imagery for the
    first and last time steps in the dataset, useful for comparing temporal
    changes in lake appearance or extent.

    Args:
        ds (xr.Dataset): An xarray Dataset containing Sentinel-2 bands as data
            variables with a 'time' dimension. Expected bands include B2, B3, B4,
            and optionally B8 for vegetation analysis.
        style (str, optional): Visualization style - either 'rgb' for true color
            composite (B4, B3, B2) or any other value for vegetation false color
            composite (B8, B4, B3). Defaults to 'rgb'.

    Returns:
        plt.Figure: A matplotlib Figure object containing two subplots with the
            visualized satellite imagery. The figure should be displayed with
            plt.show() or saved with fig.savefig().

    Example:
        >>> ds = get_rioxarray_ds_from_lake(gdf, "c22iz2n", "2026-05-01", "2026-06-01")
        >>> fig = visualize_s2_first_and_last(ds, style="rgb")
        >>> fig.savefig("comparison.png", dpi=150, bbox_inches="tight")

    Notes:
        - The RGB visualization scales values to 0-1 range by dividing by 1000
        - Values outside 0-1 are clipped
        - Y-axis aspect ratio is set to 'equal' for accurate spatial representation
    """
    fig, axes = plt.subplots(ncols=2)
    for date in [0, -1]:
        ax = axes[date]
        ds_rio = ds.isel(time=date)  # .rio.write_crs("EPSG:32604")
        if style == "rgb":
            (ds_rio[["B4", "B3", "B2"]].to_array() / 1000).clip(0, 1).plot.imshow(ax=ax)
        else:
            (ds_rio[["B8", "B4", "B3"]].to_array() / 3000).clip(0, 1).plot.imshow(ax=ax)
        ax.set_aspect("equal")
        ax.set_title(str(ds_rio.time.dt.strftime("%Y-%m-%d").data))
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    return fig


def get_rioxarray_ds_from_lake(
    lake_gdf: gpd.GeoDataFrame,
    id_geohash: str,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 20,
    buffer: float = 200,
    bands: Optional[Sequence[str]] = None,
    date_windows: Optional[Sequence[Tuple[str, str]]] = None,
    grid_scale: float = 10,
) -> xr.Dataset:
    """
    Load Sentinel-2 satellite imagery for a specific lake as an xarray Dataset.

    This function retrieves Sentinel-2 SR (Surface Reflectance) data from Google Earth Engine
    for a single lake, automatically determining the appropriate UTM coordinate reference
    system based on the lake's location. The data is returned as a rioxarray-enabled
    xarray Dataset with proper georeferencing.

    Args:
        lake_gdf (gpd.GeoDataFrame): A GeoDataFrame containing lake polygons with
            an 'id_geohash' column to identify individual lakes.
        id_geohash (str): The geohash identifier for the specific lake to retrieve
            satellite data for.
        start_date (str): Start date for the imagery query in 'YYYY-MM-DD' format.
        end_date (str): End date for the imagery query in 'YYYY-MM-DD' format.
        max_cloud_cover (float, optional): Maximum cloud cover percentage to filter
            images (0-100). Defaults to 20.
        buffer (float, optional): Buffer distance in meters to expand the lake's
            bounding box for image extraction. Defaults to 200.
        bands (Sequence[str], optional): Subset of S2 bands to open (e.g.
            ("B2", "B3", "B4")). Defaults to None, which opens all bands.
        date_windows (Sequence[Tuple[str, str]], optional): List of
            (start, end) date pairs; when given, the collection is restricted
            to the union of these windows (within start_date/end_date), so only
            imagery near the dates of interest is fetched.
        grid_scale (float, optional): Pixel size in meters for the output grid.
            Defaults to 10 (native S2 resolution); use 20-30 for thumbnails.

    Returns:
        xr.Dataset: An xarray Dataset with Sentinel-2 bands as data variables,
            dimensions (time, y, x), and proper georeferencing (CRS set via rioxarray).
            The Dataset includes all available bands (B1-B12) from the
            COPERNICUS/S2_SR_HARMONIZED collection.

    Example:
        >>> import geopandas as gpd
        >>> gdf = gpd.read_file("lakes.parquet")
        >>> ds = get_rioxarray_ds_from_lake(
        ...     lake_gdf=gdf,
        ...     id_geohash="c22iz2n",
        ...     start_date="2026-05-01",
        ...     end_date="2026-06-01",
        ...     max_cloud_cover=20
        ... )
        >>> print(ds)
    """

    local_gdf = lake_gdf[lake_gdf["id_geohash"] == id_geohash]

    # crs = 'EPSG:32604'
    crs_object = local_gdf.estimate_utm_crs()
    crs = f"EPSG:{crs_object.to_epsg()}"
    aoi = local_gdf.to_crs(crs).buffer(buffer).to_crs(4326).iloc[0]
    fc = geemap.gdf_to_ee(local_gdf)

    grid = helpers.fit_geometry(geometry=aoi, grid_crs=crs, grid_scale=(grid_scale, grid_scale))
    grid = fix_xee_grid_utm(grid)

    ic = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(fc)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover))
        .filter(ee.Filter.calendarRange(6, 9, "month"))
    )
    if date_windows:
        ic = ic.filter(ee.Filter.Or(*[ee.Filter.date(s, e) for s, e in date_windows]))
    if bands:
        ic = ic.select(list(bands))
    ds_rio = xr.open_dataset(ic, engine="ee", **grid).rio.write_crs(crs).sortby("time")

    return ds_rio


@st.cache_resource(show_spinner="Loading Sentinel-2 satellite imagery for a specific lake ")
def cached_get_rioxarray_ds_from_lake(
    _lake_gdf: gpd.GeoDataFrame,
    id_geohash: str,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 20,
    buffer: float = 200,
    bands: Optional[Tuple[str, ...]] = None,
    date_windows: Optional[Tuple[Tuple[str, str], ...]] = None,
    grid_scale: float = 10,
) -> xr.Dataset:
    """Cached wrapper around get_rioxarray_ds_from_lake.

    The GeoDataFrame is excluded from the cache key (underscore prefix);
    id_geohash together with the query parameters identifies the result, so
    re-selecting a lake does not re-hit Earth Engine.
    """
    return get_rioxarray_ds_from_lake(
        lake_gdf=_lake_gdf,
        id_geohash=id_geohash,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=max_cloud_cover,
        buffer=buffer,
        bands=bands,
        date_windows=date_windows,
        grid_scale=grid_scale,
    )


def visualize_s2_xee_cube(ds: xr.Dataset, dates: List[str], style: str = "rgb") -> plt.Figure:
    """
    Visualize Sentinel-2 acquisitions from an xarray Dataset for specified dates.

    This function creates a multi-panel figure showing Sentinel-2 imagery for
    the dates specified in the dates list, useful for comparing temporal changes
    in lake appearance or extent.

    Args:
        ds (xr.Dataset): An xarray Dataset containing Sentinel-2 bands as data
            variables with a 'time' dimension. Expected bands include B2, B3, B4,
            and optionally B8 for vegetation analysis.
        dates (List[str]): List of dates to visualize, in any format accepted by
            xarray's `sel(time=..., method='nearest')`.
        style (str, optional): Visualization style - either 'rgb' for true color
            composite (B4, B3, B2) or any other value for vegetation false color
            composite (B8, B4, B3). Defaults to 'rgb'.

    Returns:
        plt.Figure: A matplotlib Figure object containing subplots with the
            visualized satellite imagery.

    Example:
        >>> ds = get_rioxarray_ds_from_lake(gdf, "c22iz2n", "2026-05-01", "2026-06-01")
        >>> dates = ["2026-05-10", "2026-05-20", "2026-06-01"]
        >>> fig = visualize_s2_xee_cube(ds, dates, style="rgb")
        >>> fig.savefig("comparison.png", dpi=150, bbox_inches="tight")

    Notes:
        - The RGB visualization scales values to 0-1 range by dividing by 1000
        - Values outside 0-1 are clipped
        - Y-axis aspect ratio is set to 'equal' for accurate spatial representation
    """
    fig, axes = plt.subplots(ncols=len(dates))
    i = 0
    for date in dates:
        date_string = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date)
        ax = axes[i]
        # check initial date and extract nearest date
        ds_init = ds.sel(time=date, method="nearest")  # .rio.write_crs("EPSG:32604")
        # check if there are more images for the same date and pull all from the date
        ds_rio = ds.sel(time=ds_init.time.dt.date.astype(str))
        if style == "rgb":
            (ds_rio[["B4", "B3", "B2"]].median(dim="time", skipna=True).to_array() / 1000).clip(0, 1).plot.imshow(ax=ax)
        else:
            (ds_rio[["B8", "B4", "B3"]].median(dim="time", skipna=True).to_array() / 3000).clip(0, 1).plot.imshow(ax=ax)
        ax.set_aspect("equal")
        ax.set_title(date_string)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        i += 1
    return fig


@st.cache_resource(show_spinner=False)
def cached_visualize_cube(_ds: xr.Dataset, dates: List[str], style: str = "rgb", id_geohash: Optional[str] = None):
    # _ds is excluded from the cache key (underscore prefix), so id_geohash must be
    # part of the key — otherwise two lakes with the same dates share one figure.
    return visualize_s2_xee_cube(_ds, dates=tuple(dates), style=style)
