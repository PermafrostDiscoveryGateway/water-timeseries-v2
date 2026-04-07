from datetime import datetime, timedelta
from pathlib import Path

import ee
import eemont  # noqa: F401
import geemap
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from shapely.geometry import box


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
) -> ee.Image:
    """
    Generates a monthly dynamic world composite and then returns a binary mask where
    the value is 1 for each pixel if water, snow or ice was the top probability
    class according to dynamic world. The pixel in the mask is 0 otherwise.

    start_date: ee.Date, the mask will be calculated for start_date to start_date + 1 month.
    dw: ee.ImageCollection, the dynamic world image collection.
    crs: str, optional. The coordinate reference system.
    """
    # Cast startDate back to an ee.Date, type erasure happens when mapping on a list.
    start_date = ee.Date(start_date)
    end_date = start_date.advance(1, "month")
    dw_filtered_image_collection = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(polygons).filterDate(start_date, end_date)
    )

    image = ee.Algorithms.If(
        dw_filtered_image_collection.size().eq(0),
        create_no_data_image(),
        dw_filtered_image_collection.select("label")
        .reduce(ee.Reducer.mode())
        .set("system:time_start", start_date.millis())
        .setDefaultProjection(crs=crs, scale=scale),
    )
    return image


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
    Generates a monthly dynamic world composite and then returns a binary mask where
    the value is 1 for each pixel if water, snow or ice was the top probability
    class according to dynamic world. The pixel in the mask is 0 otherwise.

    start_date: ee.Date, the mask will be calculated for start_date to start_date + 1 month.
    dw: ee.ImageCollection, the dynamic world image collection.
    crs: str, optional. The coordinate reference system.
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
    Generates a Dynamic World composite reduced by mode.
    Returns an ee.Image, or None if no images are available for the period.
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
    fc_lake =  geemap.gdf_to_ee(feature)

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
