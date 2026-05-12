"""Spatial utilities for working with geospatial data."""

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import sklearn
from shapely.geometry import box
from tqdm import tqdm


def filter_gdf_by_bbox(
    gdf: gpd.GeoDataFrame,
    bbox_west: Optional[float] = None,
    bbox_south: Optional[float] = None,
    bbox_east: Optional[float] = None,
    bbox_north: Optional[float] = None,
    id_column: str = "id_geohash",
) -> gpd.GeoDataFrame:
    """Filter a GeoDataFrame by bounding box coordinates.

       Filters features based on their centroid falling within the specified
       bounding box. At least one bbox boundary must be provided.

    rt   Args:
           gdf: Input GeoDataFrame with geometry column.
           bbox_west: Minimum longitude (west) boundary.
           bbox_south: Minimum latitude (south) boundary.
           bbox_east: Maximum longitude (east) boundary.
           bbox_north: Maximum latitude (north) boundary.
           id_column: Name of the column containing unique identifiers.
               Defaults to "id_geohash".

       Returns:
           Filtered GeoDataFrame containing only features whose centroids
           fall within the bounding box.

       Raises:
           ValueError: If no bbox parameters are provided.

       Example:
           >>> import geopandas as gpd
           >>> gdf = gpd.read_file("lakes.gpkg")
           >>> filtered = filter_gdf_by_bbox(gdf, bbox_west=10, bbox_south=50, bbox_east=20, bbox_north=60)
    """
    # Check if at least one bbox parameter is provided
    if all(v is None for v in [bbox_west, bbox_south, bbox_east, bbox_north]):
        raise ValueError("At least one bbox parameter must be provided")

    # Calculate centroids
    cent = gdf.geometry.centroid

    # Build mask for filtering
    mask = True
    if bbox_west is not None:
        mask &= cent.x >= bbox_west
    if bbox_east is not None:
        mask &= cent.x <= bbox_east
    if bbox_south is not None:
        mask &= cent.y >= bbox_south
    if bbox_north is not None:
        mask &= cent.y <= bbox_north

    return gdf[mask]


# create area calculation function for bounding box in arctic projection
def bbox_area_km2_arctic(gdf: gpd.GeoDataFrame) -> float:
    """Calculate the area of a bounding box in square kilometers for a GeoDataFrame in Arctic projection.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame with geometry column.

    Returns:
        float: Area of the bounding box in square kilometers.
    """
    gdf_proj = gdf.to_crs("EPSG:3995")
    minx, miny, maxx, maxy = gdf_proj.total_bounds
    bbox = box(minx, miny, maxx, maxy)
    return bbox.area / 1_000_000.0


def assign_classes_by_size(gdf: gpd.GeoDataFrame, size: int) -> np.ndarray:
    gdf = gdf.copy()
    gdf["class"] = (np.arange(len(gdf)) // size) + 1
    return (np.arange(len(gdf)) // size) + 1


def chunk_gdf_spatial_kmeans(gdf: gpd.GeoDataFrame, chunk_size: int, epsg: int | str = 3995) -> list[gpd.GeoDataFrame]:
    """Split a GeoDataFrame into spatially coherent chunks using K-means clustering.

    This function groups features into chunks based on the spatial distribution
    of their centroids. It uses K-means clustering on the projected centroid
    coordinates to create spatially contiguous groups, then splits them into
    chunks of approximately the specified size.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame with geometry column.
        chunk_size (int): Target number of features per chunk. Actual chunk
            sizes may vary slightly depending on clustering results.
        epsg (int | str, optional): EPSG code for the coordinate reference
            system to use for calculating centroids and clustering.
            Defaults to 3995 (Arctic projection).

    Returns:
        list[gpd.GeoDataFrame]: List of GeoDataFrames, each containing a
            subset of the original features. The chunks are spatially
            coherent and approximately equal in size.

    Note:
        The function uses K-means clustering to group features by proximity,
        then splits oversized clusters into additional chunks to match the
        target chunk size.
    """
    # create
    centroids = gdf[["geometry"]]
    centroids.loc[:, "geometry"] = gdf.centroid.to_crs(epsg)
    centroids.loc[:, "x"] = centroids.geometry.x
    centroids.loc[:, "y"] = centroids.geometry.y
    # get number of clusters based on chunk size and total number of rows
    n_clusters = int(len(gdf) / chunk_size)
    # run clustering
    cl = sklearn.cluster.KMeans(n_clusters=n_clusters).fit_predict(centroids[["x", "y"]])
    # find class names and calculate area for each class
    centroids.loc[:, "cluster_kmeans"] = cl
    classes_kmeans = pd.unique(centroids["cluster_kmeans"])
    # split data into chunks based on class names and chunk size
    chunks = []
    for c in tqdm(classes_kmeans):
        chunk = centroids.query(f"cluster_kmeans == {c}")
        # split data into smaller chunks if size is larger than chunk size
        if len(chunk) > chunk_size:
            sub_chunks = [chunk.iloc[i : i + chunk_size] for i in range(0, len(chunk), chunk_size)]
            chunks.extend([gdf.loc[sub_chunk.index] for sub_chunk in sub_chunks])
        else:
            chunks.append(gdf.loc[chunk.index])

    return chunks


def chunk_gdf_simple(gdf: gpd.GeoDataFrame, chunk_size: int) -> list[gpd.GeoDataFrame]:
    """Split a GeoDataFrame into simple chunks.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame with geometry column.
        chunk_size (int): Target number of features per chunk.

    Returns:
        list[gpd.GeoDataFrame]: List of GeoDataFrames, each containing a
            subset of the original features.
    """
    chunks = []
    for i in range(0, len(gdf), chunk_size):
        chunk = gdf.iloc[i : i + chunk_size]
        chunks.append(chunk)
    return chunks
