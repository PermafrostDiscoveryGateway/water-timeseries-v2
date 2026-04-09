"""Visualization utilities for GeoDataFrames and map rendering."""

from typing import Any, Dict, List, Optional

import pandas as pd

# Color scales configuration
COLOR_SCALES = {
    "RdYlBu": {
        "name": "RdYlBu",
        "zmid": 0,  # Center the colormap at 0 for diverging data
        "description": "Red-Yellow-Blue: red for negative, blue for positive",
    },
    "Viridis": {
        "name": "Viridis",
        "zmid": None,
        "description": "Perceptually uniform colormap",
    },
}


# Default styling for map polygons
MAP_STYLING = {
    "unselected": {
        "opacity": 0.7,
        "line_width": 1,
        "line_color": "gray",
    },
    "selected": {
        "opacity": 0.8,
        "line_width": 2,
        "line_color": "black",
    },
}


# Default hover columns for lake data
DEFAULT_HOVER_COLUMNS = [
    "id_geohash",
    "Area_start_ha",
    "Area_end_ha",
    "NetChange_ha",
    "NetChange_perc",
]


# Add legend for NetChange_perc color scale
def get_legend_html_net_change() -> str:
    LEGEND_HTML_NET_CHANGE = """
        <div style="
            position: fixed;
            bottom: 40px;
            right: 10px;
            width: 180px;
            height: auto;
            border: 2px solid grey;
            z-index: 9999;
            font-size: 12px;
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
        ">
        <p style="margin: 0 0 5px 0; font-weight: bold;">Net Change (%)</p>
        <div style="
            background: linear-gradient(to right, #d73027, #f46d43, #fdae61, #fee090, #e0f3f8, #abd9e9, #74add1, #4575b4);
            width: 100%;
            height: 20px;
            border: 1px solid #ccc;
        "></div>
        <div style="display: flex; justify-content: space-between; width: 100%; margin-top: 3px;">
            <span>-40%</span>
            <span>0%</span>
            <span>+40%</span>
        </div>
        <p style="margin: 8px 0 0 0; font-size: 10px; color: #666;">
            Red = Decrease<br>
            Blue = Increase
        </p>
        </div>
    """
    return LEGEND_HTML_NET_CHANGE


def build_hover_template(
    id_column: str,
    hover_fields: List[str],
    extra_template: str = "",
) -> str:
    """
    Build a hover template for Plotly maps.

    Args:
        id_column: Name of the ID column (shown in bold).
        hover_fields: List of field names to include in hover.
        extra_template: Additional template string to append.

    Returns:
        Formatted hover template string.

    Example:
        >>> template = build_hover_template("id_geohash", ["Area_ha", "NetChange_perc"])
    """
    hover_template = f"<b>{id_column}: %{{customdata[0]}}</b><br>"

    for i, field in enumerate(hover_fields):
        hover_template += f"{field}: %{{customdata[{i + 1}]}}<br>"

    hover_template += "<extra></extra>"  # Hide secondary hover info

    if extra_template:
        hover_template += extra_template

    return hover_template


def prepare_custom_data_for_plotly(
    df: pd.DataFrame,
    id_column: str,
    hover_fields: List[str],
) -> List[tuple]:
    """
    Prepare custom data for Plotly hover tooltips.

    Converts DataFrame columns to a format suitable for Plotly's customdata.
    Handles NaN values and converts complex types (list, tuple, etc.) to strings.

    Args:
        df: GeoDataFrame or DataFrame to extract data from.
        id_column: Name of the ID column.
        hover_fields: List of field names to include in hover.

    Returns:
        List of tuples where each tuple contains values for one row.

    Example:
        >>> custom_data = prepare_custom_data_for_plotly(gdf, "id_geohash", ["Area_ha"])
    """
    custom_data = []
    hover_cols = [id_column] + hover_fields

    for col in hover_cols:
        col_data = []
        for val in df[col]:
            if pd.isna(val):
                col_data.append("")
            elif isinstance(val, (list, tuple, set, bytes)):
                col_data.append(str(list(val)))
            else:
                col_data.append(str(val))
        custom_data.append(col_data)

    # Transpose to get rows as tuples
    return list(zip(*custom_data))


def get_z_values_for_coloring(
    gdf: pd.DataFrame,
    column: str = "NetChange_perc",
    clip_range: tuple = (-50, 50),
    default_value: float = 0,
) -> List[float]:
    """
    Extract and prepare z-values for polygon coloring.

    Args:
        gdf: GeoDataFrame to extract values from.
        column: Column name to use for coloring.
        clip_range: Min/max values to clip the data to (optional).
        default_value: Default value if column doesn't exist.

    Returns:
        List of z-values for coloring.
    """
    if column in gdf.columns:
        z_values = gdf[column].fillna(default_value)
        if clip_range:
            z_values = z_values.clip(lower=clip_range[0], upper=clip_range[1])
        return z_values.tolist()
    else:
        return [default_value] * len(gdf)


def get_colorbar_config(
    title: str = "Value",
    color_scale: str = "RdYlBu",
    zmid: Optional[float] = 0,
) -> Dict[str, Any]:
    """
    Get colorbar configuration for Plotly maps.

    Args:
        title: Title for the colorbar.
        color_scale: Name of the color scale.
        zmid: Center point for diverging color scales (optional).

    Returns:
        Dictionary with colorbar configuration.
    """
    config = {
        "colorscale": color_scale,
        "showscale": True,
        "colorbar_title": title,
    }

    # Add zmid for diverging color scales
    if zmid is not None and color_scale in ["RdYlBu", "RdBu", "PiYG", "PRGn"]:
        config["zmid"] = zmid

    return config


def gdf_to_geojson_feature_collection(gdf: pd.DataFrame) -> Dict:
    """
    Convert GeoDataFrame to GeoJSON feature collection.

    Args:
        gdf: GeoDataFrame to convert.

    Returns:
        GeoJSON feature collection dictionary.
    """
    return gdf.__geo_interface__
