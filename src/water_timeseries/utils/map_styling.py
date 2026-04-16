"""Map styling utilities for interactive visualization."""


def get_colored_style_function(
    color_column: str = "NetChange_perc",
    vmin: float = -40,
    vmax: float = 40,
    colormap=None,
    default_color: str = "#cccccc",
    fill_opacity: float = 0.6,
    edge_color: str = "#dddddd",
    edge_weight: float = 1,
):
    """Create a style function for folium polygons based on a numeric column.

    Args:
        color_column: Column name to use for coloring
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization
        colormap: Matplotlib colormap (defaults to RdBu_r)
        default_color: Color for missing/null values
        fill_opacity: Opacity of polygon fill (0-1)
        edge_color: Color of polygon edges
        edge_weight: Width of polygon edges

    Returns:
        style_function: Function that can be passed to folium.GeoJson style_function parameter
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if colormap is None:
        colormap = plt.cm.RdBu_r

    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    def style_function(feature):
        props = feature.get("properties", {})
        value = props.get(color_column, None)

        if value is None or pd.isna(value):
            return {
                "fillColor": default_color,
                "color": edge_color,
                "weight": edge_weight,
                "fillOpacity": 0.5,
            }

        # Normalize value and get color from colormap
        color = colormap(norm(value))
        # Convert RGBA to hex manually to avoid JSON serialization issues
        r, g, b, a = color
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

        return {
            "fillColor": hex_color,
            "color": edge_color,
            "weight": edge_weight,
            "fillOpacity": fill_opacity,
        }

    return style_function


def get_default_style_function(
    fill_color: str = "blue",
    edge_color: str = "#dddddd",
    edge_weight: float = 1,
    fill_opacity: float = 0.5,
):
    """Create a default style function for folium polygons.

    Args:
        fill_color: Fill color for all polygons
        edge_color: Color of polygon edges
        edge_weight: Width of polygon edges
        fill_opacity: Opacity of polygon fill (0-1)

    Returns:
        style_function: Function that can be passed to folium.GeoJson style_function parameter
    """

    def style_function(feature):
        return {
            "fillColor": fill_color,
            "color": edge_color,
            "weight": edge_weight,
            "fillOpacity": fill_opacity,
        }

    return style_function


def format_tooltip_columns(
    valid_gdf,
    id_column: str,
    tooltip_columns=None,
):
    """Format columns for tooltip display to avoid JSON serialization issues.

    Args:
        valid_gdf: GeoDataFrame to format
        id_column: Name of the ID column (always shown first)
        tooltip_columns: List of tuples (original_col, display_alias, format_string, unit)
                        If None, uses default NetChange columns

    Returns:
        formatted_gdf: GeoDataFrame with display columns added
        fields_to_show: List of field names for tooltip
        aliases_to_show: List of field aliases for tooltip
    """
    import pandas as pd

    if tooltip_columns is None:
        # Default tooltip columns if NetChange data exists
        tooltip_columns = [
            ("NetChange_perc", "Net Change (%):", "{:.2f}", "%"),
            ("NetChange_ha", "Net Change (ha):", "{:.2f}", " ha"),
        ]

    # Check if we have any of the tooltip columns
    has_tooltip_data = any(col[0] in valid_gdf.columns for col in tooltip_columns)

    if has_tooltip_data:
        valid_gdf = valid_gdf.copy()
        display_columns = []
        alias_mapping = []

        for orig_col, alias, fmt, unit in tooltip_columns:
            if orig_col in valid_gdf.columns:
                display_col = f"{orig_col}_display"
                valid_gdf[display_col] = valid_gdf[orig_col].apply(
                    lambda x: f"{fmt.format(x)}{unit}" if pd.notna(x) else "N/A"
                )
                display_columns.append(display_col)
                alias_mapping.append(alias)

        # Show ID first, then formatted columns
        fields_to_show = [id_column] + display_columns
        aliases_to_show = ["ID:"] + alias_mapping
    else:
        # Fallback to ID only
        fields_to_show = [id_column]
        aliases_to_show = ["ID:"]

    return valid_gdf, fields_to_show, aliases_to_show


def create_tile_layers():
    """Create tile layers for folium maps.

    Returns:
        List of tile layer names that can be added to folium.Map
    """
    return ["CartoDB.DarkMatter", "Esri.WorldImagery"]
