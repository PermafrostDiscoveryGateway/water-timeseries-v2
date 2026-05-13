"""Map Viewer dashboard component using Streamlit and lonboard for high-performance mapping."""

import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import folium
import geemap
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
from streamlit_folium import st_folium

from water_timeseries.breakpoint import NRTBreakpoint
from water_timeseries.dataset import DWDataset, JRCDataset
from water_timeseries.downloader import EarthEngineDownloader
from water_timeseries.utils.dashboard import (
    check_dataset_availability,
    load_dataset,
    plot_time_series_data,
)
from water_timeseries.utils.io import load_vector_dataset
from water_timeseries.utils.map_styling import (
    create_tile_layers,
    format_tooltip_columns,
    get_colored_style_function,
    get_default_style_function,
)
from water_timeseries.utils.visualization import (
    DEFAULT_HOVER_COLUMNS,
    get_legend_html_net_change,
)

# Initialize Earth Engine - only if running in Streamlit context
# Check environment variable first (works outside Streamlit)
if "EARTHENGINE_TOKEN" in os.environ.keys():
    print("setting up with TOKEN")
    geemap.ee_initialize()
else:
    # Try Streamlit secrets (only works when running in Streamlit)
    try:
        if "EARTHENGINE_TOKEN" in st.secrets.keys():
            print("setting up with TOKEN from secrets")
            os.environ["EARTHENGINE_TOKEN"] = st.secrets["EARTHENGINE_TOKEN"]
            geemap.ee_initialize()
    except Exception:
        # Not running in Streamlit context, skip initialization
        pass


class MapViewer:
    """Interactive map viewer for GeoDataFrames using Streamlit and lonboard.

    Features:
    - Display GeoDataFrame on an interactive map (high performance for large datasets)
    - Hover tooltips showing feature attributes
    - Click to select features and store their id_geohash value
    - Supports multiple backends: folium, pydeck (WebGL), st.map
    """

    def __init__(
        self,
        gdf: Optional[gpd.GeoDataFrame] = None,
        parquet_path: Optional[Path | str] = None,
        geometry_column: str = "geometry",
        id_column: str = "id_geohash",
        hover_columns: Optional[List[str]] = None,
        map_center: Optional[dict] = None,
        zoom: int = 10,
        map_backend: str = "folium",  # "folium", or "st_map"
        max_features: Optional[int] = None,  # Limit features for faster loading
        drained_gdf: Optional[gpd.GeoDataFrame] = None,
        drained_label: Optional[str] = None,
        show_main_layer: bool = True,
    ):
        """Initialize the MapViewer.

        Args:
            gdf: GeoDataFrame to display. If None, will load from parquet_path.
            parquet_path: Path to parquet file to load as GeoDataFrame.
            geometry_column: Name of the geometry column in the GeoDataFrame.
            id_column: Name of the column containing unique identifiers.
            hover_columns: List of column names to show on hover. If None, shows all.
            map_center: Dictionary with 'lat' and 'lon' keys for map center.
            zoom: Initial zoom level for the map.
        """
        self.geometry_column = geometry_column
        self.id_column = id_column
        # Default hover columns if not specified (use from visualization module)
        self.hover_columns = hover_columns or DEFAULT_HOVER_COLUMNS
        self.zoom = zoom
        self.map_center = map_center
        self.map_backend = map_backend  # "folium" or "st_map"
        self.max_features = max_features  # Limit features for faster loading
        self.drained_gdf = drained_gdf
        self.drained_label = drained_label
        self.show_main_layer = show_main_layer

        # Load data if parquet_path provided
        if gdf is None and parquet_path is not None:
            self.gdf = self._load_parquet(parquet_path)
        elif gdf is not None:
            self.gdf = gdf
        else:
            raise ValueError("Either gdf or parquet_path must be provided")

        # Initialize session state for storing selected id_geohash
        if "selected_geohash" not in st.session_state:
            st.session_state.selected_geohash = None
        if "clicked_features" not in st.session_state:
            st.session_state.clicked_features = []

    def _load_parquet(self, path: Path | str) -> gpd.GeoDataFrame:
        """Load a GeoDataFrame from a parquet file.

        Args:
            path: Path to the parquet file.

        Returns:
            GeoDataFrame loaded from parquet.
        """
        # Use the utility function to load the vector dataset
        gdf = load_vector_dataset(path)

        # Ensure the geometry column is properly set
        if self.geometry_column in gdf.columns:
            gdf = gdf.set_geometry(self.geometry_column)

        # Set CRS if not already set
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)

        # Filter out rows with invalid/empty geometries
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        gdf.sort_values(by="Area_start_ha", ascending=False, inplace=True)

        return gdf

    def render(self) -> Optional[str]:
        """Render the interactive map in Streamlit using the selected backend.

        Returns:
            The selected id_geohash value if a feature was clicked, None otherwise.
        """
        st.subheader("Interactive Map Viewer")

        # Get valid indices (after filtering out invalid geometries)
        valid_mask = self.gdf.geometry.notna() & ~self.gdf.geometry.is_empty
        valid_gdf = self.gdf[valid_mask].copy()

        # Apply sampling if max_features specified (for faster loading)
        if self.max_features and len(valid_gdf) > self.max_features:
            # valid_gdf = valid_gdf.sample(n=self.max_features, random_state=42).reset_index(drop=True)
            valid_gdf = valid_gdf.head(n=self.max_features).reset_index(drop=True)
            st.caption(f"Showing largest {self.max_features} of {len(self.gdf)} features (use max_features to change)")

        # Ensure geometry is in proper shapely format
        valid_gdf = valid_gdf.reset_index(drop=True)

        if len(valid_gdf) == 0:
            st.warning("No valid geometries found.")
            return None

        # Use st.map for simple point rendering
        if self.map_backend == "st_map":
            st.map(valid_gdf)
            return None

        # Default: use folium
        return self._render_folium(valid_gdf, layer_column=getattr(self, "layer_column", None))

    def _render_folium(self, valid_gdf: gpd.GeoDataFrame, layer_column: Optional[str] = None) -> Optional[str]:
        """Render using folium with optional layer selection.

        Args:
            valid_gdf: The GeoDataFrame to render.
            layer_column: Column name to split into separate layers. Each unique
                         value becomes a toggleable layer. If None, shows single layer.

        Returns:
            The selected id_geohash value if a feature was clicked, None otherwise.
        """

        # Determine center of map
        if self.map_center is None:
            centroid = valid_gdf.geometry.unary_union.centroid
            center = [centroid.y, centroid.x]  # [lat, lon]
        else:
            center = [self.map_center.get("lat", 0), self.map_center.get("lon", 0)]

        m = folium.Map(location=center, zoom_start=self.zoom)

        # Add tile layers using utility function
        for tile_name in create_tile_layers():
            folium.TileLayer(tile_name).add_to(m)

        # Add WMS layer for permafrost data
        wms_url = "https://maps.awi.de/services/common/permafrost/ows"
        folium.WmsTileLayer(
            url=wms_url,
            name="TCVIS Landsat Trends 2005-2024 (AWI)",
            styles="composite",
            transparent=True,
            overlay=False,
            layers="tcvis",
        ).add_to(m)

        # Create style function based on whether NetChange_perc column exists
        if "NetChange_perc" in valid_gdf.columns:
            style_function = get_colored_style_function(
                color_column="NetChange_perc",
                vmin=-40,
                vmax=40,
                colormap=plt.cm.RdYlBu,
            )
        else:
            style_function = get_default_style_function()

        # Format tooltip columns using utility function
        # Include Area columns for full tooltip display
        tooltip_columns = [
            ("NetChange_perc", "Net Change (%):", "{:.2f}", "%"),
            ("NetChange_ha", "Net Change (ha):", "{:.2f}", " ha"),
            ("Area_start_ha", "Lake Area year 2000 (ha):", "{:.2f}", " ha"),
            ("Area_end_ha", "Lake Area year 2020 (ha):", "{:.2f}", " ha"),
        ]
        valid_gdf = _sanitize_geojson_properties(valid_gdf)
        valid_gdf, fields_to_show, aliases_to_show = format_tooltip_columns(
            valid_gdf,
            id_column=self.id_column,
            tooltip_columns=tooltip_columns,
        )

        folium.GeoJson(
            valid_gdf,
            name="Lakes",
            show=self.show_main_layer,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=fields_to_show,
                aliases=aliases_to_show,
            ),
        ).add_to(m)

        drained_gdf = getattr(self, "drained_gdf", None)
        if drained_gdf is not None and len(drained_gdf) > 0:
            drained_gdf = drained_gdf[drained_gdf.geometry.notna() & ~drained_gdf.geometry.is_empty].copy()
            if len(drained_gdf) > 0:
                drained_gdf = _sanitize_geojson_properties(drained_gdf)
                drained_tooltip_columns = [
                    ("water_residual", "Water residual:", "{:.2f}", ""),
                    ("water_observed", "Observed water:", "{:.2f}", ""),
                    ("water_predicted", "Predicted water:", "{:.2f}", ""),
                ]
                drained_gdf, drained_fields, drained_aliases = format_tooltip_columns(
                    drained_gdf,
                    id_column=self.id_column,
                    tooltip_columns=drained_tooltip_columns,
                )
                drained_label = getattr(self, "drained_label", None)
                drained_layer_name = "Drained last month"
                if drained_label:
                    drained_layer_name = f"{drained_layer_name} ({drained_label})"
                folium.GeoJson(
                    drained_gdf,
                    name=drained_layer_name,
                    style_function=get_default_style_function(
                        fill_color="#d73027",
                        edge_color="#7f0000",
                        edge_weight=2,
                        fill_opacity=0.7,
                    ),
                    tooltip=folium.GeoJsonTooltip(
                        fields=drained_fields,
                        aliases=drained_aliases,
                    ),
                ).add_to(m)

        folium.LayerControl().add_to(m)

        m.get_root().html.add_child(folium.Element(get_legend_html_net_change()))

        # Render the map and get click data
        # Note: returned_objects includes 'last_active_drawing' for click detection
        result = st_folium(m, height=600, width="100%", key="map_viewer", returned_objects=["last_active_drawing"])

        # Extract id_geohash from clicked feature
        clicked_id = None
        if result and "last_active_drawing" in result:
            clicked_data = result["last_active_drawing"]
            if clicked_data and "properties" in clicked_data:
                clicked_id = clicked_data["properties"].get(self.id_column)

        # Update session state only if a NEW feature was clicked (not the same one)
        if clicked_id and clicked_id != st.session_state.get("selected_geohash"):
            st.session_state.selected_geohash = clicked_id
            if clicked_id not in st.session_state.clicked_features:
                st.session_state.clicked_features.append(clicked_id)
            st.rerun()

        return None

    def get_selected_geohash(self) -> Optional[str]:
        """Get the currently selected geohash from session state.

        Returns:
            The selected id_geohash value or None.
        """
        return st.session_state.get("selected_geohash")

    def get_clicked_features(self) -> List[str]:
        """Get list of all clicked features.

        Returns:
            List of clicked id_geohash values.
        """
        return st.session_state.get("clicked_features", [])

    def clear_selection(self) -> None:
        """Clear the current selection.

        Removes the currently selected geohash from the session state,
        allowing the user to make a new selection.
        """
        st.session_state.selected_geohash = None


def _get_latest_dw_analysis_date(dw_dataset: DWDataset) -> Optional[pd.Timestamp]:
    """Return the most recent DW date that has at least one non-NaN water value."""
    available_dates = _get_available_dw_analysis_dates(dw_dataset)
    if len(available_dates) == 0:
        return None
    return available_dates[-1]


def _get_available_dw_analysis_dates(dw_dataset: DWDataset) -> List[pd.Timestamp]:
    """Return DW dates that have at least one non-NaN water value."""
    water_col = dw_dataset.water_column
    water_da = dw_dataset.ds_normalized[water_col]
    valid_counts = water_da.notnull().sum(dim="id_geohash").compute()
    available_dates = valid_counts["date"].values[np.asarray(valid_counts.values) > 0]
    if len(available_dates) == 0:
        return []
    available_dates = pd.to_datetime(available_dates)
    return sorted(list(available_dates))


def _sanitize_geojson_properties(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert non-JSON-serializable values (e.g., Timestamp) to strings."""
    sanitized = gdf.copy()
    geometry_col = sanitized.geometry.name

    datetime_cols = sanitized.select_dtypes(include=["datetime", "datetimetz"]).columns
    for col in datetime_cols:
        if col != geometry_col:
            sanitized[col] = sanitized[col].astype(str)

    object_cols = sanitized.select_dtypes(include=["object"]).columns
    for col in object_cols:
        if col == geometry_col:
            continue
        sanitized[col] = sanitized[col].apply(
            lambda value: (
                pd.to_datetime(value).isoformat()
                if isinstance(value, (pd.Timestamp, np.datetime64))
                else value
            )
        )

    return sanitized


def create_app(
    data_path: str | Path = "tests/data/lake_polygons.parquet",
    zarr_path: str | Path = "tests/data/lakes_dw_test.zarr",
    zarr_path_jrc: str | Path = "tests/data/lakes_jrc_test.zarr",
):
    """Create the Streamlit app with map viewer.

    Args:
        data_path: Path to the parquet file containing lake polygons.
        zarr_path: Path to the zarr file containing Dynamic World time series data.
        zarr_path_jrc: Path to the zarr file containing JRC time series data.
    """
    st.set_page_config(page_title="Lake Polygon Map Viewer", page_icon="🗺️", layout="wide")

    st.title("🗺️ Lake Polygon Map Viewer")
    st.markdown("""
    This dashboard displays lake polygons from a GeoDataFrame. 
    - Polygons are **colored by NetChange_perc** (red = decrease, blue = increase)
    - **Hover** over a feature to see its attributes
    - **Click** on a feature to select it and view time series & create timelapses
    """)

    # Create sidebar for controls
    st.sidebar.header("Settings")

    # Plotting mode selection (static vs dynamic/interactive) - defaults to interactive
    is_interactive = st.sidebar.toggle(
        "Interactive Plotting",
        value=True,
        help="Enable interactive Plotly plots (hover for details, zoom, pan)",
    )
    if is_interactive:
        st.sidebar.caption("🖱️ Interactive mode - hover to see values, zoom & pan available")
    else:
        st.sidebar.caption("📊 Static mode - matplotlib plots")

    map_backend = "folium"

    # Performance settings for large datasets
    st.sidebar.divider()
    st.sidebar.subheader("Performance")
    max_features = st.sidebar.number_input(
        "Max features to load",
        min_value=10,
        max_value=50000,
        value=1000,
        step=100,
        help="Limit number of polygons for faster loading. Set to 0 for no limit.",
    )
    if max_features == 0:
        max_features = None

    # Use function parameters for data paths
    data_path_input = str(data_path)
    zarr_path_input = str(zarr_path)
    zarr_path_jrc_input = str(zarr_path_jrc)
    id_column = "id_geohash"
    zoom_level = 10

    # Initialize dataset in session state if not already
    if "dw_dataset" not in st.session_state:
        st.session_state.dw_dataset = None
    if "jrc_dataset" not in st.session_state:
        st.session_state.jrc_dataset = None
    if "show_ts_popup" not in st.session_state:
        st.session_state.show_ts_popup = False
    if "downloaded_dsdw" not in st.session_state:
        st.session_state.downloaded_dsdw = None
    if "downloaded_dsjrc" not in st.session_state:
        st.session_state.downloaded_dsjrc = None
    if "nrt_breaks" not in st.session_state:
        st.session_state.nrt_breaks = None
    if "nrt_breaks_date" not in st.session_state:
        st.session_state.nrt_breaks_date = None

    # Near-real-time drainage overlay
    st.sidebar.divider()
    st.sidebar.subheader("Near-real-time drainage")
    show_drained = st.sidebar.checkbox(
        "Show lakes drained in the last month",
        value=False,
        help="Uses NRT breakpoints on the most recent DW month (water_residual < -0.25).",
    )
    drained_breaks = None
    drained_label = None
    drain_threshold = -0.25

    if show_drained:
        dw_dataset = st.session_state.get("dw_dataset")
        if dw_dataset is None:
            dw_dataset, success = load_dataset(
                "dw",
                zarr_path_input,
                st.session_state.downloaded_dsdw,
                st.session_state.dw_dataset,
            )
            if success and dw_dataset is not None:
                st.session_state.dw_dataset = dw_dataset

        if st.session_state.dw_dataset is not None:
            available_dates = _get_available_dw_analysis_dates(st.session_state.dw_dataset)
            if len(available_dates) == 0:
                st.sidebar.warning("DW dataset has no valid water observations for NRT analysis.")
            else:
                date_options = [d.strftime("%Y-%m") for d in available_dates]
                default_idx = len(date_options) - 1
                selected_analysis_month = st.sidebar.selectbox(
                    "NRT analysis month",
                    date_options,
                    index=default_idx,
                    help="Breakpoints are computed for this month; latest with data is selected by default.",
                )
                analysis_date_ts = pd.to_datetime(selected_analysis_month)
                drained_label = analysis_date_ts.strftime("%Y-%m")
            if (
                len(available_dates) > 0
                and (
                    st.session_state.get("nrt_breaks") is None
                    or st.session_state.get("nrt_breaks_date") != drained_label
                )
            ):
                with st.spinner(f"Calculating NRT breakpoints for {drained_label}..."):
                    try:
                        st.session_state.nrt_breaks = NRTBreakpoint().calculate_break(
                            st.session_state.dw_dataset,
                            analysis_date=analysis_date_ts,
                            data_aggregation_period="all",
                        )
                        st.session_state.nrt_breaks_date = drained_label
                    except Exception as e:
                        st.session_state.nrt_breaks = pd.DataFrame(columns=["water_residual"])
                        st.session_state.nrt_breaks_date = drained_label
                        st.sidebar.error(f"Could not calculate NRT breakpoints: {e}")

            if (
                len(available_dates) > 0
                and st.session_state.nrt_breaks is not None
                and not st.session_state.nrt_breaks.empty
            ):
                drained_breaks = st.session_state.nrt_breaks.query(
                    "water_residual < @drain_threshold"
                ).copy()
                st.sidebar.caption(f"{len(drained_breaks)} lakes flagged for {drained_label}")
            elif len(available_dates) > 0:
                st.sidebar.caption(f"No NRT breakpoints available for {drained_label}")
        else:
            st.sidebar.warning("DW dataset not available; cannot compute NRT drainage.")

    # Create map viewer
    try:
        viewer = MapViewer(
            parquet_path=data_path_input,
            id_column=id_column,
            zoom=zoom_level,
            map_backend=map_backend,
            max_features=max_features,
        )
        if show_drained and drained_breaks is not None and not drained_breaks.empty:
            drained_gdf = viewer.gdf.merge(
                drained_breaks.reset_index(),
                on=id_column,
                how="inner",
            )
            viewer.drained_gdf = drained_gdf
            viewer.drained_label = drained_label
            viewer.show_main_layer = False
        elif show_drained:
            viewer.drained_gdf = None
            viewer.drained_label = drained_label
            viewer.show_main_layer = True
        # viewer.layer_column = layer_column  # Set layer column for folium

        # Render the map
        selected = viewer.render()  # noqa: F841

        # Display selected features in sidebar
        st.sidebar.divider()
        st.sidebar.subheader("Previously Selected Features")

        clicked = viewer.get_clicked_features()

        # Create a dropdown for selecting from previously clicked features
        if clicked:
            # Reverse to show latest clicked at the top
            options = list(reversed(clicked))
            current = viewer.get_selected_geohash()

            # Set default index based on current selection
            if current and current in options:
                default_idx = options.index(current)
            else:
                default_idx = 0

            selected_option = st.sidebar.selectbox(
                "Previously clicked lakes:",
                options,
                index=default_idx,
                label_visibility="collapsed",
                help="Select a previously clicked lake",
            )

            # Update selection based on dropdown choice
            if selected_option != st.session_state.selected_geohash:
                st.session_state.selected_geohash = selected_option
                st.rerun()
        else:
            st.sidebar.info("No features clicked yet. Click on a feature to select it.")

        # Current selection
        current = viewer.get_selected_geohash()
        if current:
            st.sidebar.write(f"**Current selection:** {current}")

        # Clear button
        if st.sidebar.button("Clear Selection"):
            viewer.clear_selection()
            st.rerun()

        # Time Series Plot Section
        if current:
            st.divider()
            st.subheader(f"📈 Time Series: {current}")

            # Button to open time series in popup
            if st.button("📊 Open Time Series in Popup", key="open_ts_popup"):
                st.session_state.show_ts_popup = True

            # Show inline preview
            st.caption("Preview - click button above for full view")

            # Load datasets using unified helper function
            dw_dataset = st.session_state.get("dw_dataset")
            jrc_dataset = st.session_state.get("jrc_dataset")

            # Load DW dataset if needed
            if dw_dataset is None:
                dw_dataset, success = load_dataset("dw", zarr_path_input, st.session_state.downloaded_dsdw)
                if success and dw_dataset is not None:
                    st.session_state.dw_dataset = dw_dataset

            # Load JRC dataset if needed
            if jrc_dataset is None:
                jrc_dataset, success = load_dataset("jrc", zarr_path_jrc_input, st.session_state.downloaded_dsjrc)
                if success and jrc_dataset is not None:
                    st.session_state.jrc_dataset = jrc_dataset

            # Re-check availability after loading
            id_available_dw = check_dataset_availability(st.session_state.dw_dataset, current)
            id_available_jrc = check_dataset_availability(st.session_state.jrc_dataset, current)
            st.caption(f"DW availability: {id_available_dw}, JRC availability: {id_available_jrc}")

            ""  # Already handled in the unified loading code above"

            # Automatically download if not available
            if not id_available_dw or not id_available_jrc:
                st.caption("Downloading...")

                # Download data for the specific geohash
                try:
                    # Create downloader with the project from environment
                    downloader = EarthEngineDownloader(ee_auth=True)

                    if not id_available_dw:
                        # Download data for the specific geohash
                        st.caption("Downloading DW data ...")
                        dsdw_downloaded = downloader.download_dw_monthly(
                            vector_dataset=data_path_input,
                            name_attribute=id_column,
                            id_list=[current],
                            years=list(range(2017, 2026)),
                            months=[6, 7, 8, 9],
                            date_list=None,
                        )

                        if dsdw_downloaded is not None:
                            # Convert downloaded data to DWDataset
                            downloaded_dataset_dw = DWDataset(dsdw_downloaded)

                            # Merge with existing cached data if available
                            if st.session_state.dw_dataset is not None:
                                try:
                                    st.session_state.dw_dataset = st.session_state.dw_dataset.merge(
                                        downloaded_dataset_dw, how="id_geohash"
                                    )
                                except Exception as merge_e:
                                    # If merge fails, use downloaded data only
                                    st.sidebar.warning(f"Could not merge data: {merge_e}")
                                    st.session_state.dw_dataset = downloaded_dataset_dw
                            else:
                                st.session_state.dw_dataset = downloaded_dataset_dw

                            st.session_state.downloaded_dsdw = dsdw_downloaded
                            id_available_dw = True
                            st.rerun()
                        else:
                            st.error("Download returned no data.")

                    if not id_available_jrc:
                        # JRC Download
                        st.caption("Downloading JRC data ...")
                        dsjrc_downloaded = downloader.download_jrc_annual(
                            vector_dataset=data_path_input,
                            name_attribute=id_column,
                            id_list=[current],
                            years=range(1984, 2022),
                        )

                        # add dw dataset to session state
                        if dsjrc_downloaded is not None:
                            # Convert downloaded data to JRCDataset
                            downloaded_dataset_jrc = JRCDataset(dsjrc_downloaded)

                            # Merge with existing cached data if available
                            if st.session_state.jrc_dataset is not None:
                                try:
                                    st.session_state.jrc_dataset = st.session_state.jrc_dataset.merge(
                                        downloaded_dataset_jrc, how="id_geohash"
                                    )
                                except Exception as merge_e:
                                    # If merge fails, use downloaded data only
                                    st.sidebar.warning(f"Could not merge data: {merge_e}")
                                    st.session_state.jrc_dataset = downloaded_dataset_jrc
                            else:
                                st.session_state.jrc_dataset = downloaded_dataset_jrc

                            st.session_state.downloaded_dsjrc = dsjrc_downloaded
                            id_available_jrc = True

                            # Also set id_available_dw = True since DW data was also downloaded
                            id_available_dw = True

                            st.caption("Both DW and JRC data downloaded successfully!")
                            st.rerun()
                        else:
                            st.error("Download returned no data.")

                except Exception as e:
                    st.error(f"Error downloading data: {e}")
                    st.info("Make sure you have Google Earth Engine authentication configured.")

            # Plot time series if available
            if st.session_state.dw_dataset is not None and id_available_dw:
                try:
                    # Create container with one row and two columns for time series plots
                    ts_col1, ts_col2 = st.columns(2)

                    # Plot Dynamic World time series in first column
                    with ts_col1:
                        st.subheader("Dynamic World")
                        plot_time_series_data(
                            st.session_state.dw_dataset,
                            current,
                            "dw",
                            is_interactive,
                            show_success=True,
                            show_caption=True,
                        )

                    # Plot JRC time series in second column if available
                    if st.session_state.jrc_dataset is not None and id_available_jrc:
                        with ts_col2:
                            st.subheader("JRC")
                            plot_time_series_data(
                                st.session_state.jrc_dataset,
                                current,
                                "jrc",
                                is_interactive,
                                show_success=False,
                                show_caption=True,
                            )
                    else:
                        with ts_col2:
                            st.caption("JRC data not available for this feature")
                except Exception as e:
                    st.error(f"Error plotting time series: {e}")

                    # ============================================
            # Timelapse Section
            # ============================================
            st.divider()
            st.subheader("🛰️ Timelapse")

            # Create checkboxes in one column (vertically stacked)
            col_checkbox = st.container()
            with col_checkbox:
                create_sentinel2 = st.checkbox("Sentinel-2 (2016-2025)", value=True, key="sentinel2_checkbox")
                create_landsat = st.checkbox("Landsat (2000-2025)", value=True, key="landsat_checkbox")

            # Define GIF paths early so they're available for both creation and display
            gif_dir = Path("gifs")
            potential_gif_s2 = gif_dir / f"{current}_S2.gif"
            potential_gif_landsat = gif_dir / f"{current}_LS.gif"

            # Button to create timelapse(s)
            timelapse_clicked = st.button("🎬 Create Timelapse", key="create_timelapse")

            if timelapse_clicked:
                if not create_sentinel2 and not create_landsat:
                    st.warning("Please select at least one data source (Sentinel-2 or Landsat)")
                else:
                    with st.spinner("Generating timelapse... This may take a up to a minute."):
                        try:
                            # Create Sentinel-2 timelapse if checked
                            gif_path_s2 = None
                            gif_path_landsat = None

                            if create_sentinel2:
                                gif_path_s2 = st.session_state.dw_dataset.create_timelapse(
                                    lake_gdf=viewer.gdf,
                                    id_geohash=current,
                                    timelapse_source="sentinel2",
                                    gif_outdir="gifs",
                                    buffer=100,
                                    start_year=2016,
                                    end_year=2025,
                                    start_date="07-01",
                                    end_date="08-31",
                                    frames_per_second=1,
                                    dimensions=512,
                                    overwrite_exists=False,
                                )

                            # Create Landsat timelapse if checked
                            if create_landsat:
                                gif_path_landsat = st.session_state.dw_dataset.create_timelapse(
                                    lake_gdf=viewer.gdf,
                                    id_geohash=current,
                                    timelapse_source="landsat",
                                    gif_outdir="gifs",
                                    buffer=100,
                                    start_year=2000,
                                    end_year=2025,
                                    start_date="07-01",
                                    end_date="08-31",
                                    frames_per_second=1,
                                    dimensions=512,
                                    overwrite_exists=False,
                                )

                            # Display GIFs side by side with headers
                            display_col_s2, display_col_ls = st.columns(2)

                            # Sentinel-2 GIF
                            with display_col_s2:
                                st.subheader("Sentinel-2 (2016-2025)")
                                gif_s2_path = gif_path_s2 if gif_path_s2 is not None else potential_gif_s2
                                if gif_s2_path and gif_s2_path.exists():
                                    if gif_path_s2 is not None:
                                        st.success(f"Timelapse created: {gif_path_s2}")
                                    else:
                                        st.info("Timelapse already exists")

                                    # Use simple path like the working version
                                    st.image(str(gif_s2_path), caption=f"Timelapse: {current}", width=512)

                                    with open(gif_s2_path, "rb") as f:
                                        st.download_button(
                                            label="💾 Download GIF",
                                            data=f,
                                            file_name=gif_s2_path.name,
                                            mime="image/gif",
                                            key="download_s2",
                                        )

                            # Landsat GIF
                            with display_col_ls:
                                if create_landsat:
                                    st.subheader("Landsat (2000-2025)")
                                    gif_ls_path = (
                                        gif_path_landsat if gif_path_landsat is not None else potential_gif_landsat
                                    )
                                    if gif_ls_path and gif_ls_path.exists():
                                        if gif_path_landsat is not None:
                                            st.success(f"Timelapse created: {gif_path_landsat}")
                                        else:
                                            st.info("Timelapse already exists")

                                        # Use simple path like the working version
                                        st.image(str(gif_ls_path), caption=f"Timelapse: {current}", width=512)

                                        with open(gif_ls_path, "rb") as f:
                                            st.download_button(
                                                label="💾 Download GIF",
                                                data=f,
                                                file_name=gif_ls_path.name,
                                                mime="image/gif",
                                                key="download_landsat",
                                            )

                        except Exception as e:
                            st.error(f"Error creating timelapse: {e}")
                            st.info("Make sure you have Google Earth Engine authentication configured.")
            else:
                # Display existing GIFs (when button wasn't clicked)
                if potential_gif_s2.exists() or potential_gif_landsat.exists():
                    existing_col_s2, existing_col_ls = st.columns(2)

                    with existing_col_s2:
                        if potential_gif_s2.exists():
                            st.subheader("Sentinel-2 (2016-2025)")
                            st.info("Timelapse already exists")
                            # Use simple path like the working version
                            st.image(str(potential_gif_s2), caption=f"Timelapse: {current}", width=512)

                            with open(potential_gif_s2, "rb") as f:
                                st.download_button(
                                    label="💾 Download GIF",
                                    data=f,
                                    file_name=potential_gif_s2.name,
                                    mime="image/gif",
                                    key="download_existing_s2",
                                )

                    with existing_col_ls:
                        if potential_gif_landsat.exists():
                            st.subheader("Landsat (2000-2025)")
                            st.info("Timelapse already exists")
                            # Use simple path like the working version
                            st.image(str(potential_gif_landsat), caption=f"Timelapse: {current}", width=512)

                            with open(potential_gif_landsat, "rb") as f:
                                st.download_button(
                                    label="💾 Download GIF",
                                    data=f,
                                    file_name=potential_gif_landsat.name,
                                    mime="image/gif",
                                    key="download_existing_landsat",
                                )

        # Popup dialog for time series
        if st.session_state.get("show_ts_popup", False) and current:

            @st.dialog("Time Series Plot", width="large")
            def ts_popup():
                st.subheader(f"📈 Time Series: {current}")

                # Load dataset if not already loaded
                # Prioritize downloaded data over cached zarr
                if st.session_state.dw_dataset is None and st.session_state.downloaded_dsdw is not None:
                    try:
                        st.session_state.dw_dataset = DWDataset(st.session_state.downloaded_dsdw)
                    except Exception as e:
                        st.error(f"Error processing downloaded data: {e}")
                elif st.session_state.dw_dataset is None:
                    try:
                        ds = xr.open_zarr(zarr_path_input)
                        st.session_state.dw_dataset = DWDataset(ds)
                    except Exception as e:
                        st.error(f"Error loading time series data: {e}")

                # Load JRC dataset if not already loaded
                # Prioritize downloaded data over cached zarr
                if st.session_state.jrc_dataset is None and st.session_state.downloaded_dsjrc is not None:
                    try:
                        st.session_state.jrc_dataset = JRCDataset(st.session_state.downloaded_dsjrc)
                    except Exception as e:
                        st.error(f"Error processing downloaded JRC data: {e}")
                elif st.session_state.jrc_dataset is None:
                    try:
                        ds_jrc = xr.open_zarr(zarr_path_jrc_input)
                        st.session_state.jrc_dataset = JRCDataset(ds_jrc)
                    except Exception as e:
                        st.error(f"Error loading JRC time series data: {e}")

                # Check if ids are available
                id_available = False
                if st.session_state.dw_dataset is not None:
                    available_ids = st.session_state.dw_dataset.object_ids_
                    id_available = current in available_ids

                id_available_jrc = False
                if st.session_state.jrc_dataset is not None:
                    available_ids_jrc = st.session_state.jrc_dataset.object_ids_
                    id_available_jrc = current in available_ids_jrc

                # Automatically download if not available
                if not id_available:
                    st.caption("Downloading...")
                    try:
                        downloader = EarthEngineDownloader(ee_auth=True)
                        ds_downloaded = downloader.download_dw_monthly(
                            vector_dataset=data_path_input,
                            name_attribute=id_column,
                            id_list=[current],
                            years=list(range(2017, 2026)),
                            months=[6, 7, 8, 9],
                            date_list=None,
                        )
                        if ds_downloaded is not None:
                            st.session_state.downloaded_dsdw = ds_downloaded
                            st.session_state.dw_dataset = DWDataset(ds_downloaded)
                            id_available = True
                            st.rerun()
                        else:
                            st.error("Download returned no data.")
                    except Exception as e:
                        st.error(f"Error downloading data: {e}")

                # Plot time series
                if st.session_state.dw_dataset is not None and id_available:
                    try:
                        # Use interactive or static plotting based on toggle
                        if is_interactive:
                            fig = st.session_state.dw_dataset.plot_timeseries_interactive(current)
                            st.plotly_chart(fig, width="stretch")

                            # Convert figure to HTML for download (only when requested)
                            html_buffer = fig.to_html(full_html=False, include_plotlyjs="cdn")
                            st.download_button(
                                label="💾 Save Interactive Plot (HTML)",
                                data=html_buffer,
                                file_name=f"timeseries_{current}.html",
                                mime="text/html",
                            )
                        else:
                            fig = st.session_state.dw_dataset.plot_timeseries(current)

                            # Save figure to bytes buffer for download
                            img_buffer = BytesIO()
                            fig.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
                            img_buffer.seek(0)

                            # Display and offer download
                            st.pyplot(fig)

                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.download_button(
                                    label="💾 Save Figure",
                                    data=img_buffer,
                                    file_name=f"timeseries_{current}.png",
                                    mime="image/png",
                                )

                            plt.close(fig)  # Close matplotlib figure
                    except Exception as e:
                        st.error(f"Error plotting time series: {e}")

                # Plot JRC time series if available
                if st.session_state.jrc_dataset is not None and id_available_jrc:
                    try:
                        # Use interactive or static plotting based on toggle
                        if is_interactive:
                            fig_jrc = st.session_state.jrc_dataset.plot_timeseries_interactive(current)
                            st.plotly_chart(fig_jrc, width="stretch")

                            # Convert figure to HTML for download (only when requested)
                            html_buffer = fig_jrc.to_html(full_html=False, include_plotlyjs="cdn")
                            st.download_button(
                                label="💾 Save JRC Interactive Plot (HTML)",
                                data=html_buffer,
                                file_name=f"timeseries_jrc_{current}.html",
                                mime="text/html",
                            )
                        else:
                            fig_jrc = st.session_state.jrc_dataset.plot_timeseries(current)

                            # Save figure to bytes buffer for download
                            img_buffer = BytesIO()
                            fig_jrc.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
                            img_buffer.seek(0)

                            # Display and offer download
                            st.pyplot(fig_jrc)

                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.download_button(
                                    label="💾 Save JRC Figure",
                                    data=img_buffer,
                                    file_name=f"timeseries_jrc_{current}.png",
                                    mime="image/png",
                                )

                            plt.close(fig_jrc)  # Close matplotlib figure
                    except Exception as e:
                        st.error(f"Error plotting JRC time series: {e}")
                else:
                    st.caption("⚠️ JRC data not available for this feature")

                if st.button("Close", key="close_ts_popup"):
                    st.session_state.show_ts_popup = False
                    st.rerun()

            ts_popup()

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check the file path and ensure the parquet file exists.")


if __name__ == "__main__":
    create_app()
