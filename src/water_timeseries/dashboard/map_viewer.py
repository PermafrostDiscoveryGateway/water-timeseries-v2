"""Map Viewer dashboard component using Streamlit and lonboard for high-performance mapping."""

import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import folium
import geemap
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import xarray as xr
from lonboard import Map, PolygonLayer
from streamlit_folium import st_folium

from water_timeseries.dataset import DWDataset
from water_timeseries.downloader import EarthEngineDownloader
from water_timeseries.utils.io import load_vector_dataset
from water_timeseries.utils.map_styling import (
    create_tile_layers,
    format_tooltip_columns,
    get_colored_style_function,
    get_default_style_function,
)
from water_timeseries.utils.visualization import (
    DEFAULT_HOVER_COLUMNS,
)


def render_map_html(map_view: Map) -> None:
    """Fallback: Render lonboard map as HTML component."""
    # Get the HTML representation of the map
    html = map_view.to_html()
    st.components.v1.html(html, height=600)


def visualize_gdf(
    gdf: gpd.GeoDataFrame,
    fill_color: List[int] = [0, 120, 255, 180],
    line_color: List[int] = [255, 255, 255, 255],
    line_width: float = 1.0,
    height: int = 600,
    zoom: Optional[int] = None,
    map_center: Optional[List[float]] = None,
    use_st_map: bool = False,
    use_folium: bool = True,
    max_features: Optional[int] = None,  # Limit features for faster loading
) -> None:
    """Visualize a GeoDataFrame with polygons using folium, pydeck, or st.map.

    A simple function to display your polygon GeoDataFrame on an interactive map.

    Args:
        gdf: GeoDataFrame containing polygon geometries.
        fill_color: RGBA fill color for polygons [r, g, b, a].
        line_color: RGBA line color for polygon edges [r, g, b, a].
        line_width: Width of polygon edges.
        height: Height of the map in pixels.
        zoom: Initial zoom level. If None, auto-calculated from bounds.
        map_center: [lat, lon] center of the map. If None, auto-calculated from centroid.
        use_st_map: If True, use Streamlit's native st.map() (shows points at centroids).
        use_folium: If True and use_pydeck is False, use folium for polygon rendering.
        max_features: Maximum number of features to display (for large datasets).

    Example:
        >>> import geopandas as gpd
        >>> from water_timeseries.dashboard.map_viewer import visualize_gdf
        >>> gdf = gpd.read_file("lakes.parquet")
        >>> visualize_gdf(gdf, max_features=1000)  # Load max 1000 features
    """

    # Filter out invalid geometries
    valid_mask = gdf.geometry.notna() & ~gdf.geometry.is_empty
    valid_gdf = gdf[valid_mask].copy().reset_index(drop=True)

    if len(valid_gdf) == 0:
        st.warning("No valid geometries found in the GeoDataFrame.")
        return

    # Apply sampling if max_features specified (for faster loading)
    if max_features and len(valid_gdf) > max_features:
        valid_gdf = valid_gdf.sample(n=max_features, random_state=42).reset_index(drop=True)
        st.caption(f"Showing {max_features} of {len(gdf)} features (use max_features to change)")

    # Use Streamlit's native st.map for simple visualization
    if use_st_map:
        st.map(valid_gdf)
        return

    # Use folium for full polygon rendering
    if use_folium:
        # Calculate center
        if map_center is None:
            centroid = valid_gdf.geometry.unary_union.centroid
            center = [centroid.y, centroid.x]  # [lat, lon]
        else:
            center = map_center

        # Create folium map
        m = folium.Map(location=center, zoom_start=zoom if zoom else 10, tiles="Esri.WorldImagery")

        # Add polygons with simple styling
        folium.GeoJson(
            valid_gdf,
            style_function=lambda x: {
                "fillColor": "blue",
                "color": "black",
                "weight": line_width,
                "fillOpacity": 0.5,
            },
        ).add_to(m)

        st_folium(m, height=height, width="100%")
        return

    # Calculate center if not provided
    if map_center is None:
        centroid = valid_gdf.geometry.unary_union.centroid
        center = [centroid.y, centroid.x]  # [lat, lon]
    else:
        center = map_center

    # Create the polygon layer
    polygon_layer = PolygonLayer.from_geopandas(
        valid_gdf,
        get_fill_color=fill_color,
        get_line_color=line_color,
        get_line_width=line_width,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 0, 150],  # Yellow highlight on hover
    )

    # Create the map
    map_view = Map(
        polygon_layer=polygon_layer,
        center=center,
        zoom=zoom if zoom is not None else 10,
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

        return gdf

    def _prepare_hover_data(self) -> pd.DataFrame:
        """Prepare hover data for the map.

        Returns:
            DataFrame with columns to show on hover.
        """
        # Get columns to include (exclude geometry)
        if self.hover_columns:
            cols = [col for col in self.hover_columns if col in self.gdf.columns and col != self.geometry_column]
        else:
            cols = [col for col in self.gdf.columns if col != self.geometry_column]

        # Create a copy with only the needed columns
        plot_df = self.gdf[cols].copy()

        # Convert all columns to strings and handle NaN/None/arrays
        # This is handled by prepare_custom_data_for_plotly when rendering
        return plot_df

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
            valid_gdf = valid_gdf.sample(n=self.max_features, random_state=42).reset_index(drop=True)
            st.caption(f"Showing {self.max_features} of {len(self.gdf)} features (use max_features to change)")

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
        valid_gdf, fields_to_show, aliases_to_show = format_tooltip_columns(
            valid_gdf,
            id_column=self.id_column,
            tooltip_columns=tooltip_columns,
        )

        folium.GeoJson(
            valid_gdf,
            name="Lakes",
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=fields_to_show,
                aliases=aliases_to_show,
            ),
        ).add_to(m)

        folium.LayerControl().add_to(m)

        # Add legend for NetChange_perc color scale
        legend_html = """
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
        m.get_root().html.add_child(folium.Element(legend_html))

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

    def clear_selection(self):
        """Clear the current selection."""
        st.session_state.selected_geohash = None


def create_app(
    data_path: str | Path = "tests/data/lake_polygons.parquet", zarr_path: str | Path = "tests/data/lakes_dw_test.zarr"
):
    """Create the Streamlit app with map viewer.

    Args:
        data_path: Path to the parquet file containing lake polygons.
        zarr_path: Path to the zarr file containing time series data.
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

    # Map backend selection
    st.sidebar.divider()
    st.sidebar.subheader("Map Backend")
    map_backend = st.sidebar.radio(
        "Select map renderer:",
        options=["folium", "st_map"],
        index=0,
        format_func=lambda x: {
            "folium": "Folium (recommended for small/medium)",
            "st_map": "st.map (simple, points only)",
        }[x],
        help="Folium is best for small datasets, pydeck for large datasets with many polygons.",
    )

    # Performance settings for large datasets
    st.sidebar.divider()
    st.sidebar.subheader("Performance")
    max_features = st.sidebar.number_input(
        "Max features to load",
        min_value=10,
        max_value=50000,
        value=5000,
        step=100,
        help="Limit number of polygons for faster loading. Set to 0 for no limit.",
    )
    if max_features == 0:
        max_features = None

    # Use function parameters for data paths
    data_path_input = str(data_path)
    zarr_path_input = str(zarr_path)
    id_column = "id_geohash"
    zoom_level = 10

    # Layer selection for folium (only when using folium)
    st.sidebar.divider()
    st.sidebar.subheader("Layer Selection")
    if map_backend == "folium":
        # Get available columns for layer selection (exclude geometry and id columns)
        viewer_for_cols = MapViewer(
            parquet_path=str(data_path),
            id_column=id_column,
            zoom=zoom_level,
            map_backend=map_backend,
            max_features=min(100, max_features) if max_features else 100,  # Use smaller sample for column detection
        )
        available_cols = [
            c
            for c in viewer_for_cols.gdf.columns
            if c not in ["geometry", id_column, "id_geohash"]
            and viewer_for_cols.gdf[c].dtype in ["object", "int64", "float64", "int32", "float32"]
        ]

        layer_column = st.sidebar.selectbox(
            "Split into layers by column:",
            options=["None"] + available_cols,
            index=0,
            help="Select a column to split polygons into separate toggleable layers. Each unique value becomes a layer.",
        )
        if layer_column == "None":
            layer_column = None
    else:
        layer_column = None
        st.sidebar.caption("Layer selection only available with Folium backend")

    # Initialize dataset in session state if not already
    if "dw_dataset" not in st.session_state:
        st.session_state.dw_dataset = None
    if "show_ts_popup" not in st.session_state:
        st.session_state.show_ts_popup = False
    if "downloaded_ds" not in st.session_state:
        st.session_state.downloaded_ds = None

    # Create map viewer
    try:
        viewer = MapViewer(
            parquet_path=data_path_input,
            id_column=id_column,
            zoom=zoom_level,
            map_backend=map_backend,
            max_features=max_features,
        )
        viewer.layer_column = layer_column  # Set layer column for folium

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

            # Load dataset if not already loaded
            # Prioritize downloaded data over cached zarr
            if st.session_state.dw_dataset is None and st.session_state.downloaded_ds is not None:
                try:
                    st.session_state.dw_dataset = DWDataset(st.session_state.downloaded_ds)
                except Exception as e:
                    st.error(f"Error processing downloaded data: {e}")
            elif st.session_state.dw_dataset is None:
                try:
                    ds = xr.open_zarr(zarr_path_input)
                    st.session_state.dw_dataset = DWDataset(ds)
                except Exception as e:
                    st.error(f"Error loading time series data: {e}")

            # Check if selected id_geohash is available in the dataset
            id_available = False
            if st.session_state.dw_dataset is not None:
                available_ids = st.session_state.dw_dataset.object_ids_
                id_available = current in available_ids

            # Automatically download if not available
            if not id_available:
                st.caption("Downloading...")

                # Download data for the specific geohash
                try:
                    # Create downloader with the project from environment
                    downloader = EarthEngineDownloader(ee_auth=True)

                    # Download data for the specific geohash
                    ds_downloaded = downloader.download_dw_monthly(
                        vector_dataset=data_path_input,
                        name_attribute=id_column,
                        id_list=[current],
                        years=list(range(2017, 2026)),
                        months=[6, 7, 8, 9],
                        date_list=None,
                    )

                    if ds_downloaded is not None:
                        # Convert downloaded data to DWDataset
                        downloaded_dataset = DWDataset(ds_downloaded)

                        # Merge with existing cached data if available
                        if st.session_state.dw_dataset is not None:
                            try:
                                st.session_state.dw_dataset = st.session_state.dw_dataset.merge(
                                    downloaded_dataset, how="id_geohash"
                                )
                            except Exception as merge_e:
                                # If merge fails, use downloaded data only
                                st.sidebar.warning(f"Could not merge data: {merge_e}")
                                st.session_state.dw_dataset = downloaded_dataset
                        else:
                            st.session_state.dw_dataset = downloaded_dataset

                        st.session_state.downloaded_ds = ds_downloaded
                        id_available = True
                        st.rerun()
                    else:
                        st.error("Download returned no data.")

                except Exception as e:
                    st.error(f"Error downloading data: {e}")
                    st.info("Make sure you have Google Earth Engine authentication configured.")

            # Plot time series if available
            if st.session_state.dw_dataset is not None and id_available:
                try:
                    # Use interactive or static plotting based on toggle
                    if is_interactive:
                        fig = st.session_state.dw_dataset.plot_timeseries_interactive(current)
                        st.plotly_chart(fig, width="stretch")

                        # Convert figure to HTML for download
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

                        st.download_button(
                            label="💾 Save Figure",
                            data=img_buffer,
                            file_name=f"timeseries_{current}.png",
                            mime="image/png",
                        )

                        plt.close(fig)  # Close matplotlib figure

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
                                                gif_path_landsat
                                                if gif_path_landsat is not None
                                                else potential_gif_landsat
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
                except Exception as e:
                    st.error(f"Error plotting time series: {e}")

        # Popup dialog for time series
        if st.session_state.get("show_ts_popup", False) and current:

            @st.dialog("Time Series Plot", width="large")
            def ts_popup():
                st.subheader(f"📈 Time Series: {current}")

                # Load dataset if not already loaded
                # Prioritize downloaded data over cached zarr
                if st.session_state.dw_dataset is None and st.session_state.downloaded_ds is not None:
                    try:
                        st.session_state.dw_dataset = DWDataset(st.session_state.downloaded_ds)
                    except Exception as e:
                        st.error(f"Error processing downloaded data: {e}")
                elif st.session_state.dw_dataset is None:
                    try:
                        ds = xr.open_zarr(zarr_path_input)
                        st.session_state.dw_dataset = DWDataset(ds)
                    except Exception as e:
                        st.error(f"Error loading time series data: {e}")

                # Check if id is available
                id_available = False
                if st.session_state.dw_dataset is not None:
                    available_ids = st.session_state.dw_dataset.object_ids_
                    id_available = current in available_ids

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
                            st.session_state.downloaded_ds = ds_downloaded
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

                            # Convert figure to HTML for download
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

                if st.button("Close", key="close_ts_popup"):
                    st.session_state.show_ts_popup = False
                    st.rerun()

            ts_popup()

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check the file path and ensure the parquet file exists.")


if __name__ == "__main__":
    create_app()
