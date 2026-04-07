"""Map Viewer dashboard component using Streamlit and Plotly."""

from io import BytesIO
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from water_timeseries.dataset import DWDataset
from water_timeseries.downloader import EarthEngineDownloader
from water_timeseries.utils.io import load_vector_dataset
from water_timeseries.utils.visualization import (
    DEFAULT_HOVER_COLUMNS,
    MAP_STYLING,
    build_hover_template,
    gdf_to_geojson_feature_collection,
    get_colorbar_config,
    get_z_values_for_coloring,
    prepare_custom_data_for_plotly,
)


class MapViewer:
    """Interactive map viewer for GeoDataFrames using Streamlit and Plotly.

    Features:
    - Display GeoDataFrame on an interactive map
    - Hover tooltips showing feature attributes
    - Click to select features and store their id_geohash value
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
        """Render the interactive map in Streamlit.

        Returns:
            The selected id_geohash value if a feature was clicked, None otherwise.
        """
        st.subheader("Interactive Map Viewer")

        # Get valid indices (after filtering out invalid geometries)
        valid_mask = self.gdf.geometry.notna() & ~self.gdf.geometry.is_empty
        valid_gdf = self.gdf[valid_mask].copy()
        valid_indices = valid_gdf.index.tolist()

        # Prepare hover data
        hover_df_all = self._prepare_hover_data()
        all_indices = self.gdf.index.tolist()
        positions = [all_indices.index(idx) for idx in valid_indices]
        plot_df = hover_df_all.iloc[positions].reset_index(drop=True)

        # Determine center of map
        if self.map_center is None:
            centroid = valid_gdf.geometry.unary_union.centroid
            center = {"lat": centroid.y, "lon": centroid.x}
        else:
            center = self.map_center

        # Build hover fields list
        hover_fields = [col for col in plot_df.columns if col != self.id_column]

        # Use utility function to build hover template
        hover_template = build_hover_template(self.id_column, hover_fields)

        # Use utility function to prepare custom data
        custom_data = prepare_custom_data_for_plotly(plot_df, self.id_column, hover_fields)

        # Use utility function to convert to GeoJSON
        geojson = gdf_to_geojson_feature_collection(valid_gdf)

        # Use utility function to get z-values for coloring
        z_values = get_z_values_for_coloring(valid_gdf, "NetChange_perc", clip_range=(-50, 50))

        # Create the map using Plotly graph_objects for polygon rendering
        # Get styling from visualization module
        unselected_style = MAP_STYLING["unselected"]
        colorbar_config = get_colorbar_config("NetChange %", "RdYlBu", zmid=0)

        fig = go.Figure(
            go.Choroplethmapbox(
                geojson=geojson,
                locations=valid_gdf.index.tolist(),
                z=z_values,
                customdata=custom_data,
                hovertemplate=hover_template,
                marker_opacity=unselected_style["opacity"],
                marker_line_width=unselected_style["line_width"],
                marker_line_color=unselected_style["line_color"],
                colorscale=colorbar_config["colorscale"],
                zmid=colorbar_config.get("zmid"),
                showscale=colorbar_config["showscale"],
                colorbar_title=colorbar_config["colorbar_title"],
            )
        )

        # Add a second layer to highlight selected polygon with slightly thicker/darker outline
        # This layer will be updated when selection changes
        selected_geohash = st.session_state.get("selected_geohash")
        if selected_geohash and selected_geohash in valid_gdf[self.id_column].values:
            # Get the selected feature
            selected_feature = valid_gdf[valid_gdf[self.id_column] == selected_geohash]
            selected_geojson = gdf_to_geojson_feature_collection(selected_feature)
            selected_idx = valid_gdf[valid_gdf[self.id_column] == selected_geohash].index.tolist()

            # Use utility to get z-values for selected feature
            selected_z = get_z_values_for_coloring(selected_feature, "NetChange_perc", clip_range=(-50, 50))

            # Get styling from visualization module
            selected_style = MAP_STYLING["selected"]

            # Add highlight layer with slightly thicker and darker outline
            fig.add_trace(
                go.Choroplethmapbox(
                    geojson=selected_geojson,
                    locations=selected_idx,
                    z=selected_z,
                    marker_opacity=selected_style["opacity"],
                    marker_line_width=selected_style["line_width"],
                    marker_line_color=selected_style["line_color"],
                    colorscale="RdYlBu",
                    zmid=0,
                    showscale=False,
                    hoverinfo="skip",
                )
            )

        # Update layout
        fig.update_layout(
            mapbox=dict(style="open-street-map", zoom=self.zoom, center=center),
            height=600,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            clickmode="event+select",
        )

        # Render the map
        selected_points = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

        # Process selection
        if selected_points and len(selected_points.get("selection", {}).get("points", [])) > 0:
            # Get the first selected point's index
            point = selected_points["selection"]["points"][0]
            point_index = point.get("point_index")

            if point_index is not None and point_index < len(valid_indices):
                # Map back to original dataframe index
                original_index = valid_indices[point_index]
                # Get the id_geohash value
                selected_geohash = self.gdf.iloc[original_index][self.id_column]

                # Store in session state
                st.session_state.selected_geohash = selected_geohash

                # Add to clicked features list if not already there
                if selected_geohash not in st.session_state.clicked_features:
                    st.session_state.clicked_features.append(selected_geohash)

                st.success(f"Selected feature: {selected_geohash}")

                return selected_geohash

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

    # Use function parameters for data paths
    data_path_input = str(data_path)
    zarr_path_input = str(zarr_path)
    id_column = "id_geohash"
    zoom_level = 10

    # Initialize dataset in session state if not already
    if "dw_dataset" not in st.session_state:
        st.session_state.dw_dataset = None
    if "show_ts_popup" not in st.session_state:
        st.session_state.show_ts_popup = False
    if "downloaded_ds" not in st.session_state:
        st.session_state.downloaded_ds = None

    # Create map viewer
    try:
        viewer = MapViewer(parquet_path=data_path_input, id_column=id_column, zoom=zoom_level)

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
                        st.plotly_chart(fig, use_container_width=True)

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
                        create_landsat = st.checkbox("Landsat (2000-2025)", value=False, key="landsat_checkbox")

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
                            with st.spinner("Generating timelapse... This may take a few minutes."):
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

                                    # Display GIFs side by side (single row)
                                    display_col_s2, display_col_ls = st.columns(2)

                                    # Display GIFs with headers
                                    display_col_s2, display_col_ls = st.columns(2)

                                    # Sentinel-2 GIF
                                    with display_col_s2:
                                        st.subheader("Sentinel-2 (2016-2025)")
                                        gif_s2_path = gif_path_s2 if gif_path_s2 is not None else potential_gif_s2
                                        if gif_s2_path and gif_s2_path.exists():
                                            if gif_path_s2 is not None:
                                                st.success(f"Timelapse created: {gif_path_s2}")
                                            else:
                                                st.info(f"Timelapse already exists")

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
                                                    st.info(f"Timelapse already exists")

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
                                    st.info(f"Timelapse already exists")
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
                                    st.info(f"Timelapse already exists")
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
                            st.plotly_chart(fig, use_container_width=True)

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
