"""Map Viewer dashboard component using Streamlit and Plotly."""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from pathlib import Path
from typing import Optional, List

from water_timeseries.dataset import DWDataset
from water_timeseries.utils.io import load_vector_dataset


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
        zoom: int = 10
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
        # Default hover columns if not specified
        self.hover_columns = hover_columns or ["id_geohash", "Area_start_ha", "Area_end_ha", "NetChange_ha", "NetChange_perc"]
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
        for col in plot_df.columns:
            col_data = []
            for val in plot_df[col]:
                if pd.isna(val):
                    col_data.append("")
                elif isinstance(val, (list, tuple, set, bytes)):
                    col_data.append(str(list(val)))
                else:
                    col_data.append(str(val))
            plot_df[col] = col_data
        
        return plot_df
    
    def _gdf_to_geojson(self, gdf: gpd.GeoDataFrame) -> dict:
        """Convert GeoDataFrame to GeoJSON feature collection.
        
        Args:
            gdf: GeoDataFrame to convert.
            
        Returns:
            GeoJSON feature collection dictionary.
        """
        return gdf.__geo_interface__
    
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
        
        # Create hover template
        hover_template = "<b>" + self.id_column + ": %{customdata[0]}</b><br>"
        for i, field in enumerate(hover_fields):
            hover_template += f"{field}: %{{customdata[{i + 1}]}}<br>"
        hover_template += "<extra></extra>"
        
        # Prepare custom data for hover
        custom_data = []
        hover_cols = [self.id_column] + hover_fields
        for col in hover_cols:
            custom_data.append(plot_df[col].tolist())
        
        # Transpose to get rows as tuples
        custom_data = list(zip(*custom_data))
        
        # Convert to GeoJSON
        geojson = self._gdf_to_geojson(valid_gdf)
        
        # Create the map using Plotly graph_objects for polygon rendering
        fig = go.Figure(go.Choroplethmapbox(
            geojson=geojson,
            locations=valid_gdf.index.tolist(),
            z=[1] * len(valid_gdf),  # Dummy value for coloring
            customdata=custom_data,
            hovertemplate=hover_template,
            marker_opacity=0.5,
            marker_line_width=1,
            marker_line_color="blue",
            colorscale=[[0, "blue"], [1, "blue"]],
            showscale=False,
        ))
        
        # Update layout
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                zoom=self.zoom,
                center=center
            ),
            height=600,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            clickmode="event+select"
        )
        
        # Render the map
        selected_points = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun"
        )
        
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
    data_path: str | Path = "tests/data/lake_polygons.parquet",
    zarr_path: str | Path = "tests/data/lakes_dw_test.zarr"
):
    """Create the Streamlit app with map viewer.
    
    Args:
        data_path: Path to the parquet file containing lake polygons.
        zarr_path: Path to the zarr file containing time series data.
    """
    st.set_page_config(
        page_title="Lake Polygon Map Viewer",
        page_icon="🗺️",
        layout="wide"
    )
    
    st.title("🗺️ Lake Polygon Map Viewer")
    st.markdown("""
    This dashboard displays lake polygons from a GeoDataFrame. 
    - **Hover** over a feature to see its attributes
    - **Click** on a feature to select it and store its id_geohash
    """)
    
    # Create sidebar for controls
    st.sidebar.header("Settings")
    
    # Data path input
    default_path = str(data_path)
    data_path_input = st.sidebar.text_input(
        "Parquet File Path",
        value=default_path
    )
    
    # Zarr data path input
    default_zarr_path = str(zarr_path)
    zarr_path_input = st.sidebar.text_input(
        "Zarr Time Series Path",
        value=default_zarr_path
    )
    
    # ID column input
    id_column = st.sidebar.text_input("ID Column Name", value="id_geohash")
    
    # Zoom level
    zoom_level = st.sidebar.slider("Initial Zoom Level", min_value=1, max_value=20, value=10)
    
    # Initialize dataset in session state if not already
    if "dw_dataset" not in st.session_state:
        st.session_state.dw_dataset = None
    if "show_ts_popup" not in st.session_state:
        st.session_state.show_ts_popup = False
    
    # Create map viewer
    try:
        viewer = MapViewer(
            parquet_path=data_path_input,
            id_column=id_column,
            zoom=zoom_level
        )
        
        # Render the map
        selected = viewer.render()
        
        # Display selected features in sidebar
        st.sidebar.divider()
        st.sidebar.subheader("Selected Features")
        
        clicked = viewer.get_clicked_features()
        if clicked:
            st.sidebar.write("Clicked id_geohash values:")
            for i, geohash in enumerate(clicked):
                st.sidebar.code(geohash)
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
        
        # Data info
        st.sidebar.divider()
        st.sidebar.subheader("Data Info")
        st.sidebar.write(f"Total features: {len(viewer.gdf)}")
        st.sidebar.write(f"Columns: {list(viewer.gdf.columns)}")
        
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
            if st.session_state.dw_dataset is None:
                try:
                    ds = xr.open_zarr(zarr_path_input)
                    st.session_state.dw_dataset = DWDataset(ds)
                    st.sidebar.write("Dataset loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading time series data: {e}")
            
            # Plot time series preview if dataset is loaded
            if st.session_state.dw_dataset is not None:
                try:
                    fig = st.session_state.dw_dataset.plot_timeseries(current)
                    # Display matplotlib figure in Streamlit
                    st.pyplot(fig)
                    plt.close(fig)  # Close figure to free memory
                except Exception as e:
                    st.error(f"Error plotting time series: {e}")
            else:
                st.warning("Unable to load time series dataset.")
        
        # Popup dialog for time series
        if st.session_state.get("show_ts_popup", False) and current:
            @st.dialog("Time Series Plot", width="large")
            def ts_popup():
                st.subheader(f"📈 Time Series: {current}")
                
                # Load dataset if not already loaded
                if st.session_state.dw_dataset is None:
                    try:
                        ds = xr.open_zarr(zarr_path_input)
                        st.session_state.dw_dataset = DWDataset(ds)
                    except Exception as e:
                        st.error(f"Error loading time series data: {e}")
                        st.stop()
                
                # Plot time series
                if st.session_state.dw_dataset is not None:
                    try:
                        fig = st.session_state.dw_dataset.plot_timeseries(current)
                        st.pyplot(fig)
                        plt.close(fig)
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
