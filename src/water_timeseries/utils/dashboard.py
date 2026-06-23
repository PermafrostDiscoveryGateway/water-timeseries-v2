"""Dashboard helper functions to reduce code duplication in map_viewer.py."""

from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import streamlit as st
import xarray as xr

from water_timeseries.dataset import DWDataset, JRCDataset
from water_timeseries.downloader import EarthEngineDownloader
from water_timeseries.utils.io import load_vector_dataset, load_xarray_dataset


@st.cache_data(ttl=3600, show_spinner="Loading GeoDataframe dataset from parquet...")
def load_lake_polygons_cached(file_path: str):
    gdf = load_vector_dataset(file_path)
    return gdf


@st.cache_data(ttl=3600, show_spinner="Loading xarray dataset")
def load_xarray_dataset_cached(file_path: str):
    return load_xarray_dataset(file_path)


# loading slow for large datasets
# @st.cache_data(ttl=3600, show_spinner="Loading xarray dataset from zarr...")
def load_dataset(
    dataset_type: str,
    zarr_path: str | Path,
    downloaded_data: Optional[object] = None,
    dataset_obj: Optional[DWDataset | JRCDataset] = None,
) -> Tuple[DWDataset | JRCDataset | None, bool]:
    """
    Load a dataset with fallback logic: downloaded data > cached dataset > zarr file.

    Args:
        dataset_type: Either 'dw' or 'jrc' to specify which dataset to load
        zarr_path: Path to the zarr file for cached data
        downloaded_data: Previously downloaded data (if any)
        dataset_obj: Already loaded dataset object (if any)

    Returns:
        Tuple of (loaded_dataset, success)
    """
    dataset_class = DWDataset if dataset_type == "dw" else JRCDataset
    # dataset_key = f"{'dw' if dataset_type == 'dw' else 'jrc'}_dataset"

    # Priority 1: Use downloaded data if available
    if downloaded_data is not None:
        try:
            return dataset_class(downloaded_data), True
        except Exception as e:
            st.error(f"Error processing downloaded {dataset_type.upper()} data: {e}")
            return None, False

    # Priority 2: Use existing dataset object if available
    if dataset_obj is not None:
        return dataset_obj, True

    # Priority 3: Load from zarr file
    try:
        ds = load_xarray_dataset(str(zarr_path))
        return dataset_class(ds), True
    except Exception as e:
        st.error(f"Error loading {dataset_type.upper()} time series data: {e}")
        return None, False


def check_dataset_availability(dataset: Optional[DWDataset | JRCDataset], id_geohash: str) -> bool:
    """Check if the specific ID is available in the dataset."""
    if dataset is None:
        return False
    return id_geohash in dataset.object_ids_


def check_dataset_availability_ds_raw(dataset: xr.Dataset, id_geohash: str) -> bool:
    """Check if the specific ID is available in the dataset."""
    if dataset is None:
        return False
    return id_geohash in dataset.id_geohash


def plot_time_series_data(
    dataset: Optional[DWDataset | JRCDataset],
    id_geohash: str,
    dataset_type: str,
    is_interactive: bool,
    show_success: bool = True,
    show_caption: bool = True,
) -> bool:
    """
    Unified function to plot time series data with automatic format handling.

    Args:
        dataset: Dataset object (DWDataset or JRCDataset)
        id_geohash: ID of the feature to plot
        dataset_type: 'dw' or 'jrc' for labeling
        is_interactive: Whether to use interactive (plotly) or static (matplotlib) plots
        show_success: Whether to show success message for DW data
        show_caption: Whether to show captions and availability information

    Returns:
        True if plotting succeeded, False otherwise
    """
    if dataset is None:
        if show_caption:
            st.caption(f"⚠️ {dataset_type.upper()} dataset not loaded")
        return False

    # Check availability
    id_available = check_dataset_availability(dataset, id_geohash)

    if not id_available:
        if show_caption:
            st.caption(f"⚠️ {dataset_type.upper()} data not available for this feature")
        return False

    # Success message for DW dataset
    # if show_success and dataset_type == 'dw':
    #     st.success("✅ Dynamic World data available")

    try:
        if is_interactive:
            # Plot interactive plotly chart
            fig = dataset.plot_timeseries_interactive(id_geohash)
            st.plotly_chart(fig, width="stretch")

            # Save as HTML for download
            html_buffer = fig.to_html(full_html=False, include_plotlyjs="cdn")
            file_suffix = "jrc_" if dataset_type == "jrc" else ""
            st.download_button(
                label=f"💾 Save {file_suffix.upper()}Interactive Plot (HTML)",
                data=html_buffer,
                file_name=f"timeseries_{file_suffix}{id_geohash}.html",
                mime="text/html",
                key=f"download_{dataset_type}_interactive_{id_geohash}",
            )
        else:
            # Plot static matplotlib figure
            fig = dataset.plot_timeseries(id_geohash)
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
            img_buffer.seek(0)

            st.pyplot(fig)
            plt.close(fig)

            # Save as PNG for download
            file_suffix = "jrc_" if dataset_type == "jrc" else ""
            st.download_button(
                label=f"💾 Save {file_suffix.upper()}Figure",
                data=img_buffer,
                file_name=f"timeseries_{file_suffix}{id_geohash}.png",
                mime="image/png",
                key=f"download_{dataset_type}_static_{id_geohash}",
            )

        return True

    except Exception as e:
        st.error(f"Error plotting {dataset_type.upper()} time series: {e}")
        return False
    return id_geohash in dataset.object_ids_


def plot_timeseries_with_fallback(
    dataset: Optional[DWDataset | JRCDataset],
    id_geohash: str,
    dataset_type: str,
    zarr_path: str | Path,
    downloaded_data: Optional[object] = None,
    is_interactive: bool = True,
    show_download: bool = True,
    layer_title: str | None = None,
) -> bool:
    """
    Plot time series with automatic dataset loading and download fallback.

    Args:
        dataset: Current dataset object
        id_geohash: ID of the feature to plot
        dataset_type: 'dw' or 'jrc'
        zarr_path: Path to zarr file for fallback loading
        downloaded_data: Previously downloaded data
        is_interactive: Whether to use interactive (plotly) or static (matplotlib) plots
        show_download: Whether to show download buttons
        layer_title: Title for the plot layer

    Returns:
        True if plotting succeeded, False otherwise
    """
    # Load dataset if needed
    if dataset is None:
        dataset, success = load_dataset(dataset_type, zarr_path, downloaded_data)
        if not success or dataset is None:
            return False


def download_dataset_if_needed(
    id_geohash: str,
    dataset_type: str,
    data_path: str | Path,
    id_column: str,
    dataset_obj: Optional[DWDataset | JRCDataset] = None,
    downloaded_data: Optional[object] = None,
    download_func=None,
) -> Tuple[Optional[object], DWDataset | JRCDataset | None, bool]:
    """
    Download dataset data for a specific ID if not already available.

    Args:
        id_geohash: ID to download data for
        dataset_type: 'dw' or 'jrc'
        data_path: Path to vector dataset
        id_column: Name of ID column
        dataset_obj: Existing dataset object
        downloaded_data: Previously downloaded data
        download_func: Function to call for downloading

    Returns:
        Tuple of (downloaded_data, new_dataset, success)
    """
    if download_func is None:
        return downloaded_data, dataset_obj, False

    try:
        downloader = EarthEngineDownloader(ee_auth=True)
        new_data = download_func(downloader, id_geohash, data_path, id_column)

        if new_data is not None:
            dataset_class = DWDataset if dataset_type == "dw" else JRCDataset
            new_dataset = dataset_class(new_data)

            # Merge with existing data if available
            if dataset_obj is not None:
                try:
                    merged_dataset = dataset_obj.merge(new_dataset, how="id_geohash")
                    st.session_state[f"downloaded_ds{dataset_type}"] = new_data
                    return new_data, merged_dataset, True
                except Exception as merge_e:
                    st.sidebar.warning(f"Could not merge {dataset_type.upper()} data: {merge_e}")

            st.session_state[f"downloaded_ds{dataset_type}"] = new_data
            return new_data, new_dataset, True

        return downloaded_data, dataset_obj, False

    except Exception as e:
        st.error(f"Error downloading {dataset_type.upper()} data: {e}")
        return downloaded_data, dataset_obj, False


def create_timelapse_handler(
    dataset_obj: DWDataset | JRCDataset,
    viewer_gdf,
    id_geohash: str,
    source: str,
    start_year: int,
    end_year: int,
    start_date: str,
    end_date: str,
    buffer: int = 100,
    gif_outdir: str = "gifs",
) -> Optional[Path]:
    """
    Create a timelapse GIF and return its path.

    Args:
        dataset_obj: Dataset object with create_timelapse method
        viewer_gdf: GeoDataFrame from viewer
        id_geohash: ID of the lake
        source: 'sentinel2' or 'landsat'
        start_year: Start year for timelapse
        end_year: End year for timelapse
        start_date: Start date (month-day format)
        end_date: End date (month-day format)
        buffer: Buffer size in meters
        gif_outdir: Output directory for GIF

    Returns:
        Path to created GIF or None if failed
    """
    try:
        gif_path = dataset_obj.create_timelapse(
            lake_gdf=viewer_gdf,
            id_geohash=id_geohash,
            timelapse_source=source,
            gif_outdir=gif_outdir,
            buffer=buffer,
            start_year=start_year,
            end_year=end_year,
            start_date=start_date,
            end_date=end_date,
            frames_per_second=1,
            dimensions=512,
            overwrite_exists=False,
        )
        return Path(gif_path) if gif_path else None
    except Exception as e:
        st.error(f"Error creating {source.upper()} timelapse: {e}")
        return None


def display_gif_row(
    gif_path_s2: Optional[Path], gif_path_landsat: Optional[Path], current: str, gif_dir: Path = Path("gifs")
):
    """
    Display GIFs in a two-column layout.

    Args:
        gif_path_s2: Path to Sentinel-2 GIF or None
        gif_path_landsat: Path to Landsat GIF or None
        current: Current ID_geohash
        gif_dir: Directory where GIFs are stored
    """
    display_col_s2, display_col_ls = st.columns(2)

    # Sentinel-2 GIF
    with display_col_s2:
        st.subheader("Sentinel-2 (2016-2025)")
        gif_s2_path = gif_path_s2 if gif_path_s2 else gif_dir / f"{current}_S2.gif"

        if gif_s2_path and gif_s2_path.exists():
            if gif_path_s2:
                st.success(f"Timelapse created: {gif_path_s2}")
            else:
                st.info("Timelapse already exists")

            st.image(str(gif_s2_path), caption=f"Timelapse: {current}", width=512)

            with open(gif_s2_path, "rb") as f:
                st.download_button(
                    label="💾 Download GIF",
                    data=f,
                    file_name=gif_s2_path.name,
                    mime="image/gif",
                    key=f"download_s2_{current}",
                )

    # Landsat GIF
    with display_col_ls:
        st.subheader("Landsat (2000-2025)")
        gif_ls_path = gif_path_landsat if gif_path_landsat else gif_dir / f"{current}_LS.gif"

        if gif_ls_path and gif_ls_path.exists():
            if gif_path_landsat:
                st.success(f"Timelapse created: {gif_path_landsat}")
            else:
                st.info("Timelapse already exists")

            st.image(str(gif_ls_path), caption=f"Timelapse: {current}", width=512)

            with open(gif_ls_path, "rb") as f:
                st.download_button(
                    label="💾 Download GIF",
                    data=f,
                    file_name=gif_ls_path.name,
                    mime="image/gif",
                    key=f"download_landsat_{current}",
                )


def display_existing_gifs(current: str, gif_dir: Path = Path("gifs")):
    """
    Display existing GIFs in a two-column layout.

    Args:
        current: Current ID_geohash
        gif_dir: Directory where GIFs are stored
    """
    potential_gif_s2 = gif_dir / f"{current}_S2.gif"
    potential_gif_landsat = gif_dir / f"{current}_LS.gif"

    existing_col_s2, existing_col_ls = st.columns(2)

    with existing_col_s2:
        if potential_gif_s2.exists():
            st.subheader("Sentinel-2 (2016-2025)")
            st.info("Timelapse already exists")
            st.image(str(potential_gif_s2), caption=f"Timelapse: {current}", width=512)

            with open(potential_gif_s2, "rb") as f:
                st.download_button(
                    label="💾 Download GIF",
                    data=f,
                    file_name=potential_gif_s2.name,
                    mime="image/gif",
                    key=f"download_existing_s2_{current}",
                )

    with existing_col_ls:
        if potential_gif_landsat.exists():
            st.subheader("Landsat (2000-2025)")
            st.info("Timelapse already exists")
            st.image(str(potential_gif_landsat), caption=f"Timelapse: {current}", width=512)

            with open(potential_gif_landsat, "rb") as f:
                st.download_button(
                    label="💾 Download GIF",
                    data=f,
                    file_name=potential_gif_landsat.name,
                    mime="image/gif",
                    key=f"download_existing_landsat_{current}",
                )


def plot_jrc_timeseries(
    jrc_dataset: Optional[JRCDataset],
    id_geohash: str,
    zarr_path_jrc: str | Path,
    downloaded_dsjrc: Optional[object],
    is_interactive: bool,
) -> bool:
    """
    Plot JRC time series with proper availability checking and fallback loading.

    Args:
        jrc_dataset: Current JRC dataset object
        id_geohash: ID of the feature to plot
        zarr_path_jrc: Path to JRC zarr file
        downloaded_dsjrc: Previously downloaded JRC data
        is_interactive: Whether to use interactive plots

    Returns:
        True if plotting succeeded, False otherwise
    """
    # Check JRC availability
    id_available_jrc = check_dataset_availability(jrc_dataset, id_geohash)

    if not id_available_jrc:
        st.caption("⚠️ JRC data not available for this feature")
        return False
