"""Explore lake drainage pipeline NetCDF outputs (merged regional tiles)."""

from __future__ import annotations

import glob as glob_std
import os
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import xarray as xr
from streamlit_folium import st_folium

from water_timeseries.dataset import DWDataset
from water_timeseries.utils.map_styling import (
    create_tile_layers,
    get_colored_style_function,
    patch_folium_leaflet_default_icon_urls,
)

DW_BANDS = [
    "water",
    "bare",
    "snow_and_ice",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_glob() -> str:
    env = os.environ.get("LAKE_DRAINAGE_NC_GLOB")
    if env:
        return env
    return str(
        _project_root()
        / "downloads"
        / "annual_water_data"
        / "lake_drainage"
        / "results"
        / "output"
        / "2026"
        / "*.nc"
    )


@st.cache_data(ttl=600, show_spinner="Loading NetCDF files…")
def load_merged_nc(sorted_resolved_paths: tuple[str, ...]) -> xr.Dataset:
    paths = [Path(p) for p in sorted_resolved_paths]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing NetCDF files: {missing[:3]}{'…' if len(missing) > 3 else ''}")
    ds = xr.open_mfdataset(
        paths,
        combine="nested",
        concat_dim="id_geohash",
        join="outer",
        parallel=False,
        engine="netcdf4",
    )
    return ds.load()


def points_gdf_from_ds(ds: xr.Dataset, water_delta_scale: float = 100.0) -> gpd.GeoDataFrame:
    """Point GeoDataFrame at lake centroids; water change as pseudo NetChange_perc for styling."""
    lon = np.asarray(ds["centroid_lon"].values)
    lat = np.asarray(ds["centroid_lat"].values)
    ids = np.asarray(ds["id_geohash"].values)
    w = np.asarray(ds["water"].values, dtype=float)
    delta = (w[:, -1] - w[:, 0]) * float(water_delta_scale)
    delta = np.where(np.isfinite(delta), delta, 0.0)
    area_sq_m = np.asarray(ds["area_sq_m"].values, dtype=float)
    gdf = gpd.GeoDataFrame(
        {
            "id_geohash": ids,
            "NetChange_perc": delta,
            "water_t0": w[:, 0],
            "water_t1": w[:, -1],
            "area_sq_m": area_sq_m,
        },
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )
    gdf = gdf[np.isfinite(lon) & np.isfinite(lat)].copy()
    gdf.sort_values("area_sq_m", ascending=False, inplace=True)
    return gdf


st.title("Lake drainage (NetCDF)")
st.caption(
    "Merged Dynamic World monthly extracts from the lake drainage pipeline. "
    "Map colors show change in water fraction (×100) from the first to the last time step."
)

glob_default = _default_glob()
glob_input = st.sidebar.text_input("NetCDF glob", value=glob_default, help="Shell-style path glob, e.g. …/2026/*.nc")

pattern = glob_input.strip()
paths = sorted({Path(p).resolve() for p in glob_std.glob(pattern, recursive=False)})
if not paths and not os.path.isabs(pattern):
    for root in (_project_root(), Path("/app")):
        alt = root / pattern
        paths = sorted({Path(p).resolve() for p in glob_std.glob(str(alt), recursive=False)})
        if paths:
            break
if not paths:
    st.warning(f"No files matched: `{glob_input}`. Adjust the glob or set `LAKE_DRAINAGE_NC_GLOB`.")
    st.stop()

st.sidebar.metric("NetCDF files", len(paths))
st.sidebar.metric("Total size (MB)", round(sum(p.stat().st_size for p in paths) / (1024 * 1024), 1))

max_features = st.sidebar.number_input(
    "Max lakes on map",
    min_value=50,
    max_value=50000,
    value=2000,
    step=50,
    help="Subsample largest lakes by `area_sq_m` for responsiveness.",
)

vmin = st.sidebar.number_input("Color scale min (Δ water ×100)", value=-40.0, step=5.0)
vmax = st.sidebar.number_input("Color scale max (Δ water ×100)", value=40.0, step=5.0)
is_interactive = st.sidebar.toggle("Interactive time-series (Plotly)", value=True)

path_key = tuple(str(p.resolve()) for p in paths)
try:
    raw_ds = load_merged_nc(path_key)
except Exception as e:
    st.error(f"Could not load NetCDF: {e}")
    st.stop()

gdf_full = points_gdf_from_ds(raw_ds)
gdf_map = gdf_full.head(max_features).reset_index(drop=True)
if gdf_map.empty:
    st.error("No lakes with valid coordinates found in the merged dataset.")
    st.stop()

if "selected_geohash_nc" not in st.session_state:
    st.session_state.selected_geohash_nc = None

center_lat = float(gdf_map.geometry.y.mean())
center_lon = float(gdf_map.geometry.x.mean())
m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
patch_folium_leaflet_default_icon_urls(m)
for tile_name in create_tile_layers():
    folium.TileLayer(tile_name).add_to(m)

style_fn = get_colored_style_function(
    color_column="NetChange_perc",
    vmin=vmin,
    vmax=vmax,
    colormap=plt.cm.RdYlBu,
)
folium.GeoJson(
    gdf_map,
    name="Lakes",
    style_function=style_fn,
    tooltip=folium.GeoJsonTooltip(
        fields=["id_geohash", "NetChange_perc", "area_sq_m", "water_t0", "water_t1"],
        aliases=["id", "Δ water (×100)", "area m²", "water (first)", "water (last)"],
    ),
).add_to(m)
folium.LayerControl().add_to(m)

result = st_folium(
    m,
    height=560,
    width="100%",
    key="lake_drainage_map",
    returned_objects=["last_active_drawing"],
)

clicked_id = None
if result and result.get("last_active_drawing"):
    props = result["last_active_drawing"].get("properties") or {}
    raw_id = props.get("id_geohash")
    clicked_id = str(raw_id) if raw_id is not None else None

if clicked_id and clicked_id != st.session_state.selected_geohash_nc:
    st.session_state.selected_geohash_nc = clicked_id
    st.rerun()

st.sidebar.divider()
current = st.session_state.selected_geohash_nc
id_list = gdf_map["id_geohash"].astype(str).tolist()
if current is None and id_list:
    current = id_list[0]
try:
    sel_index = id_list.index(str(current)) if current is not None else 0
except ValueError:
    sel_index = 0

idx = min(sel_index, len(id_list) - 1) if id_list else 0
picked = st.sidebar.selectbox(
    "Selected lake",
    options=id_list,
    index=idx,
    label_visibility="collapsed",
)
if picked != st.session_state.selected_geohash_nc:
    st.session_state.selected_geohash_nc = picked
    st.rerun()

current = st.session_state.selected_geohash_nc
st.sidebar.write(f"**Selection:** `{current}`")

try:
    dw = DWDataset(raw_ds[DW_BANDS])
except Exception as e:
    st.error(f"DWDataset failed: {e}")
    st.stop()

id_set = {str(x) for x in dw.object_ids_}

if current and str(current) not in id_set:
    st.warning("Selected id is not in the merged dataset (try clicking the map or widen files).")
elif current:
    st.subheader(f"Time series — `{current}`")
    try:
        if is_interactive:
            fig = dw.plot_timeseries_interactive(str(current))
            st.plotly_chart(fig, width="stretch")
        else:
            fig = dw.plot_timeseries(str(current))
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.error(f"Plot failed: {e}")

st.sidebar.caption("Tip: run via Docker Compose with the repo mounted so `downloads/…` is visible at `/app`.")
