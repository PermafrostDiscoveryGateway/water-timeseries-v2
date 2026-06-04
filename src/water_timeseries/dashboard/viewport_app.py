"""Streamlit dashboard with DuckDB viewport filtering for large lake datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import duckdb
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import xarray as xr
from streamlit_folium import st_folium

from water_timeseries.dataset import DWDataset, JRCDataset
from water_timeseries.downloader import EarthEngineDownloader
from water_timeseries.utils.dashboard import (
    check_dataset_availability,
    display_existing_gifs,
    plot_time_series_data,
)
from water_timeseries.utils.data import dw_bandnames
from water_timeseries.utils.io import load_xarray_dataset
from water_timeseries.utils.map_styling import get_colored_style_function
from water_timeseries.utils.visualization import get_legend_html_net_change

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "data"
_DEFAULT_LAKES = _DATA_DIR / "optimized_lakes.parquet"
_DEFAULT_TIMESERIES = _DATA_DIR / "lake_timeseries.parquet"
_PREPROCESSED_TIMESERIES = _REPO_ROOT / "preprocessed" / "lake_timeseries.parquet"
_DEFAULT_TIMESERIES_NC = _DATA_DIR / "lakes_dw_V2d.nc"

DEFAULT_VIEWPORT = {
    "min_lat": 60.0,
    "min_lon": -170.0,
    "max_lat": 75.0,
    "max_lon": -120.0,
}

# Simplify complex lake polygons before GeoJSON export (degrees, EPSG:4326).
# Values near 0.01 collapse most lakes to invalid rings; capped per viewport in
# _effective_simplify_tolerance.
DEFAULT_SIMPLIFY_TOLERANCE = 0.001

# Ignore degenerate map bounds from streamlit-folium before the map has settled.
MIN_VIEWPORT_LAT_SPAN = 0.05
MIN_VIEWPORT_LON_SPAN = 0.05


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _viewport_tuple(viewport: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        round(viewport["min_lat"], 4),
        round(viewport["min_lon"], 4),
        round(viewport["max_lat"], 4),
        round(viewport["max_lon"], 4),
    )


def _is_valid_viewport(viewport: dict[str, float]) -> bool:
    """Reject degenerate bounds that would return no lakes or cause rerun loops."""
    try:
        min_lat = float(viewport["min_lat"])
        min_lon = float(viewport["min_lon"])
        max_lat = float(viewport["max_lat"])
        max_lon = float(viewport["max_lon"])
    except (KeyError, TypeError, ValueError):
        return False

    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon
    if lat_span < MIN_VIEWPORT_LAT_SPAN or lon_span < MIN_VIEWPORT_LON_SPAN:
        return False
    if min_lat < -90 or max_lat > 90 or min_lon < -180 or max_lon > 180:
        return False
    return True


def _effective_simplify_tolerance(
    viewport: dict[str, float],
    simplify_tolerance: float | None,
) -> float | None:
    """Cap ST_Simplify so rings stay valid for the current map scale."""
    if simplify_tolerance is None:
        return None
    span = min(
        viewport["max_lat"] - viewport["min_lat"],
        viewport["max_lon"] - viewport["min_lon"],
    )
    # ~0.01° tolerance destroys most lakes at regional zoom; scale to viewport.
    return min(simplify_tolerance, max(span / 50_000.0, 1e-7))


def _geometry_has_valid_ring(geometry: dict[str, Any]) -> bool:
    """Leaflet requires at least four positions per polygon ring."""
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return False
    if geom_type == "Polygon":
        outer = coords[0] if coords else []
        return isinstance(outer, list) and len(outer) >= 4
    if geom_type == "MultiPolygon":
        return any(
            isinstance(poly, list) and poly and isinstance(poly[0], list) and len(poly[0]) >= 4
            for poly in coords
        )
    return True


def _viewport_changed(old: dict[str, float], new: dict[str, float]) -> bool:
    """Return True if the map bounds changed enough to warrant a data refresh."""
    if not _is_valid_viewport(new):
        return False
    return _viewport_tuple(old) != _viewport_tuple(new)


def _viewport_center(viewport: dict[str, float]) -> tuple[float, float]:
    return (
        (viewport["min_lat"] + viewport["max_lat"]) / 2,
        (viewport["min_lon"] + viewport["max_lon"]) / 2,
    )


def _estimate_zoom_from_viewport(viewport: dict[str, float]) -> int:
    """Rough initial zoom from viewport lat span (degrees)."""
    lat_span = viewport["max_lat"] - viewport["min_lat"]
    if lat_span > 20:
        return 4
    if lat_span > 10:
        return 5
    if lat_span > 5:
        return 6
    if lat_span > 2:
        return 7
    if lat_span > 1:
        return 8
    return 9


def _sync_map_state_from_widget() -> None:
    """Update session state from the latest st_folium widget output (runs in on_change)."""
    map_data = st.session_state.get("arctic_lake_map")
    if not map_data:
        return

    new_bounds = _parse_folium_bounds(map_data)
    if (
        new_bounds is not None
        and _is_valid_viewport(new_bounds)
        and _viewport_changed(st.session_state.map_viewport, new_bounds)
    ):
        st.session_state.map_viewport = new_bounds

    center = map_data.get("center")
    if isinstance(center, dict):
        lat = _safe_float(center.get("lat"))
        lng = _safe_float(center.get("lng", center.get("lon")))
        if lat is not None and lng is not None:
            st.session_state.map_center = [lat, lng]

    zoom = map_data.get("zoom")
    if zoom is not None:
        try:
            st.session_state.map_zoom = int(zoom)
        except (TypeError, ValueError):
            pass

    clicked_id = _extract_clicked_lake_id(map_data)
    if clicked_id:
        st.session_state.selected_lake_id = clicked_id


@st.cache_resource
def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Persistent in-memory DuckDB with spatial extension."""
    conn = duckdb.connect(database=":memory:")
    conn.execute("INSTALL spatial;")
    conn.execute("LOAD spatial;")
    return conn


@st.cache_data(show_spinner=False)
def _lake_id_column(lakes_parquet: str) -> str:
    """Return the lake identifier column name in the lakes parquet file."""
    conn = get_duckdb_connection()
    columns = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet(?))",
            [lakes_parquet],
        ).fetchall()
    }
    if "lake_id" in columns:
        return "lake_id"
    if "id_geohash" in columns:
        return "id_geohash"
    raise ValueError(f"No lake id column found in {lakes_parquet}. Columns: {sorted(columns)}")


@st.cache_data(show_spinner=False)
def _lake_id_sql_expression(lakes_parquet: str) -> str:
    """Resolve the lake identifier column for DuckDB queries."""
    conn = get_duckdb_connection()
    columns = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet(?))",
            [lakes_parquet],
        ).fetchall()
    }
    if "lake_id" in columns:
        return "CAST(lake_id AS VARCHAR)"
    if "id_geohash" in columns:
        return "CAST(id_geohash AS VARCHAR)"
    raise ValueError(f"No lake id column found in {lakes_parquet}. Columns: {sorted(columns)}")


_LAKE_MAP_ATTRS = ("NetChange_perc", "NetChange_ha", "Area_start_ha", "Area_end_ha")
_LAKE_TOOLTIP_COLUMNS = [
    ("NetChange_perc", "Net Change (%):", "{:.2f}", "%"),
    ("NetChange_ha", "Net Change (ha):", "{:.2f}", " ha"),
    ("Area_start_ha", "Lake Area year 2000 (ha):", "{:.2f}", " ha"),
    ("Area_end_ha", "Lake Area year 2020 (ha):", "{:.2f}", " ha"),
]


@st.cache_data(show_spinner=False)
def _lake_map_attribute_columns(lakes_parquet: str) -> tuple[str, ...]:
    """Attribute columns to include in viewport GeoJSON (for coloring and tooltips)."""
    conn = get_duckdb_connection()
    columns = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet(?))",
            [lakes_parquet],
        ).fetchall()
    }
    return tuple(col for col in _LAKE_MAP_ATTRS if col in columns)


def _feature_properties(lake_id: str, attributes: dict[str, Any]) -> dict[str, Any]:
    """Build GeoJSON properties with formatted tooltip fields."""
    props: dict[str, Any] = {"lake_id": str(lake_id)}
    for col, _alias, fmt, unit in _LAKE_TOOLTIP_COLUMNS:
        if col not in attributes:
            continue
        value = attributes[col]
        props[col] = value
        if value is None or (isinstance(value, float) and pd.isna(value)):
            props[f"{col}_display"] = "N/A"
        else:
            props[f"{col}_display"] = f"{fmt.format(float(value))}{unit}"
    return props


def query_viewport_polygons(
    conn: duckdb.DuckDBPyConnection,
    lakes_parquet: Path,
    bounds: dict[str, float],
    limit: int = 200,
    simplify_tolerance: float | None = None,
) -> list[dict[str, Any]]:
    """Return GeoJSON features intersecting the map viewport (bounded row count)."""
    min_lon = bounds["min_lon"]
    min_lat = bounds["min_lat"]
    max_lon = bounds["max_lon"]
    max_lat = bounds["max_lat"]

    lakes_str = str(lakes_parquet)
    lake_id_expr = _lake_id_sql_expression(lakes_str)
    attr_columns = _lake_map_attribute_columns(lakes_str)
    geom_expr = "geometry"
    if simplify_tolerance is not None:
        geom_expr = f"ST_Simplify(geometry, {simplify_tolerance})"

    attr_select = ""
    if attr_columns:
        attr_select = ", " + ", ".join(attr_columns)

    sql = f"""
        SELECT
            {lake_id_expr} AS lake_id{attr_select},
            ST_AsGeoJSON({geom_expr}) AS geojson
        FROM read_parquet(?)
        WHERE geometry && ST_MakeEnvelope(?, ?, ?, ?)
        LIMIT ?
    """
    rows = conn.execute(
        sql,
        [
            lakes_str,
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            limit,
        ],
    ).fetchall()

    features = []
    for row in rows:
        lake_id = row[0]
        geojson_str = row[-1]
        attr_values = {}
        if attr_columns:
            for col, value in zip(attr_columns, row[1:-1], strict=True):
                attr_values[col] = value
        if geojson_str is None:
            continue
        try:
            geometry = json.loads(geojson_str)
        except json.JSONDecodeError:
            continue
        if not _geometry_has_valid_ring(geometry):
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": _feature_properties(str(lake_id), attr_values),
            }
        )
    return features


def get_lake_timeseries(
    conn: duckdb.DuckDBPyConnection,
    timeseries_parquet: Path,
    lake_id: str,
) -> pd.DataFrame:
    """Fast indexed scan of pre-sorted lake time series."""
    return conn.execute(
        """
        SELECT lake_id, date, water
        FROM read_parquet(?)
        WHERE lake_id = ?
        ORDER BY date
        """,
        [str(timeseries_parquet), str(lake_id)],
    ).df()


@st.cache_resource
def _open_timeseries_netcdf(nc_path: str) -> xr.Dataset:
    return xr.open_dataset(nc_path)


@st.cache_data(show_spinner=False)
def get_lake_timeseries_from_netcdf(
    nc_path: str,
    lake_id: str,
    water_column: str = "water",
) -> pd.DataFrame:
    """Load one lake's time series directly from the NetCDF source."""
    ds = _open_timeseries_netcdf(nc_path)
    try:
        sub = ds.sel(id_geohash=lake_id)
    except KeyError:
        return pd.DataFrame(columns=["lake_id", "date", "water"])
    df = sub[water_column].to_dataframe(name="water").reset_index()
    df = df.rename(columns={"id_geohash": "lake_id"})
    return df[["lake_id", "date", "water"]].sort_values("date")


def _configure_ee_credentials() -> None:
    """Use Streamlit secrets for EE when the token is not already in the environment."""
    if "EARTHENGINE_TOKEN" in os.environ:
        return
    try:
        if "EARTHENGINE_TOKEN" in st.secrets:
            os.environ["EARTHENGINE_TOKEN"] = st.secrets["EARTHENGINE_TOKEN"]
    except Exception:
        pass


@st.cache_resource
def _get_ee_downloader(ee_project: str | None) -> EarthEngineDownloader:
    _configure_ee_credentials()
    return EarthEngineDownloader(ee_auth=True, ee_project=ee_project)


def _load_lake_geodataframe(lakes_parquet: Path, lake_id: str, id_column: str) -> gpd.GeoDataFrame:
    """Load a single lake polygon for Earth Engine zonal statistics."""
    lake_id = str(lake_id)
    try:
        gdf = gpd.read_parquet(lakes_parquet, filters=[(id_column, "==", lake_id)])
    except Exception:
        gdf = gpd.read_parquet(lakes_parquet, columns=[id_column, "geometry"])
        gdf = gdf[gdf[id_column].astype(str) == lake_id]
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def _dataset_to_timeseries_df(ds: xr.Dataset, lake_id: str, id_column: str) -> pd.DataFrame:
    """Convert a Dynamic World xarray dataset to lake_id/date/water rows."""
    if id_column not in ds.dims and id_column not in ds.coords:
        return pd.DataFrame(columns=["lake_id", "date", "water"])
    if "water" not in ds.data_vars:
        return pd.DataFrame(columns=["lake_id", "date", "water"])

    try:
        water = ds["water"].sel({id_column: lake_id})
    except KeyError:
        return pd.DataFrame(columns=["lake_id", "date", "water"])

    df = water.to_dataframe(name="water").reset_index()
    if id_column in df.columns:
        df = df.rename(columns={id_column: "lake_id"})
    if "lake_id" not in df.columns:
        df["lake_id"] = lake_id
    return df[["lake_id", "date", "water"]].sort_values("date")


@st.cache_data(show_spinner="Querying Google Earth Engine…")
def get_lake_timeseries_from_gee(
    lakes_parquet: str,
    lake_id: str,
    ee_project: str | None,
    dw_start_year: int,
    dw_end_year: int,
    dw_start_month: int,
    dw_end_month: int,
) -> pd.DataFrame:
    """Download monthly Dynamic World water area for one lake from Google Earth Engine."""
    lakes_path = Path(lakes_parquet)
    id_column = _lake_id_column(lakes_parquet)
    gdf = _load_lake_geodataframe(lakes_path, lake_id, id_column)
    if gdf.empty:
        return pd.DataFrame(columns=["lake_id", "date", "water"])

    downloader = _get_ee_downloader(ee_project)
    months = list(range(dw_start_month, dw_end_month + 1))
    ds = downloader.download_dw_monthly(
        gdf=gdf,
        name_attribute=id_column,
        id_list=[lake_id],
        years=list(range(dw_start_year, dw_end_year + 1)),
        months=months,
    )
    return _dataset_to_timeseries_df(ds, str(lake_id), id_column)


def _load_lake_timeseries(
    *,
    lake_id: str,
    conn: duckdb.DuckDBPyConnection,
    lakes_path: Path,
    timeseries_path: Path,
    nc_path: Path,
    has_timeseries_parquet: bool,
    has_timeseries_nc: bool,
    offline_mode: bool,
    ee_project: str | None,
    dw_start_year: int,
    dw_end_year: int,
    dw_start_month: int,
    dw_end_month: int,
) -> tuple[pd.DataFrame, str | None]:
    """Load time series locally when available, otherwise query Earth Engine."""
    ts_df = pd.DataFrame(columns=["lake_id", "date", "water"])
    source: str | None = None

    if has_timeseries_parquet:
        ts_df = get_lake_timeseries(conn, timeseries_path, lake_id)
        source = "local parquet"
    elif has_timeseries_nc:
        ts_df = get_lake_timeseries_from_netcdf(str(nc_path), lake_id)
        source = "local NetCDF"

    if ts_df.empty and not offline_mode:
        ts_df = get_lake_timeseries_from_gee(
            str(lakes_path.resolve()),
            lake_id,
            ee_project,
            dw_start_year,
            dw_end_year,
            dw_start_month,
            dw_end_month,
        )
        source = "Google Earth Engine"

    return ts_df, source


def _bounds_from_corners(sw: Any, ne: Any) -> dict[str, float] | None:
    """Build a viewport dict from southwest / northeast corner objects."""
    if not sw or not ne:
        return None

    if isinstance(sw, dict):
        sw_lat = _safe_float(sw.get("lat", sw.get(0)))
        sw_lng = _safe_float(sw.get("lng", sw.get("lon", sw.get(1))))
        ne_lat = _safe_float(ne.get("lat", ne.get(0)))
        ne_lng = _safe_float(ne.get("lng", ne.get("lon", ne.get(1))))
    elif isinstance(sw, (list, tuple)) and isinstance(ne, (list, tuple)) and len(sw) >= 2 and len(ne) >= 2:
        sw_lat = _safe_float(sw[0])
        sw_lng = _safe_float(sw[1])
        ne_lat = _safe_float(ne[0])
        ne_lng = _safe_float(ne[1])
    else:
        return None

    if None in (sw_lat, sw_lng, ne_lat, ne_lng):
        return None

    return {
        "min_lat": min(sw_lat, ne_lat),
        "min_lon": min(sw_lng, ne_lng),
        "max_lat": max(sw_lat, ne_lat),
        "max_lon": max(sw_lng, ne_lng),
    }


def _parse_folium_bounds(map_data: dict) -> dict[str, float] | None:
    """Normalize streamlit-folium bounds to min_lat/min_lon/max_lat/max_lon."""
    if not map_data:
        return None

    bounds = map_data.get("bounds")
    if bounds is None:
        parsed = _bounds_from_corners(map_data.get("_southWest"), map_data.get("_northEast"))
        if parsed is not None:
            return parsed
        clicked = map_data.get("last_clicked")
        if isinstance(clicked, dict):
            lat = _safe_float(clicked.get("lat"))
            lng = _safe_float(clicked.get("lng"))
            if lat is not None and lng is not None:
                pad = 0.25
                return {
                    "min_lat": lat - pad,
                    "min_lon": lng - pad,
                    "max_lat": lat + pad,
                    "max_lon": lng + pad,
                }
        return None

    if isinstance(bounds, dict):
        parsed = _bounds_from_corners(
            bounds.get("_southWest") or bounds.get("southWest") or bounds.get("sw"),
            bounds.get("_northEast") or bounds.get("northEast") or bounds.get("ne"),
        )
        if parsed is not None:
            return parsed

    if isinstance(bounds, list) and len(bounds) == 2:
        return _bounds_from_corners(bounds[0], bounds[1])

    return None


def _extract_clicked_lake_id(map_data: dict) -> str | None:
    """Read lake_id from GeoJSON click events returned by streamlit-folium."""
    for key in ("last_object_clicked", "last_active_drawing"):
        clicked = map_data.get(key)
        if not clicked:
            continue
        props = clicked.get("properties") if isinstance(clicked, dict) else None
        if props and props.get("lake_id"):
            return str(props["lake_id"])
    return None


def _map_has_netchange(features: list[dict[str, Any]]) -> bool:
    if not features:
        return False
    return "NetChange_perc" in features[0].get("properties", {})


def _tooltip_fields_and_aliases(features: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    fields = ["lake_id"]
    aliases = ["Lake ID:"]
    if not features:
        return fields, aliases
    props = features[0].get("properties", {})
    for col, alias, _fmt, _unit in _LAKE_TOOLTIP_COLUMNS:
        display_key = f"{col}_display"
        if display_key in props:
            fields.append(display_key)
            aliases.append(alias)
    return fields, aliases


def _make_style_function(selected_lake_id: str | None, features: list[dict[str, Any]]):
    """Color by NetChange_perc (main dashboard) and highlight the selected lake."""
    selected = str(selected_lake_id) if selected_lake_id is not None else None
    if _map_has_netchange(features):
        base_style = get_colored_style_function(
            color_column="NetChange_perc",
            vmin=-40,
            vmax=40,
            colormap=plt.cm.RdYlBu,
        )
    else:
        def base_style(_feature: dict) -> dict:
            return {
                "fillColor": "#4da6ff",
                "color": "#1a6bb5",
                "weight": 2,
                "fillOpacity": 0.35,
            }

    def style_fn(feature: dict) -> dict:
        style = dict(base_style(feature))
        lake_id = feature.get("properties", {}).get("lake_id")
        if lake_id is not None and str(lake_id) == selected:
            style["color"] = "#ff4b4b"
            style["weight"] = 4
            style["fillOpacity"] = min(float(style.get("fillOpacity", 0.6)) + 0.15, 0.9)
        return style

    return style_fn


@st.cache_data(show_spinner=False)
def _dataset_has_dw_bands(path: str) -> bool:
    """True when a zarr/NetCDF file contains the full Dynamic World band set."""
    path_obj = Path(path)
    if not path_obj.exists():
        return False
    try:
        path_str = str(path)
        if path_str.endswith(".zarr"):
            ds = xr.open_zarr(path_str, consolidated=True)
        else:
            ds = xr.open_dataset(path_str)
        return all(band in ds.data_vars for band in dw_bandnames)
    except Exception:
        return False


def _dw_subset_for_lake(ds: xr.Dataset, lake_id: str) -> xr.Dataset:
    """Select one lake and keep the id_geohash dimension for DWDataset."""
    sub = ds.sel(id_geohash=lake_id)
    if "id_geohash" not in sub.dims:
        sub = sub.expand_dims(id_geohash=[lake_id])
    return sub.load()


@st.cache_resource
def _open_dw_backend_dataset(path: str) -> xr.Dataset:
    if str(path).endswith(".zarr"):
        return xr.open_zarr(path, consolidated=True)
    return xr.open_dataset(path)


def _load_dw_dataset_for_lake(
    lake_id: str,
    *,
    dw_zarr_path: Path | None,
    timeseries_nc: Path | None,
    downloaded_ds: xr.Dataset | None,
) -> DWDataset | None:
    if downloaded_ds is not None:
        try:
            return DWDataset(downloaded_ds)
        except Exception:
            pass

    for path in (dw_zarr_path, timeseries_nc):
        if path is None or not path.exists() or not _dataset_has_dw_bands(str(path)):
            continue
        try:
            ds = _open_dw_backend_dataset(str(path))
            return DWDataset(_dw_subset_for_lake(ds, lake_id))
        except (KeyError, Exception):
            continue
    return None


def _load_jrc_dataset_for_lake(
    lake_id: str,
    *,
    jrc_zarr_path: Path | None,
    downloaded_ds: xr.Dataset | None,
) -> JRCDataset | None:
    if downloaded_ds is not None:
        try:
            return JRCDataset(downloaded_ds)
        except Exception:
            pass
    if jrc_zarr_path is None or not jrc_zarr_path.exists():
        return None
    try:
        ds = load_xarray_dataset(str(jrc_zarr_path))
        return JRCDataset(_dw_subset_for_lake(ds, lake_id))
    except (KeyError, Exception):
        return None


def _render_lake_detail_panel(
    *,
    lake_id: str,
    lakes_path: Path,
    lakes_id_column: str,
    dw_zarr_path: Path | None,
    jrc_zarr_path: Path | None,
    timeseries_nc: Path | None,
    offline_mode: bool,
    ee_project: str | None,
    dw_start_year: int,
    dw_end_year: int,
    dw_start_month: int,
    dw_end_month: int,
    is_interactive: bool,
    water_only_df: pd.DataFrame | None,
    water_only_source: str | None,
) -> None:
    """Render the same lake-click visualizations as the main dashboard."""
    st.subheader(f"Time series: {lake_id}")

    if st.button("Open time series in popup", key="open_ts_popup"):
        st.session_state.show_ts_popup = True

    if "dw_dataset" not in st.session_state:
        st.session_state.dw_dataset = None
    if "jrc_dataset" not in st.session_state:
        st.session_state.jrc_dataset = None
    if "downloaded_dsdw" not in st.session_state:
        st.session_state.downloaded_dsdw = None
    if "downloaded_dsjrc" not in st.session_state:
        st.session_state.downloaded_dsjrc = None
    if "show_ts_popup" not in st.session_state:
        st.session_state.show_ts_popup = False

    cache_key = (lake_id, str(dw_zarr_path), str(timeseries_nc), str(jrc_zarr_path))
    if st.session_state.get("dw_dataset_lake_key") != cache_key:
        st.session_state.dw_dataset = _load_dw_dataset_for_lake(
            lake_id,
            dw_zarr_path=dw_zarr_path,
            timeseries_nc=timeseries_nc,
            downloaded_ds=st.session_state.downloaded_dsdw,
        )
        st.session_state.jrc_dataset = _load_jrc_dataset_for_lake(
            lake_id,
            jrc_zarr_path=jrc_zarr_path,
            downloaded_ds=st.session_state.downloaded_dsjrc,
        )
        st.session_state.dw_dataset_lake_key = cache_key

    dw_dataset = st.session_state.dw_dataset
    jrc_dataset = st.session_state.jrc_dataset

    id_available_dw = check_dataset_availability(dw_dataset, lake_id)
    id_available_jrc = check_dataset_availability(jrc_dataset, lake_id)

    if not id_available_dw and not offline_mode:
        try:
            downloader = _get_ee_downloader(ee_project)
            id_col = lakes_id_column
            gdf = _load_lake_geodataframe(lakes_path, lake_id, id_col)
            if not gdf.empty:
                with st.spinner("Downloading Dynamic World data from Earth Engine…"):
                    dsdw = downloader.download_dw_monthly(
                        gdf=gdf,
                        name_attribute=id_col,
                        id_list=[lake_id],
                        years=list(range(dw_start_year, dw_end_year + 1)),
                        months=list(range(dw_start_month, dw_end_month + 1)),
                    )
                if dsdw is not None:
                    st.session_state.downloaded_dsdw = dsdw
                    st.session_state.dw_dataset = DWDataset(dsdw)
                    st.session_state.dw_dataset_lake_key = cache_key
                    dw_dataset = st.session_state.dw_dataset
                    id_available_dw = True
                    st.rerun()
        except Exception as exc:
            st.error(f"Failed to download Dynamic World data: {exc}")

    if id_available_dw or id_available_jrc:
        ts_col1, ts_col2 = st.columns(2)
        with ts_col1:
            st.markdown("**Dynamic World**")
            plot_time_series_data(dw_dataset, lake_id, "dw", is_interactive, show_caption=True)
        with ts_col2:
            if id_available_jrc:
                st.markdown("**JRC**")
                plot_time_series_data(jrc_dataset, lake_id, "jrc", is_interactive, show_caption=True)
            else:
                st.caption("JRC data not available for this lake")
    elif water_only_df is not None and not water_only_df.empty:
        st.caption(
            f"Water extent preview ({water_only_source}). "
            "Provide a Dynamic World zarr/NetCDF or enable Earth Engine for full class plots."
        )
        preview = water_only_df.copy()
        preview["date"] = pd.to_datetime(preview["date"])
        preview = preview.set_index("date")
        st.line_chart(preview["water"])
        with st.expander("Raw data"):
            st.dataframe(preview, use_container_width=True)
    else:
        st.warning("No time series data available for this lake.")

    if id_available_dw and dw_dataset is not None and not offline_mode:
        st.divider()
        st.subheader("Satellite timelapse")
        create_sentinel2 = st.checkbox("Sentinel-2 (2016-2025)", value=True, key="sentinel2_checkbox")
        create_landsat = st.checkbox("Landsat (2000-2025)", value=True, key="landsat_checkbox")
        gif_dir = Path("gifs")
        if st.button("Create timelapse", key="create_timelapse"):
            if not create_sentinel2 and not create_landsat:
                st.warning("Select at least one data source.")
            else:
                lake_gdf = _load_lake_geodataframe(lakes_path, lake_id, lakes_id_column)
                with st.spinner("Generating timelapse…"):
                    try:
                        gif_s2 = None
                        gif_ls = None
                        if create_sentinel2:
                            gif_s2 = dw_dataset.create_timelapse(
                                lake_gdf=lake_gdf,
                                id_geohash=lake_id,
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
                        if create_landsat:
                            gif_ls = dw_dataset.create_timelapse(
                                lake_gdf=lake_gdf,
                                id_geohash=lake_id,
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
                        display_existing_gifs(lake_id, gif_dir)
                        if gif_s2:
                            st.success(f"Sentinel-2 timelapse: {gif_s2}")
                        if gif_ls:
                            st.success(f"Landsat timelapse: {gif_ls}")
                    except Exception as exc:
                        st.error(f"Timelapse failed: {exc}")
        else:
            display_existing_gifs(lake_id, gif_dir)

    if st.session_state.get("show_ts_popup") and (id_available_dw or id_available_jrc):

        @st.dialog("Time series plot", width="large")
        def ts_popup() -> None:
            st.subheader(f"Time series: {lake_id}")
            if id_available_dw:
                if is_interactive:
                    fig = dw_dataset.plot_timeseries_interactive(lake_id)
                    st.plotly_chart(fig, width="stretch")
                else:
                    fig = dw_dataset.plot_timeseries(lake_id)
                    st.pyplot(fig)
                    plt.close(fig)
            if id_available_jrc and jrc_dataset is not None:
                if is_interactive:
                    fig_jrc = jrc_dataset.plot_timeseries_interactive(lake_id)
                    st.plotly_chart(fig_jrc, width="stretch")
                else:
                    fig_jrc = jrc_dataset.plot_timeseries(lake_id)
                    st.pyplot(fig_jrc)
                    plt.close(fig_jrc)
            if st.button("Close", key="close_ts_popup"):
                st.session_state.show_ts_popup = False

        ts_popup()


def run_app(
    lakes_parquet: Path | None = None,
    timeseries_parquet: Path | None = None,
    timeseries_nc: Path | None = None,
    dw_dataset_file: Path | None = None,
    jrc_dataset_file: Path | None = None,
    polygon_limit: int = 200,
    simplify_tolerance: float | None = DEFAULT_SIMPLIFY_TOLERANCE,
    offline_mode: bool = False,
    ee_project: str | None = None,
    dw_start_year: int = 2017,
    dw_end_year: int = 2025,
    dw_start_month: int = 6,
    dw_end_month: int = 9,
) -> None:
    """Run the viewport-filtered dashboard."""
    lakes_path = Path(lakes_parquet or _DEFAULT_LAKES).resolve()
    if timeseries_parquet is not None:
        timeseries_path = Path(timeseries_parquet).resolve()
    elif _DEFAULT_TIMESERIES.exists():
        timeseries_path = _DEFAULT_TIMESERIES.resolve()
    elif _PREPROCESSED_TIMESERIES.exists():
        timeseries_path = _PREPROCESSED_TIMESERIES.resolve()
    else:
        timeseries_path = _DEFAULT_TIMESERIES.resolve()
    nc_path = Path(timeseries_nc or _DEFAULT_TIMESERIES_NC).resolve()
    dw_zarr_path = Path(dw_dataset_file).resolve() if dw_dataset_file else None
    jrc_zarr_path = Path(jrc_dataset_file).resolve() if jrc_dataset_file else None
    lakes_id_column = _lake_id_column(str(lakes_path))

    st.set_page_config(layout="wide", page_title="Arctic Lakes — Viewport Dashboard")

    if not lakes_path.exists():
        st.error(
            f"Lake polygons not found at `{lakes_path}`. "
            "Run preprocessing first:\n\n"
            "```bash\n"
            "uv run water-timeseries preprocess-viewport-data\n"
            "```"
        )
        st.stop()

    if "map_viewport" not in st.session_state or not _is_valid_viewport(st.session_state.map_viewport):
        st.session_state.map_viewport = dict(DEFAULT_VIEWPORT)
    if "map_center" not in st.session_state:
        lat, lon = _viewport_center(st.session_state.map_viewport)
        st.session_state.map_center = [lat, lon]
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = _estimate_zoom_from_viewport(st.session_state.map_viewport)
    if "selected_lake_id" not in st.session_state:
        st.session_state.selected_lake_id = None

    conn = get_duckdb_connection()
    has_timeseries_parquet = timeseries_path.exists()
    has_timeseries_nc = nc_path.exists()
    has_local_timeseries = has_timeseries_parquet or has_timeseries_nc
    can_query_gee = not offline_mode

    st.session_state.offline_mode = offline_mode

    st.sidebar.header("Settings")
    is_interactive = st.sidebar.toggle(
        "Interactive plotting",
        value=True,
        help="Plotly charts with hover and zoom (same as main dashboard).",
    )

    st.title("Arctic Permafrost Lakes")
    if has_timeseries_parquet:
        ts_caption = f" · Time series: `{timeseries_path.name}`"
    elif has_timeseries_nc:
        ts_caption = f" · Time series: `{nc_path.name}`"
    elif can_query_gee:
        ts_caption = " · Time series: Google Earth Engine (on demand)"
    else:
        ts_caption = " · Time series: unavailable (offline mode)"
    st.caption(
        f"Viewport-filtered map (max {polygon_limit} polygons). "
        f"Data: `{lakes_path.name}`{ts_caption}"
    )

    col1, col2 = st.columns([2, 1])

    viewport = st.session_state.map_viewport
    effective_simplify = _effective_simplify_tolerance(viewport, simplify_tolerance)
    with st.spinner("Loading lakes in viewport…"):
        features = query_viewport_polygons(
            conn,
            lakes_path,
            viewport,
            limit=polygon_limit,
            simplify_tolerance=effective_simplify,
        )

    with col1:
        center_lat, center_lon = st.session_state.map_center
        map_zoom = st.session_state.map_zoom
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=map_zoom,
            tiles="CartoDB positron",
        )
        if _map_has_netchange(features):
            folium.TileLayer("CartoDB.DarkMatter", name="Dark Matter (CartoDB)").add_to(m)
            folium.TileLayer("Esri.WorldImagery", name="ESRI World Imagery").add_to(m)

        if features:
            geojson_fc = {"type": "FeatureCollection", "features": features}
            tooltip_fields, tooltip_aliases = _tooltip_fields_and_aliases(features)
            folium.GeoJson(
                geojson_fc,
                name="Lakes in viewport",
                style_function=_make_style_function(st.session_state.selected_lake_id, features),
                tooltip=folium.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=tooltip_aliases,
                ),
            ).add_to(m)
            if _map_has_netchange(features):
                m.get_root().html.add_child(folium.Element(get_legend_html_net_change()))
            folium.LayerControl().add_to(m)
        else:
            st.info("No lakes in the current viewport. Pan or zoom the map.")

        # Rebuild the map each rerun from session center/zoom (updated in on_change).
        # Do not pass center/zoom to st_folium — in-place updates leave GeoJSON paths at
        # zero screen size until a full remount.
        st_folium(
            m,
            key="arctic_lake_map",
            height=600,
            width=None,
            use_container_width=True,
            returned_objects=[
                "last_object_clicked",
                "last_active_drawing",
                "bounds",
                "center",
                "zoom",
            ],
            on_change=_sync_map_state_from_widget,
        )

    with col2:
        st.subheader("Selection")
        selected = st.session_state.selected_lake_id
        if selected is None:
            st.info("Click a lake polygon on the map to view time series and timelapses.")
        else:
            st.markdown(f"**Lake ID:** `{selected}`")

        st.divider()
        st.markdown("**Viewport**")
        st.json(st.session_state.map_viewport)
        st.caption(f"Showing {len(features)} / {polygon_limit} max polygons in view.")

    if selected:
        water_df: pd.DataFrame | None = None
        water_source: str | None = None
        has_dw_source = (
            (dw_zarr_path is not None and dw_zarr_path.exists())
            or (nc_path.exists() and _dataset_has_dw_bands(str(nc_path)))
            or can_query_gee
        )
        if not has_dw_source and (has_local_timeseries or can_query_gee):
            try:
                water_df, water_source = _load_lake_timeseries(
                    lake_id=selected,
                    conn=conn,
                    lakes_path=lakes_path,
                    timeseries_path=timeseries_path,
                    nc_path=nc_path,
                    has_timeseries_parquet=has_timeseries_parquet,
                    has_timeseries_nc=has_timeseries_nc and not _dataset_has_dw_bands(str(nc_path)),
                    offline_mode=offline_mode,
                    ee_project=ee_project,
                    dw_start_year=dw_start_year,
                    dw_end_year=dw_end_year,
                    dw_start_month=dw_start_month,
                    dw_end_month=dw_end_month,
                )
            except Exception as exc:
                st.error(f"Failed to load water time series: {exc}")

        st.divider()
        _render_lake_detail_panel(
            lake_id=selected,
            lakes_path=lakes_path,
            lakes_id_column=lakes_id_column,
            dw_zarr_path=dw_zarr_path,
            jrc_zarr_path=jrc_zarr_path,
            timeseries_nc=nc_path if nc_path.exists() and _dataset_has_dw_bands(str(nc_path)) else None,
            offline_mode=offline_mode,
            ee_project=ee_project,
            dw_start_year=dw_start_year,
            dw_end_year=dw_end_year,
            dw_start_month=dw_start_month,
            dw_end_month=dw_end_month,
            is_interactive=is_interactive,
            water_only_df=water_df,
            water_only_source=water_source,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lakes-parquet", type=Path, default=None)
    parser.add_argument("--timeseries-parquet", type=Path, default=None)
    parser.add_argument(
        "--timeseries-nc",
        type=Path,
        default=None,
        help="NetCDF time series (used when lake_timeseries.parquet is absent).",
    )
    parser.add_argument("--polygon-limit", type=int, default=200)
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        default=DEFAULT_SIMPLIFY_TOLERANCE,
        help="ST_Simplify tolerance in degrees (capped to viewport scale; default 0.001).",
    )
    parser.add_argument(
        "--offline-mode",
        action="store_true",
        help="Disable Google Earth Engine fallback when local time series are missing.",
    )
    parser.add_argument("--ee-project", type=str, default=None, help="Google Earth Engine project ID.")
    parser.add_argument(
        "--dw-dataset-file",
        type=Path,
        default=None,
        help="Dynamic World zarr store for full class time series plots.",
    )
    parser.add_argument(
        "--jrc-dataset-file",
        type=Path,
        default=None,
        help="JRC zarr store for annual water time series plots.",
    )
    parser.add_argument("--dw-start-year", type=int, default=2017)
    parser.add_argument("--dw-end-year", type=int, default=2025)
    parser.add_argument("--dw-start-month", type=int, default=6)
    parser.add_argument("--dw-end-month", type=int, default=9)
    args, _unknown = parser.parse_known_args()
    run_app(
        lakes_parquet=args.lakes_parquet,
        timeseries_parquet=args.timeseries_parquet,
        timeseries_nc=args.timeseries_nc,
        dw_dataset_file=args.dw_dataset_file,
        jrc_dataset_file=args.jrc_dataset_file,
        polygon_limit=args.polygon_limit,
        simplify_tolerance=args.simplify_tolerance,
        offline_mode=args.offline_mode,
        ee_project=args.ee_project,
        dw_start_year=args.dw_start_year,
        dw_end_year=args.dw_end_year,
        dw_start_month=args.dw_start_month,
        dw_end_month=args.dw_end_month,
    )
