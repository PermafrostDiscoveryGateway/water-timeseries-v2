"""MapLibre + PMTiles map viewer for Streamlit and standalone use."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import streamlit as st

from water_timeseries.utils.io import load_vector_dataset
from water_timeseries.utils.pmtiles_reader import read_pmtiles_header, read_pmtiles_header_remote
from water_timeseries.utils.pmtiles_serve import PmtilesServer

_SESSION_SERVER_KEY = "_pmtiles_map_server"

# Listens for postMessage from the map iframe (cross-origin) and sets ?selected_lake= on
# the Streamlit page so sync_query_param_selection can load time series.
_SELECTION_BRIDGE_HTML = """
<script>
(function() {
  if (window.__waterTimeseriesSelectionBridge) return;
  window.__waterTimeseriesSelectionBridge = true;
  window.addEventListener("message", function(event) {
    var data = event.data;
    if (!data || data.type !== "water_timeseries_select_lake") return;
    var id = data.id != null ? String(data.id) : "";
    if (!id) return;
    var url = new URL(window.location.href);
    if (url.searchParams.get("selected_lake") === id) return;
    url.searchParams.set("selected_lake", id);
    window.location.href = url.toString();
  });
})();
</script>
"""


def _inject_selection_bridge() -> None:
    """Register a listener on the Streamlit page for map iframe selection events."""
    if hasattr(st, "html"):
        st.html(_SELECTION_BRIDGE_HTML, unsafe_allow_javascript=True)
    else:
        import streamlit.components.v1 as components

        components.html(_SELECTION_BRIDGE_HTML, height=0)


def _bounds_center_zoom(gdf: gpd.GeoDataFrame) -> tuple[list[float], float]:
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    center = [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]
    lat_span = max(bounds[3] - bounds[1], 0.5)
    zoom = max(2, min(12, 8 - lat_span / 10))
    return center, zoom


def _get_or_start_server(pmtiles_file: Optional[Path | str] = None) -> PmtilesServer:
    """Start (or reuse) a local server that hosts the map page (and optionally the .pmtiles file)."""
    if pmtiles_file is not None:
        pmtiles_path = Path(pmtiles_file).resolve()
        if not pmtiles_path.is_file():
            raise FileNotFoundError(f"PMTiles file not found: {pmtiles_path}")
    else:
        pmtiles_path = None

    server: Optional[PmtilesServer] = st.session_state.get(_SESSION_SERVER_KEY)
    if server is None or server.pmtiles_path != pmtiles_path:
        if server is not None:
            server.stop()
        server = PmtilesServer(pmtiles_path).start()
        st.session_state[_SESSION_SERVER_KEY] = server

    return server


def _build_map_config(
    *,
    pmtiles_file: Optional[Path | str] = None,
    pmtiles_url: Optional[str] = None,
    vector_file_for_bounds: Optional[Path | str] = None,
    id_column: str = "id_geohash",
    viz_configuration: str = "colored_historical",
    height: int = 620,
    center: Optional[list[float]] = None,
    zoom: Optional[float] = None,
    drained_data: Optional[dict[str, Any]] = None,
    show_main_layer: bool = True,
) -> dict[str, Any]:
    bounds = None
    if not show_main_layer and drained_data:
        pass  # Without GeoJSON, we cannot easily calculate bounds. We will rely on default center/zoom or the PMTiles header.

    if bounds is None and pmtiles_file is not None:
        header = read_pmtiles_header(pmtiles_file)
        bounds = header["bounds"]
        if center is None:
            center = header["center"]
        if zoom is None:
            zoom = float(header["zoom"])
    elif bounds is None and pmtiles_url:
        try:
            header = read_pmtiles_header_remote(pmtiles_url)
            bounds = header["bounds"]
            if center is None:
                center = header["center"]
            if zoom is None:
                zoom = float(header["zoom"])
        except Exception as e:
            print(f"Could not fetch remote PMTiles header: {e}")
            # Fall back to hardcoded Alaska defaults
    elif bounds is None and vector_file_for_bounds is not None:
        gdf = load_vector_dataset(vector_file_for_bounds)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        else:
            gdf = gdf.to_crs(epsg=4326)
        minx, miny, maxx, maxy = gdf.total_bounds
        bounds = [[float(minx), float(miny)], [float(maxx), float(maxy)]]
        if center is None:
            center, zoom_est = _bounds_center_zoom(gdf)
            if zoom is None:
                zoom = zoom_est

    config: dict[str, Any] = {
        "pmtiles_url": pmtiles_url or "",
        "id_column": id_column,
        "viz_configuration": viz_configuration,
        "source_layer": "lakes",
        "center": center or [-164.1, 66.5],
        "zoom": zoom if zoom is not None else 8,
        "bounds": bounds,
        "height": height,
        "drained_data": drained_data,
        "show_main_layer": show_main_layer,
    }
    return config


def render_pmtiles_map(
    *,
    pmtiles_file: Optional[Path | str] = None,
    pmtiles_url: Optional[str] = None,
    vector_file_for_bounds: Optional[Path | str] = None,
    id_column: str = "id_geohash",
    viz_configuration: str = "colored_historical",
    height: int = 620,
    center: Optional[list[float]] = None,
    zoom: Optional[float] = None,
    drained_data: Optional[dict[str, Any]] = None,
    show_main_layer: bool = True,
) -> None:
    """Embed a MapLibre map that loads lake polygons from PMTiles on demand.

    Uses ``st.iframe`` pointed at a local map server so the map and PMTiles archive
    share one origin. Streamlit's ``components.html`` sandbox often blocks cross-origin
    fetches to a separate tile port (basemap loads, vector lakes do not).
    """
    if pmtiles_url:
        server = _get_or_start_server(None)  # Server running in HTML-only remote mode
        config = _build_map_config(
            pmtiles_url=pmtiles_url,
            vector_file_for_bounds=vector_file_for_bounds,
            id_column=id_column,
            viz_configuration=viz_configuration,
            height=height,
            center=center,
            zoom=zoom,
            drained_data=drained_data,
            show_main_layer=show_main_layer,
        )
        map_url = server.map_iframe_url(config)
        _inject_selection_bridge()
        
        if hasattr(st, "iframe"):
            st.iframe(map_url, height=height)
        else:
            import streamlit.components.v1 as components
            components.iframe(map_url, height=height)
        return

    if pmtiles_file is None:
        raise ValueError("Either pmtiles_file or pmtiles_url must be provided")

    server = _get_or_start_server(pmtiles_file)
    config = _build_map_config(
        pmtiles_file=pmtiles_file,
        vector_file_for_bounds=vector_file_for_bounds if pmtiles_file is None else None,
        id_column=id_column,
        viz_configuration=viz_configuration,
        height=height,
        center=center,
        zoom=zoom,
        drained_data=drained_data,
        show_main_layer=show_main_layer,
    )
    map_url = server.map_iframe_url(config)

    st.markdown(f"🗺️ [Open map in full tab]({map_url}) (if the embed is blank)")

    _inject_selection_bridge()

    # Same-origin iframe: map page + PMTiles on one port (works reliably in Streamlit).
    if hasattr(st, "iframe"):
        st.iframe(map_url, height=height)
    else:
        import streamlit.components.v1 as components

        components.iframe(map_url, height=height)


def _query_param_lake_id() -> Optional[str]:
    raw = st.query_params.get("selected_lake")
    if raw is None:
        return None
    if isinstance(raw, list):
        return str(raw[0]) if raw else None
    return str(raw)


def sync_query_param_selection(id_column: str = "id_geohash") -> Optional[str]:
    """Read ``selected_lake`` from URL query params into session state."""
    selected = _query_param_lake_id()
    if not selected:
        return st.session_state.get("selected_geohash")

    if selected != st.session_state.get("selected_geohash"):
        st.session_state.selected_geohash = selected
        clicked = st.session_state.get("clicked_features", [])
        if selected not in clicked:
            clicked.append(selected)
            st.session_state.clicked_features = clicked

    return selected