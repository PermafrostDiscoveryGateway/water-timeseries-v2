import os
from pathlib import Path

import folium
from folium_pmtiles.vector import PMTilesMapLibreLayer, PMTilesMapLibreTooltip


def build_pmtiles_map(
    pmtiles_url: str,
    center: tuple[float, float] = (70.0, -140.0),
    zoom_start: int = 4,
    source_layer: str = "lakes",
) -> folium.Map:
    """Return a Folium map with a PMTiles vector layer for lake polygons."""
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="CartoDB positron",  # lightweight basemap
    )

    tooltip = PMTilesMapLibreTooltip()
    lake_layer = PMTilesMapLibreLayer(
        pmtiles_url,
        layer_name="lakes_pmtiles",
        style={
            "version": 8,
            "sources": {
                "lakes_pmtiles": {
                    "type": "vector",
                    "url": "pmtiles://" + pmtiles_url,
                }
            },
            "layers": [
                {
                    "id": "lakes-fill",
                    "source": "lakes_pmtiles",
                    "source-layer": source_layer,
                    "type": "fill",
                    "paint": {
                        "fill-color": [
                            "interpolate",
                            ["linear"],
                            ["get", "net_change"],
                            -1.0,
                            "#d73027",
                            0.0,
                            "#ffffbf",
                            1.0,
                            "#1a9850",
                        ],
                        "fill-opacity": 0.7,
                    },
                },
                {
                    "id": "lakes-line",
                    "source": "lakes_pmtiles",
                    "source-layer": source_layer,
                    "type": "line",
                    "paint": {
                        "line-color": "#333333",
                        "line-width": 0.5,
                    },
                },
            ],
        },
        tooltip=tooltip,
    )
    m.add_child(lake_layer)
    return m


def resolve_pmtiles_url(pmtiles_file: str) -> str:
    """Given a local path or existing URL, return a URL the browser can fetch.

    Priority: explicit http(s) URL > GCS gs:// > local server.
    """
    if pmtiles_file.startswith(("http://", "https://")):
        return pmtiles_file
    if pmtiles_file.startswith("gs://"):
        # Convert to public GCS URL (assumes bucket is publicly readable)
        path = pmtiles_file[5:]
        return f"https://storage.googleapis.com/{path}"

    from pathlib import Path
    import streamlit as st
    from water_timeseries.utils.pmtiles_serve import PmtilesServer

    pmtiles_path = Path(pmtiles_file).resolve()
    if not pmtiles_path.is_file():
        raise FileNotFoundError(f"PMTiles file not found: {pmtiles_path}")

    server_key = "_pmtiles_map_server"
    server = st.session_state.get(server_key)
    if server is None or server.pmtiles_path != pmtiles_path:
        if server is not None:
            server.stop()
        # Serve the file
        server = PmtilesServer(pmtiles_path).start()
        st.session_state[server_key] = server

    return server.url_for(pmtiles_path.name)
