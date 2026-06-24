from pathlib import Path

import branca.element
import folium
import folium.elements
import leafmap.foliumap as leafmap
import pygeohash
from folium_pmtiles.vector import PMTilesMapLibreLayer

from water_timeseries.utils.map_styles.pmtiles import (
    get_style_pmtiles_colored_historical,
    get_style_pmtiles_drainage_year,
)
from water_timeseries.utils.visualization import get_legend_html_date_drainage_year, get_legend_html_net_change


class PMTilesMapLibreTooltipWithRounding(folium.elements.JSCSSMixin, branca.element.MacroElement):
    _template = branca.element.Template(
        """
            {% macro header(this, kwargs) %}
            <style>
            .maplibregl-popup {
                font: 12px/20px 'Helvetica Neue', Arial, Helvetica, sans-serif;
                z-index: 651;
            }
            .feature-row{
                margin-bottom: 0.5em;
                &:not(:last-of-type) {
                    border-bottom: 1px solid black;
                }
            }
            </style>
            {% endmacro %}
            {% macro script(this, kwargs) -%}
                var {{ this.get_name() }} = {{ this._parent.get_name() }}.getMaplibreMap();
                const popup_{{ this.get_name() }} = new maplibregl.Popup({
                    closeButton: false,
                    closeOnClick: false
                });

                function setTooltipForPMTilesMapLibreLayer_{{ this.get_name() }}(maplibreLayer) {
                    var mlMap = maplibreLayer.getMaplibreMap();
                    var popup = popup_{{ this.get_name() }};

                    mlMap.on('mousemove', (e) => {
                        mlMap.getCanvas().style.cursor = 'pointer';
                        const { x, y } = e.point;
                        const r = 2; // radius around the point
                        const features = mlMap.queryRenderedFeatures([
                            [x - r, y - r],
                            [x + r, y + r],
                        ]);

                        const {lng, lat}  = e.lngLat;
                        const coordinates = [lng, lat]
                        const html = features.map(f=>`
                        <div class="feature-row">
                            <span>
                                <strong>${f.layer['source-layer']}</strong>
                                <span style="fontSize: 0.8em" }> (${f.geometry.type})</span>
                            </span>
                            <table>
                                ${Object.entries(f.properties).map(([key, value]) => {
                                    let displayVal = value;
                                    if (typeof value === 'number') {
                                        displayVal = value.toFixed(2);
                                    } else if (typeof value === 'string' && !isNaN(value) && value.includes('.')) {
                                        displayVal = parseFloat(value).toFixed(2);
                                    }
                                    return `<tr><td>${key}</td><td style="text-align: right">${displayVal}</td></tr>`;
                                }).join("")}
                            </table>
                        </div>
                        `).join("")
                        if(features.length){
                            popup.setLngLat(e.lngLat).setHTML(html).addTo(mlMap);
                        } else {
                            popup.remove();
                        }
                    });
                    mlMap.on('mouseleave', () => {popup.remove();});
                }

                // maplibre map object
                {{ this.get_name() }}.on("load", (e) => {
                    setTooltipForPMTilesMapLibreLayer_{{ this.get_name() }}({{ this._parent.get_name() }});
                })

                // leaflet map object
                {{ this._parent._parent.get_name() }}.on("layeradd", (e) => {
                    setTooltipForPMTilesMapLibreLayer_{{ this.get_name() }}({{ this._parent.get_name() }});
                });
            {%- endmacro %}
            """
    )

    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self._name = name if name else "PMTilesTooltip"


def build_pmtiles_map(
    pmtiles_url: str,
    center: tuple[float, float] = (70.0, -140.0),
    zoom_start: int = 4,
    source_layer: str = "lakes",
    drained_ids: list[str] | None = None,
    viz_configuration_name: str = "colored_historical",
    tooltip=None,
) -> folium.Map:
    """Return a Folium map with a PMTiles vector layer for lake polygons."""
    m = leafmap.Map(
        location=center,
        zoom_start=zoom_start,  # lightweight basemap
    )
    print("running render pmtiles")
    # Add background map types
    wms_url = "https://maps.awi.de/services/common/permafrost/ows"
    tcvis_tile_layer = folium.WmsTileLayer(
        url=wms_url,
        name="TCVIS Landsat Trends 2005-2024 (AWI)",
        styles="composite",
        transparent=True,
        overlay=False,
        layers="tcvis",
    )
    tile_layer_darkmatter = folium.TileLayer("CartoDB.DarkMatter", name="Dark Matter (CartoDB)")
    tile_layer_esriworld = folium.TileLayer("Esri.WorldImagery", name="ESRI World Imagery")

    # # Add background tiles
    # tile_layer_darkmatter.add_to(m)
    # tile_layer_esriworld.add_to(m)
    # tcvis_tile_layer.add_to(m)

    tooltip = PMTilesMapLibreTooltipWithRounding()
    if viz_configuration_name == "colored_historical":
        fill_color, fill_opacity, line_color, line_width, line_opacity = get_style_pmtiles_colored_historical()
        legend = get_legend_html_net_change()
        # Use only one basemap to avoid overlap
        tile_layer_darkmatter.add_to(m)
        tile_layer_esriworld.add_to(m)
        tcvis_tile_layer.add_to(m)

    elif viz_configuration_name == "drainage_year":
        # Convert to number to handle string values in PMTiles
        fill_color, fill_opacity, line_color, line_width, line_opacity = get_style_pmtiles_drainage_year()
        legend = get_legend_html_date_drainage_year()

        # Use only one basemap to avoid overlap
        tile_layer_darkmatter.add_to(m)
        tcvis_tile_layer.add_to(m)
        tile_layer_esriworld.add_to(m)

    else:
        # Define default paint values
        fill_color, fill_opacity, line_color, line_width, line_opacity = get_style_pmtiles_colored_historical()
        legend = get_legend_html_net_change()
        # Add background tiles
        tile_layer_darkmatter.add_to(m)
        tcvis_tile_layer.add_to(m)
        tile_layer_esriworld.add_to(m)

    if drained_ids:
        # Highlight drained lakes in red, dim others
        fill_color = [
            "match",
            ["get", "id_geohash"],
            drained_ids,
            "#d73027",  # Red fill for drained
            fill_color,  # Default color ramp for non-drained
        ]
        fill_opacity = [
            "match",
            ["get", "id_geohash"],
            drained_ids,
            0.9,  # High opacity for drained
            0.3,  # Dimmer opacity for non-drained
        ]
        line_color = [
            "match",
            ["get", "id_geohash"],
            drained_ids,
            "#7f0000",  # Dark red border for drained
            "#333333",  # Default border color
        ]
        line_width = [
            "match",
            ["get", "id_geohash"],
            drained_ids,
            2.0,  # Thicker border for drained
            0.5,  # Default border width
        ]

    lake_layer = PMTilesMapLibreLayer(
        pmtiles_url,
        "Lakes",
        overlay=True,
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
                        "fill-color": fill_color,
                        "fill-opacity": fill_opacity,
                    },
                },
                {
                    "id": "lakes-line",
                    "source": "lakes_pmtiles",
                    "source-layer": source_layer,
                    "type": "line",
                    "paint": {
                        "line-color": line_color,
                        "line-width": line_width,
                        "line-opacity": line_opacity,
                    },
                },
            ],
        },
        tooltip=tooltip,
    )
    m.add_child(lake_layer)

    if drained_ids:
        drained_markers = folium.FeatureGroup(name="Drained Lake Markers")
        for gid in drained_ids:
            # Decode the geohash into latitude and longitude coordinates
            lat, lon = pygeohash.decode(gid)
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color="red", icon="info-sign"),
                tooltip=f"Drained Lake: {gid}",
            ).add_to(drained_markers)
        drained_markers.add_to(m)
    # -----------------------------------------------

    folium.LayerControl().add_to(m)

    m.get_root().html.add_child(folium.Element(legend))
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
