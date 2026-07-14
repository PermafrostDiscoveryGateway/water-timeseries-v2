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
    get_style_pmtiles_generic_water,
    get_style_pmtiles_nrt_drainage,
)
from water_timeseries.utils.visualization import (
    get_legend_html_date_drainage_year,
    get_legend_html_net_change,
    get_legend_html_nrt_drainage,
)


class PMTilesMapLibreTooltipWithRounding(folium.elements.JSCSSMixin, branca.element.MacroElement):
    _template = branca.element.Template(
        """
    {% macro header(this, kwargs) %}
    <style>
    .maplibregl-popup {
    font: 11px/16px 'Helvetica Neue', Arial, Helvetica, sans-serif;
    z-index: 651;
    border: 1px solid #ddd;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .maplibregl-popup .maplibregl-popup-content {
    padding: 0;
    margin: 0;
    background: transparent;
    overflow: visible;
    }
    .maplibregl-popup .maplibregl-popup-tip {
    display: none;
    }
    .lakes-tooltip {
    max-width: none;
    }
    .feature-row{
    background: white;
    border-radius: 6px;
    overflow: hidden;
    }
    .feature-row table {
    border-collapse: collapse;
    width: 100%;
    }
    .feature-row table tr:nth-child(even) {
    background-color: #f8f8f8;
    }
    .feature-row table tr:last-child td {
    border-bottom: none;
    }
    .feature-row table td {
    padding: 4px 8px;
    }
    .feature-row table td:first-child {
    font-weight: 500;
    color: #555;
    }
    .feature-row table td:last-child {
    text-align: right;
    }
    </style>
    {% endmacro %}
    {% macro script(this, kwargs) -%}
    var {{ this.get_name() }} = {{ this._parent.get_name() }}.getMaplibreMap();
    const popup_{{ this.get_name() }} = new maplibregl.Popup({
    closeButton: false,
    closeOnClick: false,
    offset: 20,
    autoPan: true,
    autoPanPadding: [50, 50]
    });
    var columnAliases_{{ this.get_name() }} = {{ this.column_aliases_json }};
    var filterLayers_{{ this.get_name() }} = {{ this.filter_layers_json }};
    var minZoom_{{ this.get_name() }} = {{ this.min_zoom_json }};
    var maxZoom_{{ this.get_name() }} = {{ this.max_zoom_json }};
    function setTooltipForPMTilesMapLibreLayer_{{ this.get_name() }}(maplibreLayer) {
    var mlMap = maplibreLayer.getMaplibreMap();
    var popup = popup_{{ this.get_name() }};
    mlMap.on('mousemove', (e) => {
    var zoom = mlMap.getZoom();
    if (minZoom_{{ this.get_name() }} !== null && zoom < minZoom_{{ this.get_name() }}) { popup.remove(); return; }
    if (maxZoom_{{ this.get_name() }} !== null && zoom > maxZoom_{{ this.get_name() }}) { popup.remove(); return; }
    mlMap.getCanvas().style.cursor = 'pointer';
    const { x, y } = e.point;
    const r = 2; // radius around the point
    var features = mlMap.queryRenderedFeatures([
    [x - r, y - r],
    [x + r, y + r],
    ]);
    // Filter by layer if filterLayers is set
    var filterLayers = filterLayers_{{ this.get_name() }};
    if (filterLayers && filterLayers.length > 0) {
    features = features.filter(f => filterLayers.includes(f.layer.id));
    }
    const {lng, lat}  = e.lngLat;
    const coordinates = [lng, lat]
    const aliases = columnAliases_{{ this.get_name() }};
    const html = features.map(f=>`
    <div class="feature-row">
    <table>
    ${Object.entries(f.properties).map(([key, value]) => {
    let displayKey = aliases[key] || key;
    let displayVal = value;
    if (typeof value === 'number') {
    displayVal = value.toFixed(2);
    } else if (typeof value === 'string' && !isNaN(value) && value.includes('.')) {
    displayVal = parseFloat(value).toFixed(2);
    }
    return `<tr><td>${displayKey}</td><td style="text-align: right">${displayVal}</td></tr>`;
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

    def __init__(self, name=None, column_aliases=None, filter_layers=None, min_zoom=None, max_zoom=None, **kwargs):
        # Pop custom kwargs before passing to parent
        kwargs.pop("column_aliases", None)
        kwargs.pop("filter_layers", None)
        kwargs.pop("min_zoom", None)
        kwargs.pop("max_zoom", None)
        super().__init__(**kwargs)
        self._name = name if name else "PMTilesTooltip"
        self.column_aliases = column_aliases if column_aliases else {}
        self.filter_layers = filter_layers if filter_layers else []
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

    @property
    def column_aliases_json(self):
        import json

        return json.dumps(self.column_aliases)

    @property
    def filter_layers_json(self):
        import json

        return json.dumps(self.filter_layers)

    @property
    def min_zoom_json(self):
        import json

        return json.dumps(self.min_zoom)

    @property
    def max_zoom_json(self):
        import json

        return json.dumps(self.max_zoom)


def build_pmtiles_map(
    pmtiles_url: str,
    center: tuple[float, float] = (70.0, -140.0),
    zoom_start: int = 4,
    source_layer: str = "lakes",
    drained_ids: list[str] | None = None,
    viz_configuration_name: str = "colored_historical",
    tooltip=None,
    min_zoom=4,
    max_zoom=15,
    hide_stable_lakes: bool = False,
) -> folium.Map:
    """Return a Folium map with a PMTiles vector layer for lake polygons."""

    m = leafmap.Map(
        location=center,
        zoom_start=zoom_start,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
    )
    # m.clear_layers()
    # logger.info("running render pmtiles")
    # Add background map types
    wms_url = "https://maps.awi.de/services/common/permafrost/ows"
    tcvis_tile_layer = folium.WmsTileLayer(
        url=wms_url,
        name="TCVIS Landsat Trends 2005-2024 (AWI)",
        styles="composite",
        transparent=True,
        overlay=False,
        layers="tcvis",
        min_zoom=min_zoom,
        max_zoom=max_zoom,
    )
    tile_layer_darkmatter = folium.TileLayer(
        "CartoDB.DarkMatter", name="Dark Matter (CartoDB)", min_zoom=min_zoom, max_zoom=max_zoom
    )
    tile_layer_esriworld = folium.TileLayer(
        "Esri.WorldImagery", name="ESRI World Imagery", min_zoom=min_zoom, max_zoom=max_zoom
    )

    if viz_configuration_name == "colored_historical" and not drained_ids:
        aliases = {
            "NetChange_perc": "Net Change (%)",
            "NetChange_ha": "Net Change (ha)",
            "Area_start_ha": "Lake Area year 2000 (ha)",
            "Area_end_ha": "Lake Area year 2020 (ha)",
            "date_break_year": "Drainage Year",
        }
        tooltip = PMTilesMapLibreTooltipWithRounding(
            column_aliases=aliases, filter_layers=["lakes-fill"], min_zoom=8, max_zoom=14
        )
        fill_color, fill_opacity, line_color, line_width, line_opacity = get_style_pmtiles_colored_historical()
        legend = get_legend_html_net_change()
        # Use only one basemap to avoid overlap
        tile_layer_darkmatter.add_to(m)
        tile_layer_esriworld.add_to(m)
        tcvis_tile_layer.add_to(m)

    elif viz_configuration_name == "drainage_year" and not drained_ids:
        aliases = {
            "id_geohash": "Lake ID",
            "date_break": "Break date [YYYY-MM]",
            "date_break_year": "Year of change",
            "pre_break_median": "Lake area before break [ha]",
            "post_break_median": "Lake area after break [ha]",
            "water_change_ha": "Change of water area [ha]",
            "water_change_perc": "Change of water area [%]",
        }
        tooltip = PMTilesMapLibreTooltipWithRounding(
            column_aliases=aliases, filter_layers=["lakes-fill"], min_zoom=8, max_zoom=14
        )
        # Convert to number to handle string values in PMTiles
        fill_color, fill_opacity, line_color, line_width, line_opacity = get_style_pmtiles_drainage_year(
            hide_stable_lakes=hide_stable_lakes
        )
        legend = get_legend_html_date_drainage_year()

        # Use only one basemap to avoid overlap
        tile_layer_darkmatter.add_to(m)
        tcvis_tile_layer.add_to(m)
        tile_layer_esriworld.add_to(m)

    elif viz_configuration_name == "nrt_drainage" and not drained_ids:
        aliases = {
            "id_geohash": "Lake ID",
            "date": "Analysis date [YYYY-MM]",
            "water_observed_absolute": "Observed water area [ha]",
            "water_predicted_absolute": "Predicted water area [ha]",
            "water_predicted_ci_absolute": "Predicted water area range [ha]",
            "water_residual_absolute": "Difference of lake area from prediction [ha]",
            "drainage_confidence": "Confidence of drainage detection [0 (low) to 3 (high)]",
        }
        tooltip = PMTilesMapLibreTooltipWithRounding(
            column_aliases=aliases, filter_layers=["lakes-fill"], min_zoom=8, max_zoom=14
        )
        # Convert to number to handle string values in PMTiles
        fill_color, fill_opacity, line_color, line_width, line_opacity = get_style_pmtiles_nrt_drainage(
            hide_stable_lakes=hide_stable_lakes
        )
        legend = get_legend_html_nrt_drainage()

        # Use only one basemap to avoid overlap
        tile_layer_darkmatter.add_to(m)
        tcvis_tile_layer.add_to(m)
        tile_layer_esriworld.add_to(m)

    else:
        tooltip = PMTilesMapLibreTooltipWithRounding(filter_layers=["lakes-fill"])
        # Define default paint values
        fill_color, fill_opacity, line_color, line_width, line_opacity = get_style_pmtiles_generic_water()
        # legend = get_legend_html_net_change()
        legend = None
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
            "#ADD8E6",  # Default color ramp for non-drained
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
            "#eeeeee",  # Default border color
        ]
        line_width = [
            "match",
            ["get", "id_geohash"],
            drained_ids,
            2.0,  # Thicker border for drained
            0.5,  # Default border width
        ]

    # Build layer definitions
    lakes_fill_layer = {
        "id": "lakes-fill",
        "source": "lakes_pmtiles",
        "source-layer": source_layer,
        "type": "fill",
        "paint": {
            "fill-color": fill_color,
            "fill-opacity": fill_opacity,
        },
    }
    lakes_line_layer = {
        "id": "lakes-line",
        "source": "lakes_pmtiles",
        "source-layer": source_layer,
        "type": "line",
        "paint": {
            "line-color": line_color,
            "line-width": line_width,
            "line-opacity": line_opacity,
        },
    }

    # Apply filter for drainage_year viz to hide stable lakes
    if viz_configuration_name == "drainage_year" and hide_stable_lakes:
        nan_filter = [
            "all",
            ["!=", ["get", "date_break_year"], None],
            ["!=", ["to-string", ["get", "date_break_year"]], "NaN"],
            ["!=", ["to-string", ["get", "date_break_year"]], ""],
        ]
        lakes_fill_layer["filter"] = nan_filter
        lakes_line_layer["filter"] = nan_filter

    # setup PMTiles Layer
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
            "layers": [lakes_fill_layer, lakes_line_layer],
        },
        tooltip=tooltip,
    )
    m.add_child(lake_layer)

    if drained_ids:
        drained_markers = folium.FeatureGroup(name="Drained Lake Markers", control=True)
        for gid in drained_ids:
            # Decode the geohash into latitude and longitude coordinates
            lat, lon = pygeohash.decode(gid)
            # set marker
            marker = folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color="darkred",
                fill=True,
                fill_color="red",
                fill_opacity=0.6,
                border_width=0.5,
                icon=folium.Icon(color="red", icon="tint", prefix="fa"),
            )
            # add to group
            marker.add_to(drained_markers)

        # add marker to map
        drained_markers.add_to(m)
        # get bounds of layer
        ul, lr = drained_markers.get_bounds()
        print(ul, lr)
    # -----------------------------------------------

    # m.add_child(lake_layer)
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

    import os

    base_url = os.environ.get("PMTILES_BASE_URL")
    if base_url:
        return f"{base_url.rstrip('/')}/{Path(pmtiles_file).name}"

    import streamlit as st

    from water_timeseries.utils.pmtiles_serve import PmtilesServer

    pmtiles_path = Path(pmtiles_file).resolve()
    if not pmtiles_path.is_file():
        raise FileNotFoundError(f"PMTiles file not found: {pmtiles_path}")

    @st.cache_resource
    def _get_pmtiles_server(path_str: str):
        return PmtilesServer(Path(path_str)).start()

    server = _get_pmtiles_server(str(pmtiles_path))

    return server.url_for(pmtiles_path.name)


def geohash_to_human_readable_name(geohash: str) -> str:
    """Convert a geohash to a human-readable name."""
    lat, lon = pygeohash.decode(geohash)
    return f"{geohash} | {lat:.3f} : {lon:.3f}"


def human_readable_name_to_geohash(human_readable_name: str) -> str:
    """Convert a human-readable name to a geohash."""
    # Extract the geohash from the human-readable name
    geohash = human_readable_name.split(" | ")[0]
    return geohash
