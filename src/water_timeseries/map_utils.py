import functools
import os
from pathlib import Path

import branca.element
import folium
import folium.elements
import leafmap.foliumap as leafmap
import pygeohash
from branca.element import Element  # <--- Added this import
from folium_pmtiles.vector import PMTilesMapLibreLayer

from water_timeseries.utils.map_styles.pmtiles import (
    get_style_pmtiles_colored_historical,
    get_style_pmtiles_drainage_year,
    get_style_pmtiles_generic_water,
    get_style_pmtiles_nrt_base_lakes,
    get_style_pmtiles_nrt_confidence_featurestate,
    get_style_pmtiles_nrt_drainage,
    get_style_pmtiles_nrt_monthly_tiles,
)
from water_timeseries.utils.pmtiles_build import NRT_POINT_POLY_SWITCH_ZOOM
from water_timeseries.utils.visualization import (
    get_legend_html_date_drainage_year,
    get_legend_html_net_change,
    get_legend_html_nrt_drainage,
    get_legend_html_nrt_drainage_magnitude,
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
    color: #222;
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
    var propertyOverrides_{{ this.get_name() }} = {{ this.property_overrides_json }};
    var filterLayers_{{ this.get_name() }} = {{ this.filter_layers_json }};
    var suppressedProperties_{{ this.get_name() }} = {{ this.suppressed_properties_json }};
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
    // filterLayers doubles as a priority list: show only the first listed
    // layer that has a hit, so a lake covered by an overlay renders the
    // overlay's values instead of two stacked tables.
    for (const layerId of filterLayers) {
    const preferred = features.filter(f => f.layer.id === layerId);
    if (preferred.length) { features = preferred; break; }
    }
    }
    const {lng, lat}  = e.lngLat;
    const coordinates = [lng, lat]
    const aliases = columnAliases_{{ this.get_name() }};
    const overrides = propertyOverrides_{{ this.get_name() }};
    const suppressed = suppressedProperties_{{ this.get_name() }};
    const html = features.map(f=>{
    const props = Object.assign({}, f.properties, overrides[f.properties["id_geohash"]] || {});
    for (const key of (suppressed[f.layer.id] || [])) { delete props[key]; }
    return `
    <div class="feature-row">
    <table>
    ${Object.entries(props).map(([key, value]) => {
    // Tilesets bake missing values as placeholder strings ("NaT" for null
    // dates, "nan" for null floats); showing those as data is worse than
    // omitting the row.
    if (value === null || value === undefined) { return ""; }
    const asText = String(value).trim();
    if (asText === "" || ["nat", "nan", "none", "null"].includes(asText.toLowerCase())) { return ""; }
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
    `;
    }).join("")
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

    def __init__(
        self,
        name=None,
        column_aliases=None,
        filter_layers=None,
        min_zoom=None,
        max_zoom=None,
        property_overrides=None,
        suppressed_properties=None,
        **kwargs,
    ):
        # Pop custom kwargs before passing to parent
        kwargs.pop("column_aliases", None)
        kwargs.pop("filter_layers", None)
        kwargs.pop("min_zoom", None)
        kwargs.pop("max_zoom", None)
        super().__init__(**kwargs)
        self._name = name if name else "PMTilesTooltip"
        self.column_aliases = column_aliases if column_aliases else {}
        # Ordered by priority: only the first layer with a hit under the cursor
        # contributes to the popup.
        self.filter_layers = filter_layers if filter_layers else []
        # {layer_id: [property, ...]} to drop from the popup for that layer --
        # for properties a tileset bakes that would be stale or misleading in
        # the current view.
        self.suppressed_properties = suppressed_properties if suppressed_properties else {}
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        # Per-feature tooltip content overrides, keyed by id_geohash: the
        # matching feature's tile-baked properties are merged with (and
        # superseded by) the given key/value pairs at hover time.
        self.property_overrides = property_overrides if property_overrides else {}

    @property
    def property_overrides_json(self):
        import json

        return json.dumps(self.property_overrides)

    @property
    def suppressed_properties_json(self):
        import json

        return json.dumps(self.suppressed_properties)

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


class PMTilesMapLibreFeatureState(branca.element.MacroElement):
    """Push per-feature ``feature-state`` values into a PMTiles MapLibre layer.

    Must be added as a child of a ``PMTilesMapLibreLayer`` so that
    ``this._parent`` resolves to the layer (same wiring as
    ``PMTilesMapLibreTooltipWithRounding``), and the layer's source must
    declare ``promoteId`` so features have a stable string id to address.

    ``state_by_id`` maps a feature id (``id_geohash``) to the state object to
    set for it, e.g. ``{"b7g0abc12345": {"confidence": 2}}``. Any state left
    over from a previous push is cleared first, so features absent from the
    current mapping fall back to the paint expression's no-state default.
    """

    _template = branca.element.Template(
        """
    {% macro script(this, kwargs) -%}
    var stateById_{{ this.get_name() }} = {{ this.state_by_id_json }};
    function applyFeatureState_{{ this.get_name() }}(maplibreLayer) {
    var mlMap = maplibreLayer.getMaplibreMap();
    function pushState() {
    mlMap.removeFeatureState({source: {{ this.source_json }}, sourceLayer: {{ this.source_layer_json }}});
    for (const [id, state] of Object.entries(stateById_{{ this.get_name() }})) {
    mlMap.setFeatureState(
    {source: {{ this.source_json }}, sourceLayer: {{ this.source_layer_json }}, id: id},
    state
    );
    }
    }
    if (mlMap.isStyleLoaded()) { pushState(); } else { mlMap.on("load", pushState); }
    }
    // maplibre map object
    applyFeatureState_{{ this.get_name() }}({{ this._parent.get_name() }});
    // leaflet map object
    {{ this._parent._parent.get_name() }}.on("layeradd", (e) => {
    applyFeatureState_{{ this.get_name() }}({{ this._parent.get_name() }});
    });
    {%- endmacro %}
    """
    )

    def __init__(self, state_by_id, source="lakes_pmtiles", source_layer="lakes", name=None, **kwargs):
        super().__init__(**kwargs)
        self._name = name if name else "PMTilesFeatureState"
        self.state_by_id = state_by_id if state_by_id else {}
        self.source = source
        self.source_layer = source_layer

    @property
    def state_by_id_json(self):
        import json

        return json.dumps(self.state_by_id)

    @property
    def source_json(self):
        import json

        return json.dumps(self.source)

    @property
    def source_layer_json(self):
        import json

        return json.dumps(self.source_layer)


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
    hidden_categories: frozenset[str] = frozenset(),
    nrt_confidence_by_id: dict[str, int | None] | None = None,
    nrt_tooltip_overrides: dict[str, dict] | None = None,
    nrt_magnitude_by_id: dict[str, float] | None = None,
    nrt_monthly_tiles_url: str | None = None,
    nrt_month_has_confidence: bool = True,
) -> folium.Map:
    """Return a Folium map with a PMTiles vector layer for lake polygons.

    When ``nrt_confidence_by_id`` is given (``nrt_drainage`` viz only), the
    lakes are colored by that per-month mapping of ``id_geohash`` to drainage
    confidence (1-3, or ``None`` for "drained, confidence unknown") instead of
    the tile-baked ``drainage_confidence`` property. The values are delivered
    via MapLibre ``feature-state``, so the tiles and paint expressions stay
    static across month switches. ``nrt_tooltip_overrides`` optionally swaps
    hovered features' tooltip content by ``id_geohash`` to match.

    ``nrt_magnitude_by_id`` carries ``water_change_perc`` (relative water
    loss, negative) for unknown-confidence lakes: those render as a red
    intensity gradient by loss magnitude instead of flat red, and the map
    legend switches to the gradient variant.

    ``nrt_monthly_tiles_url`` is the preferred path for the monthly overlay:
    given a per-month drained-lakes tileset (built by
    ``build_pmtiles_nrt_monthly``), the month's confidence is read from baked
    tile properties, so none of the ``nrt_*_by_id`` dicts are needed and
    nothing per-lake is serialized into the page. The dict-based arguments
    above are the fallback for deployments without those tilesets.
    """
    m = leafmap.Map(
        location=center,
        zoom_start=zoom_start,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
    )

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
        fill_color, fill_opacity, line_color, line_width, line_opacity = get_style_pmtiles_drainage_year(
            hide_stable_lakes=hide_stable_lakes
        )
        legend = get_legend_html_date_drainage_year()

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
            "water_change_ha": "Change of water area [ha]",
            "water_change_perc": "Change of water area [%]",
            "pre_break_median": "Lake area before break [ha]",
            "post_break_median": "Lake area after break [ha]",
            # Carried by the base lake tiles, which non-drained lakes hover.
            "date_break": "Historical break date",
            "date_break_year": "Historical break year",
            "Area_start_ha": "Lake area year 2000 [ha]",
            "Area_end_ha": "Lake area year 2020 [ha]",
            "NetChange_perc": "Net change [%]",
            "NetChange_ha": "Net change [ha]",
            # Baked into the per-month drainage tilesets: the NRT months carry
            # these rather than the "_absolute" variants above.
            "analysis_month": "Analysis month [YYYY-MM]",
            "water_observed": "Observed water area",
            "water_predicted": "Predicted water area",
            "water_residual": "Difference of lake area from prediction",
            "water_predicted_lower_90": "Predicted water area, lower 90%",
            "water_predicted_upper_90": "Predicted water area, upper 90%",
        }
        if nrt_monthly_tiles_url:
            # Prefer the overlay, fall back to the base lakes: a drained lake
            # hovers the selected month's values, any other lake still hovers
            # whatever the base tiles carry. `date`/`drainage_confidence` are
            # suppressed for the base layer because some base tilesets bake
            # them from a single NRT run, which would report the wrong month.
            tooltip = PMTilesMapLibreTooltipWithRounding(
                column_aliases=aliases,
                filter_layers=["nrt-drained-fill", "lakes-fill"],
                suppressed_properties={"lakes-fill": ["date", "drainage_confidence"]},
                min_zoom=8,
                max_zoom=14,
            )
            # The month's drained lakes come from their own tileset (added
            # below); the base tiles are the backdrop of all other lakes.
            fill_color, fill_opacity, line_color, line_width, line_opacity = get_style_pmtiles_nrt_base_lakes()
            has_magnitude_legend = not nrt_month_has_confidence
        else:
            tooltip = PMTilesMapLibreTooltipWithRounding(
                column_aliases=aliases,
                filter_layers=["lakes-fill"],
                min_zoom=8,
                max_zoom=14,
                property_overrides=nrt_tooltip_overrides,
            )
            if nrt_confidence_by_id is not None:
                # Monthly overlay: static feature-state paint, values pushed at
                # runtime by PMTilesMapLibreFeatureState (added below).
                fill_color, fill_opacity, line_color, line_width, line_opacity = (
                    get_style_pmtiles_nrt_confidence_featurestate(hidden_categories=hidden_categories)
                )
            else:
                # Convert to number to handle string values in PMTiles
                fill_color, fill_opacity, line_color, line_width, line_opacity = get_style_pmtiles_nrt_drainage(
                    hidden_categories=hidden_categories
                )
            has_magnitude_legend = bool(nrt_magnitude_by_id)
        legend = get_legend_html_nrt_drainage_magnitude() if has_magnitude_legend else get_legend_html_nrt_drainage()

        tile_layer_darkmatter.add_to(m)
        tcvis_tile_layer.add_to(m)
        tile_layer_esriworld.add_to(m)

    else:
        tooltip = PMTilesMapLibreTooltipWithRounding(filter_layers=["lakes-fill"])
        fill_color, fill_opacity, line_color, line_width, line_opacity = get_style_pmtiles_generic_water()
        legend = None
        tile_layer_darkmatter.add_to(m)
        tcvis_tile_layer.add_to(m)
        tile_layer_esriworld.add_to(m)

    if drained_ids:
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

    if viz_configuration_name == "drainage_year" and hide_stable_lakes:
        nan_filter = [
            "all",
            ["!=", ["get", "date_break_year"], None],
            ["!=", ["to-string", ["get", "date_break_year"]], "NaN"],
            ["!=", ["to-string", ["get", "date_break_year"]], ""],
        ]
        lakes_fill_layer["filter"] = nan_filter
        lakes_line_layer["filter"] = nan_filter

    sources = {
        "lakes_pmtiles": {
            "type": "vector",
            "url": "pmtiles://" + pmtiles_url,
            # Stable per-feature identity for setFeatureState, consistent
            # across zoom levels/tiles.
            "promoteId": "id_geohash",
        }
    }
    layers = [lakes_fill_layer, lakes_line_layer]

    if nrt_monthly_tiles_url:
        drained_fill, drained_opacity, drained_line, drained_width, drained_line_opacity = (
            get_style_pmtiles_nrt_monthly_tiles()
        )
        sources["nrt_pmtiles"] = {
            "type": "vector",
            "url": "pmtiles://" + nrt_monthly_tiles_url,
        }
        # Base-lake centroids below z6, where the base tileset has no polygons
        # (it bakes `lakes` at z6-14 and `lakes_points` at z0-5). Without this
        # the zoomed-out view would show only the drained lakes, which do have
        # a low-zoom layer. Drawn first so drained lakes stay on top.
        base_points_layer = {
            "id": "lakes-points",
            "source": "lakes_pmtiles",
            "source-layer": "lakes_points",
            "type": "circle",
            "maxzoom": 6,
            "paint": {
                "circle-color": fill_color,
                "circle-opacity": fill_opacity,
                "circle-radius": ["interpolate", ["linear"], ["zoom"], 0, 0.6, 5, 2.5],
            },
        }
        layers.insert(0, base_points_layer)

        if hide_stable_lakes or "stable" in hidden_categories:
            # Switch the base layers off rather than painting them at zero
            # opacity: a zero-opacity layer still counts as rendered, so
            # queryRenderedFeatures would keep producing hover popups for the
            # lakes the user asked to hide. The drained overlay is unaffected.
            for layer in (base_points_layer, lakes_fill_layer, lakes_line_layer):
                layer["layout"] = {"visibility": "none"}

        nrt_drained_points_layer = {
            # Centroids below the switch zoom, where the polygons are sub-pixel:
            # this is what keeps drained lakes findable when zoomed out, without
            # per-lake browser markers. maxzoom is exclusive and the polygon
            # layers' minzoom is inclusive, so both using the same value hands
            # off in one step with no zoom left uncovered.
            "id": "nrt-drained-points",
            "source": "nrt_pmtiles",
            "source-layer": "drained_points",
            "type": "circle",
            "maxzoom": NRT_POINT_POLY_SWITCH_ZOOM,
            "paint": {
                "circle-color": drained_fill,
                "circle-opacity": drained_opacity,
                "circle-radius": ["interpolate", ["linear"], ["zoom"], 0, 1.5, 5, 4],
                "circle-stroke-color": drained_line,
                "circle-stroke-width": 0.5,
            },
        }
        nrt_drained_fill_layer = {
            "id": "nrt-drained-fill",
            "source": "nrt_pmtiles",
            "source-layer": "drained",
            "type": "fill",
            "minzoom": NRT_POINT_POLY_SWITCH_ZOOM,
            "paint": {"fill-color": drained_fill, "fill-opacity": drained_opacity},
        }
        nrt_drained_line_layer = {
            "id": "nrt-drained-line",
            "source": "nrt_pmtiles",
            "source-layer": "drained",
            "type": "line",
            "minzoom": NRT_POINT_POLY_SWITCH_ZOOM,
            "paint": {
                "line-color": drained_line,
                "line-width": drained_width,
                "line-opacity": drained_line_opacity,
            },
        }

        # drainage_confidence is absent (coalesced to -1) for drained lakes
        # with no ARIMA confidence score -- that's the "no data" category.
        hidden_confidence_values = {
            "no_data": -1,
            "low": 1,
            "medium": 2,
            "high": 3,
        }
        hidden_values = [v for cat, v in hidden_confidence_values.items() if cat in hidden_categories]
        if hidden_values:
            drained_filter = ["!", ["in", ["coalesce", ["get", "drainage_confidence"], -1], ["literal", hidden_values]]]
            for layer in (nrt_drained_points_layer, nrt_drained_fill_layer, nrt_drained_line_layer):
                layer["filter"] = drained_filter

        layers.extend([nrt_drained_points_layer, nrt_drained_fill_layer, nrt_drained_line_layer])

    lake_layer = PMTilesMapLibreLayer(
        pmtiles_url,
        "Lakes",
        overlay=True,
        style={"version": 8, "sources": sources, "layers": layers},
        tooltip=tooltip,
    )

    # --- FIXED LINE BELOW ---
    lake_layer.add_to(m)
    # ------------------------

    if viz_configuration_name == "nrt_drainage" and not drained_ids and nrt_confidence_by_id is not None:
        # None means "drained, confidence unknown" — encoded as 0 (rendered as
        # a red gradient by water_change_perc when available, else mid red).
        state_by_id = {
            gid: {"confidence": 0 if conf is None else int(conf)} for gid, conf in nrt_confidence_by_id.items()
        }
        for gid, perc in (nrt_magnitude_by_id or {}).items():
            if gid in state_by_id:
                state_by_id[gid]["water_change_perc"] = perc
        feature_state = PMTilesMapLibreFeatureState(
            state_by_id=state_by_id,
            source="lakes_pmtiles",
            source_layer=source_layer,
        )
        feature_state.add_to(lake_layer)

    if drained_ids:
        drained_markers = folium.FeatureGroup(name="Drained Lake Markers", control=True)
        for gid in drained_ids:
            lat, lon = pygeohash.decode(gid)
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
            marker.add_to(drained_markers)

        drained_markers.add_to(m)
        ul, lr = drained_markers.get_bounds()
        print(ul, lr)

    folium.LayerControl().add_to(m)

    # Injecting the legend
    if legend is not None:
        m.get_root().html.add_child(folium.Element(legend))

    # --- UPDATED CSS INJECTION FOR ATTRIBUTION BAR ---
    style = """
    <style>
        .leaflet-control-attribution {
            font-size: 10px !important;
            opacity: 0.6;
            padding: 3px !important;
            
            /* Positioning logic */
            right: 0 !important;         /* Anchor to the right edge */
            left: auto !important;        /* Ensure it's not stretched left */
            width: 70% !important;       /* Occupy only 70% of the width */
            text-align: right !important; /* Align text inside that 70% block to the right */
        }
    </style>
    """
    m.get_root().html.add_child(Element(style))

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

    from water_timeseries.utils.pmtiles_serve import PmtilesServer  # noqa: F401

    pmtiles_path = Path(pmtiles_file).resolve()
    if not pmtiles_path.is_file():
        raise FileNotFoundError(f"PMTiles file not found: {pmtiles_path}")

    return _get_pmtiles_server(str(pmtiles_path)).url_for(pmtiles_path.name)


def resolve_nrt_monthly_tiles_url(location: str | Path | None, month: str) -> str | None:
    """Return a browser-fetchable URL for ``month``'s drainage tileset, or None.

    ``location`` is where ``build_pmtiles_nrt_monthly`` wrote its output: a
    local directory, an ``http(s)://`` prefix, or a ``gs://`` prefix. Returns
    None when no tileset exists for the month, which is the signal for callers
    to fall back to the runtime feature-state path.

    For a local directory, the tiles are served by a background
    ``PmtilesServer`` rooted at that directory — unless ``PMTILES_BASE_URL`` is
    set, in which case the monthly tilesets are expected to be reachable by
    filename under that base URL (same convention as ``resolve_pmtiles_url``).
    """
    if not location:
        return None

    from water_timeseries.utils.pmtiles_build import nrt_monthly_tiles_filename

    filename = nrt_monthly_tiles_filename(month)
    location_str = str(location)

    if location_str.startswith(("http://", "https://")):
        return f"{location_str.rstrip('/')}/{filename}"
    if location_str.startswith("gs://"):
        return f"https://storage.googleapis.com/{location_str[5:].strip('/')}/{filename}"

    tiles_dir = Path(location_str)
    if not (tiles_dir / filename).is_file():
        return None

    base_url = os.environ.get("PMTILES_BASE_URL")
    if base_url:
        return f"{base_url.rstrip('/')}/{filename}"

    return _get_pmtiles_server(str(tiles_dir.resolve())).url_for(filename)


@functools.cache
def _get_pmtiles_server(path_str: str):
    from water_timeseries.utils.pmtiles_serve import PmtilesServer

    return PmtilesServer(Path(path_str)).start()


def geohash_to_human_readable_name(geohash: str) -> str:
    """Convert a geohash to a human-readable name."""
    lat, lon = pygeohash.decode(geohash)
    return f"{geohash} | {lat:.3f} : {lon:.3f}"


def human_readable_name_to_geohash(human_readable_name: str) -> str:
    """Convert a human-readable name to a geohash."""
    # Extract the geohash from the human-readable name
    geohash = human_readable_name.split(" | ")[0]
    return geohash
