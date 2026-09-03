import functools
import os
from pathlib import Path
from typing import ClassVar

import branca.element
import folium
import folium.elements
import leafmap.foliumap as leafmap
import pygeohash
from branca.element import Element  # <--- Added this import
from folium_pmtiles.vector import PMTilesMapLibreLayer

from water_timeseries.utils.map_styles.pmtiles import (
    DRAINED_LAKE_FILTER,
    STABLE_LAKE_FILTER,
    get_style_pmtiles_colored_historical,
    get_style_pmtiles_drainage_year,
    get_style_pmtiles_generic_water,
    get_style_pmtiles_nrt_confidence_featurestate,
    get_style_pmtiles_nrt_drainage,
    get_style_pmtiles_nrt_monthly_tiles,
    get_style_pmtiles_stable_lakes,
)
from water_timeseries.utils.pmtiles_build import (
    POINT_POLY_SWITCH_ZOOM,
    TILE_MAX_ZOOM,
    archive_bakes_low_zoom_centroids,
)
from water_timeseries.utils.visualization import (
    get_legend_html_date_drainage_year,
    get_legend_html_drained_month,
    get_legend_html_net_change,
    get_legend_html_nrt_drainage,
)

# Exponent the base centroids' opacity is raised to, relative to the fill opacity
# of the polygons they stand in for (see base_points_layer in build_pmtiles_map).
# Below 1 it brightens; the smaller it is the harder the faint end is lifted.
CENTROID_OPACITY_EXPONENT = 1 / 3


class PMTilesMapLibreLayerSynced(PMTilesMapLibreLayer):
    """PMTilesMapLibreLayer with a fix for the GL layer drifting away from the basemap.

    maplibre-gl-leaflet only syncs the GL canvas on throttled Leaflet ``move``
    events, so the final update of a drag (especially a fast one, or during
    inertia) can be lost and the polygons stay offset from the basemap until
    the next interaction. A single ``moveend`` resync is not reliable on its
    own: ``_update()`` mutates ``transform.center``/``transform.zoom``
    directly from whatever the map's current animation state is, so calling
    it once at the "wrong" instant (e.g. mid-inertia, before layout settles)
    can itself desync the GL layer. We therefore also register our own
    unthrottled ``move`` listener -- calling ``_update()`` on every move is
    cheap and means the transform is recomputed continuously through the
    drag rather than only once at the end, so a single bad sample doesn't
    stick. Also bumps maplibre-gl-leaflet 0.0.17 -> 0.0.22, which rounds
    fractional container positions (sub-pixel misalignment at certain
    window widths).

    Also disables MapLibre's own interaction handlers: the layer needs
    ``interactive: true`` for tooltips, but the library passes that flag on to
    the MapLibre constructor, which then drives the GL view in parallel with
    Leaflet -- most visibly at Leaflet's zoom limits, where the basemap stops
    but the polygons keep zooming.
    """

    _template = branca.element.Template(
        """
            {% macro script(this, kwargs) -%}
            if (!("pmtiles" in maplibregl.config.REGISTERED_PROTOCOLS)) {
                var protocol = new pmtiles.Protocol();
                maplibregl.addProtocol("pmtiles", protocol.tile);
            }

            // see: https://github.com/maplibre/maplibre-gl-leaflet/issues/19
            {{ this._parent.get_name() }}.createPane('overlay_{{ this.get_name() }}');
            {{ this._parent.get_name() }}.getPane('overlay_{{ this.get_name() }}').style.zIndex = 650;
            {{ this._parent.get_name() }}.getPane('overlay_{{ this.get_name() }}').style.pointerEvents = 'none';

            var {{ this.get_name() }} = L.maplibreGL({
                pane: 'overlay_{{ this.get_name() }}',
                style: {{ this.style|tojson}},
                interactive: true,
            }).addTo({{ this._parent.get_name() }});

            // The GL map is built with `interactive: true` so its canvas keeps
            // pointer events (the tooltip listens for mousemove/click on it),
            // but maplibre-gl-leaflet forwards that flag straight into the
            // MapLibre constructor, which also switches on MapLibre's *own*
            // scroll/drag/rotate handlers. Those then run alongside Leaflet's
            // on the very same gesture, using MapLibre's zoom range (-2..22)
            // instead of the basemap's. While Leaflet is still moving, the
            // resync below hides it -- but once Leaflet clamps at its min/max
            // zoom it stops emitting 'move', so the GL map goes on zooming by
            // itself and the polygons shrink away from the basemap for good.
            // Rotate/pitch never resync at all, since the library's
            // `_transformGL` only writes center and zoom. Leaflet owns the
            // view, so turn MapLibre's handlers off.
            (function (glMap) {
                ['scrollZoom', 'boxZoom', 'dragRotate', 'dragPan', 'keyboard',
                 'doubleClickZoom', 'touchZoomRotate', 'touchPitch'].forEach(function (handler) {
                    if (glMap[handler] && glMap[handler].disable) {
                        glMap[handler].disable();
                    }
                });
            })({{ this.get_name() }}.getMaplibreMap());

            // Resync on every 'move' (unthrottled, unlike the library's own
            // handler) plus once more after 'moveend' settles, so the GL
            // transform tracks the drag continuously instead of depending on
            // a single throttled or end-of-drag sample landing correctly.
            {{ this._parent.get_name() }}.on('move', function () {
                if ({{ this.get_name() }}._glMap) {
                    {{ this.get_name() }}._update();
                }
            });
            {{ this._parent.get_name() }}.on('moveend', function () {
                requestAnimationFrame(function () {
                    if ({{ this.get_name() }}._glMap) {
                        {{ this.get_name() }}._update();
                    }
                }.bind(this));
            });
            {%- endmacro %}
            """
    )

    default_js: ClassVar = [
        ("pmtiles", "https://unpkg.com/pmtiles@2.5.0/dist/index.js"),
        ("maplibre-lib", "https://unpkg.com/maplibre-gl@2.2.1/dist/maplibre-gl.js"),
        (
            "maplibre-leaflet",
            "https://unpkg.com/@maplibre/maplibre-gl-leaflet@0.0.22/leaflet-maplibre-gl.js",
        ),
    ]


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
    # One level of overzoom past the deepest baked tile: MapLibre scales the
    # TILE_MAX_ZOOM tile rather than dropping its features, so the extra level
    # costs nothing and the hover gate (below) follows this ceiling, not the
    # tileset's.
    max_zoom=TILE_MAX_ZOOM + 1,
    hide_stable_lakes: bool = False,
    drained_label: str | None = None,
    hidden_categories: frozenset[str] = frozenset(),
    nrt_confidence_by_id: dict[str, int | None] | None = None,
    nrt_tooltip_overrides: dict[str, dict] | None = None,
    nrt_monthly_tiles_url: str | None = None,
    historical_drained_tiles_url: str | None = None,
    nrt_month_has_confidence: bool = True,
    selected_id: str | None = None,
    id_column: str = "id_geohash",
    base_has_centroids: bool = True,
) -> folium.Map:
    """Return a Folium map with a PMTiles vector layer for lake polygons.

    When ``nrt_confidence_by_id`` is given (``nrt_drainage`` viz only), the
    lakes are colored by that per-month mapping of ``id_geohash`` to drainage
    confidence (1-3, or ``None`` for "drained, confidence unknown") instead of
    the tile-baked ``drainage_confidence`` property. The values are delivered
    via MapLibre ``feature-state``, so the tiles and paint expressions stay
    static across month switches. ``nrt_tooltip_overrides`` optionally swaps
    hovered features' tooltip content by ``id_geohash`` to match.

    ``nrt_monthly_tiles_url`` is the preferred path for the monthly overlay:
    given a per-month drained-lakes tileset (built by
    ``build_pmtiles_nrt_monthly``), the month's confidence is read from baked
    tile properties, so none of the ``nrt_*_by_id`` dicts are needed and
    nothing per-lake is serialized into the page. The dict-based arguments
    above are the fallback for deployments without those tilesets.

    Args:
        selected_id: Lake whose outline is highlighted on top of every other
            layer, so the current selection stays identifiable after the map
            re-centers on it.
        id_column: Tile property holding the lake id (matched against
            ``selected_id``).
        historical_drained_tiles_url: Optional drained-lakes tileset for
            ``drainage_year`` (built by ``build_pmtiles_historical_drained``).
            The base archive samples its 4M lakes to fit the tile budget, which
            takes drained lakes off the map when zoomed out; served from their
            own unlimited tileset instead they are all present at every zoom,
            over the base archive's sampled grey. Without it the drained lakes
            are filtered out of the base archive as before.
        base_has_centroids: Whether the base archive bakes usable centroids
            below ``POINT_POLY_SWITCH_ZOOM``, from
            ``archive_bakes_low_zoom_centroids``. When it does not -- every
            archive built before the per-feature zoom ranges, including the
            shipped pan-arctic one -- the dot/polygon handoff has nothing to
            hand off to, so the polygons are drawn at every zoom instead (which
            is what those archives bake) and the centroid layer is left out.
            Callers that know the archive should pass the detected value;
            ``map_viewer`` does.
    """
    # Set only by viz modes that separate stable lakes from drained ones (see the
    # drainage_year branch below); None means one set of layers covers both.
    stable_style: tuple | None = None

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
            column_aliases=aliases, filter_layers=["lakes-fill"], min_zoom=POINT_POLY_SWITCH_ZOOM, max_zoom=max_zoom
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
            column_aliases=aliases,
            filter_layers=["lakes-fill", "lakes-stable-fill"],
            min_zoom=POINT_POLY_SWITCH_ZOOM,
            max_zoom=max_zoom,
        )
        fill_color, fill_opacity, line_color, line_width, line_opacity = get_style_pmtiles_drainage_year()
        # Stable lakes are a separate, neutral-grey layer underneath rather than
        # a branch inside the paint above, so every drained lake draws over every
        # stable one instead of losing to whichever came later in the tile.
        stable_style = get_style_pmtiles_stable_lakes()
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
                min_zoom=POINT_POLY_SWITCH_ZOOM,
                max_zoom=max_zoom,
            )
            # The month's drained lakes come from their own tileset (added
            # below); the base tiles are the backdrop of all other lakes.
            fill_color, fill_opacity, line_color, line_width, line_opacity = get_style_pmtiles_stable_lakes()
        else:
            tooltip = PMTilesMapLibreTooltipWithRounding(
                column_aliases=aliases,
                filter_layers=["lakes-fill"],
                min_zoom=POINT_POLY_SWITCH_ZOOM,
                max_zoom=max_zoom,
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
        legend = get_legend_html_nrt_drainage()

        tile_layer_darkmatter.add_to(m)
        tcvis_tile_layer.add_to(m)
        tile_layer_esriworld.add_to(m)

    else:
        tooltip = PMTilesMapLibreTooltipWithRounding(
            filter_layers=["lakes-fill"], min_zoom=POINT_POLY_SWITCH_ZOOM, max_zoom=max_zoom
        )
        fill_color, fill_opacity, line_color, line_width, line_opacity = get_style_pmtiles_generic_water()
        legend = None
        tile_layer_darkmatter.add_to(m)
        tcvis_tile_layer.add_to(m)
        tile_layer_esriworld.add_to(m)

    # The drained set can run to tens of thousands of ids for a single month.
    # Styling it via per-property ["match", ..., drained_ids, ...] expressions
    # serialized the id list once per paint property (four copies, ~0.5 MB
    # each) into the map HTML. Instead the base layers get plain constant
    # paint for "not drained", and the drained lakes are painted by two
    # overlay layers whose *filter* carries the id list -- two copies, and
    # MapLibre applies it as a feature filter rather than a per-property
    # expression evaluated four times over.
    drained_overlay_layers: list[dict] = []
    if drained_ids:
        legend = get_legend_html_drained_month(drained_label)
        fill_color = "#ADD8E6"  # Default color ramp for non-drained
        fill_opacity = 0.3  # Dimmer opacity for non-drained
        line_color = "#eeeeee"  # Default border color
        line_width = 0.5  # Default border width

    # Below the switch the base lakes are drawn as centroids -- but only if the
    # archive baked any (see archive_bakes_low_zoom_centroids). When it did not,
    # gating the polygons here would leave those zooms blank, so they stay
    # ungated and cover the whole range on their own, as they did before the
    # handoff existed. Every base-source polygon layer takes the same gate.
    poly_gate = {"minzoom": POINT_POLY_SWITCH_ZOOM} if base_has_centroids else {}

    # Where the coloured layers read their drained lakes from. Their own tileset
    # when there is one -- every feature in it is drained, so it needs no filter
    # and, being small enough to build with no tile budget, it never has lakes
    # sampled out from under it. Otherwise they come out of the base archive
    # behind a filter, which is correct but shares the base archive's sampling.
    use_drained_tiles = bool(historical_drained_tiles_url) and stable_style is not None
    if use_drained_tiles:
        drained_source = "drained_pmtiles"
        drained_poly_layer = "drained"
        drained_point_layer = "drained_points"
        lake_filter: dict = {}
        # That tileset only exists as built by the current builder, so its
        # centroids are always there to hand off to (unlike base_has_centroids).
        drained_poly_gate = {"minzoom": POINT_POLY_SWITCH_ZOOM}
        drained_has_centroids = True
    else:
        drained_source = "lakes_pmtiles"
        drained_poly_layer = source_layer
        drained_point_layer = "lakes_points"
        lake_filter = {"filter": DRAINED_LAKE_FILTER} if stable_style else {}
        drained_poly_gate = poly_gate
        drained_has_centroids = base_has_centroids

    lakes_fill_layer = {
        "id": "lakes-fill",
        "source": drained_source,
        "source-layer": drained_poly_layer,
        "type": "fill",
        **drained_poly_gate,
        **lake_filter,
        "paint": {
            "fill-color": fill_color,
            "fill-opacity": fill_opacity,
        },
    }
    lakes_line_layer = {
        "id": "lakes-line",
        "source": drained_source,
        "source-layer": drained_poly_layer,
        "type": "line",
        **drained_poly_gate,
        **lake_filter,
        "paint": {
            "line-color": line_color,
            "line-width": line_width,
            "line-opacity": line_opacity,
        },
    }

    # A dot covers a few pixels where a polygon covers hundreds, so the fill
    # opacity each viz mode picked -- tuned as a wash of colour over an area --
    # does not survive the change of mark: drainage_year paints stable lakes at
    # 0.05, a legible tint across a lake and nothing at all on a 2px circle.
    #
    # A root curve rather than a multiplier, because the modes start from very
    # different places and a single factor big enough for 0.05 pins everything
    # else at fully opaque -- which would cost the nrt_drainage base lakes their
    # whole job of staying muted under the drained overlay. A root lifts the
    # faint end hard, leaves the opaque end nearly alone, never exceeds 1, and is
    # monotonic, so every mode keeps its own ordering:
    #
    #   drainage_year  stable 0.05 -> 0.37   drained 0.20 -> 0.58
    #   nrt_drainage   base   0.35 -> 0.70   (overlay stays at 0.85, still the figure)
    #   colored_historical /generic_water 0.70 -> 0.89
    circle_opacity = ["^", fill_opacity, CENTROID_OPACITY_EXPONENT]

    # Centroids below the switch zoom, where the base tileset has no polygons
    # and a lake polygon would be sub-pixel anyway. maxzoom is exclusive and the
    # polygon layers' minzoom is inclusive, so the shared POINT_POLY_SWITCH_ZOOM
    # hands off in one step with no zoom drawn twice or left blank. Listed first
    # so any overlay stays on top of it. Only reaches the style when the archive
    # can back it (see poly_gate above); an unbacked circle layer would draw the
    # handful of dots that survived a rate-dropped build and nothing else.
    base_points_layer = {
        "id": "lakes-points",
        "source": drained_source,
        "source-layer": drained_point_layer,
        "type": "circle",
        "maxzoom": POINT_POLY_SWITCH_ZOOM,
        **lake_filter,
        "paint": {
            "circle-color": fill_color,
            "circle-opacity": circle_opacity,
            # Ramp up to the last zoom the circles are drawn at, so the dots are
            # at their largest just before the polygons take over.
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 0, 1.2, POINT_POLY_SWITCH_ZOOM - 1, 3.0],
            # A dark ring, the same trick the selection highlight uses: the dot's
            # own colour carries it on Dark Matter, and the ring is what separates
            # it from the mid-tone clutter of the satellite and TCVIS basemaps,
            # where an unringed dot in lake colours disappears. Tracks the fill so
            # a muted dot does not get a hard outline.
            "circle-stroke-color": "#1a1a1a",
            "circle-stroke-opacity": circle_opacity,
            "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 0, 0.4, POINT_POLY_SWITCH_ZOOM - 1, 0.9],
        },
    }

    if drained_ids:
        drained_filter = ["in", ["get", "id_geohash"], ["literal", drained_ids]]
        drained_overlay_layers = [
            {
                "id": "lakes-fill-drained",
                "source": "lakes_pmtiles",
                "source-layer": source_layer,
                "type": "fill",
                **poly_gate,
                "filter": drained_filter,
                "paint": {
                    "fill-color": "#d73027",  # Red fill for drained
                    "fill-opacity": 0.3,  # High opacity for drained
                },
            },
            {
                "id": "lakes-line-drained",
                "source": "lakes_pmtiles",
                "source-layer": source_layer,
                "type": "line",
                **poly_gate,
                "filter": drained_filter,
                "paint": {
                    "line-color": "#7f0000",  # Dark red border for drained
                    "line-width": 2.0,  # Thicker border for drained
                    "line-opacity": line_opacity,
                },
            },
        ]

    # The selected lake is outlined on top of everything else. Its own filter
    # means the highlight survives the "hide stable lakes" filter below, and the
    # dark casing under the red core keeps it readable over the red drained
    # overlay as well as over the light basemaps.
    selected_overlay_layers: list[dict] = []
    if selected_id:
        selected_filter = ["==", ["get", id_column], selected_id]
        selected_overlay_layers = [
            {
                "id": "lakes-line-selected-casing",
                "source": "lakes_pmtiles",
                "source-layer": source_layer,
                "type": "line",
                **poly_gate,
                "filter": selected_filter,
                "paint": {
                    "line-color": "#1a1a1a",
                    "line-width": 6.0,
                    "line-opacity": 0.8,
                },
            },
            {
                "id": "lakes-line-selected",
                "source": "lakes_pmtiles",
                "source-layer": source_layer,
                "type": "line",
                **poly_gate,
                "filter": selected_filter,
                "paint": {
                    "line-color": "#ff2d2d",
                    "line-width": 2.5,
                    "line-opacity": 1.0,
                },
            },
        ]

    # The neutral-grey lakes, under everything else. Same geometry, gates and
    # centroid handling as the coloured layers above -- only the paint and the
    # filter differ -- so a lake looks the same either side of the switch zoom
    # whichever of the two layers is drawing it.
    #
    # "Hide stable lakes" drops them from the style outright rather than painting
    # them transparent: a zero-opacity layer still answers queryRenderedFeatures,
    # so hidden lakes would keep producing hover popups.
    stable_layers: list[dict] = []
    if stable_style and not hide_stable_lakes:
        stable_fill, stable_opacity, stable_line, stable_line_width, stable_line_opacity = stable_style
        if base_has_centroids:
            stable_layers.append(
                {
                    "id": "lakes-stable-points",
                    "source": "lakes_pmtiles",
                    "source-layer": "lakes_points",
                    "type": "circle",
                    "maxzoom": POINT_POLY_SWITCH_ZOOM,
                    "filter": STABLE_LAKE_FILTER,
                    "paint": {
                        "circle-color": stable_fill,
                        "circle-opacity": ["^", stable_opacity, CENTROID_OPACITY_EXPONENT],
                        "circle-radius": base_points_layer["paint"]["circle-radius"],
                        "circle-stroke-color": "#1a1a1a",
                        "circle-stroke-opacity": ["^", stable_opacity, CENTROID_OPACITY_EXPONENT],
                        "circle-stroke-width": base_points_layer["paint"]["circle-stroke-width"],
                    },
                }
            )
        stable_layers += [
            {
                "id": "lakes-stable-fill",
                "source": "lakes_pmtiles",
                "source-layer": source_layer,
                "type": "fill",
                **poly_gate,
                "filter": STABLE_LAKE_FILTER,
                "paint": {"fill-color": stable_fill, "fill-opacity": stable_opacity},
            },
            {
                "id": "lakes-stable-line",
                "source": "lakes_pmtiles",
                "source-layer": source_layer,
                "type": "line",
                **poly_gate,
                "filter": STABLE_LAKE_FILTER,
                "paint": {
                    "line-color": stable_line,
                    "line-width": stable_line_width,
                    "line-opacity": stable_line_opacity,
                },
            },
        ]

    # The centroid layer is listed first so any overlay stays on top of it, and
    # is left out entirely when the archive cannot back it (see poly_gate).
    # Which archive that is depends on where the coloured lakes come from: with
    # a drained tileset these circles read from it, not from the base, and it
    # always bakes centroids -- gating them on the base archive's centroids
    # instead would blank the drained lakes below the switch whenever an older
    # base archive is paired with a freshly built overlay.
    lake_points_backed = drained_has_centroids if use_drained_tiles else base_has_centroids
    base_layers = ([base_points_layer] if lake_points_backed else []) + [lakes_fill_layer, lakes_line_layer]

    sources: dict[str, dict] = {}
    if use_drained_tiles:
        sources["drained_pmtiles"] = {
            "type": "vector",
            "url": "pmtiles://" + str(historical_drained_tiles_url),
            "promoteId": "id_geohash",
        }
    sources |= {
        "lakes_pmtiles": {
            "type": "vector",
            "url": "pmtiles://" + pmtiles_url,
            # Stable per-feature identity for setFeatureState, consistent
            # across zoom levels/tiles.
            "promoteId": "id_geohash",
        }
    }
    layers = [*stable_layers, *base_layers, *drained_overlay_layers]

    if nrt_monthly_tiles_url:
        drained_fill, drained_opacity, drained_line, drained_width, drained_line_opacity = (
            get_style_pmtiles_nrt_monthly_tiles()
        )
        sources["nrt_pmtiles"] = {
            "type": "vector",
            "url": "pmtiles://" + nrt_monthly_tiles_url,
        }
        if hide_stable_lakes or "stable" in hidden_categories:
            # Switch the base layers off rather than painting them at zero
            # opacity: a zero-opacity layer still counts as rendered, so
            # queryRenderedFeatures would keep producing hover popups for the
            # lakes the user asked to hide. The drained overlay is unaffected.
            for layer in base_layers:
                layer["layout"] = {"visibility": "none"}

        nrt_drained_points_layer = {
            # The overlay's own centroids, switching at the same zoom as the
            # base lakes below it (see base_points_layer): this is what keeps
            # drained lakes findable when zoomed out, without per-lake markers.
            "id": "nrt-drained-points",
            "source": "nrt_pmtiles",
            "source-layer": "drained_points",
            "type": "circle",
            "maxzoom": POINT_POLY_SWITCH_ZOOM,
            "paint": {
                "circle-color": drained_fill,
                "circle-opacity": drained_opacity,
                "circle-radius": ["interpolate", ["linear"], ["zoom"], 0, 1.5, POINT_POLY_SWITCH_ZOOM - 1, 4],
                "circle-stroke-color": drained_line,
                "circle-stroke-width": 0.5,
            },
        }
        nrt_drained_fill_layer = {
            "id": "nrt-drained-fill",
            "source": "nrt_pmtiles",
            "source-layer": "drained",
            "type": "fill",
            "minzoom": POINT_POLY_SWITCH_ZOOM,
            "paint": {"fill-color": drained_fill, "fill-opacity": drained_opacity},
        }
        nrt_drained_line_layer = {
            "id": "nrt-drained-line",
            "source": "nrt_pmtiles",
            "source-layer": "drained",
            "type": "line",
            "minzoom": POINT_POLY_SWITCH_ZOOM,
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

    layers.extend(selected_overlay_layers)

    lake_layer = PMTilesMapLibreLayerSynced(
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
        # None means "drained, confidence unknown" — encoded as 0 (rendered grey).
        state_by_id = {
            gid: {"confidence": 0 if conf is None else int(conf)} for gid, conf in nrt_confidence_by_id.items()
        }
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


@functools.lru_cache(maxsize=32)
def pmtiles_has_low_zoom_centroids(pmtiles_source: str, points_layer: str = "lakes_points") -> bool:
    """Whether ``pmtiles_source`` backs the dot/polygon handoff below the switch zoom.

    ``pmtiles_source`` is what the config names -- a local path, an http(s) URL
    or a ``gs://`` one -- read the same way the map's bounds are, from the
    archive's own metadata rather than from a flag kept alongside it. So an
    archive rebuilt with the current builder starts drawing centroids without
    anything else having to be changed. Cached because Streamlit rebuilds the
    map on every rerun and a remote archive costs two range requests.

    False when the archive cannot be read at all: the whole point is to keep
    lakes on screen, and the archives that fail this check are the ones that
    bake polygons at every zoom.
    """
    from water_timeseries.utils.pmtiles_reader import read_pmtiles_metadata, read_pmtiles_metadata_remote

    try:
        if pmtiles_source.startswith(("http://", "https://", "gs://")):
            metadata = read_pmtiles_metadata_remote(resolve_pmtiles_url(pmtiles_source))
        else:
            metadata = read_pmtiles_metadata(pmtiles_source)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Could not read PMTiles metadata for {pmtiles_source}: {exc}")
        return False

    return archive_bakes_low_zoom_centroids(metadata, points_layer=points_layer)


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

    return _get_pmtiles_server().pmtiles_url_for(pmtiles_path)


def resolve_nrt_monthly_tiles_url(location: str | Path | None, month: str) -> str | None:
    """Return a browser-fetchable URL for ``month``'s drainage tileset, or None.

    ``location`` is where ``build_pmtiles_nrt_monthly`` wrote its output: a
    local directory, an ``http(s)://`` prefix, or a ``gs://`` prefix. Returns
    None when no tileset exists for the month, which is the signal for callers
    to fall back to the runtime feature-state path.

    For a local directory, the month's archive is mounted onto the process-wide
    ``PmtilesServer`` (same one the base tileset uses) — unless
    ``PMTILES_BASE_URL`` is set, in which case the monthly tilesets are expected
    to be reachable by filename under that base URL (same convention as
    ``resolve_pmtiles_url``).
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

    return _get_pmtiles_server().pmtiles_url_for(tiles_dir / filename)


@functools.cache
def _get_pmtiles_server():
    """The one tile server for this process; archives are mounted onto it.

    Not one server per archive: switching dashboard modes brings in a second
    tileset, and the port may be pinned via ``PMTILES_PORT`` (Docker), so a
    second server could not bind.
    """
    from water_timeseries.utils.pmtiles_serve import PmtilesServer

    return PmtilesServer(None).start()


def geohash_to_human_readable_name(geohash: str) -> str:
    """Convert a geohash to a human-readable name."""
    lat, lon = pygeohash.decode(geohash)
    return f"{geohash} | {lat:.3f} : {lon:.3f}"


def human_readable_name_to_geohash(human_readable_name: str) -> str:
    """Convert a human-readable name to a geohash."""
    # Extract the geohash from the human-readable name
    geohash = human_readable_name.split(" | ")[0]
    return geohash
