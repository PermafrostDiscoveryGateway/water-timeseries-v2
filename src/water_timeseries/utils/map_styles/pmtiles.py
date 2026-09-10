def get_style_pmtiles_colored_historical() -> tuple:
    # Define default paint values
    fill_color = [
        "interpolate",
        ["linear"],
        ["get", "NetChange_perc"],
        -40.0,
        "#d73027",
        -20.0,
        "#f46d43",
        0.0,
        "#fee090",
        20.0,
        "#74add1",
        40.0,
        "#4575b4",
    ]
    fill_opacity = 0.7
    line_color = "#333333"
    line_width = 0.5
    line_opacity = 1
    return fill_color, fill_opacity, line_color, line_width, line_opacity


# A lake with no break year is a stable lake. Three shapes mean "no break year":
# the property is absent (tippecanoe drops nulls when it bakes a tile), or it
# survived as the string "" or "NaN" from the parquet. All three have to count,
# or stable lakes leak into the drained styling and vice versa.
STABLE_LAKE_FILTER: list = [
    "any",
    ["!", ["has", "date_break_year"]],
    ["==", ["to-string", ["get", "date_break_year"]], ""],
    ["==", ["to-string", ["get", "date_break_year"]], "NaN"],
]
DRAINED_LAKE_FILTER: list = ["!", STABLE_LAKE_FILTER]


def get_style_pmtiles_drainage_year(hide_stable_lakes: bool = False) -> tuple:
    fill_color = [
        "interpolate",
        ["linear"],
        ["to-number", ["get", "date_break_year"]],
        2016,
        "#313695",
        2017,
        "#4575b4",
        2018,
        "#74add1",
        2019,
        "#abd9e9",
        2020,
        "#e0f3f8",
        2021,
        "#fee090",
        2022,
        "#fdae61",
        2023,
        "#f46d43",
        2024,
        "#d73027",
        2025,
        "#a50026",
    ]
    # fill_color_no_date = "#ADD8E6"
    fill_opacity = [
        "case",
        [
            "any",
            ["==", ["to-string", ["get", "date_break_year"]], ""],
            ["==", ["to-string", ["get", "date_break_year"]], "NaN"],
        ],
        0.05,
        0.2,
    ]
    line_color = [
        "case",
        [
            "any",
            ["==", ["to-string", ["get", "date_break_year"]], ""],
            ["==", ["to-string", ["get", "date_break_year"]], "NaN"],
        ],
        "#9e9e9e",  # default line color for stable lakes
        [
            "interpolate",
            ["linear"],
            ["to-number", ["get", "date_break_year"]],
            2017,
            "#4575b4",
            2018,
            "#74add1",
            2019,
            "#abd9e9",
            2020,
            "#e0f3f8",
            2021,
            "#ffffbf",
            2022,
            "#fee090",
            2023,
            "#fdae61",
            2024,
            "#f46d43",
            2025,
            "#d73027",
        ],
    ]
    line_opacity = 1
    # switch to disable non drained lakes (stable lakes) from being displayed on the map
    if hide_stable_lakes:
        line_width = [
            "case",
            [
                "any",
                ["==", ["to-string", ["get", "date_break_year"]], ""],
                ["==", ["to-string", ["get", "date_break_year"]], "NaN"],
            ],
            0,
            3,
        ]
    else:
        line_width = [
            "case",
            [
                "any",
                ["==", ["to-string", ["get", "date_break_year"]], ""],
                ["==", ["to-string", ["get", "date_break_year"]], "NaN"],
            ],
            0.6,
            3,
        ]
    return fill_color, fill_opacity, line_color, line_width, line_opacity


def get_style_pmtiles_nrt_drainage(hidden_categories: frozenset[str] = frozenset()) -> tuple:
    """Paint for the baked ``drainage_confidence`` property (0=stable, 1-3=drained, NaN=no data).

    ``hidden_categories`` drops opacity/width to 0 for any of
    ``{"no_data", "stable", "low", "medium", "high"}`` the user unchecked in
    the sidebar legend controls, so hidden features are invisible to both
    rendering and ``queryRenderedFeatures`` hover.
    """
    hide_no_data = "no_data" in hidden_categories
    hide_stable = "stable" in hidden_categories
    hide_low = "low" in hidden_categories
    hide_medium = "medium" in hidden_categories
    hide_high = "high" in hidden_categories

    # Helper expressions for NaN vs zero
    is_nan = [
        "any",
        ["==", ["to-string", ["get", "drainage_confidence"]], ""],
        ["==", ["to-string", ["get", "drainage_confidence"]], "NaN"],
    ]

    # UPDATED COLORS:
    # 1: Saturated Gold (#f9c80e) - Much clearer than the previous pale gold.
    # 2: Deep Amber (#e67e22) - Distinctly darker and more orange.
    # 3: Strong Red (#d73027) - High-risk anchor.
    fill_color = [
        "case",
        is_nan,
        "#9e9e9e",  # Medium grey for NaN (Visible but neutral)
        [
            "interpolate",
            ["linear"],
            ["to-number", ["get", "drainage_confidence"]],
            1,
            "#f9c80e",
            2,
            "#e67e22",
            3,
            "#d73027",
        ],
    ]

    fill_opacity = [
        "case",
        is_nan,
        0 if hide_no_data else 0.5,
        ["==", ["to-number", ["get", "drainage_confidence"]], 0],
        0 if hide_stable else 0.15,
        ["==", ["to-number", ["get", "drainage_confidence"]], 1],
        0 if hide_low else 0.2,
        ["==", ["to-number", ["get", "drainage_confidence"]], 2],
        0 if hide_medium else 0.2,
        0 if hide_high else 0.2,
    ]

    line_color = [
        "case",
        is_nan,
        "#9e9e9e",  # Medium grey for NaN
        ["==", ["to-number", ["get", "drainage_confidence"]], 0],
        "#bdbdbd",  # Light grey for stable lakes (Distinct from NaN)
        [
            "interpolate",
            ["linear"],
            ["to-number", ["get", "drainage_confidence"]],
            1,
            "#f9c80e",
            2,
            "#e67e22",
            3,
            "#d73027",
        ],
    ]

    line_opacity = [
        "case",
        is_nan,
        0 if hide_no_data else 0.5,
        ["==", ["to-number", ["get", "drainage_confidence"]], 0],
        0 if hide_stable else 1,
        ["==", ["to-number", ["get", "drainage_confidence"]], 1],
        0 if hide_low else 1,
        ["==", ["to-number", ["get", "drainage_confidence"]], 2],
        0 if hide_medium else 1,
        0 if hide_high else 1,
    ]
    line_width = [
        "case",
        is_nan,
        0 if hide_no_data else 0.5,
        ["==", ["to-number", ["get", "drainage_confidence"]], 0],
        0 if hide_stable else 1,
        ["==", ["to-number", ["get", "drainage_confidence"]], 1],
        0 if hide_low else 1,
        ["==", ["to-number", ["get", "drainage_confidence"]], 2],
        0 if hide_medium else 1,
        0 if hide_high else 3,
    ]

    return fill_color, fill_opacity, line_color, line_width, line_opacity


def get_style_pmtiles_nrt_confidence_featurestate(hidden_categories: frozenset[str] = frozenset()) -> tuple:
    """Static paint for the NRT monthly drainage overlay driven by feature-state.

    The per-month, per-lake confidence values are pushed into the map at
    runtime via ``map.setFeatureState`` (see ``PMTilesMapLibreFeatureState``
    in ``map_utils.py``), so this paint tuple never changes between months —
    it reads ``["feature-state", "confidence"]`` instead of a baked tile
    property. Features with no state set are stable lakes for the selected
    month. Confidence 0 means "drained, confidence unknown" (no real
    ``drainage_confidence`` available for this month) and renders grey.

    Confidence colors match ``get_legend_html_nrt_drainage`` and
    ``get_style_pmtiles_nrt_monthly_tiles``, the preferred baked-tile path.

    This is the fallback for deployments without per-month drainage tilesets;
    see ``build_pmtiles_nrt_monthly``.
    """
    hide_no_data = "no_data" in hidden_categories
    hide_low = "low" in hidden_categories
    hide_medium = "medium" in hidden_categories
    hide_high = "high" in hidden_categories
    hide_stable = "stable" in hidden_categories

    conf = ["coalesce", ["feature-state", "confidence"], -1]

    fill_color = [
        "match",
        conf,
        0,
        "#bdbdbd",  # grey - no confidence value for this lake
        1,
        "#f9c80e",  # gold - low confidence
        2,
        "#e67e22",  # amber - medium confidence
        3,
        "#d73027",  # red - high confidence
        "#9e9e9e",  # stable lake (no state set this month)
    ]
    fill_opacity = [
        "match",
        conf,
        0,
        0 if hide_stable else 0.2,
        1,
        0 if hide_low else 0.15,
        2,
        0 if hide_medium else 0.815,
        3,
        0 if hide_high else 0.15,
        0 if hide_no_data else 0.2,
    ]
    line_color = [
        "match",
        conf,
        0,
        "#bdbdbd",  # dark red border for unknown-confidence drained
        1,
        "#f9c80e",
        2,
        "#e67e22",
        3,
        "#d73027",
        "#9e9e9e",
    ]
    line_width = [
        "match",
        conf,
        0,
        0 if hide_stable else 2.0,
        1,
        0 if hide_low else 1,
        2,
        0 if hide_medium else 1,
        3,
        0 if hide_high else 3,
        0 if hide_no_data else 1,
    ]
    line_opacity = [
        "match",
        conf,
        0,
        0 if hide_stable else 1,
        1,
        0 if hide_low else 1,
        2,
        0 if hide_medium else 1,
        3,
        0 if hide_high else 1,
        0 if hide_no_data else 0.9,
    ]
    return fill_color, fill_opacity, line_color, line_width, line_opacity


def get_style_pmtiles_nrt_monthly_tiles() -> tuple:
    """Paint for the per-month NRT drainage overlay tileset.

    Reads ``drainage_confidence`` straight from the tile properties baked by
    ``build_pmtiles_nrt_monthly``, so switching months
    means pointing the source at a different archive — nothing per-lake is sent
    to the browser and no ``setFeatureState`` push is needed (contrast
    ``get_style_pmtiles_nrt_confidence_featurestate``, the older runtime path).

    Every feature in this tileset is a drained lake for its month, so there is
    no "stable lake" case here; stable lakes stay in the base tiles underneath
    (see ``get_style_pmtiles_stable_lakes``). A lake with no
    ``drainage_confidence`` renders grey; every month served so far has full
    confidence coverage, so that is a defensive case rather than a real one.

    Colors, border colors and the thick-border-for-confidence-3 signature match
    ``get_legend_html_nrt_drainage``. Fill opacity deliberately does not: the
    legend swatches sit on white at 0.2, while these polygons sit on satellite
    imagery and need to stay legible.
    """
    # -1 stands in for "no drainage_confidence property on this feature".
    conf = ["coalesce", ["get", "drainage_confidence"], -1]

    fill_color = [
        "case",
        ["==", conf, 0],
        "#bdbdbd",  # gold - low confidence
        ["==", conf, 1],
        "#f9c80e",  # gold - low confidence
        ["==", conf, 2],
        "#e67e22",  # amber - medium confidence
        ["==", conf, 3],
        "#d73027",  # red - high confidence
        "#9e9e9e",  # grey - no confidence value for this lake
    ]
    fill_opacity = [
        "case",
        ["==", conf, 0],
        0.2,  # gold - low confidence
        ["==", conf, 1],
        0.15,  # gold - low confidence
        ["==", conf, 2],
        0.15,  # amber - medium confidence
        ["==", conf, 3],
        0.15,  # red - high confidence
        0.35,  # grey - no confidence value for this lake
    ]
    line_color = [
        "case",
        ["==", conf, 0],
        "#bdbdbd",
        ["==", conf, 1],
        "#f9c80e",
        ["==", conf, 2],
        "#e67e22",
        ["==", conf, 3],
        "#d73027",
        "#9e9e9e",  # grey border for unknown-confidence drained
    ]
    line_width = [
        "case",
        ["==", conf, 3],
        3,
        ["==", conf, -1],
        2.0,
        1,
    ]
    line_opacity = 1
    return fill_color, fill_opacity, line_color, line_width, line_opacity


def get_style_pmtiles_stable_lakes() -> tuple:
    """Paint for lakes that are context rather than subject: neutral grey.

    Used by both modes that draw drained lakes on top of everything else -- the
    base tiles under the per-month NRT overlay, and the stable lakes under the
    drainage_year colours. They read as neutral context, matching the "0 -
    Stable lake" swatch in ``get_legend_html_nrt_drainage``: visible enough to
    show where lakes are, muted enough that the coloured drained lakes stay the
    figure.

    "Hide stable lakes" is *not* handled here: hiding via zero opacity would
    leave the layer rendered as far as ``queryRenderedFeatures`` is concerned,
    so invisible lakes would still produce hover popups. ``build_pmtiles_map``
    switches the base layers' ``layout.visibility`` off instead.
    """
    return "#bdbdbd", 0.35, "#bdbdbd", 1, 0.9


def get_style_pmtiles_generic_water() -> tuple:
    fill_color = "#ADD8E6"
    fill_opacity = 0.7
    # line_color = "#1E90FF"
    line_color = "#eeeeee"
    line_width = 1
    line_opacity = 1
    return fill_color, fill_opacity, line_color, line_width, line_opacity
