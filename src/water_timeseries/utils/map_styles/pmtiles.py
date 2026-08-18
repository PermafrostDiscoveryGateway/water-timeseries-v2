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


def get_style_pmtiles_nrt_drainage(hide_stable_lakes: bool = False) -> tuple:
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

    if hide_stable_lakes:
        fill_opacity = [
            "case",
            is_nan,
            0,  # Hide NaN
            ["==", ["to-number", ["get", "drainage_confidence"]], 0],
            0,  # Hide stable
            0.2,  # Show drained lakes
        ]
    else:
        fill_opacity = [
            "case",
            is_nan,
            0.5,  # Show NaN at medium opacity
            ["==", ["to-number", ["get", "drainage_confidence"]], 0],
            0.15,  # Stable lakes (Very faint)
            0.2,  # Drained lakes
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

    if hide_stable_lakes:
        line_opacity = [
            "case",
            is_nan,
            0,  # Hide NaN line
            ["==", ["to-number", ["get", "drainage_confidence"]], 0],
            0,  # Hide stable line
            1,  # Show drained lake lines
        ]
        line_width = [
            "case",
            is_nan,
            0,  # Width 0 for NaN
            ["==", ["to-number", ["get", "drainage_confidence"]], 0],
            0,  # Width 0 for stable
            [
                "case",
                ["==", ["to-number", ["get", "drainage_confidence"]], 1],
                1,
                ["==", ["to-number", ["get", "drainage_confidence"]], 2],
                1,
                3,  # Width 3 for confidence 3
            ],
        ]
    else:
        line_opacity = [
            "case",
            is_nan,
            0.5,  # Show NaN lines
            1,  # Show all others at full opacity
        ]
        line_width = [
            "case",
            is_nan,
            0.5,  # Width 0.5 for NaN
            ["==", ["to-number", ["get", "drainage_confidence"]], 0],
            1,  # Width 1 for stable
            ["==", ["to-number", ["get", "drainage_confidence"]], 1],
            1,  # Width 1 for confidence 1
            ["==", ["to-number", ["get", "drainage_confidence"]], 2],
            1,  # Width 1 for confidence 2
            3,  # Width 3 for confidence 3
        ]

    return fill_color, fill_opacity, line_color, line_width, line_opacity


def get_style_pmtiles_generic_water() -> tuple:
    fill_color = "#ADD8E6"
    fill_opacity = 0.7
    # line_color = "#1E90FF"
    line_color = "#eeeeee"
    line_width = 1
    line_opacity = 1
    return fill_color, fill_opacity, line_color, line_width, line_opacity


def get_style_pmtiles_drained_ids(drained_ids: list[str]):
    fill_color = [
        "match",
        ["get", "id_geohash"],
        drained_ids,
        "#d73027",  # Red fill for drained
        # ADD8E6,  # Default color ramp for non-drained
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
    line_opacity = 1
    return fill_color, fill_opacity, line_color, line_width, line_opacity
