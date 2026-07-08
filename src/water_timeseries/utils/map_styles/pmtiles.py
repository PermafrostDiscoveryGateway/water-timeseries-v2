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
        2017.0,
        "#fff5f0",
        2025.0,
        "#67000d",
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
        "#ADD8E6",  # default line color for stable lakes
        [
            "interpolate",
            ["linear"],
            ["to-number", ["get", "date_break_year"]],
            2017,
            "#fff5f0",
            2021,
            "#f46d43",
            2025,
            "#67000d",
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
    fill_color = [
        "interpolate",
        ["linear"],
        ["to-number", ["get", "drainage_confidence"]],
        0,
        "#525252",
        1,
        "#969696",
        2,
        "#bdbdbd",
        3,
        "#ffffff",
    ]
    # fill_color_no_date = "#ADD8E6"
    fill_opacity = [
        "case",
        [
            "any",
            ["==", ["to-string", ["get", "drainage_confidence"]], ""],
            ["==", ["to-string", ["get", "drainage_confidence"]], "NaN"],
            ["==", ["to-number", ["get", "drainage_confidence"]], 0],
        ],
        0.05,
        0.2,
    ]
    line_color = [
        "case",
        [
            "any",
            ["==", ["to-string", ["get", "drainage_confidence"]], ""],
            ["==", ["to-string", ["get", "drainage_confidence"]], "NaN"],
            ["==", ["to-number", ["get", "drainage_confidence"]], 0],
        ],
        "#ADD8E6",  # default line color for stable lakes
        [
            "interpolate",
            ["linear"],
            ["to-number", ["get", "drainage_confidence"]],
            0,
            "#525252",
            1,
            "#969696",
            2,
            "#bdbdbd",
            3,
            "#ffffff",
        ],
    ]
    line_opacity = 1
    # switch to disable non drained lakes (stable lakes) from being displayed on the map
    # line width: 0.5 for confidence 0, 1 for 1-2, 3 for 3
    if hide_stable_lakes:
        line_width = [
            "case",
            [
                "any",
                ["==", ["to-string", ["get", "drainage_confidence"]], ""],
                ["==", ["to-string", ["get", "drainage_confidence"]], "NaN"],
                ["==", ["to-number", ["get", "drainage_confidence"]], 0],
            ],
            0,
            [
                "case",
                ["==", ["to-number", ["get", "drainage_confidence"]], 1],
                1,
                ["==", ["to-number", ["get", "drainage_confidence"]], 2],
                1,
                3,  # confidence 3
            ],
        ]
    else:
        line_width = [
            "case",
            ["==", ["to-number", ["get", "drainage_confidence"]], 0],
            0.5,
            ["==", ["to-number", ["get", "drainage_confidence"]], 1],
            1,
            ["==", ["to-number", ["get", "drainage_confidence"]], 2],
            1,
            3,  # confidence 3
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
