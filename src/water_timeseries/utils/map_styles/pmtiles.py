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


def get_style_pmtiles_drainage_year() -> tuple:
    fill_color = [
        "interpolate",
        ["linear"],
        ["to-number", ["get", "date_break_year"]],
        2017.0,
        "#fff5f0",
        2025.0,
        "#67000d",
    ]
    fill_opacity = 0.4
    line_color = [
        "interpolate",
        ["linear"],
        ["to-number", ["get", "date_break_year"]],
        2017,
        "#fff5f0",
        2021,
        "#f46d43",
        2025,
        "#67000d",
    ]
    # line_color = "#dddddd"
    line_width = 3
    line_opacity = 1
    return fill_color, fill_opacity, line_color, line_width, line_opacity


def get_style_pmtiles_generic_water() -> tuple:
    fill_color = "#ADD8E6"
    fill_opacity = 0.7
    # line_color = "#1E90FF"
    line_color = "#eeeeee"
    line_width = 20
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
