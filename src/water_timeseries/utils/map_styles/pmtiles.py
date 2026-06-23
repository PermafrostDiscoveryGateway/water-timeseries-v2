        
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
    line_opacity=1
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