def test_build_pmtiles_map_returns_folium_map():
    import folium

    from water_timeseries.map_utils import build_pmtiles_map

    m = build_pmtiles_map("https://example.com/lakes.pmtiles")
    assert isinstance(m, folium.Map)
    # PMTiles child should be present
    children = list(m._children.values())
    assert any("pmtiles" in str(type(c)).lower() for c in children)


def test_build_pmtiles_map_with_drained_ids():
    import folium

    from water_timeseries.map_utils import build_pmtiles_map

    drained_ids = ["geohash1", "geohash2"]
    m = build_pmtiles_map("https://example.com/lakes.pmtiles", drained_ids=drained_ids)
    assert isinstance(m, folium.Map)
    children = list(m._children.values())

    # Find the PMTilesMapLibreLayer child
    lake_layer = None
    for c in children:
        if "pmtiles" in str(type(c)).lower():
            lake_layer = c
            break

    assert lake_layer is not None
    # Let's inspect the style passed to PMTilesMapLibreLayer
    style = getattr(lake_layer, "style", None)
    assert style is not None
    assert "layers" in style

    fill_layer = next(lyr for lyr in style["layers"] if lyr["id"] == "lakes-fill")
    line_layer = next(lyr for lyr in style["layers"] if lyr["id"] == "lakes-line")

    # Check that fill-color is a match expression with drained_ids
    fill_color = fill_layer["paint"]["fill-color"]
    assert fill_color[0] == "match"
    assert fill_color[1] == ["get", "id_geohash"]
    assert fill_color[2] == drained_ids
    assert fill_color[3] == "#d73027"

    # Check that line-color is a match expression with drained_ids
    line_color = line_layer["paint"]["line-color"]
    assert line_color[0] == "match"
    assert line_color[1] == ["get", "id_geohash"]
    assert line_color[2] == drained_ids
    assert line_color[3] == "#7f0000"


def test_resolve_pmtiles_url_passthrough():
    from water_timeseries.map_utils import resolve_pmtiles_url

    url = "https://storage.googleapis.com/bucket/lakes.pmtiles"
    assert resolve_pmtiles_url(url) == url
