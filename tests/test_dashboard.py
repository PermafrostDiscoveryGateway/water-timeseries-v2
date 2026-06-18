def test_build_pmtiles_map_returns_folium_map():
    from water_timeseries.map_utils import build_pmtiles_map
    import folium

    m = build_pmtiles_map("https://example.com/lakes.pmtiles")
    assert isinstance(m, folium.Map)
    # PMTiles child should be present
    children = list(m._children.values())
    assert any("pmtiles" in str(type(c)).lower() for c in children)


def test_resolve_pmtiles_url_passthrough():
    from water_timeseries.map_utils import resolve_pmtiles_url

    url = "https://storage.googleapis.com/bucket/lakes.pmtiles"
    assert resolve_pmtiles_url(url) == url
