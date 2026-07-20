"""Tests for shareable-link state encoding/decoding (dashboard.share_state)."""

import pygeohash
from streamlit.testing.v1 import AppTest

from water_timeseries.dashboard.share_state import (
    DEFAULT_LAT,
    DEFAULT_LON,
    DEFAULT_ZOOM,
    UrlState,
    decode_url_state,
    encode_view,
)


class TestEncodeView:
    def test_rounds_lat_lon_to_5_decimals(self):
        encoded = encode_view(66.5123456789, -164.0876543, 12)
        assert encoded["lat"] == "66.51235"
        assert encoded["lon"] == "-164.08765"

    def test_rounds_zoom_to_2_decimals(self):
        assert encode_view(66.5, -164.1, 7.4361)["zoom"] == "7.44"

    def test_strips_trailing_zeros(self):
        encoded = encode_view(66.5, -164.0, 12.0)
        assert encoded == {"lat": "66.5", "lon": "-164", "zoom": "12"}

    def test_zero_values(self):
        encoded = encode_view(0.0, 0.0, 0.0)
        assert encoded == {"lat": "0", "lon": "0", "zoom": "0"}


class TestDecodeUrlState:
    def test_empty_params(self):
        state = decode_url_state({})
        assert state == UrlState()
        assert state.lat is None and state.drained is False

    def test_full_valid_params(self):
        state = decode_url_state(
            {
                "selected_lake": "b7zpm2xq4k9d",
                "lat": "66.512",
                "lon": "-164.087",
                "zoom": "12",
                "drained": "1",
                "month": "2024-06",
                "hide_stable": "1",
                "basemap": "tcvis",
                "hidden_layers": "drained_polygons,drained_markers",
            }
        )
        assert state.selected_lake == "b7zpm2xq4k9d"
        assert state.lat == 66.512
        assert state.lon == -164.087
        assert state.zoom == 12
        assert state.drained is True
        assert state.month == "2024-06"
        assert state.hide_stable is True
        assert state.basemap == "tcvis"
        assert state.hidden_layers == ("drained_markers", "drained_polygons")

    def test_malformed_floats_dropped(self):
        state = decode_url_state({"lat": "abc", "lon": "1e999", "zoom": "nan"})
        assert state.lat is None and state.lon is None and state.zoom is None

    def test_out_of_range_coords_dropped(self):
        state = decode_url_state({"lat": "95", "lon": "-200", "zoom": "50"})
        assert state.lat is None and state.lon is None and state.zoom is None

    def test_bad_month_dropped(self):
        for bad in ("2024-6", "202406", "June 2024", "2024-06-01", "<script>"):
            assert decode_url_state({"month": bad}).month is None

    def test_good_month_kept(self):
        assert decode_url_state({"month": "2019-11"}).month == "2019-11"

    def test_bad_geohash_dropped(self):
        for bad in ("b7zpm2xq4k9d0", "geo hash!", "a/b", ""):
            assert decode_url_state({"selected_lake": bad}).selected_lake is None

    def test_flags_only_accept_1(self):
        state = decode_url_state({"drained": "true", "hide_stable": "yes"})
        assert state.drained is False and state.hide_stable is False

    def test_invalid_basemap_dropped(self):
        assert decode_url_state({"basemap": "satellite"}).basemap is None
        assert decode_url_state({}).basemap is None

    def test_valid_basemap_kept(self):
        assert decode_url_state({"basemap": "esri_world_imagery"}).basemap == "esri_world_imagery"

    def test_hidden_layers_filters_unknown_keys_and_sorts(self):
        # "lakes" isn't a real layer key -- it's the map's primary content and
        # always shown, with no toggle -- so it's filtered out like any other
        # unrecognized token.
        state = decode_url_state({"hidden_layers": "drained_markers,bogus,lakes,drained_polygons,drained_polygons"})
        assert state.hidden_layers == ("drained_markers", "drained_polygons")

    def test_empty_hidden_layers_is_empty_tuple(self):
        assert decode_url_state({"hidden_layers": ""}).hidden_layers == ()
        assert decode_url_state({}).hidden_layers == ()


class TestRoundTrip:
    def test_encode_decode_round_trip(self):
        encoded = encode_view(66.51234, -164.08765, 11.5)
        state = decode_url_state(encoded)
        assert state.lat == 66.51234
        assert state.lon == -164.08765
        assert state.zoom == 11.5

    def test_defaults_round_trip(self):
        encoded = encode_view(DEFAULT_LAT, DEFAULT_LON, DEFAULT_ZOOM)
        state = decode_url_state(encoded)
        assert state.lat == DEFAULT_LAT
        assert state.lon == DEFAULT_LON
        assert state.zoom == DEFAULT_ZOOM


def _restore_app():
    import streamlit as st

    from water_timeseries.dashboard.share_state import apply_url_state_once

    apply_url_state_once()
    st.write("ok")


def _sync_text_app():
    import streamlit as st

    from water_timeseries.dashboard.share_state import sync_text_param

    sync_text_param("basemap", st.query_params.get("_value"), "dark_matter")
    st.write("ok")


def _sync_hidden_layers_app():
    import streamlit as st

    from water_timeseries.dashboard.share_state import sync_hidden_layers_param

    hidden = set((st.query_params.get("_value") or "").split(",")) - {""}
    sync_hidden_layers_param(hidden)
    st.write("ok")


class TestSyncTextParam:
    def test_non_default_value_is_written(self):
        at = AppTest.from_function(_sync_text_app, default_timeout=60)
        at.query_params["_value"] = "tcvis"
        at.run()
        assert not at.exception
        assert at.query_params["basemap"] == ["tcvis"]

    def test_default_value_is_elided(self):
        at = AppTest.from_function(_sync_text_app, default_timeout=60)
        at.query_params["basemap"] = "tcvis"  # stale value from a previous run
        at.query_params["_value"] = "dark_matter"
        at.run()
        assert not at.exception
        assert "basemap" not in at.query_params

    def test_empty_value_is_elided(self):
        at = AppTest.from_function(_sync_text_app, default_timeout=60)
        at.run()
        assert not at.exception
        assert "basemap" not in at.query_params


class TestSyncHiddenLayersParam:
    def test_hidden_set_is_written_sorted(self):
        at = AppTest.from_function(_sync_hidden_layers_app, default_timeout=60)
        at.query_params["_value"] = "drained_polygons,drained_markers"
        at.run()
        assert not at.exception
        assert at.query_params["hidden_layers"] == ["drained_markers,drained_polygons"]

    def test_empty_hidden_set_elides_param(self):
        at = AppTest.from_function(_sync_hidden_layers_app, default_timeout=60)
        at.query_params["hidden_layers"] = "drained_markers"  # stale value
        at.run()
        assert not at.exception
        assert "hidden_layers" not in at.query_params

    def test_unknown_keys_are_ignored(self):
        at = AppTest.from_function(_sync_hidden_layers_app, default_timeout=60)
        # "lakes" isn't a real layer key (no toggle for the map's primary
        # content), so it's filtered out just like any other unknown token.
        at.query_params["_value"] = "drained_markers,lakes,bogus"
        at.run()
        assert not at.exception
        assert at.query_params["hidden_layers"] == ["drained_markers"]


class TestApplyUrlStateOnce:
    def test_full_url_state_seeds_session_state(self):
        at = AppTest.from_function(_restore_app, default_timeout=60)
        for key, value in {
            "lat": "66.512",
            "lon": "-164.087",
            "zoom": "12",
            "drained": "1",
            "month": "2024-06",
            "hide_stable": "1",
            "selected_lake": "b7zpm2xq4k9d",
            "basemap": "esri_world_imagery",
            "hidden_layers": "drained_polygons",
        }.items():
            at.query_params[key] = value
        at.run()
        assert not at.exception
        ss = at.session_state
        assert ss["map_center"] == {"lat": 66.512, "lon": -164.087}
        assert ss["zoom_level"] == 12.0
        assert ss["live_map_zoom"] == 12.0
        assert ss["show_drained_toggle"] is True
        # zoom-6 drainage-overview override must be suppressed on restore
        assert ss["_prev_show_drained"] is True
        assert ss["heatmap_selected_cell"] == "2024-06"
        assert ss["heatmap_sync_dropdown"] is True
        assert ss["toggle_hide_stable_lakes"] is True
        # the shared live view wins over the automatic zoom-12 recenter
        assert ss["_centered_selection"] == "b7zpm2xq4k9d"
        assert ss["map_basemap_choice"] == "esri_world_imagery"
        assert ss["show_layer_drained_polygons"] is False
        # drained_markers wasn't in hidden_layers, so it's left untouched (not seeded)
        assert "show_layer_drained_markers" not in ss

    def test_selected_lake_only_centers_at_zoom_12(self):
        geohash = pygeohash.encode(67.1, -160.2, precision=12)
        at = AppTest.from_function(_restore_app, default_timeout=60)
        at.query_params["selected_lake"] = geohash
        at.run()
        assert not at.exception
        ss = at.session_state
        assert ss["zoom_level"] == 12
        assert abs(ss["map_center"]["lat"] - 67.1) < 1e-6
        assert abs(ss["map_center"]["lon"] + 160.2) < 1e-6
        assert ss["_centered_selection"] == geohash

    def test_no_params_is_a_noop(self):
        at = AppTest.from_function(_restore_app, default_timeout=60)
        at.run()
        assert not at.exception
        ss = at.session_state
        assert "map_center" not in ss
        assert "show_drained_toggle" not in ss
        assert ss["_url_state_applied"] is True
