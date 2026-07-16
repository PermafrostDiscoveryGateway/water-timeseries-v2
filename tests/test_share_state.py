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
            }
        )
        assert state.selected_lake == "b7zpm2xq4k9d"
        assert state.lat == 66.512
        assert state.lon == -164.087
        assert state.zoom == 12
        assert state.drained is True
        assert state.month == "2024-06"
        assert state.hide_stable is True

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
