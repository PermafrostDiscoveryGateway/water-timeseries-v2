"""Shareable-link state: sync window state (map view, toggles, selection) with URL query params.

The dashboard keeps its restorable state in readable query params on its own URL:

    ?selected_lake=<geohash>&lat=<f>&lon=<f>&zoom=<f>&drained=1&month=YYYY-MM&hide_stable=1

Params at their default value are removed to keep URLs clean. When the app is
embedded in an iframe on a cooperating parent site (see ``embed/``), the same
params are mirrored onto the parent URL with a ``wt_`` prefix via postMessage,
and the "Copy link" button produces a parent-site URL instead.

Map view state is two-tiered to avoid feedback loops with streamlit-folium:

- Construction keys (``map_center``/``zoom_level``) are baked into the folium
  HTML; changing them remounts the Leaflet map, so they may only change on full
  reruns (programmatic jumps, URL restore, live-view adoption).
- Live keys (``live_map_center``/``live_map_zoom``) mirror what the user
  actually sees (captured from ``st_folium``'s returned center/zoom on fragment
  reruns); they feed the URL but never the fragment's map construction.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Mapping, Optional

import pygeohash
import streamlit as st

# Defaults matching create_app's initial view; params equal to these are elided.
DEFAULT_LAT = 66.5
DEFAULT_LON = -164.1
DEFAULT_ZOOM = 10.0

#: All query-param keys owned by this module (plus the pre-existing selected_lake).
STATE_PARAM_KEYS = ("selected_lake", "lat", "lon", "zoom", "drained", "month", "hide_stable")

#: Prefix applied to state params when mirrored onto a parent site's URL.
PARENT_PARAM_PREFIX = "wt_"

#: Env var restricting which parent origin the app will postMessage to.
PARENT_ORIGIN_ENV = "WT_PARENT_ORIGIN"

_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_GEOHASH_RE = re.compile(r"^[0-9a-zA-Z]{1,12}$")

_COORD_EPS = 1e-5
_ZOOM_EPS = 0.01


@dataclass
class UrlState:
    """Decoded, validated window state from URL query params."""

    lat: Optional[float] = None
    lon: Optional[float] = None
    zoom: Optional[float] = None
    selected_lake: Optional[str] = None
    drained: bool = False
    month: Optional[str] = None
    hide_stable: bool = False


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):  # NaN/inf guard
        return None
    return result


def decode_url_state(params: Mapping[str, str]) -> UrlState:
    """Parse query params into a UrlState, silently dropping malformed values."""
    lat = _parse_float(params.get("lat"))
    lon = _parse_float(params.get("lon"))
    zoom = _parse_float(params.get("zoom"))
    if lat is not None and not -90 <= lat <= 90:
        lat = None
    if lon is not None and not -180 <= lon <= 180:
        lon = None
    if zoom is not None and not 0 <= zoom <= 24:
        zoom = None

    selected_lake = params.get("selected_lake")
    if selected_lake and not _GEOHASH_RE.match(str(selected_lake)):
        selected_lake = None

    month = params.get("month")
    if month and not _MONTH_RE.match(str(month)):
        month = None

    return UrlState(
        lat=lat,
        lon=lon,
        zoom=zoom,
        selected_lake=str(selected_lake) if selected_lake else None,
        drained=params.get("drained") == "1",
        month=str(month) if month else None,
        hide_stable=params.get("hide_stable") == "1",
    )


def _fmt(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def encode_view(lat: float, lon: float, zoom: float) -> dict:
    """Round view values to URL precision (lat/lon 5 dp ~1 m, zoom 2 dp)."""
    return {"lat": _fmt(lat, 5), "lon": _fmt(lon, 5), "zoom": _fmt(zoom, 2)}


def _view_is_default(lat: float, lon: float, zoom: float) -> bool:
    return (
        abs(lat - DEFAULT_LAT) < _COORD_EPS
        and abs(lon - DEFAULT_LON) < _COORD_EPS
        and abs(zoom - DEFAULT_ZOOM) < _ZOOM_EPS
    )


def _write_view_params(lat: float, lon: float, zoom: float) -> None:
    if _view_is_default(lat, lon, zoom):
        for key in ("lat", "lon", "zoom"):
            st.query_params.pop(key, None)
        return
    encoded = encode_view(lat, lon, zoom)
    for key, value in encoded.items():
        if st.query_params.get(key) != value:
            st.query_params[key] = value


def sync_flag_param(name: str, on: bool) -> None:
    """Keep a boolean query param in sync: '1' when on, absent when off."""
    if on:
        if st.query_params.get(name) != "1":
            st.query_params[name] = "1"
    else:
        st.query_params.pop(name, None)


def set_desired_view(lat: float, lon: float, zoom: float) -> None:
    """Programmatic jump: set construction keys, live keys, and URL params.

    Single choke point for all programmatic view changes (lake selection, URL
    restore). Writing the live keys too prevents a stale live view from
    clobbering the jump when the next full run adopts live state.
    """
    st.session_state.map_center = {"lat": lat, "lon": lon}
    st.session_state.zoom_level = zoom
    st.session_state.live_map_center = {"lat": lat, "lon": lon}
    st.session_state.live_map_zoom = zoom
    _write_view_params(lat, lon, zoom)


def update_live_view(center: Optional[Mapping], zoom: Optional[float]) -> None:
    """Record the user's live view (from st_folium) without touching map construction.

    Called on fragment reruns after pan/zoom. Only updates live keys and URL
    params, so the folium HTML is unchanged and the map does not remount.
    """
    if not center:
        return
    lat = center.get("lat")
    lon = center.get("lng", center.get("lon"))
    if lat is None or lon is None:
        return
    if zoom is None:
        zoom = st.session_state.get("live_map_zoom", st.session_state.get("zoom_level", DEFAULT_ZOOM))

    prev_center = st.session_state.get("live_map_center") or {}
    prev_zoom = st.session_state.get("live_map_zoom")
    unchanged = (
        prev_zoom is not None
        and abs(lat - prev_center.get("lat", 1e9)) < _COORD_EPS
        and abs(lon - prev_center.get("lon", 1e9)) < _COORD_EPS
        and abs(zoom - prev_zoom) < _ZOOM_EPS
    )
    if unchanged:
        return

    st.session_state.live_map_center = {"lat": lat, "lon": lon}
    st.session_state.live_map_zoom = zoom
    _write_view_params(lat, lon, zoom)


def adopt_live_view() -> None:
    """On a full run, rebuild the map at the user's live view (if it moved).

    Copies live keys into construction keys so the rebuilt folium map is
    already positioned where the user left it.
    """
    live_center = st.session_state.get("live_map_center")
    live_zoom = st.session_state.get("live_map_zoom")
    if live_center is not None:
        st.session_state.map_center = dict(live_center)
    if live_zoom is not None:
        st.session_state.zoom_level = live_zoom


def apply_url_state_once() -> None:
    """Restore window state from URL query params (once per session).

    Must run in create_app BEFORE the recenter-to-selection block and before
    any widget it seeds (show_drained_toggle, toggle_hide_stable_lakes,
    heatmap_selected_cell) is instantiated.
    """
    if st.session_state.get("_url_state_applied"):
        return
    st.session_state["_url_state_applied"] = True

    state = decode_url_state(st.query_params)

    if state.lat is not None and state.lon is not None:
        set_desired_view(state.lat, state.lon, state.zoom if state.zoom is not None else DEFAULT_ZOOM)
        if state.selected_lake:
            # The shared live view wins over the automatic zoom-12 recenter.
            st.session_state["_centered_selection"] = state.selected_lake
    elif state.selected_lake:
        try:
            glat, glon = pygeohash.decode(state.selected_lake)
            set_desired_view(glat, glon, 12)
            st.session_state["_centered_selection"] = state.selected_lake
        except Exception:
            pass

    if state.drained:
        st.session_state["show_drained_toggle"] = True
        # Suppress the one-time zoom-6 drainage-overview override on restore.
        st.session_state["_prev_show_drained"] = True

    if state.month:
        st.session_state["heatmap_selected_cell"] = state.month
        st.session_state["heatmap_sync_dropdown"] = True

    if state.hide_stable:
        st.session_state["toggle_hide_stable_lakes"] = True


def current_state_params() -> dict:
    """Current shareable-state params from the URL (source of truth for sharing)."""
    return {key: st.query_params[key] for key in STATE_PARAM_KEYS if key in st.query_params}


def _target_origin() -> str:
    return os.environ.get(PARENT_ORIGIN_ENV) or "*"


def render_state_bridge() -> None:
    """Post the current state params to a cooperating parent page (wt:state).

    Rendered inside the map fragment so pan/zoom fragment reruns re-post. The
    component only re-executes when its HTML (i.e. the params JSON) changes,
    so unchanged state produces no message spam. Harmless when not framed.
    """
    params_json = json.dumps(current_state_params(), sort_keys=True)
    target_json = json.dumps(_target_origin())
    st.iframe(
        f"""
        <script>
        (function() {{
            try {{
                window.top.postMessage(
                    {{ type: "wt:state", version: 1, params: {params_json} }},
                    {target_json}
                );
            }} catch (e) {{ /* not framed or origin mismatch: nothing to do */ }}
        }})();
        </script>
        """,
        height=1,
    )


def render_copy_link_button() -> None:
    """Sidebar "Copy link" button restoring the exact window state.

    Standalone: copies the app's own URL (embed params stripped). Inside an
    iframe on a cooperating parent page (which answers wt:hello with
    wt:hello-ack), copies a parent-site URL carrying wt_-prefixed params.
    Falls back to a selectable text input when the clipboard is unavailable.
    """
    fallback_params_json = json.dumps(current_state_params(), sort_keys=True)
    app_url = ""
    try:
        app_url = str(st.context.url or "")
    except Exception:
        pass
    config = json.dumps(
        {
            "fallbackParams": json.loads(fallback_params_json),
            "fallbackAppUrl": app_url,
            "targetOrigin": _target_origin(),
            "stateKeys": list(STATE_PARAM_KEYS),
        }
    )
    st.iframe(
        """
        <style>
            body { margin: 0; font-family: "Source Sans Pro", sans-serif; }
            #wt-copy {
                width: 100%; padding: 0.4rem 0.75rem; cursor: pointer;
                border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 0.5rem;
                background: transparent; color: inherit; font-size: 0.875rem;
            }
            #wt-copy:hover { border-color: #ff4b4b; color: #ff4b4b; }
            #wt-url {
                width: 100%; margin-top: 0.25rem; padding: 0.25rem;
                font-size: 0.75rem; box-sizing: border-box; display: none;
            }
            @media (prefers-color-scheme: dark) {
                #wt-copy { border-color: rgba(250, 250, 250, 0.2); color: #fafafa; }
            }
        </style>
        <button id="wt-copy" title="Copy a link that restores this exact view">🔗 Copy link to this view</button>
        <input id="wt-url" readonly>
        <script>
        (function() {
            var CFG = __WT_CONFIG__;
            var parentInfo = null;

            window.addEventListener("message", function(ev) {
                var d = ev.data;
                if (!d || d.type !== "wt:hello-ack") return;
                if (CFG.targetOrigin !== "*" && ev.origin !== CFG.targetOrigin) return;
                parentInfo = { href: d.href, prefix: d.prefix || "wt_" };
            });
            try {
                window.top.postMessage({ type: "wt:hello", version: 1 }, CFG.targetOrigin);
            } catch (e) {}

            function currentParams() {
                try {
                    var sp = new URLSearchParams(window.parent.location.search);
                    var out = {};
                    CFG.stateKeys.forEach(function(k) {
                        var v = sp.get(k);
                        if (v !== null) out[k] = v;
                    });
                    return out;
                } catch (e) {
                    return CFG.fallbackParams;
                }
            }

            function buildUrl() {
                var params = currentParams();
                if (parentInfo && parentInfo.href) {
                    var u = new URL(parentInfo.href);
                    var stale = [];
                    u.searchParams.forEach(function(v, k) {
                        if (k.indexOf(parentInfo.prefix) === 0) stale.push(k);
                    });
                    stale.forEach(function(k) { u.searchParams.delete(k); });
                    Object.keys(params).forEach(function(k) {
                        u.searchParams.set(parentInfo.prefix + k, params[k]);
                    });
                    return u.toString();
                }
                var base = CFG.fallbackAppUrl;
                try { base = window.parent.location.href; } catch (e) {}
                if (!base) return null;
                var u2 = new URL(base);
                u2.searchParams.delete("embed");
                u2.searchParams.delete("embed_options");
                CFG.stateKeys.forEach(function(k) { u2.searchParams.delete(k); });
                Object.keys(params).forEach(function(k) { u2.searchParams.set(k, params[k]); });
                return u2.toString();
            }

            var btn = document.getElementById("wt-copy");
            var inp = document.getElementById("wt-url");
            function flash(text) {
                var old = "🔗 Copy link to this view";
                btn.textContent = text;
                setTimeout(function() { btn.textContent = old; }, 1600);
            }
            function showFallback(url) {
                inp.style.display = "block";
                inp.value = url;
                inp.focus();
                inp.select();
                try {
                    document.execCommand("copy");
                    flash("✓ Copied!");
                } catch (e) {
                    flash("Press Ctrl/Cmd+C to copy");
                }
            }
            btn.addEventListener("click", function() {
                var url = buildUrl();
                if (!url) { flash("No URL available"); return; }
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(url).then(
                        function() { flash("✓ Copied!"); },
                        function() { showFallback(url); }
                    );
                } else {
                    showFallback(url);
                }
            });
        })();
        </script>
        """.replace("__WT_CONFIG__", config),
        height=70,
    )
