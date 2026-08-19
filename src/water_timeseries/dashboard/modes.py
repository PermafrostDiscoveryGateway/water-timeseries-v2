"""Switchable dashboard modes (Historical vs Near Real-Time).

The dashboard is launched with a single config YAML (``--config-file``), which
pins the dataset, the PMTiles archive and the ``viz_configuration`` styling.
A mode is just *another* such config: switching modes re-reads the other YAML
and overlays its data keys onto the launch settings, so one running instance
can serve both views.

The active mode lives in the ``mode`` query param, which keeps it shareable and
survives a browser reload (see :mod:`water_timeseries.dashboard.share_state`).

Modes come from the ``modes:`` block of the launch config, so one mounted file
carries every view (see :func:`water_timeseries.utils.cli.mode_configs`). A flat
single-mode config has no such block; those deployments fall back to the default
config below, or to ``DASHBOARD_MODES`` when it names one file per mode::

    DASHBOARD_MODES="drainage_year=configs/a.yaml,nrt_drainage=configs/b.yaml"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from loguru import logger

from water_timeseries.utils.cli import flatten_mode, load_config, mode_configs
from water_timeseries.utils.io import is_remote_path

_REPO_ROOT = Path(__file__).parent.parent.parent.parent

#: Query param holding the active mode key.
MODE_PARAM = "mode"

#: Env var overriding mode discovery: ``key=path`` pairs, comma separated.
MODES_ENV = "DASHBOARD_MODES"

#: Config keys a mode owns. Everything else (EE project, offline flag, logging)
#: stays as launched — those are deployment settings, not view settings.
_MODE_SETTING_KEYS = (
    "vector_file",
    "pmtiles_file",
    "pmtiles_url",
    "dw_dataset_file",
    "jrc_dataset_file",
    "precomputed_nrt_dir",
    "viz_configuration",
    "dw_start_year",
    "dw_end_year",
    "dw_start_month",
    "dw_end_month",
)

#: Multi-mode config used when the launch config carries no ``modes:`` block.
_DEFAULT_CONFIG = "configs/dashboard_panarctic.yaml"

#: Mode settings that name a local file or directory, and so need resolving
#: against the repo root before the mode can be offered.
_MODE_PATH_KEYS = (
    "vector_file",
    "pmtiles_file",
    "dw_dataset_file",
    "jrc_dataset_file",
    "precomputed_nrt_dir",
)

#: Paths a mode cannot work without: the tiles it paints and the polygons that
#: turn a map click into a lake id (see ``MapViewer.find_lake_id_at_point``).
#: A mode missing either would render but not respond, so it isn't offered.
_MODE_REQUIRED_PATH_KEYS = ("vector_file", "pmtiles_file")

#: ``viz_configuration`` -> mode key, used to name the launch config's mode when
#: it isn't one of the discovered YAMLs.
_VIZ_TO_MODE = {
    "colored_historical": "drainage_year",
    "drainage_year": "drainage_year",
    "nrt_drainage": "nrt_drainage",
}

_LABELS = {"drainage_year": "Historical Drainage (2016-2025)", "nrt_drainage": "Near Real-Time Anomalies (2026)"}

#: Session/query state that describes one mode's view and must not leak into
#: the other (different lakes, different months, different toggles).
_MODE_SCOPED_SESSION_KEYS = (
    "dw_dataset",
    "dw_dataset_raw",
    "downloaded_dsdw",
    "jrc_dataset",
    "jrc_dataset_raw",
    "downloaded_dsjrc",
    "precomputed_nrt_counts",
    "precomputed_nrt_breaks",
    "selected_geohash",
    "clicked_features",
    "clicked_lakes_dropdown",
    "nrt_month_selector",
    "heatmap_selected_cell",
    "heatmap_sync_dropdown",
    "show_drained_toggle",
    "_prev_show_drained",
    "toggle_hide_stable_lakes",
    "show_ts_popup",
)

_MODE_SCOPED_QUERY_PARAMS = ("selected_lake", "drained", "month", "hide_stable")


@dataclass(frozen=True)
class DashboardMode:
    """One selectable view, backed by its own dashboard config YAML."""

    key: str
    label: str
    config_path: Path
    settings: dict


def _parse_modes_env(raw: str) -> list[tuple[str, str]]:
    """Parse ``DASHBOARD_MODES`` into ``(key, path)`` pairs, skipping junk."""
    pairs = []
    for chunk in raw.split(","):
        key, _, path = chunk.partition("=")
        key, path = key.strip(), path.strip()
        if key and path:
            pairs.append((key, path))
    return pairs


def _resolve(path: str | Path) -> Path:
    """Resolve a config path against the CWD first, then the repo root."""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return _REPO_ROOT / candidate


def _resolve_mode_paths(settings: dict) -> dict:
    """Return ``settings`` with local data paths resolved against the repo root.

    Mode YAMLs spell their data paths relative to the repo root (``data/...``),
    but the dashboard's working directory is whatever it was launched from, so
    the raw values only work by luck. Remote URIs are left untouched.
    """
    resolved = dict(settings)
    for key in _MODE_PATH_KEYS:
        value = resolved.get(key)
        if value and not is_remote_path(value):
            resolved[key] = str(_resolve(value))
    return resolved


def _missing_required_paths(settings: dict) -> list[str]:
    """Required local paths this mode names but that don't exist on disk."""
    missing = []
    for key in _MODE_REQUIRED_PATH_KEYS:
        value = settings.get(key)
        if not value:
            missing.append(key)
        elif not is_remote_path(value) and not Path(value).exists():
            missing.append(f"{key}={value}")
    # A hosted tileset stands in for a local pmtiles_file.
    if settings.get("pmtiles_url"):
        missing = [item for item in missing if not item.startswith("pmtiles_file")]
    return missing


def _mode_specs(launch_config: dict | None) -> list[tuple[str, str, Path, dict]]:
    """Candidate ``(key, label, config_path, settings)`` tuples, before filtering.

    Preference order: the launch config's own ``modes:`` block, then
    ``DASHBOARD_MODES``, then the built-in multi-mode config.
    """
    config_path = Path(launch_config.get("config_file")) if (launch_config or {}).get("config_file") else None
    modes = mode_configs(launch_config or {})
    if modes:
        return [
            (
                key,
                values.get("label") or _LABELS.get(key, key.replace("_", " ").title()),
                config_path or _resolve(_DEFAULT_CONFIG),
                flatten_mode(launch_config, key),
            )
            for key, values in modes.items()
        ]

    env_raw = os.environ.get(MODES_ENV, "").strip()
    if env_raw:
        specs = []
        for key, path in _parse_modes_env(env_raw):
            path = _resolve(path)
            specs.append((key, _LABELS.get(key, key.replace("_", " ").title()), path, load_config(path, logger)))
        return specs

    # Flat launch config: fall back to the shipped multi-mode file.
    default_path = _resolve(_DEFAULT_CONFIG)
    default_config = load_config(default_path, logger)
    return [
        (
            key,
            values.get("label") or _LABELS.get(key, key.replace("_", " ").title()),
            default_path,
            flatten_mode(default_config, key),
        )
        for key, values in mode_configs(default_config).items()
    ]


def available_modes(launch_config: dict | None = None) -> list[DashboardMode]:
    """Discover selectable modes, dropping any that aren't usable here.

    A mode is dropped when its data isn't present: switching to a mode whose
    ``vector_file`` is absent used to leave a map that painted but ignored every
    click, because :func:`app.main` quietly substituted the tiny test fixture
    for the missing parquet (issue #229).

    Args:
        launch_config: The config the dashboard was launched with. Its
            ``modes:`` block, when present, defines the modes.

    Returns:
        The usable modes, or an empty list when fewer than two are usable --
        a switcher with nothing to switch to would only be noise.
    """
    modes = []
    for key, label, config_path, settings in _mode_specs(launch_config):
        if not settings:
            logger.debug(f"Skipping mode {key!r}: no usable config at {config_path}")
            continue
        settings = _resolve_mode_paths(settings)
        missing = _missing_required_paths(settings)
        if missing:
            logger.warning(f"Skipping mode {key!r} ({config_path}): missing data -- {', '.join(missing)}")
            continue
        modes.append(DashboardMode(key=key, label=label, config_path=config_path, settings=settings))

    if len(modes) < 2:
        return []
    return modes


def mode_key_for_viz(viz_configuration: str | None) -> str:
    """Mode key implied by a ``viz_configuration`` name."""
    return _VIZ_TO_MODE.get(viz_configuration or "", "drainage_year")


def resolve_mode(requested: str | None, modes: list[DashboardMode]) -> DashboardMode | None:
    """Return the requested mode, or None if it isn't offered."""
    if not requested:
        return None
    for mode in modes:
        if mode.key == requested:
            return mode
    logger.warning(f"Unknown dashboard mode {requested!r}; keeping the launch mode.")
    return None


def apply_mode_override(
    launch_settings: dict,
    *,
    requested_mode: str | None,
    launch_config: dict | None = None,
) -> tuple[dict, str, list[DashboardMode]]:
    """Overlay the requested mode's config onto the launch settings.

    Args:
        launch_settings: Data settings the dashboard was launched with.
        requested_mode: Mode key from the ``mode`` query param (may be None).
        launch_config: The launch config file's contents, whose ``modes:``
            block defines the selectable modes.

    Returns:
        ``(settings, active_mode_key, modes)`` — settings with the mode's data
        keys applied, the key now in effect, and the selectable modes.
    """
    modes = available_modes(launch_config)
    launch_key = mode_key_for_viz(launch_settings.get("viz_configuration"))

    active = resolve_mode(requested_mode, modes)
    if active is None:
        return dict(launch_settings), launch_key, modes
    # Same key: the launch config is that mode's authoritative version, so keep
    # its paths -- unless it named none, as when launched on the shared
    # `dashboard_panarctic_base.yaml`. Short-circuiting there stranded the run
    # on the test fixture with no way to reach the real data from the switcher.
    if active.key == launch_key and any(launch_settings.get(k) for k in _MODE_REQUIRED_PATH_KEYS + ("pmtiles_url",)):
        return dict(launch_settings), launch_key, modes

    settings = dict(launch_settings)
    for key in _MODE_SETTING_KEYS:
        if key in active.settings:
            settings[key] = active.settings[key]
    logger.info(f"Switched dashboard mode to {active.key!r} using {active.config_path}")
    return settings, active.key, modes


def clear_mode_scoped_state() -> None:
    """Drop selection/overlay state that only makes sense within one mode."""
    for key in _MODE_SCOPED_SESSION_KEYS:
        st.session_state.pop(key, None)
    for key in _MODE_SCOPED_QUERY_PARAMS:
        st.query_params.pop(key, None)


def render_mode_switcher(
    modes: list[DashboardMode],
    active_key: str,
    container=None,
) -> None:
    """Render the sidebar mode picker; switching rewrites ``?mode=`` and reruns."""
    if len(modes) < 2:
        return

    target = container if container is not None else st.sidebar
    keys = [mode.key for mode in modes]
    labels = {mode.key: mode.label for mode in modes}
    index = keys.index(active_key) if active_key in keys else 0

    choice = target.radio(
        "View mode",
        keys,
        index=index,
        format_func=lambda key: labels.get(key, key),
        horizontal=True,
        key="dashboard_mode_selector",
        help="Historical shows the 2017-2025 breakpoint analysis; Near Real-Time shows the latest monthly drainage run.",
    )

    if choice != active_key:
        clear_mode_scoped_state()
        st.query_params[MODE_PARAM] = choice
        st.rerun()
