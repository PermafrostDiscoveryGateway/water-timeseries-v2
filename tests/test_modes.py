"""Tests for switching between the Historical and Near Real-Time dashboards."""

import yaml

from water_timeseries.dashboard import modes as modes_mod
from water_timeseries.dashboard.modes import (
    MODES_ENV,
    apply_mode_override,
    available_modes,
    mode_key_for_viz,
    resolve_mode,
)


def _write_config(path, **values):
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def _two_configs(tmp_path):
    historical = _write_config(
        tmp_path / "historical.yaml",
        vector_file="data/historical.parquet",
        pmtiles_file="data/historical.pmtiles",
        viz_configuration="drainage_year",
        ee_project="should-not-be-copied",
    )
    nrt = _write_config(
        tmp_path / "nrt.yaml",
        vector_file="data/nrt.parquet",
        pmtiles_file="data/nrt.pmtiles",
        viz_configuration="nrt_drainage",
        dw_end_year=2026,
    )
    return historical, nrt


def test_mode_key_for_viz():
    assert mode_key_for_viz("drainage_year") == "historical"
    assert mode_key_for_viz("colored_historical") == "historical"
    assert mode_key_for_viz("nrt_drainage") == "nrt"
    assert mode_key_for_viz(None) == "historical"


def test_available_modes_from_env(tmp_path, monkeypatch):
    historical, nrt = _two_configs(tmp_path)
    monkeypatch.setenv(MODES_ENV, f"historical={historical},nrt={nrt}")

    found = available_modes()

    assert [mode.key for mode in found] == ["historical", "nrt"]
    assert [mode.label for mode in found] == ["Historical", "Near Real-Time"]
    assert found[1].settings["viz_configuration"] == "nrt_drainage"


def test_available_modes_empty_when_config_missing(tmp_path, monkeypatch):
    historical, _ = _two_configs(tmp_path)
    monkeypatch.setenv(MODES_ENV, f"historical={historical},nrt={tmp_path / 'gone.yaml'}")

    # A lone mode is nothing to switch to, so no switcher is offered.
    assert available_modes() == []


def test_resolve_mode_ignores_unknown_key(tmp_path, monkeypatch):
    historical, nrt = _two_configs(tmp_path)
    monkeypatch.setenv(MODES_ENV, f"historical={historical},nrt={nrt}")
    found = available_modes()

    assert resolve_mode("nrt", found).key == "nrt"
    assert resolve_mode("bogus", found) is None
    assert resolve_mode(None, found) is None


def test_apply_mode_override_swaps_data_settings(tmp_path, monkeypatch):
    historical, nrt = _two_configs(tmp_path)
    monkeypatch.setenv(MODES_ENV, f"historical={historical},nrt={nrt}")

    launch = {
        "vector_file": "data/historical.parquet",
        "pmtiles_file": "data/historical.pmtiles",
        "viz_configuration": "drainage_year",
        "ee_project": "launch-project",
        "dw_end_year": 2025,
    }
    settings, active, found = apply_mode_override(launch, requested_mode="nrt")

    assert active == "nrt"
    assert settings["vector_file"] == "data/nrt.parquet"
    assert settings["pmtiles_file"] == "data/nrt.pmtiles"
    assert settings["viz_configuration"] == "nrt_drainage"
    assert settings["dw_end_year"] == 2026
    # Deployment settings stay as launched.
    assert settings["ee_project"] == "launch-project"
    assert len(found) == 2
    # The caller's dict is not mutated.
    assert launch["vector_file"] == "data/historical.parquet"


def test_apply_mode_override_keeps_launch_settings_without_request(tmp_path, monkeypatch):
    historical, nrt = _two_configs(tmp_path)
    monkeypatch.setenv(MODES_ENV, f"historical={historical},nrt={nrt}")

    launch = {"vector_file": "data/historical.parquet", "viz_configuration": "drainage_year"}

    for requested in (None, "historical", "nonsense"):
        settings, active, _ = apply_mode_override(launch, requested_mode=requested)
        assert active == "historical"
        assert settings["vector_file"] == "data/historical.parquet"


def test_apply_mode_override_without_configs(tmp_path, monkeypatch):
    monkeypatch.setenv(MODES_ENV, f"historical={tmp_path / 'nope.yaml'}")

    launch = {"vector_file": "data/nrt.parquet", "viz_configuration": "nrt_drainage"}
    settings, active, found = apply_mode_override(launch, requested_mode="historical")

    assert found == []
    assert active == "nrt"
    assert settings == launch


def test_default_modes_point_at_repo_configs():
    """The shipped configs are the two modes users switch between."""
    keys = [key for key, _label, _path in modes_mod._DEFAULT_MODES]
    assert keys == ["historical", "nrt"]

    found = available_modes()
    assert [mode.key for mode in found] == ["historical", "nrt"]
    assert found[0].settings["viz_configuration"] == "drainage_year"
    assert found[1].settings["viz_configuration"] == "nrt_drainage"
