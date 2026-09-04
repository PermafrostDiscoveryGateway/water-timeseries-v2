"""Tests for switching between the Historical and Near Real-Time dashboards."""

from pathlib import Path

import yaml

from water_timeseries.dashboard import modes as modes_mod
from water_timeseries.dashboard.modes import (
    MODES_ENV,
    apply_mode_override,
    available_modes,
    mode_key_for_viz,
    resolve_mode,
)
from water_timeseries.utils.cli import flatten_mode, mode_configs
from water_timeseries.utils.pmtiles_build import historical_drained_tiles_path


def _write_config(path, **values):
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def _touch(path):
    """Create a stand-in data file, since a mode with missing data is skipped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _two_configs(tmp_path):
    data = tmp_path / "data"
    paths = {
        name: _touch(data / name) for name in ("historical.parquet", "historical.pmtiles", "nrt.parquet", "nrt.pmtiles")
    }
    historical = _write_config(
        tmp_path / "historical.yaml",
        vector_file=str(paths["historical.parquet"]),
        pmtiles_file=str(paths["historical.pmtiles"]),
        viz_configuration="drainage_year",
        ee_project="should-not-be-copied",
    )
    nrt = _write_config(
        tmp_path / "nrt.yaml",
        vector_file=str(paths["nrt.parquet"]),
        pmtiles_file=str(paths["nrt.pmtiles"]),
        viz_configuration="nrt_drainage",
        dw_end_year=2026,
    )
    return historical, nrt


def _multi_mode_config(tmp_path):
    """One config holding both modes, as configs/dashboard_panarctic.yaml does."""
    paths = {
        name: _touch(tmp_path / "data" / name)
        for name in ("historical.parquet", "historical.pmtiles", "nrt.parquet", "nrt.pmtiles")
    }
    return _write_config(
        tmp_path / "panarctic.yaml",
        ee_project="shared-project",
        dw_end_year=2026,
        default_mode="drainage_year",
        modes={
            "drainage_year": {
                "label": "Historical",
                "vector_file": str(paths["historical.parquet"]),
                "pmtiles_file": str(paths["historical.pmtiles"]),
                "viz_configuration": "drainage_year",
            },
            "nrt_drainage": {
                "label": "Near Real-Time",
                "vector_file": str(paths["nrt.parquet"]),
                "pmtiles_file": str(paths["nrt.pmtiles"]),
                "viz_configuration": "nrt_drainage",
            },
        },
    )


def _paths(tmp_path):
    data = tmp_path / "data"
    return {
        name: str(data / name) for name in ("historical.parquet", "historical.pmtiles", "nrt.parquet", "nrt.pmtiles")
    }


def test_mode_key_for_viz():
    assert mode_key_for_viz("drainage_year") == "drainage_year"
    assert mode_key_for_viz("colored_historical") == "drainage_year"
    assert mode_key_for_viz("nrt_drainage") == "nrt_drainage"
    assert mode_key_for_viz(None) == "drainage_year"


def test_available_modes_from_env(tmp_path, monkeypatch):
    historical, nrt = _two_configs(tmp_path)
    monkeypatch.setenv(MODES_ENV, f"drainage_year={historical},nrt_drainage={nrt}")

    found = available_modes()

    assert [mode.key for mode in found] == ["drainage_year", "nrt_drainage"]
    assert [mode.label for mode in found] == [
        "Historical Drainage (2016-2025)",
        "Near Real-Time Anomalies (2026)",
    ]
    assert found[1].settings["viz_configuration"] == "nrt_drainage"


def test_available_modes_empty_when_config_missing(tmp_path, monkeypatch):
    historical, _ = _two_configs(tmp_path)
    monkeypatch.setenv(MODES_ENV, f"drainage_year={historical},nrt_drainage={tmp_path / 'gone.yaml'}")

    # A lone mode is nothing to switch to, so no switcher is offered.
    assert available_modes() == []


def test_resolve_mode_ignores_unknown_key(tmp_path, monkeypatch):
    historical, nrt = _two_configs(tmp_path)
    monkeypatch.setenv(MODES_ENV, f"drainage_year={historical},nrt_drainage={nrt}")
    found = available_modes()

    assert resolve_mode("nrt_drainage", found).key == "nrt_drainage"
    assert resolve_mode("bogus", found) is None
    assert resolve_mode(None, found) is None


def test_apply_mode_override_swaps_data_settings(tmp_path, monkeypatch):
    historical, nrt = _two_configs(tmp_path)
    monkeypatch.setenv(MODES_ENV, f"drainage_year={historical},nrt_drainage={nrt}")

    expected = _paths(tmp_path)
    launch = {
        "vector_file": expected["historical.parquet"],
        "pmtiles_file": expected["historical.pmtiles"],
        "viz_configuration": "drainage_year",
        "ee_project": "launch-project",
        "dw_end_year": 2025,
    }
    settings, active, found = apply_mode_override(launch, requested_mode="nrt_drainage")

    assert active == "nrt_drainage"
    assert settings["vector_file"] == expected["nrt.parquet"]
    assert settings["pmtiles_file"] == expected["nrt.pmtiles"]
    assert settings["viz_configuration"] == "nrt_drainage"
    assert settings["dw_end_year"] == 2026
    # Deployment settings stay as launched.
    assert settings["ee_project"] == "launch-project"
    assert len(found) == 2
    # The caller's dict is not mutated.
    assert launch["vector_file"] == expected["historical.parquet"]


def test_apply_mode_override_keeps_launch_settings_without_request(tmp_path, monkeypatch):
    historical, nrt = _two_configs(tmp_path)
    monkeypatch.setenv(MODES_ENV, f"drainage_year={historical},nrt_drainage={nrt}")

    launch = {"vector_file": _paths(tmp_path)["historical.parquet"], "viz_configuration": "drainage_year"}

    for requested in (None, "drainage_year", "nonsense"):
        settings, active, _ = apply_mode_override(launch, requested_mode=requested)
        assert active == "drainage_year"
        assert settings["vector_file"] == launch["vector_file"]


def test_apply_mode_override_without_configs(tmp_path, monkeypatch):
    monkeypatch.setenv(MODES_ENV, f"drainage_year={tmp_path / 'nope.yaml'}")

    launch = {"vector_file": "data/nrt.parquet", "viz_configuration": "nrt_drainage"}
    settings, active, found = apply_mode_override(launch, requested_mode="drainage_year")

    assert found == []
    assert active == "nrt_drainage"
    assert settings == launch


def test_available_modes_skips_mode_with_missing_data(tmp_path, monkeypatch):
    """A mode whose vector_file is absent is not offered (issue #229).

    Without its polygons, the map paints from its tiles but no click can
    resolve to a lake -- and app.main silently substitutes the test fixture,
    so nothing tells the user why the map went dead.
    """
    historical, _ = _two_configs(tmp_path)
    nrt = _write_config(
        tmp_path / "nrt_broken.yaml",
        vector_file=str(tmp_path / "data" / "does_not_exist.parquet"),
        pmtiles_file=str(tmp_path / "data" / "nrt.pmtiles"),
        viz_configuration="nrt_drainage",
    )
    monkeypatch.setenv(MODES_ENV, f"drainage_year={historical},nrt_drainage={nrt}")

    assert available_modes() == []


def test_available_modes_accepts_hosted_tiles(tmp_path, monkeypatch):
    """A remote pmtiles_url stands in for a local archive."""
    historical, _ = _two_configs(tmp_path)
    nrt = _write_config(
        tmp_path / "nrt_hosted.yaml",
        vector_file=str(tmp_path / "data" / "nrt.parquet"),
        pmtiles_url="https://example.org/nrt.pmtiles",
        viz_configuration="nrt_drainage",
    )
    monkeypatch.setenv(MODES_ENV, f"drainage_year={historical},nrt_drainage={nrt}")

    assert [mode.key for mode in available_modes()] == ["drainage_year", "nrt_drainage"]


def test_shipped_config_defines_both_modes():
    """The shipped config carries both modes users switch between."""
    config = modes_mod.load_config(modes_mod._resolve(modes_mod._DEFAULT_CONFIG), modes_mod.logger)
    assert config, "missing configs/dashboard_panarctic.yaml"
    assert list(mode_configs(config)) == ["drainage_year", "nrt_drainage"]
    assert config["default_mode"] in mode_configs(config)

    # Data presence is deployment-specific (available_modes() drops modes whose
    # datasets are absent), so check the shipped config itself.
    for key in mode_configs(config):
        settings = flatten_mode(config, key)
        assert modes_mod.mode_key_for_viz(settings["viz_configuration"]) == key
        for required in modes_mod._MODE_REQUIRED_PATH_KEYS:
            assert settings.get(required) or settings.get("pmtiles_url")
        # Shared top-level keys reach every mode.
        assert settings["ee_project"] == config["ee_project"]
        assert "modes" not in settings and "label" not in settings


def test_shipped_config_pairs_the_shared_archive_with_a_drained_overlay():
    """drainage_year renders from the shared base archive, plus its own drained overlay.

    The base archive is mode-agnostic and carries no per-year values, so the
    overlay is what makes the drainage years visible. Sharing the base is only
    safe while that overlay stays per-mode, so this pins both halves.

    nrt_drainage joins the shared archive once the per-month tilesets can answer
    for a non-drained lake -- see
    ``test_shipped_config_shares_one_base_archive_between_modes``.
    """
    config = modes_mod.load_config(modes_mod._resolve(modes_mod._DEFAULT_CONFIG), modes_mod.logger)
    settings = {key: flatten_mode(config, key) for key in mode_configs(config)}

    base_archives = {key: values["pmtiles_file"] for key, values in settings.items()}
    assert base_archives["drainage_year"] == "data/lake_geometry/lakes.pmtiles"

    # The drained overlay is not optional once the base archive is shared: the
    # fallback filters the base on date_break_year, which the shared archive
    # carries only for drained lakes, and a geometry-only one not at all.
    drained = settings["drainage_year"]["drained_pmtiles_file"]
    assert drained, "drainage_year must name its drained overlay, not fall back to the base archive"
    # Both resolution paths have to agree: the explicit key (which only reaches a
    # fresh page load because cli.py forwards it) and the <base>_drained.pmtiles
    # convention app.py falls back on when no key is passed.
    assert Path(drained) == historical_drained_tiles_path(base_archives["drainage_year"])

    assert settings["nrt_drainage"]["nrt_pmtiles_dir"]
    assert "nrt_pmtiles_dir" not in mode_configs(config)["drainage_year"]


def test_dashboard_entrypoints_carry_the_drained_overlay_through(monkeypatch):
    """The config key is useless unless both hops forward it.

    ``cli.py dashboard`` spawns the streamlit script, so a key it does not pass
    on only takes effect once the user switches modes (which re-reads the YAML).
    That gap is what left historical mode grey on first load.
    """
    import inspect
    import sys

    from water_timeseries.dashboard.app import parse_args
    from water_timeseries.scripts.cli import dashboard

    assert "drained_pmtiles_file" in inspect.signature(dashboard).parameters

    monkeypatch.setattr(sys, "argv", ["app.py", "--drained-pmtiles-file", "x.pmtiles"])
    assert parse_args().drained_pmtiles_file == "x.pmtiles"


def test_apply_mode_override_fills_in_dataless_launch(tmp_path, monkeypatch):
    """Launching the shared base config leaves no data; picking a mode supplies it."""
    historical, nrt = _two_configs(tmp_path)
    monkeypatch.setenv(MODES_ENV, f"drainage_year={historical},nrt_drainage={nrt}")

    # What a config naming no data produces: no vector_file/pmtiles_file, and
    # the CLI's default viz_configuration.
    launch = {"vector_file": None, "pmtiles_file": None, "viz_configuration": "colored_historical"}

    settings, active, _ = apply_mode_override(launch, requested_mode="drainage_year")

    assert active == "drainage_year"
    assert settings["vector_file"] == _paths(tmp_path)["historical.parquet"]
    assert settings["viz_configuration"] == "drainage_year"


def _load(path):
    return modes_mod.load_config(path, modes_mod.logger)


def test_available_modes_from_multi_mode_config(tmp_path, monkeypatch):
    """Both modes come out of one config, with no DASHBOARD_MODES involved."""
    monkeypatch.delenv(MODES_ENV, raising=False)
    config = _load(_multi_mode_config(tmp_path))

    found = available_modes(config)

    assert [mode.key for mode in found] == ["drainage_year", "nrt_drainage"]
    # The mode's own `label` wins over the built-in default.
    assert [mode.label for mode in found] == ["Historical", "Near Real-Time"]
    assert found[1].settings["vector_file"] == _paths(tmp_path)["nrt.parquet"]
    # Shared top-level keys are inherited by each mode.
    assert found[0].settings["ee_project"] == "shared-project"


def test_apply_mode_override_swaps_within_one_config(tmp_path, monkeypatch):
    monkeypatch.delenv(MODES_ENV, raising=False)
    config = _load(_multi_mode_config(tmp_path))
    expected = _paths(tmp_path)

    launch = flatten_mode(config)
    assert launch["vector_file"] == expected["historical.parquet"]

    settings, active, found = apply_mode_override(launch, requested_mode="nrt_drainage", launch_config=config)

    assert active == "nrt_drainage"
    assert len(found) == 2
    assert settings["vector_file"] == expected["nrt.parquet"]
    assert settings["viz_configuration"] == "nrt_drainage"
    # Deployment settings stay as launched.
    assert settings["ee_project"] == "shared-project"


def test_multi_mode_config_takes_precedence_over_env(tmp_path, monkeypatch):
    """A config carrying its own modes ignores DASHBOARD_MODES."""
    historical, nrt = _two_configs(tmp_path / "flat")
    monkeypatch.setenv(MODES_ENV, f"drainage_year={historical},nrt_drainage={nrt}")
    config = _load(_multi_mode_config(tmp_path))

    found = available_modes(config)

    assert [mode.label for mode in found] == ["Historical", "Near Real-Time"]
    assert found[0].settings["vector_file"] == _paths(tmp_path)["historical.parquet"]


def test_flatten_mode_leaves_flat_config_alone():
    """A single-mode config has no modes: block and passes through untouched."""
    flat = {"vector_file": "a.parquet", "viz_configuration": "drainage_year"}

    assert mode_configs(flat) == {}
    assert flatten_mode(flat) == flat
    assert flatten_mode(flat, "nrt_drainage") == flat


def test_flatten_mode_falls_back_to_a_real_mode(tmp_path):
    """An unknown or absent mode key resolves to default_mode, not a crash."""
    config = _load(_multi_mode_config(tmp_path))

    assert flatten_mode(config, "bogus")["viz_configuration"] == "drainage_year"
    assert flatten_mode(config, None)["viz_configuration"] == "drainage_year"

    del config["default_mode"]
    assert flatten_mode(config)["viz_configuration"] == "drainage_year"


def test_missing_config_file_is_fatal(tmp_path):
    """An explicitly named config that isn't there stops the launch (not a silent fixture fallback)."""
    import pytest

    from water_timeseries.utils.cli import load_required_config

    _write_config(tmp_path / "real.yaml", vector_file="a.parquet")

    with pytest.raises(SystemExit):
        load_required_config(tmp_path / "gone.yaml", modes_mod.logger)

    # An empty/unparseable config is just as useless as a missing one.
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_required_config(tmp_path / "empty.yaml", modes_mod.logger)

    # No config at all stays valid -- that's "use the defaults".
    assert load_required_config(None, modes_mod.logger) == {}
    assert load_required_config(tmp_path / "real.yaml", modes_mod.logger)["vector_file"] == "a.parquet"


def test_shipped_config_shares_one_base_archive_between_modes():
    """Both modes render from the same base tiles; only their overlays differ.

    NRT mode could not share it until the per-month tilesets grew their
    ``scored`` layer: the shared archive answers with the historical area
    columns, so before that a non-drained lake in NRT mode hovered numbers from
    the wrong mode. With the layer in place the month answers for every lake it
    scored, and the base archive is just geometry underneath.
    """
    config = modes_mod.load_config(modes_mod._resolve(modes_mod._DEFAULT_CONFIG), modes_mod.logger)
    settings = {key: flatten_mode(config, key) for key in mode_configs(config)}

    base_archives = {key: values["pmtiles_file"] for key, values in settings.items()}
    assert len(set(base_archives.values())) == 1, f"one shared base archive, got {base_archives}"

    # Sharing the base is only safe while the drained overlays stay per-mode --
    # they are what carry the per-year and per-month values.
    assert settings["nrt_drainage"]["nrt_pmtiles_dir"]
    assert settings["drainage_year"]["drained_pmtiles_file"]
