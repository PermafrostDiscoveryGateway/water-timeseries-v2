# imports
import json
from pathlib import Path

import yaml


def load_config(config_path: Path | None, logger) -> dict:
    """Load configuration from YAML or JSON file.

    Args:
        config_path: Path to config file.
        logger: Logger instance.

    Returns:
        Dictionary with configuration values.
    """
    if not config_path or not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            if config_path.suffix in (".yaml", ".yml"):
                return yaml.safe_load(f) or {}
            elif config_path.suffix == ".json":
                return json.load(f)
    except (OSError, ValueError, yaml.YAMLError) as e:
        logger.warning(f"Failed to load config file {config_path}: {e}")
    return {}


#: Key holding the per-mode setting overlays in a multi-mode dashboard config.
MODES_KEY = "modes"

#: Key naming which mode a multi-mode config launches on.
DEFAULT_MODE_KEY = "default_mode"

#: Per-mode key that labels the mode in the UI rather than configuring data.
MODE_LABEL_KEY = "label"


def mode_configs(config: dict) -> dict:
    """Per-mode setting overlays from a config, or ``{}`` if it isn't multi-mode.

    A multi-mode config keeps everything the modes share at the top level and
    nests only the differences under ``modes:``::

        ee_project: pdg-project-406720   # shared by every mode
        default_mode: drainage_year
        modes:
          drainage_year: {label: Historical, vector_file: ..., ...}
          nrt_drainage: {label: Near Real-Time, vector_file: ..., ...}

    A flat single-mode config has no ``modes:`` key and is returned as-is by
    :func:`flatten_mode`, so both layouts stay launchable.
    """
    modes = config.get(MODES_KEY)
    if not isinstance(modes, dict):
        return {}
    return {key: values for key, values in modes.items() if isinstance(values, dict)}


def flatten_mode(config: dict, mode_key: str | None = None) -> dict:
    """Collapse a multi-mode config down to flat settings for one mode.

    Shared top-level keys form the base and the mode's own keys override them,
    which is the same precedence a reader would assume from the indentation.

    Args:
        config: Loaded config, multi-mode or flat.
        mode_key: Mode to select. Defaults to ``default_mode``, then to the
            first mode defined.

    Returns:
        Flat settings dict, with ``modes``/``default_mode``/``label`` removed.
    """
    modes = mode_configs(config)
    if not modes:
        return dict(config)

    key = mode_key if mode_key in modes else config.get(DEFAULT_MODE_KEY)
    if key not in modes:
        key = next(iter(modes))

    flat = {k: v for k, v in config.items() if k not in (MODES_KEY, DEFAULT_MODE_KEY)}
    flat.update(modes[key])
    flat.pop(MODE_LABEL_KEY, None)
    return flat


def load_required_config(config_path: Path | str | None, logger) -> dict:
    """Load a config the caller explicitly asked for, failing if it isn't usable.

    :func:`load_config` is deliberately lenient -- mode discovery probes paths
    that may legitimately be absent and wants ``{}`` back. An explicit
    ``--config-file`` is the opposite: silently returning ``{}`` there drops the
    dashboard onto the test-data fixture with the default styling, which reads
    as "my data vanished" rather than "that path is wrong".

    Args:
        config_path: Path the user named, or None for "no config".
        logger: Logger instance.

    Returns:
        The loaded config, or ``{}`` when ``config_path`` is None.

    Raises:
        SystemExit: If the named file is missing or contains no settings.
    """
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        siblings = sorted(p.name for p in path.parent.glob("*.y*ml")) if path.parent.is_dir() else []
        hint = f" Available in {path.parent}/: {', '.join(siblings)}" if siblings else ""
        logger.error(f"Config file not found: {path}.{hint}")
        raise SystemExit(1)

    config = load_config(path, logger)
    if not config:
        logger.error(f"Config file {path} could not be parsed, or holds no settings.")
        raise SystemExit(1)
    return config


def merge_config_with_args(config: dict, **kwargs) -> dict:
    """Merge config with CLI args, CLI args take priority.

    Args:
        config: Configuration dictionary from config file.
        **kwargs: CLI arguments (None values are ignored).

    Returns:
        Merged dictionary with CLI args taking priority.
    """
    result = config.copy()
    for key, value in kwargs.items():
        if value is not None:
            result[key] = value
    return result
