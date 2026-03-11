# imports
import json
from pathlib import Path
from typing import Optional

import yaml


def load_config(config_path: Optional[Path], logger) -> dict:
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
    except Exception as e:
        logger.warning(f"Failed to load config file {config_path}: {e}")
    return {}


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
