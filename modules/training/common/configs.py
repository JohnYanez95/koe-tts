"""
Training configuration utilities.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load training configuration from YAML.

    Args:
        config_path: Path to config file

    Returns:
        Config dict
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """
    Recursively merge override config into base.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result
