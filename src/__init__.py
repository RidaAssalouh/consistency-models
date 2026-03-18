"""Top-level consistency package blueprint."""

from .config import ConsistencyConfig, load_config, save_config

__all__ = ["ConsistencyConfig", "load_config", "save_config"]
