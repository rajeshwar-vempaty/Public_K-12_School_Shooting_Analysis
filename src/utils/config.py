"""
Configuration Management Module
Handles loading and accessing configuration from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


class Config:
    """Configuration manager for the application."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        # Load environment variables
        load_dotenv()

        # Determine config file path
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)

            # Replace environment variable placeholders
            self._replace_env_vars(self._config)

        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")

    def _replace_env_vars(self, config: Any) -> None:
        """
        Recursively replace environment variable placeholders in config.

        Args:
            config: Configuration dictionary or value
        """
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    config[key] = os.getenv(env_var, value)
                elif isinstance(value, (dict, list)):
                    self._replace_env_vars(value)
        elif isinstance(config, list):
            for i, item in enumerate(config):
                if isinstance(item, str) and item.startswith("${") and item.endswith("}"):
                    env_var = item[2:-1]
                    config[i] = os.getenv(env_var, item)
                elif isinstance(item, (dict, list)):
                    self._replace_env_vars(item)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., 'model.best_model.kernel')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., 'model.random_state')
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get_all(self) -> Dict[str, Any]:
        """
        Get entire configuration dictionary.

        Returns:
            Complete configuration
        """
        return self._config.copy()

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()


# Global configuration instance
_global_config: Optional[Config] = None


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Config instance
    """
    global _global_config
    _global_config = Config(config_path)
    return _global_config


def get_config() -> Config:
    """
    Get global configuration instance.

    Returns:
        Config instance

    Raises:
        RuntimeError: If configuration not loaded
    """
    global _global_config

    if _global_config is None:
        _global_config = load_config()

    return _global_config
