"""Model Registry Module.

Loads model configurations from YAML and provides model instances based on task category.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

from backend.app.models.base import BaseModel
from backend.app.models.mock import MockModel

# Default path to the models configuration file
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "models.yaml"


def _parse_yaml_fallback(file_content: str) -> Dict[str, Any]:
    """Simple fallback parser for basic nested key-value mappings when PyYAML is not yet installed."""
    data: Dict[str, Any] = {"models": {}}
    current_category = None
    for line in file_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "models:":
            continue
        if line.endswith(":") and not any(line.startswith(k) for k in ("model_name", "backend", "category")):
            current_category = line[:-1].strip()
            data["models"][current_category] = {}
        elif ":" in line and current_category:
            key, val = line.split(":", 1)
            data["models"][current_category][key.strip()] = val.strip()
    return data


class ModelRegistry:
    """Manages model configurations and instantiation."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._configs: Optional[Dict[str, Any]] = None

    def load_configs(self) -> Dict[str, Any]:
        """Load and parse the models configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Model configuration file not found at: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            content = f.read()

        if yaml is not None:
            parsed = yaml.safe_load(content)
        else:
            parsed = _parse_yaml_fallback(content)

        if not parsed or "models" not in parsed:
            raise ValueError("Invalid configuration format: missing 'models' root key.")

        self._configs = parsed["models"]
        return self._configs

    def get_config(self, category: str) -> Dict[str, Any]:
        """Retrieve model configuration dictionary for a given task category."""
        if self._configs is None:
            self.load_configs()

        assert self._configs is not None
        if category not in self._configs:
            # Fallback to general category if category not found
            if "general" in self._configs:
                return self._configs["general"]
            raise KeyError(f"Task category '{category}' not found in configuration.")

        return self._configs[category]

    def get_model(self, category: str) -> BaseModel:
        """Return a model instance for the given task category.

        For Milestone 3, returns a MockModel initialized with the configured
        model name and backend.
        """
        config = self.get_config(category)
        return MockModel(
            model_name=config.get("model_name", "unknown-model"),
            backend=config.get("backend", "llama.cpp"),
            config=config,
        )


# Global singleton helper instance
_default_registry = ModelRegistry()


def load_model_configs(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Convenience function to load model configurations."""
    registry = ModelRegistry(config_path) if config_path else _default_registry
    return registry.load_configs()


def get_model_config(category: str, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Convenience function to retrieve configuration for a category."""
    registry = ModelRegistry(config_path) if config_path else _default_registry
    return registry.get_config(category)


def get_model_for_category(category: str, config_path: Optional[Union[str, Path]] = None) -> BaseModel:
    """Convenience function to get a model instance for a category."""
    registry = ModelRegistry(config_path) if config_path else _default_registry
    return registry.get_model(category)
