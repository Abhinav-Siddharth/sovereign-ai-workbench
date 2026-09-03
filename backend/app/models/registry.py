"""Model registry for Sovereign."""

from pathlib import Path
from typing import Any, Dict

import yaml

from backend.app.models.local_llm import LocalLLM
from backend.app.models.mock import MockModel
from backend.app.multimodal.vision import QwenVisionModel


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "models.yaml"
)


class ModelRegistry:
    """Load model configurations and create model instances."""

    def __init__(self, config_path=None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.configs = self.load_configs()

    def load_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load model configurations from YAML."""

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Model configuration file not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        return data.get("models", {})

    def get_config(self, category: str) -> Dict[str, Any]:
        """Return configuration for a category.

        Unknown categories fall back to the general model.
        """

        if category in self.configs:
            return self.configs[category]

        if "general" in self.configs:
            return self.configs["general"]

        raise KeyError(
            f"Category '{category}' not found and no general model exists."
        )

    def get_model_config(self, category: str) -> Dict[str, Any]:
        """Return model configuration for a category."""

        return self.get_config(category)

    def get_required_vram(self, category: str) -> int:
        """Return required VRAM in MiB for a category."""

        config = self.get_config(category)

        return int(config.get("required_vram_mib", 0))

    def get_model(self, category: str):
        """Create the appropriate model implementation."""

        config = self.get_config(category)

        model_type = config.get("type", "mock")
        model_name = config.get("model_name", "unknown")
        backend = config.get("backend", "unknown")

        if model_type == "local":
            return LocalLLM(
                model_name=model_name,
                backend=backend,
                config=config,
            )

        if model_type == "vision":
            return QwenVisionModel(
                model_name=model_name,
                backend=backend,
                config=config,
            )

        return MockModel(
            model_name=model_name,
            backend=backend,
            config=config,
        )


_default_registry = ModelRegistry()


def load_model_configs() -> Dict[str, Dict[str, Any]]:
    """Load model configurations using the default registry."""

    return _default_registry.configs


def get_model_config(category: str) -> Dict[str, Any]:
    """Get model configuration using the default registry."""

    return _default_registry.get_model_config(category)


def get_model_for_category(category: str):
    """Get a model instance for a category."""

    return _default_registry.get_model(category)