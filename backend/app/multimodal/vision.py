"""Vision model interface for the Sovereign multimodal pipeline."""

from pathlib import Path
from typing import Any


class VisionModel:
    """Base interface for a local vision-language model."""

    name = "vision_model"

    def process(self, image_path: str) -> str:
        """Process an image and return extracted information."""
        raise NotImplementedError


class MockVisionModel(VisionModel):
    """Mock vision model for local development."""

    name = "mock_qwen_vl"

    def process(self, image_path: str) -> str:
        """Return deterministic output for testing."""

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        return (
            f"[Mock Vision] Visual information extracted "
            f"from {path.name}"
        )