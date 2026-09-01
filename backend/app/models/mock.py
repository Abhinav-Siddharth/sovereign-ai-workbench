"""Mock model implementation for development, testing, and offline verification."""

from typing import Any, Dict, Optional

from backend.app.models.base import BaseModel


class MockModel(BaseModel):
    """Mock model that returns simulated responses without executing real AI weights."""

    def __init__(
        self,
        model_name: str = "mock-model",
        backend: str = "llama.cpp",
        config: Optional[Dict[str, Any]] = None,
        canned_response: Optional[str] = None,
    ) -> None:
        """Initialize mock model.

        Args:
            model_name: Name of the simulated model.
            backend: Name of the simulated backend engine.
            config: Optional configuration dictionary.
            canned_response: Optional static response to return for all generate calls.
        """
        super().__init__(model_name=model_name, backend=backend, config=config)
        self.canned_response = canned_response

    def generate(self, prompt: str) -> str:
        """Return a simulated generated response.

        Args:
            prompt: Input prompt text.

        Returns:
            A deterministic mock string showing the model name, backend, and prompt.
        """
        if self.canned_response is not None:
            return self.canned_response

        return f"[{self.model_name} via {self.backend}] Generated response for: '{prompt}'"
