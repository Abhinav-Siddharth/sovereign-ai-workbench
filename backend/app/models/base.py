"""Base model interfaces for Sovereign local AI engines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModel(ABC):
    """Abstract Base Class for all AI model implementations in Sovereign.

    Every model backend (whether mock, llama.cpp, vLLM, etc.) must implement
    this interface to ensure a uniform contract across the application.
    """

    def __init__(
        self,
        model_name: str,
        backend: str = "llama.cpp",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize base model attributes.

        Args:
            model_name: Name or identifier of the model (e.g. 'granite', 'qwen-vl').
            backend: The execution runtime or engine (e.g. 'llama.cpp').
            config: Additional optional model parameters or settings.
        """
        self.model_name = model_name
        self.backend = backend
        self.config = config or {}

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response text from the given prompt.

        Args:
            prompt: Input text prompt.

        Returns:
            The generated response string.
        """
        pass
