"""Local LLM implementation for Sovereign offline inference via local Ollama."""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from backend.app.models.base import BaseModel

# Default endpoint for locally hosted Ollama service
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL_NAME = "llama3.2:latest"
DEFAULT_TIMEOUT_SECONDS = 60


class LocalLLM(BaseModel):
    """Local LLM runner for air-gapped on-premise model execution via Ollama.

    Inherits from BaseModel and interfaces with a local Ollama instance running
    on localhost (http://localhost:11434/api/generate) without contacting any
    cloud or external network services.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        backend: str = "ollama",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the LocalLLM instance.

        Args:
            model_name: The identifier of the local model (e.g. 'llama3.2:latest', 'granite').
            backend: The local inference runtime engine (defaults to 'ollama').
            config: Optional configuration dictionary containing inference settings,
                    such as 'api_url', 'model_name', or 'timeout'.
        """
        super().__init__(model_name=model_name, backend=backend, config=config)

    def generate(self, prompt: str) -> str:
        """Generate a response using the local Ollama HTTP API.

        Sends a non-streaming POST request with JSON payload to the local Ollama
        endpoint on localhost and extracts the generated text response.

        Args:
            prompt: The input prompt text for the model.

        Returns:
            The generated text string from Ollama.

        Raises:
            RuntimeError: If connection to local Ollama fails or an HTTP error occurs.
            ValueError: If the response cannot be parsed or lacks the 'response' field.
        """
        # Determine model name from config, fallback to self.model_name, then default
        model = self.config.get("model_name") or self.model_name or DEFAULT_MODEL_NAME

        # API URL and timeout settings (with defaults for local operation)
        api_url = self.config.get("api_url", DEFAULT_OLLAMA_ENDPOINT)
        timeout = self.config.get("timeout", DEFAULT_TIMEOUT_SECONDS)

        # Prepare request payload for Ollama /api/generate
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        # Encode JSON data and set headers
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        req = urllib.request.Request(
            url=api_url,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status_code = response.getcode()
                response_body = response.read().decode("utf-8")

                if status_code != 200:
                    raise RuntimeError(
                        f"Local Ollama API returned unexpected status code {status_code}: {response_body}"
                    )

                parsed_response = json.loads(response_body)

                if "response" not in parsed_response:
                    raise ValueError(
                        f"Invalid response from Ollama API: missing 'response' field in {parsed_response}"
                    )

                return parsed_response["response"]

        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to communicate with local Ollama API at {api_url}. "
                f"Ensure Ollama is running locally. Error: {e}"
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to decode JSON response from Ollama API: {e}"
            ) from e
