"""Task Service Module.

Connects task routing with model selection, model lifecycle management,
and model inference execution.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from backend.app.model_ops.model_orchestrator import ModelOrchestrator
from backend.app.router.task_router import classify_task


def process_task(
    prompt: str,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Process a user prompt end-to-end.

    The request is classified, the appropriate model is selected,
    and the model is prepared for execution.

    Args:
        prompt: The raw user prompt.
        config_path: Optional custom models.yaml path.

    Returns:
        Dictionary containing category, response, model name,
        and backend.
    """

    service = TaskService(config_path=config_path)
    return service.execute(prompt)


class TaskService:
    """Service for end-to-end task routing and model execution."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        orchestrator: Optional[ModelOrchestrator] = None,
    ) -> None:
        """Initialize the task service.

        Args:
            config_path: Optional custom path to models.yaml.
            orchestrator: Optional model orchestrator for testing.
        """
        self.config_path = config_path

        if orchestrator is not None:
            self.orchestrator = orchestrator
        else:
            self.orchestrator = ModelOrchestrator()

    def execute(self, prompt: str) -> Dict[str, Any]:
        """Classify, prepare the correct model, and generate a response."""

        # Step 1: Classify the request.
        category = classify_task(prompt)

        # Step 2: Select and prepare the appropriate model.
        selection = self.orchestrator.prepare_model(category)

        model_name = selection["model_name"]
        backend = selection["backend"]

        # Step 3: Get the active model.
        model = self.orchestrator.manager.get_active_model()

        # Step 4: Generate the response.
        response_text = model.generate(prompt)

        # Step 5: Return the result.
        return {
            "category": category,
            "response": response_text,
            "model_name": model_name,
            "backend": backend,
        }