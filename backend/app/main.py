"""FastAPI entry point for Sovereign."""

from fastapi import FastAPI
from pydantic import BaseModel

from backend.app.services.task_service import TaskService


# Initialize the FastAPI application
app = FastAPI(
    title="Sovereign AI Workbench",
    description="Air-gapped AI workbench for confidential industrial work",
    version="0.1.0",
)


class TaskRequest(BaseModel):
    """Request body for a Sovereign task."""

    prompt: str


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    """Verify that the Sovereign backend is running."""

    return {
        "status": "ok",
        "service": "sovereign",
    }


# ---------------------------------------------------------
# Task endpoint
# ---------------------------------------------------------

def get_task_service() -> TaskService:
    """Create the default TaskService."""

    return TaskService()


@app.post("/task")
def execute_task(request: TaskRequest):
    """Execute a user task through the Sovereign pipeline."""

    service = get_task_service()

    result = service.execute(
        prompt=request.prompt,
    )

    return result