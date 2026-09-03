from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


class FakeTaskService:
    """Fake TaskService for API testing."""

    def execute(self, prompt):
        return {
            "category": "general",
            "response": f"Fake response for: {prompt}",
            "model_name": "phi-4-mini",
            "backend": "llama.cpp",
        }


def test_health_endpoint():
    """Verify the health endpoint."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "sovereign"


def test_task_endpoint(monkeypatch):
    """Verify that the task endpoint accepts a request."""

    monkeypatch.setattr(
        "backend.app.main.get_task_service",
        lambda: FakeTaskService(),
    )

    response = client.post(
        "/task",
        json={
            "prompt": "Hello, how are you?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["category"] == "general"
    assert data["model_name"] == "phi-4-mini"
    assert data["backend"] == "llama.cpp"
    assert "response" in data
    assert "Hello, how are you?" in data["response"]