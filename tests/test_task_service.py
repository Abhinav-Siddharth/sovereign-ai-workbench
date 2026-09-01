"""Unit tests for the Task Service orchestration flow."""

import sys
from pathlib import Path

# Ensure project root is on sys.path for backend package imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.app.router.task_router import (
    CATEGORY_CODING,
    CATEGORY_DOCUMENT,
    CATEGORY_GENERAL,
    CATEGORY_VISION,
)
from backend.app.services.task_service import TaskService, process_task


def test_process_coding_task():
    """Verify coding prompt routes to 'coding' category and executes 'granite' mock model."""
    prompt = "write Python code for a binary search algorithm"
    result = process_task(prompt)

    assert result["category"] == CATEGORY_CODING
    assert result["model_name"] == "granite"
    assert result["backend"] == "llama.cpp"
    assert "granite" in result["response"]
    assert "llama.cpp" in result["response"]
    assert prompt in result["response"]


def test_process_vision_task():
    """Verify vision prompt routes to 'vision' category and executes 'qwen-vl' mock model."""
    prompt = "read this scanned report and extract the invoice table"
    result = process_task(prompt)

    assert result["category"] == CATEGORY_VISION
    assert result["model_name"] == "qwen-vl"
    assert result["backend"] == "llama.cpp"
    assert "qwen-vl" in result["response"]
    assert "llama.cpp" in result["response"]
    assert prompt in result["response"]


def test_process_document_task():
    """Verify document prompt routes to 'document' category and executes 'nemotron' mock model."""
    prompt = "summarize this SOP document for high-voltage maintenance"
    result = process_task(prompt)

    assert result["category"] == CATEGORY_DOCUMENT
    assert result["model_name"] == "nemotron"
    assert result["backend"] == "llama.cpp"
    assert "nemotron" in result["response"]
    assert "llama.cpp" in result["response"]
    assert prompt in result["response"]


def test_process_general_task():
    """Verify general prompt routes to 'general' category and executes 'phi-4-mini' mock model."""
    prompt = "What are three safety precautions when handling pressurized gas cylinders?"
    result = process_task(prompt)

    assert result["category"] == CATEGORY_GENERAL
    assert result["model_name"] == "phi-4-mini"
    assert result["backend"] == "llama.cpp"
    assert "phi-4-mini" in result["response"]
    assert "llama.cpp" in result["response"]
    assert prompt in result["response"]


def test_task_service_class_instance():
    """Verify TaskService class instance produces identical orchestration behavior."""
    service = TaskService()
    result = service.execute("debug this Java program with a NullPointerException")

    assert result["category"] == CATEGORY_CODING
    assert result["model_name"] == "granite"
    assert result["backend"] == "llama.cpp"
    assert "granite" in result["response"]


def test_edge_case_empty_prompt():
    """Verify empty prompt falls back to general category and handles generation gracefully."""
    result = process_task("")

    assert result["category"] == CATEGORY_GENERAL
    assert result["model_name"] == "phi-4-mini"
    assert "phi-4-mini" in result["response"]
