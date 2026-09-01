"""Unit tests for model configuration loading, registry, and mock model generation."""

import sys
from pathlib import Path

# Ensure project root is on sys.path for backend package imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.app.models.base import BaseModel
from backend.app.models.mock import MockModel
from backend.app.models.registry import (
    ModelRegistry,
    get_model_config,
    get_model_for_category,
    load_model_configs,
)


def test_load_model_configs():
    """Verify that models.yaml is successfully loaded and contains all 4 categories."""
    configs = load_model_configs()
    assert isinstance(configs, dict)
    assert "coding" in configs
    assert "vision" in configs
    assert "document" in configs
    assert "general" in configs


def test_category_configurations():
    """Verify each task category has the expected model_name and backend."""
    # Coding -> granite, llama.cpp
    coding_cfg = get_model_config("coding")
    assert coding_cfg["model_name"] == "granite"
    assert coding_cfg["backend"] == "llama.cpp"
    assert coding_cfg["category"] == "coding"

    # Vision -> qwen-vl, llama.cpp
    vision_cfg = get_model_config("vision")
    assert vision_cfg["model_name"] == "qwen-vl"
    assert vision_cfg["backend"] == "llama.cpp"
    assert vision_cfg["category"] == "vision"

    # Document -> nemotron, llama.cpp
    doc_cfg = get_model_config("document")
    assert doc_cfg["model_name"] == "nemotron"
    assert doc_cfg["backend"] == "llama.cpp"
    assert doc_cfg["category"] == "document"

    # General -> phi-4-mini, llama.cpp
    gen_cfg = get_model_config("general")
    assert gen_cfg["model_name"] == "phi-4-mini"
    assert gen_cfg["backend"] == "llama.cpp"
    assert gen_cfg["category"] == "general"


def test_mock_model_generation():
    """Verify mock model returns expected response text with metadata."""
    model = MockModel(model_name="test-llm", backend="llama.cpp")
    assert isinstance(model, BaseModel)

    prompt = "Write a function to sort a list."
    response = model.generate(prompt)

    assert isinstance(response, str)
    assert "test-llm" in response
    assert "llama.cpp" in response
    assert prompt in response


def test_get_model_for_category_and_generate():
    """Verify retrieving a model for each category produces appropriate mock output."""
    categories = {
        "coding": "granite",
        "vision": "qwen-vl",
        "document": "nemotron",
        "general": "phi-4-mini",
    }

    for category, expected_model_name in categories.items():
        model = get_model_for_category(category)
        assert isinstance(model, BaseModel)
        assert model.model_name == expected_model_name
        assert model.backend == "llama.cpp"

        output = model.generate(f"Test prompt for {category}")
        assert expected_model_name in output
        assert "llama.cpp" in output


def test_fallback_category():
    """Verify unknown category falls back gracefully to general model."""
    config = get_model_config("unrecognized_category")
    assert config["model_name"] == "phi-4-mini"
    assert config["backend"] == "llama.cpp"
