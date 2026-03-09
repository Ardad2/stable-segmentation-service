"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from segmentation_service.main import create_app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Return a synchronous TestClient backed by the FastAPI app."""
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def point_payload() -> dict:
    return {
        "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "image_format": "png",
        "prompt_type": "point",
        "points": [{"x": 10, "y": 20, "label": 1}],
    }


@pytest.fixture
def box_payload() -> dict:
    return {
        "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "image_format": "png",
        "prompt_type": "box",
        "box": {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100},
    }
