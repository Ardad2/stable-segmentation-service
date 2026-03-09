"""Tests for GET /api/v1/capabilities."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_capabilities_returns_200(client: TestClient) -> None:
    resp = client.get("/api/v1/capabilities")
    assert resp.status_code == 200


def test_capabilities_schema(client: TestClient) -> None:
    data = client.get("/api/v1/capabilities").json()
    assert "backend" in data
    assert "supported_input_types" in data
    assert "supported_prompt_types" in data
    assert isinstance(data["max_image_width"], int)
    assert isinstance(data["max_image_height"], int)


def test_capabilities_supported_prompts(client: TestClient) -> None:
    data = client.get("/api/v1/capabilities").json()
    for pt in ("point", "box", "text"):
        assert pt in data["supported_prompt_types"]
