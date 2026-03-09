"""Tests for GET /api/v1/health."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200


def test_health_response_schema(client: TestClient) -> None:
    data = client.get("/api/v1/health").json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "backend" in data


def test_health_backend_is_mock(client: TestClient) -> None:
    data = client.get("/api/v1/health").json()
    assert data["backend"] == "mock"
