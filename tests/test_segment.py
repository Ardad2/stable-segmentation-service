"""Tests for POST /api/v1/segment."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_segment_point_prompt(client: TestClient, point_payload: dict) -> None:
    resp = client.post("/api/v1/segment", json=point_payload)
    assert resp.status_code == 200


def test_segment_response_schema(client: TestClient, point_payload: dict) -> None:
    data = client.post("/api/v1/segment", json=point_payload).json()
    assert "request_id" in data
    assert "backend" in data
    assert isinstance(data["masks"], list)
    assert len(data["masks"]) >= 1
    assert "latency_ms" in data


def test_segment_mask_fields(client: TestClient, point_payload: dict) -> None:
    data = client.post("/api/v1/segment", json=point_payload).json()
    mask = data["masks"][0]
    assert "mask_b64" in mask
    assert "score" in mask
    assert 0.0 <= mask["score"] <= 1.0
    assert "area" in mask


def test_segment_box_prompt(client: TestClient, box_payload: dict) -> None:
    resp = client.post("/api/v1/segment", json=box_payload)
    assert resp.status_code == 200


def test_segment_multimask_output(client: TestClient, point_payload: dict) -> None:
    payload = {**point_payload, "multimask_output": True}
    data = client.post("/api/v1/segment", json=payload).json()
    assert len(data["masks"]) == 3


def test_segment_missing_point_fails(client: TestClient) -> None:
    payload = {
        "image": "abc",
        "prompt_type": "point",
        # no 'points' key
    }
    resp = client.post("/api/v1/segment", json=payload)
    assert resp.status_code == 422


def test_segment_missing_box_fails(client: TestClient) -> None:
    payload = {
        "image": "abc",
        "prompt_type": "box",
        # no 'box' key
    }
    resp = client.post("/api/v1/segment", json=payload)
    assert resp.status_code == 422


def test_segment_invalid_image_format(client: TestClient, point_payload: dict) -> None:
    payload = {**point_payload, "image_format": "bmp"}
    resp = client.post("/api/v1/segment", json=payload)
    assert resp.status_code == 422


def test_segment_return_logits(client: TestClient, point_payload: dict) -> None:
    payload = {**point_payload, "return_logits": True}
    data = client.post("/api/v1/segment", json=payload).json()
    assert data["masks"][0]["logits_b64"] is not None
