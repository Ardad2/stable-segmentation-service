"""Versioning compatibility test matrix.

Tests the full grid of client version × server endpoint version combinations.
Each test is labelled with its expected outcome so the matrix is immediately
readable in pytest output.

Compatibility matrix
--------------------

| Client payload  | Endpoint       | Expected  | Reason                          |
|-----------------|----------------|-----------|--------------------------------|
| v1 flat         | /api/v1/...    | PASS      | Exact match                     |
| v1 flat (+ api_version field) | /api/v1/... | PASS | Additive; field present in resp |
| v2 prompt envelope | /api/v2/...  | PASS     | Exact match                     |
| v1 flat         | /api/v2/segment| FAIL 422  | 'prompt' field missing          |
| v2 envelope     | /api/v1/segment| FAIL 422  | 'prompt_type'='point', no points|
| mock v1         | /api/v1        | PASS      | Same schema, different backend  |
| mock v2         | /api/v2        | PASS      | Same schema, different backend  |

Additional tests:
- api_version field values: v1 responses carry "1.0", v2 responses carry "2.0"
- V2 mask field: responses use "mask_data" not "mask_b64"
- Prompt type routing works for point, box, text via v2 envelope

Design notes
------------
All fixtures are function-scoped (no scope="module") to prevent patch
interactions between test cases.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from segmentation_service.main import create_app

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Minimal 1×1 PNG, base64-encoded — fast and deterministic.
_IMG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient backed by the mock backend (default)."""
    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# v1 payloads (flat layout)
# ---------------------------------------------------------------------------

@pytest.fixture
def v1_point_payload() -> dict:
    return {
        "image": _IMG,
        "image_format": "png",
        "prompt_type": "point",
        "points": [{"x": 0, "y": 0, "label": 1}],
    }


@pytest.fixture
def v1_box_payload() -> dict:
    return {
        "image": _IMG,
        "image_format": "png",
        "prompt_type": "box",
        "box": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
    }


@pytest.fixture
def v1_text_payload() -> dict:
    return {
        "image": _IMG,
        "image_format": "png",
        "prompt_type": "text",
        "text_prompt": "the object",
    }


# ---------------------------------------------------------------------------
# v2 payloads (nested prompt envelope)
# ---------------------------------------------------------------------------

@pytest.fixture
def v2_point_payload() -> dict:
    return {
        "image": _IMG,
        "image_format": "png",
        "prompt": {"type": "point", "points": [{"x": 0, "y": 0, "label": 1}]},
    }


@pytest.fixture
def v2_box_payload() -> dict:
    return {
        "image": _IMG,
        "image_format": "png",
        "prompt": {"type": "box", "box": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}},
    }


@pytest.fixture
def v2_text_payload() -> dict:
    return {
        "image": _IMG,
        "image_format": "png",
        "prompt": {"type": "text", "text": "the object"},
    }


# ===========================================================================
# 1. v1 client → v1 server  (must PASS)
# ===========================================================================

class TestV1ClientV1Server:
    """v1 payload → /api/v1/* endpoints.  All must succeed."""

    def test_health_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_capabilities_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/capabilities")
        assert resp.status_code == 200

    def test_point_prompt_200(self, client: TestClient, v1_point_payload: dict) -> None:
        resp = client.post("/api/v1/segment", json=v1_point_payload)
        assert resp.status_code == 200

    def test_box_prompt_200(self, client: TestClient, v1_box_payload: dict) -> None:
        resp = client.post("/api/v1/segment", json=v1_box_payload)
        assert resp.status_code == 200

    def test_text_prompt_200(self, client: TestClient, v1_text_payload: dict) -> None:
        resp = client.post("/api/v1/segment", json=v1_text_payload)
        assert resp.status_code == 200

    def test_response_has_masks(self, client: TestClient, v1_point_payload: dict) -> None:
        data = client.post("/api/v1/segment", json=v1_point_payload).json()
        assert "masks" in data
        assert len(data["masks"]) >= 1

    def test_mask_has_mask_b64(self, client: TestClient, v1_point_payload: dict) -> None:
        """v1 mask field is mask_b64 — v1 clients depend on this."""
        data = client.post("/api/v1/segment", json=v1_point_payload).json()
        assert "mask_b64" in data["masks"][0]


# ===========================================================================
# 2. v1 additive backward-compatible change  (must PASS)
# ===========================================================================

class TestV1BackwardCompatibleAddition:
    """api_version field is now present in v1 responses.

    Old clients that do not read this field are unaffected.
    New clients can use it to verify the API contract version.
    """

    def test_health_has_api_version(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert "api_version" in data

    def test_health_api_version_is_1_0(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert data["api_version"] == "1.0"

    def test_capabilities_has_api_version(self, client: TestClient) -> None:
        data = client.get("/api/v1/capabilities").json()
        assert "api_version" in data

    def test_capabilities_api_version_is_1_0(self, client: TestClient) -> None:
        data = client.get("/api/v1/capabilities").json()
        assert data["api_version"] == "1.0"

    def test_segment_has_api_version(self, client: TestClient, v1_point_payload: dict) -> None:
        data = client.post("/api/v1/segment", json=v1_point_payload).json()
        assert "api_version" in data

    def test_segment_api_version_is_1_0(
        self, client: TestClient, v1_point_payload: dict
    ) -> None:
        data = client.post("/api/v1/segment", json=v1_point_payload).json()
        assert data["api_version"] == "1.0"

    def test_old_v1_client_fields_still_present(
        self, client: TestClient, v1_point_payload: dict
    ) -> None:
        """All pre-existing v1 fields are still present — nothing was removed."""
        data = client.post("/api/v1/segment", json=v1_point_payload).json()
        for field in ("request_id", "backend", "masks", "latency_ms", "metadata"):
            assert field in data, f"v1 field '{field}' is unexpectedly absent"


# ===========================================================================
# 3. v2 client → v2 server  (must PASS)
# ===========================================================================

class TestV2ClientV2Server:
    """v2 payload → /api/v2/* endpoints.  All must succeed."""

    def test_health_200(self, client: TestClient) -> None:
        resp = client.get("/api/v2/health")
        assert resp.status_code == 200

    def test_capabilities_200(self, client: TestClient) -> None:
        resp = client.get("/api/v2/capabilities")
        assert resp.status_code == 200

    def test_point_prompt_200(self, client: TestClient, v2_point_payload: dict) -> None:
        resp = client.post("/api/v2/segment", json=v2_point_payload)
        assert resp.status_code == 200

    def test_box_prompt_200(self, client: TestClient, v2_box_payload: dict) -> None:
        resp = client.post("/api/v2/segment", json=v2_box_payload)
        assert resp.status_code == 200

    def test_text_prompt_200(self, client: TestClient, v2_text_payload: dict) -> None:
        resp = client.post("/api/v2/segment", json=v2_text_payload)
        assert resp.status_code == 200

    def test_response_has_masks(self, client: TestClient, v2_point_payload: dict) -> None:
        data = client.post("/api/v2/segment", json=v2_point_payload).json()
        assert "masks" in data
        assert len(data["masks"]) >= 1

    def test_mask_uses_mask_data_not_mask_b64(
        self, client: TestClient, v2_point_payload: dict
    ) -> None:
        """v2 renames mask_b64 → mask_data.  Old field name must NOT appear."""
        data = client.post("/api/v2/segment", json=v2_point_payload).json()
        mask = data["masks"][0]
        assert "mask_data" in mask, "v2 mask field 'mask_data' is absent"
        assert "mask_b64" not in mask, "v1 mask field 'mask_b64' must not appear in v2"

    def test_v2_health_api_version(self, client: TestClient) -> None:
        data = client.get("/api/v2/health").json()
        assert data["api_version"] == "2.0"

    def test_v2_capabilities_api_version(self, client: TestClient) -> None:
        data = client.get("/api/v2/capabilities").json()
        assert data["api_version"] == "2.0"

    def test_v2_segment_api_version(
        self, client: TestClient, v2_point_payload: dict
    ) -> None:
        data = client.post("/api/v2/segment", json=v2_point_payload).json()
        assert data["api_version"] == "2.0"

    def test_v2_response_fields(self, client: TestClient, v2_point_payload: dict) -> None:
        data = client.post("/api/v2/segment", json=v2_point_payload).json()
        for field in ("request_id", "api_version", "backend", "masks", "latency_ms"):
            assert field in data, f"v2 response field '{field}' is absent"

    def test_multimask_output(self, client: TestClient, v2_point_payload: dict) -> None:
        payload = {**v2_point_payload, "multimask_output": True}
        data = client.post("/api/v2/segment", json=payload).json()
        assert len(data["masks"]) == 3

    def test_return_logits(self, client: TestClient, v2_point_payload: dict) -> None:
        payload = {**v2_point_payload, "return_logits": True}
        data = client.post("/api/v2/segment", json=payload).json()
        assert data["masks"][0]["logits_b64"] is not None


# ===========================================================================
# 4. v1 client → v2 server  (must FAIL clearly)
# ===========================================================================

class TestV1ClientV2Server:
    """A v1-style payload sent to the v2 endpoint must fail with 422.

    The v2 endpoint requires a nested 'prompt' field which the v1 payload
    does not include.  Pydantic rejects the request before any adapter
    code runs.
    """

    def test_v1_point_payload_to_v2_fails_422(
        self, client: TestClient, v1_point_payload: dict
    ) -> None:
        resp = client.post("/api/v2/segment", json=v1_point_payload)
        assert resp.status_code == 422, (
            f"Expected 422 for v1 payload on v2 endpoint, got {resp.status_code}"
        )

    def test_v1_box_payload_to_v2_fails_422(
        self, client: TestClient, v1_box_payload: dict
    ) -> None:
        resp = client.post("/api/v2/segment", json=v1_box_payload)
        assert resp.status_code == 422

    def test_v1_text_payload_to_v2_fails_422(
        self, client: TestClient, v1_text_payload: dict
    ) -> None:
        resp = client.post("/api/v2/segment", json=v1_text_payload)
        assert resp.status_code == 422

    def test_error_mentions_prompt_field(
        self, client: TestClient, v1_point_payload: dict
    ) -> None:
        """The 422 detail must reference the missing 'prompt' field."""
        body = client.post("/api/v2/segment", json=v1_point_payload).json()
        detail_str = str(body).lower()
        assert "prompt" in detail_str, (
            f"422 detail should mention 'prompt'; got: {body}"
        )

    def test_minimal_payload_no_prompt_fails(self, client: TestClient) -> None:
        """Even with a valid image, the missing prompt envelope is enough to fail."""
        resp = client.post("/api/v2/segment", json={"image": _IMG})
        assert resp.status_code == 422


# ===========================================================================
# 5. v2 client → v1 server  (must FAIL clearly)
# ===========================================================================

class TestV2ClientV1Server:
    """A v2-style payload sent to the v1 endpoint must also fail.

    Pydantic ignores the unknown 'prompt' extra field.  The v1 model defaults
    prompt_type to 'point' and points to None.  The v1 route guard then raises
    422 because prompt_type='point' but no points were supplied.
    """

    def test_v2_point_payload_to_v1_fails_422(
        self, client: TestClient, v2_point_payload: dict
    ) -> None:
        resp = client.post("/api/v1/segment", json=v2_point_payload)
        assert resp.status_code == 422, (
            f"Expected 422 for v2 payload on v1 endpoint, got {resp.status_code}"
        )

    def test_v2_box_payload_to_v1_fails_422(
        self, client: TestClient, v2_box_payload: dict
    ) -> None:
        resp = client.post("/api/v1/segment", json=v2_box_payload)
        assert resp.status_code == 422

    def test_v2_text_payload_to_v1_fails_422(
        self, client: TestClient, v2_text_payload: dict
    ) -> None:
        resp = client.post("/api/v1/segment", json=v2_text_payload)
        assert resp.status_code == 422


# ===========================================================================
# 6. API version × backend independence
# ===========================================================================

class TestAPIVersionBackendIndependence:
    """Changing the backend does not break the API version contract.

    Both v1 and v2 endpoints work correctly regardless of which backend is
    active, as long as the prompt type is supported.
    """

    def test_v1_mock_backend_version_is_1_0(self, client: TestClient) -> None:
        """Mock backend via v1 → api_version must be '1.0'."""
        data = client.get("/api/v1/health").json()
        assert data["backend"] == "mock"
        assert data["api_version"] == "1.0"

    def test_v2_mock_backend_version_is_2_0(self, client: TestClient) -> None:
        """Mock backend via v2 → api_version must be '2.0'."""
        data = client.get("/api/v2/health").json()
        assert data["backend"] == "mock"
        assert data["api_version"] == "2.0"

    def test_v1_and_v2_same_backend(self, client: TestClient) -> None:
        """Both versions report the same active backend."""
        v1_backend = client.get("/api/v1/health").json()["backend"]
        v2_backend = client.get("/api/v2/health").json()["backend"]
        assert v1_backend == v2_backend

    def test_v1_capabilities_and_v2_capabilities_same_prompts(
        self, client: TestClient
    ) -> None:
        """Supported prompt types are reported consistently across API versions."""
        v1_types = set(client.get("/api/v1/capabilities").json()["supported_prompt_types"])
        v2_types = set(client.get("/api/v2/capabilities").json()["supported_prompt_types"])
        assert v1_types == v2_types

    def test_v1_and_v2_produce_same_mask_content(
        self,
        client: TestClient,
        v1_point_payload: dict,
        v2_point_payload: dict,
    ) -> None:
        """Same underlying model → same mask bytes regardless of API version."""
        v1_data = client.post("/api/v1/segment", json=v1_point_payload).json()
        v2_data = client.post("/api/v2/segment", json=v2_point_payload).json()
        # v1 uses mask_b64, v2 uses mask_data — both carry the same PNG
        v1_mask = v1_data["masks"][0]["mask_b64"]
        v2_mask = v2_data["masks"][0]["mask_data"]
        assert v1_mask == v2_mask, "Same backend should produce identical mask bytes"


# ===========================================================================
# 7. Invalid prompt contents via v2 envelope (route-level guards)
# ===========================================================================

class TestV2PromptValidation:
    """The v2 route applies the same prompt content guards as v1.

    Sending a point envelope with no points, or a box envelope with no box,
    must still fail with 422.
    """

    def test_point_envelope_no_points_fails(self, client: TestClient) -> None:
        payload = {
            "image": _IMG,
            "prompt": {"type": "point"},  # points omitted
        }
        resp = client.post("/api/v2/segment", json=payload)
        assert resp.status_code == 422

    def test_box_envelope_no_box_fails(self, client: TestClient) -> None:
        payload = {
            "image": _IMG,
            "prompt": {"type": "box"},  # box omitted
        }
        resp = client.post("/api/v2/segment", json=payload)
        assert resp.status_code == 422

    def test_text_envelope_empty_text_fails(self, client: TestClient) -> None:
        payload = {
            "image": _IMG,
            "prompt": {"type": "text", "text": ""},  # empty text
        }
        resp = client.post("/api/v2/segment", json=payload)
        assert resp.status_code == 422

    def test_invalid_image_format_fails(self, client: TestClient) -> None:
        payload = {
            "image": _IMG,
            "image_format": "bmp",  # unsupported
            "prompt": {"type": "point", "points": [{"x": 0, "y": 0, "label": 1}]},
        }
        resp = client.post("/api/v2/segment", json=payload)
        assert resp.status_code == 422
