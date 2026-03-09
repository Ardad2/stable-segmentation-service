"""HTTP-level integration tests for /api/v1/* when the CLIPSeg backend is active.

No CLIPSeg library or model weights are needed. A CLIPSegSegmentationAdapter
is pre-loaded with a mock _infer method, then get_adapter is patched at every
import site so that the FastAPI routes transparently use it.

This suite also verifies that the API contract is entirely backend-agnostic:
all three endpoints must work correctly regardless of which adapter is active.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from segmentation_service.adapters.clipseg_adapter import CLIPSegSegmentationAdapter

# Minimal valid 1×1 PNG (base64).
_STUB_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

_STUB_MASK = np.ones((64, 64), dtype=bool)
_STUB_LOGITS = np.ones((64, 64), dtype=np.float32) * 3.0
_STUB_SCORE = 0.95


# ---------------------------------------------------------------------------
# Module-scoped fixture: one TestClient with a mocked CLIPSeg adapter.
# get_adapter is patched at every import site used by the API routes.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clipseg_client() -> TestClient:
    """Return a TestClient whose routes all use a mocked CLIPSeg adapter."""
    from segmentation_service.main import create_app

    adapter = CLIPSegSegmentationAdapter()
    # Skip model loading; patch _infer to return deterministic results.
    adapter._processor = MagicMock()
    adapter._model = MagicMock()

    patches = [
        patch("segmentation_service.api.v1.health.get_adapter", return_value=adapter),
        patch("segmentation_service.api.v1.capabilities.get_adapter", return_value=adapter),
        patch("segmentation_service.api.v1.segment.get_adapter", return_value=adapter),
    ]
    with patches[0], patches[1], patches[2]:
        app = create_app()
        with TestClient(app) as client:
            # Patch _infer on the adapter for the lifetime of the client
            with patch.object(
                adapter,
                "_infer",
                return_value=(_STUB_MASK, _STUB_LOGITS, _STUB_SCORE),
            ):
                yield client


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------

def test_health_returns_200(clipseg_client: TestClient) -> None:
    assert clipseg_client.get("/api/v1/health").status_code == 200


def test_health_reports_clipseg_backend(clipseg_client: TestClient) -> None:
    data = clipseg_client.get("/api/v1/health").json()
    assert data["backend"] == "clipseg"
    assert data["status"] == "ok"


def test_health_schema_fields(clipseg_client: TestClient) -> None:
    data = clipseg_client.get("/api/v1/health").json()
    for field in ("status", "version", "backend"):
        assert field in data


# ---------------------------------------------------------------------------
# GET /api/v1/capabilities
# ---------------------------------------------------------------------------

def test_capabilities_returns_200(clipseg_client: TestClient) -> None:
    assert clipseg_client.get("/api/v1/capabilities").status_code == 200


def test_capabilities_backend_is_clipseg(clipseg_client: TestClient) -> None:
    data = clipseg_client.get("/api/v1/capabilities").json()
    assert data["backend"] == "clipseg"


def test_capabilities_supports_text_prompt(clipseg_client: TestClient) -> None:
    data = clipseg_client.get("/api/v1/capabilities").json()
    assert "text" in data["supported_prompt_types"]


def test_capabilities_excludes_point_and_box(clipseg_client: TestClient) -> None:
    data = clipseg_client.get("/api/v1/capabilities").json()
    prompts = data["supported_prompt_types"]
    assert "point" not in prompts
    assert "box" not in prompts


def test_capabilities_schema_complete(clipseg_client: TestClient) -> None:
    data = clipseg_client.get("/api/v1/capabilities").json()
    for field in (
        "backend",
        "supported_input_types",
        "supported_prompt_types",
        "max_image_width",
        "max_image_height",
    ):
        assert field in data


# ---------------------------------------------------------------------------
# POST /api/v1/segment — text prompt (primary use case)
# ---------------------------------------------------------------------------

def test_segment_text_returns_200(clipseg_client: TestClient) -> None:
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "text",
        "text_prompt": "a cat sitting on a chair",
    }
    assert clipseg_client.post("/api/v1/segment", json=payload).status_code == 200


def test_segment_text_response_schema(clipseg_client: TestClient) -> None:
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "text",
        "text_prompt": "the dog",
    }
    data = clipseg_client.post("/api/v1/segment", json=payload).json()
    assert data["backend"] == "clipseg"
    assert isinstance(data["masks"], list)
    assert len(data["masks"]) == 1
    mask = data["masks"][0]
    assert "mask_b64" in mask
    assert "score" in mask
    assert "area" in mask
    assert 0.0 <= mask["score"] <= 1.0


def test_segment_text_score_matches_mock(clipseg_client: TestClient) -> None:
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "text",
        "text_prompt": "water",
    }
    data = clipseg_client.post("/api/v1/segment", json=payload).json()
    assert data["masks"][0]["score"] == pytest.approx(_STUB_SCORE)


def test_segment_text_metadata_has_prompt(clipseg_client: TestClient) -> None:
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "text",
        "text_prompt": "red bicycle",
    }
    data = clipseg_client.post("/api/v1/segment", json=payload).json()
    assert data["metadata"].get("text_prompt") == "red bicycle"


def test_segment_text_missing_prompt_returns_422(clipseg_client: TestClient) -> None:
    """text_prompt field is required when prompt_type=text."""
    payload = {"image": _STUB_IMAGE, "prompt_type": "text"}
    assert clipseg_client.post("/api/v1/segment", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/segment — unsupported prompt types (point / box)
# ---------------------------------------------------------------------------

def test_segment_point_prompt_returns_500(clipseg_client: TestClient) -> None:
    """Point prompts are rejected by the CLIPSeg adapter → HTTP 500.

    Clients should check /capabilities before sending requests.
    """
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "point",
        "points": [{"x": 100, "y": 100, "label": 1}],
    }
    assert clipseg_client.post("/api/v1/segment", json=payload).status_code == 500


def test_segment_box_prompt_returns_500(clipseg_client: TestClient) -> None:
    """Box prompts are rejected by the CLIPSeg adapter → HTTP 500."""
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "box",
        "box": {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100},
    }
    assert clipseg_client.post("/api/v1/segment", json=payload).status_code == 500


# ---------------------------------------------------------------------------
# POST /api/v1/segment — logits
# ---------------------------------------------------------------------------

def test_segment_return_logits_true(clipseg_client: TestClient) -> None:
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "text",
        "text_prompt": "sky",
        "return_logits": True,
    }
    data = clipseg_client.post("/api/v1/segment", json=payload).json()
    assert data["masks"][0]["logits_b64"] is not None


def test_segment_return_logits_false_by_default(clipseg_client: TestClient) -> None:
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "text",
        "text_prompt": "sky",
    }
    data = clipseg_client.post("/api/v1/segment", json=payload).json()
    assert data["masks"][0]["logits_b64"] is None


# ---------------------------------------------------------------------------
# Verify existing contracts are unaffected
# ---------------------------------------------------------------------------

def test_invalid_image_format_still_returns_422(clipseg_client: TestClient) -> None:
    payload = {
        "image": _STUB_IMAGE,
        "image_format": "bmp",    # unsupported — validated at schema level
        "prompt_type": "text",
        "text_prompt": "sky",
    }
    assert clipseg_client.post("/api/v1/segment", json=payload).status_code == 422
