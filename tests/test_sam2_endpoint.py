"""HTTP-level integration tests for /api/v1/* when the SAM2 backend is active.

The SAM2 library and model weights are NOT needed.  We build an
SAM2SegmentationAdapter with a mock predictor injected, then patch
``get_adapter`` at every import site so the FastAPI routes use it.

All existing endpoint contracts must hold regardless of which backend is
active — this suite verifies that the API layer remains fully backend-agnostic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from segmentation_service.adapters.sam2_adapter import SAM2SegmentationAdapter

# Minimal valid 1×1 PNG (base64).
_STUB_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _make_mock_predictor(
    *,
    n_masks: int = 1,
    score: float = 0.95,
) -> MagicMock:
    p = MagicMock()
    p.predict.return_value = (
        np.ones((n_masks, 1, 1), dtype=bool),
        np.full(n_masks, score, dtype=np.float32),
        np.zeros((n_masks, 256, 256), dtype=np.float32),
    )
    return p


# ---------------------------------------------------------------------------
# Module-scoped fixture: one TestClient for all tests in this file.
# get_adapter is patched at every place it is imported and called from.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sam2_client() -> TestClient:
    """Return a TestClient whose routes all use a mocked SAM2 adapter."""
    from segmentation_service.main import create_app

    adapter = SAM2SegmentationAdapter()
    adapter._predictor = _make_mock_predictor()

    patches = [
        patch("segmentation_service.api.v1.health.get_adapter", return_value=adapter),
        patch("segmentation_service.api.v1.capabilities.get_adapter", return_value=adapter),
        patch("segmentation_service.api.v1.segment.get_adapter", return_value=adapter),
    ]
    with patches[0], patches[1], patches[2]:
        app = create_app()
        with TestClient(app) as client:
            yield client


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------

def test_health_returns_200(sam2_client: TestClient) -> None:
    assert sam2_client.get("/api/v1/health").status_code == 200


def test_health_reports_sam2_backend(sam2_client: TestClient) -> None:
    data = sam2_client.get("/api/v1/health").json()
    assert data["backend"] == "sam2"
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# GET /api/v1/capabilities
# ---------------------------------------------------------------------------

def test_capabilities_returns_200(sam2_client: TestClient) -> None:
    assert sam2_client.get("/api/v1/capabilities").status_code == 200


def test_capabilities_backend_is_sam2(sam2_client: TestClient) -> None:
    data = sam2_client.get("/api/v1/capabilities").json()
    assert data["backend"] == "sam2"


def test_capabilities_supported_prompts(sam2_client: TestClient) -> None:
    data = sam2_client.get("/api/v1/capabilities").json()
    prompts = data["supported_prompt_types"]
    assert "point" in prompts
    assert "box" in prompts
    assert "text" not in prompts


def test_capabilities_schema_fields(sam2_client: TestClient) -> None:
    data = sam2_client.get("/api/v1/capabilities").json()
    for field in ("backend", "supported_input_types", "supported_prompt_types",
                  "max_image_width", "max_image_height"):
        assert field in data


# ---------------------------------------------------------------------------
# POST /api/v1/segment — point prompt
# ---------------------------------------------------------------------------

def test_segment_point_returns_200(sam2_client: TestClient) -> None:
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "point",
        "points": [{"x": 100, "y": 200, "label": 1}],
    }
    assert sam2_client.post("/api/v1/segment", json=payload).status_code == 200


def test_segment_point_response_schema(sam2_client: TestClient) -> None:
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "point",
        "points": [{"x": 100, "y": 200, "label": 1}],
    }
    data = sam2_client.post("/api/v1/segment", json=payload).json()
    assert data["backend"] == "sam2"
    assert isinstance(data["masks"], list)
    assert len(data["masks"]) >= 1
    mask = data["masks"][0]
    assert "mask_b64" in mask
    assert "score" in mask
    assert "area" in mask
    assert 0.0 <= mask["score"] <= 1.0


def test_segment_point_score_matches_mock(sam2_client: TestClient) -> None:
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "point",
        "points": [{"x": 0, "y": 0, "label": 1}],
    }
    data = sam2_client.post("/api/v1/segment", json=payload).json()
    assert data["masks"][0]["score"] == pytest.approx(0.95)


def test_segment_point_missing_points_returns_422(sam2_client: TestClient) -> None:
    payload = {"image": _STUB_IMAGE, "prompt_type": "point"}
    assert sam2_client.post("/api/v1/segment", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/segment — box prompt
# ---------------------------------------------------------------------------

def test_segment_box_returns_200(sam2_client: TestClient) -> None:
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "box",
        "box": {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100},
    }
    assert sam2_client.post("/api/v1/segment", json=payload).status_code == 200


def test_segment_box_response_backend(sam2_client: TestClient) -> None:
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "box",
        "box": {"x_min": 10, "y_min": 20, "x_max": 110, "y_max": 120},
    }
    data = sam2_client.post("/api/v1/segment", json=payload).json()
    assert data["backend"] == "sam2"


def test_segment_box_missing_box_returns_422(sam2_client: TestClient) -> None:
    payload = {"image": _STUB_IMAGE, "prompt_type": "box"}
    assert sam2_client.post("/api/v1/segment", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/segment — text prompt (unsupported by SAM2)
# ---------------------------------------------------------------------------

def test_segment_text_prompt_returns_500(sam2_client: TestClient) -> None:
    """Text prompts are rejected by the SAM2 adapter, surfacing as HTTP 500.

    Clients should always check /capabilities before sending requests.
    """
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "text",
        "text_prompt": "a dog",
    }
    # The SAM2 adapter raises ValueError; the endpoint converts it to 500.
    assert sam2_client.post("/api/v1/segment", json=payload).status_code == 500


# ---------------------------------------------------------------------------
# POST /api/v1/segment — logits
# ---------------------------------------------------------------------------

def test_segment_return_logits(sam2_client: TestClient) -> None:
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "point",
        "points": [{"x": 0, "y": 0, "label": 1}],
        "return_logits": True,
    }
    data = sam2_client.post("/api/v1/segment", json=payload).json()
    assert data["masks"][0]["logits_b64"] is not None


def test_segment_no_logits_by_default(sam2_client: TestClient) -> None:
    payload = {
        "image": _STUB_IMAGE,
        "prompt_type": "point",
        "points": [{"x": 0, "y": 0, "label": 1}],
    }
    data = sam2_client.post("/api/v1/segment", json=payload).json()
    assert data["masks"][0]["logits_b64"] is None
