"""Cross-backend compatibility tests.

These tests prove the central claim of the project:
  "The same client logic and the same API surface work correctly
   with any backend — without any code changes."

Architecture
------------
Each backend (mock, SAM2, CLIPSeg) is exercised via a FastAPI TestClient
that has `get_adapter` patched at every import site, consistent with the
existing backend endpoint test suites.

The key invariants tested here:
1. /capabilities accurately reports what /segment accepts.
2. /segment succeeds for every prompt type listed in capabilities.
3. /segment returns 500 for every prompt type NOT in capabilities.
4. The client's select_prompt() picks the right prompt type for each backend.
5. The SegmentationClient's public interface produces the same result
   structure regardless of which backend is active.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from segmentation_service.adapters.clipseg_adapter import CLIPSegSegmentationAdapter
from segmentation_service.adapters.mock_adapter import MockSegmentationAdapter
from segmentation_service.adapters.sam2_adapter import SAM2SegmentationAdapter
from segmentation_service.client.cli import select_prompt

# Minimal 1×1 PNG.
_STUB_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# All possible prompt types in the API.
_ALL_PROMPT_TYPES = {"point", "box", "text"}


# ---------------------------------------------------------------------------
# Payload factories — one per prompt type
# ---------------------------------------------------------------------------

_PAYLOADS: dict[str, dict] = {
    "point": {
        "image": _STUB_IMAGE,
        "prompt_type": "point",
        "points": [{"x": 0, "y": 0, "label": 1}],
    },
    "box": {
        "image": _STUB_IMAGE,
        "prompt_type": "box",
        "box": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
    },
    "text": {
        "image": _STUB_IMAGE,
        "prompt_type": "text",
        "text_prompt": "object",
    },
}


# ---------------------------------------------------------------------------
# Adapter factory helpers (mirrors existing endpoint test pattern)
# ---------------------------------------------------------------------------

def _make_mock_sam2_predictor(n: int = 1) -> MagicMock:
    p = MagicMock()
    p.predict.return_value = (
        np.ones((n, 1, 1), dtype=bool),
        np.full(n, 0.9, dtype=np.float32),
        np.zeros((n, 256, 256), dtype=np.float32),
    )
    return p


def _make_mock_infer_clipseg():
    mask = np.ones((4, 4), dtype=bool)
    logits = np.zeros((4, 4), dtype=np.float32)
    return mask, logits, 0.9


# ---------------------------------------------------------------------------
# TestClient fixtures (one per backend, function-scoped)
#
# Fixtures must be function-scoped, not module-scoped, because all three are
# used in the same test module.  Module-scoped fixtures from different
# definitions all have their patch context managers active simultaneously,
# so the last fixture to be set up would overwrite earlier patches.
# Function scope ensures only one set of patches is active per test.
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client() -> TestClient:
    """TestClient backed by the real MockSegmentationAdapter."""
    from segmentation_service.main import create_app

    adapter = MockSegmentationAdapter()
    patches = [
        patch("segmentation_service.api.v1.health.get_adapter", return_value=adapter),
        patch("segmentation_service.api.v1.capabilities.get_adapter", return_value=adapter),
        patch("segmentation_service.api.v1.segment.get_adapter", return_value=adapter),
    ]
    with patches[0], patches[1], patches[2]:
        app = create_app()
        with TestClient(app) as client:
            yield client


@pytest.fixture
def sam2_client() -> TestClient:
    """TestClient backed by SAM2 adapter with mocked predictor."""
    from segmentation_service.main import create_app

    adapter = SAM2SegmentationAdapter()
    adapter._predictor = _make_mock_sam2_predictor()
    patches = [
        patch("segmentation_service.api.v1.health.get_adapter", return_value=adapter),
        patch("segmentation_service.api.v1.capabilities.get_adapter", return_value=adapter),
        patch("segmentation_service.api.v1.segment.get_adapter", return_value=adapter),
    ]
    with patches[0], patches[1], patches[2]:
        app = create_app()
        with TestClient(app) as client:
            yield client


@pytest.fixture
def clipseg_client() -> TestClient:
    """TestClient backed by CLIPSeg adapter with mocked _infer."""
    from segmentation_service.main import create_app

    adapter = CLIPSegSegmentationAdapter()
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
            with patch.object(adapter, "_infer", return_value=_make_mock_infer_clipseg()):
                yield client


# ---------------------------------------------------------------------------
# Helper: verify capabilities ↔ segment alignment for a given TestClient
# ---------------------------------------------------------------------------

def _assert_capabilities_matches_segment_behavior(tc: TestClient) -> None:
    """Core compatibility invariant: what capabilities advertises, segment accepts."""
    caps = tc.get("/api/v1/capabilities").json()
    supported = set(caps["supported_prompt_types"])
    unsupported = _ALL_PROMPT_TYPES - supported

    for pt in supported:
        resp = tc.post("/api/v1/segment", json=_PAYLOADS[pt])
        assert resp.status_code == 200, (
            f"Backend '{caps['backend']}' claims to support '{pt}' "
            f"but /segment returned {resp.status_code}"
        )

    for pt in unsupported:
        resp = tc.post("/api/v1/segment", json=_PAYLOADS[pt])
        assert resp.status_code in (422, 500), (
            f"Backend '{caps['backend']}' claims NOT to support '{pt}' "
            f"but /segment returned {resp.status_code}"
        )


# ---------------------------------------------------------------------------
# 1. Capabilities ↔ segment alignment — per backend
# ---------------------------------------------------------------------------

def test_mock_capabilities_matches_segment(mock_client: TestClient) -> None:
    _assert_capabilities_matches_segment_behavior(mock_client)


def test_sam2_capabilities_matches_segment(sam2_client: TestClient) -> None:
    _assert_capabilities_matches_segment_behavior(sam2_client)


def test_clipseg_capabilities_matches_segment(clipseg_client: TestClient) -> None:
    _assert_capabilities_matches_segment_behavior(clipseg_client)


# ---------------------------------------------------------------------------
# 2. Backend identity — /capabilities and /health agree on backend name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name,expected_backend", [
    ("mock_client", "mock"),
    ("sam2_client", "sam2"),
    ("clipseg_client", "clipseg"),
])
def test_health_and_capabilities_agree_on_backend(
    request, fixture_name: str, expected_backend: str
) -> None:
    tc = request.getfixturevalue(fixture_name)
    health = tc.get("/api/v1/health").json()
    caps = tc.get("/api/v1/capabilities").json()
    assert health["backend"] == expected_backend
    assert caps["backend"] == expected_backend


# ---------------------------------------------------------------------------
# 3. Known prompt-type support matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt_type,should_succeed", [
    ("point", True),
    ("box", True),
    ("text", True),
])
def test_mock_supports_all_prompt_types(
    mock_client: TestClient, prompt_type: str, should_succeed: bool
) -> None:
    resp = mock_client.post("/api/v1/segment", json=_PAYLOADS[prompt_type])
    if should_succeed:
        assert resp.status_code == 200
    else:
        assert resp.status_code in (422, 500)


@pytest.mark.parametrize("prompt_type,should_succeed", [
    ("point", True),
    ("box", True),
    ("text", False),
])
def test_sam2_known_support_matrix(
    sam2_client: TestClient, prompt_type: str, should_succeed: bool
) -> None:
    resp = sam2_client.post("/api/v1/segment", json=_PAYLOADS[prompt_type])
    if should_succeed:
        assert resp.status_code == 200, f"SAM2 should support {prompt_type}"
    else:
        assert resp.status_code in (422, 500), (
            f"SAM2 should NOT support {prompt_type}"
        )


@pytest.mark.parametrize("prompt_type,should_succeed", [
    ("text", True),
    ("point", False),
    ("box", False),
])
def test_clipseg_known_support_matrix(
    clipseg_client: TestClient, prompt_type: str, should_succeed: bool
) -> None:
    resp = clipseg_client.post("/api/v1/segment", json=_PAYLOADS[prompt_type])
    if should_succeed:
        assert resp.status_code == 200, f"CLIPSeg should support {prompt_type}"
    else:
        assert resp.status_code in (422, 500), (
            f"CLIPSeg should NOT support {prompt_type}"
        )


# ---------------------------------------------------------------------------
# 4. select_prompt adapts correctly to each backend's capabilities
# ---------------------------------------------------------------------------

class TestSelectPromptAgainstRealCapabilities:
    """Verify that the client's select_prompt() is compatible with each backend.

    These tests simulate what happens when the client calls /capabilities and
    then calls select_prompt() with user-supplied inputs.
    """

    def test_select_prompt_for_mock_with_text(self, mock_client: TestClient) -> None:
        supported = mock_client.get("/api/v1/capabilities").json()["supported_prompt_types"]
        result = select_prompt(supported, text_prompt="the cat")
        assert result["prompt_type"] == "text"

    def test_select_prompt_for_mock_with_point(self, mock_client: TestClient) -> None:
        supported = mock_client.get("/api/v1/capabilities").json()["supported_prompt_types"]
        result = select_prompt(supported, point="10,20,1")
        assert result["prompt_type"] == "point"

    def test_select_prompt_for_sam2_with_point(self, sam2_client: TestClient) -> None:
        supported = sam2_client.get("/api/v1/capabilities").json()["supported_prompt_types"]
        result = select_prompt(supported, point="320,240,1")
        assert result["prompt_type"] == "point"

    def test_select_prompt_for_sam2_rejects_text(self, sam2_client: TestClient) -> None:
        supported = sam2_client.get("/api/v1/capabilities").json()["supported_prompt_types"]
        with pytest.raises(ValueError, match="not supported"):
            select_prompt(supported, text_prompt="the cat")

    def test_select_prompt_for_clipseg_with_text(self, clipseg_client: TestClient) -> None:
        supported = clipseg_client.get("/api/v1/capabilities").json()["supported_prompt_types"]
        result = select_prompt(supported, text_prompt="a red bicycle")
        assert result["prompt_type"] == "text"

    def test_select_prompt_for_clipseg_rejects_point(self, clipseg_client: TestClient) -> None:
        supported = clipseg_client.get("/api/v1/capabilities").json()["supported_prompt_types"]
        with pytest.raises(ValueError, match="not supported"):
            select_prompt(supported, point="10,20,1")

    def test_select_prompt_for_clipseg_rejects_box(self, clipseg_client: TestClient) -> None:
        supported = clipseg_client.get("/api/v1/capabilities").json()["supported_prompt_types"]
        with pytest.raises(ValueError, match="not supported"):
            select_prompt(supported, box="0,0,100,100")


# ---------------------------------------------------------------------------
# 5. Response schema is identical across all backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name,payload_key", [
    ("mock_client", "point"),
    ("sam2_client", "point"),
    ("clipseg_client", "text"),
])
def test_response_schema_is_backend_agnostic(
    request, fixture_name: str, payload_key: str
) -> None:
    """Every backend must return the same SegmentResponse shape."""
    tc = request.getfixturevalue(fixture_name)
    data = tc.post("/api/v1/segment", json=_PAYLOADS[payload_key]).json()

    # All fields must be present.
    assert "request_id" in data
    assert "backend" in data
    assert "masks" in data
    assert "latency_ms" in data
    assert isinstance(data["masks"], list)

    mask = data["masks"][0]
    assert "mask_b64" in mask
    assert "score" in mask
    assert "area" in mask
    assert 0.0 <= mask["score"] <= 1.0
    assert mask["area"] >= 0


# ---------------------------------------------------------------------------
# 6. Health endpoint is stable across all backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", ["mock_client", "sam2_client", "clipseg_client"])
def test_health_always_returns_ok(request, fixture_name: str) -> None:
    tc = request.getfixturevalue(fixture_name)
    data = tc.get("/api/v1/health").json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "backend" in data


# ---------------------------------------------------------------------------
# 7. API paths are unchanged for all backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", ["mock_client", "sam2_client", "clipseg_client"])
def test_api_paths_unchanged(request, fixture_name: str) -> None:
    tc = request.getfixturevalue(fixture_name)
    assert tc.get("/api/v1/health").status_code == 200
    assert tc.get("/api/v1/capabilities").status_code == 200
