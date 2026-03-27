"""Regression tests for segmentation_service.eval.probe_payloads.

Verifies that:
- mock always uses the built-in tiny synthetic image.
- sam2 probes load from eval_assets/requests/sam2_*.json (not the 1×1 stub).
- clipseg probes load from eval_assets/requests/clipseg_text.json.
- load_request() returns valid SegmentRequest objects.
- Fallback behaviour for unknown (backend, prompt_type) pairs is safe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from segmentation_service.eval.probe_payloads import (
    BACKEND_PROBE_TYPES,
    DEFAULT_PROMPT_TYPE,
    _ASSETS_DIR,
    _MOCK_IMAGE,
    load_payload,
    load_request,
)
from segmentation_service.schemas.segment import SegmentRequest

# The 1×1 stub image shared across all mock probes.
_STUB_1x1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


# ---------------------------------------------------------------------------
# Mock probes use the built-in tiny image, NOT asset files
# ---------------------------------------------------------------------------

class TestMockPayloads:
    def test_mock_point_uses_stub_image(self):
        p = load_payload("mock", "point")
        assert p["image"] == _MOCK_IMAGE

    def test_mock_box_uses_stub_image(self):
        p = load_payload("mock", "box")
        assert p["image"] == _MOCK_IMAGE

    def test_mock_text_uses_stub_image(self):
        p = load_payload("mock", "text")
        assert p["image"] == _MOCK_IMAGE

    def test_mock_point_prompt_type(self):
        p = load_payload("mock", "point")
        assert p["prompt_type"] == "point"
        assert "points" in p

    def test_mock_box_prompt_type(self):
        p = load_payload("mock", "box")
        assert p["prompt_type"] == "box"
        assert "box" in p

    def test_mock_text_prompt_type(self):
        p = load_payload("mock", "text")
        assert p["prompt_type"] == "text"
        assert p.get("text_prompt")

    def test_mock_payload_is_fresh_copy(self):
        """Mutating the returned dict must not affect subsequent calls."""
        p1 = load_payload("mock", "point")
        p1["image"] = "tampered"
        p2 = load_payload("mock", "point")
        assert p2["image"] == _MOCK_IMAGE


# ---------------------------------------------------------------------------
# SAM2 probes load from asset files (not the 1×1 stub)
# ---------------------------------------------------------------------------

class TestSam2Payloads:
    def test_sam2_point_loads_from_asset_file(self):
        """sam2/point payload must differ from the 1×1 mock stub image."""
        p = load_payload("sam2", "point")
        assert p["image"] != _STUB_1x1, (
            "sam2 point probe is using the 1×1 mock stub — it should use "
            "eval_assets/requests/sam2_point.json"
        )

    def test_sam2_box_loads_from_asset_file(self):
        """sam2/box payload must differ from the 1×1 mock stub image."""
        p = load_payload("sam2", "box")
        assert p["image"] != _STUB_1x1, (
            "sam2 box probe is using the 1×1 mock stub — it should use "
            "eval_assets/requests/sam2_box.json"
        )

    def test_sam2_point_matches_asset_json(self):
        expected = json.loads((_ASSETS_DIR / "sam2_point.json").read_text())
        assert load_payload("sam2", "point") == expected

    def test_sam2_box_matches_asset_json(self):
        expected = json.loads((_ASSETS_DIR / "sam2_box.json").read_text())
        assert load_payload("sam2", "box") == expected

    def test_sam2_point_prompt_type(self):
        p = load_payload("sam2", "point")
        assert p["prompt_type"] == "point"

    def test_sam2_box_prompt_type(self):
        p = load_payload("sam2", "box")
        assert p["prompt_type"] == "box"

    def test_sam2_text_falls_back_gracefully(self):
        """sam2 has no text asset file; fallback must return a valid payload."""
        p = load_payload("sam2", "text")
        assert p["prompt_type"] == "text"
        assert "image" in p


# ---------------------------------------------------------------------------
# CLIPSeg probes load from asset file
# ---------------------------------------------------------------------------

class TestClipsegPayloads:
    def test_clipseg_text_loads_from_asset_file(self):
        p = load_payload("clipseg", "text")
        assert p["image"] != _STUB_1x1, (
            "clipseg text probe is using the 1×1 mock stub — it should use "
            "eval_assets/requests/clipseg_text.json"
        )

    def test_clipseg_text_matches_asset_json(self):
        expected = json.loads((_ASSETS_DIR / "clipseg_text.json").read_text())
        assert load_payload("clipseg", "text") == expected

    def test_clipseg_point_falls_back_gracefully(self):
        """clipseg has no point asset file; fallback must return a valid payload."""
        p = load_payload("clipseg", "point")
        assert p["prompt_type"] == "point"
        assert "image" in p


# ---------------------------------------------------------------------------
# load_request returns valid SegmentRequest objects
# ---------------------------------------------------------------------------

class TestLoadRequest:
    @pytest.mark.parametrize("backend,prompt_type", [
        ("mock", "point"),
        ("mock", "box"),
        ("mock", "text"),
        ("sam2", "point"),
        ("sam2", "box"),
        ("clipseg", "text"),
    ])
    def test_returns_segment_request(self, backend, prompt_type):
        req = load_request(backend, prompt_type)
        assert isinstance(req, SegmentRequest)

    def test_sam2_point_request_has_points(self):
        req = load_request("sam2", "point")
        assert req.points and len(req.points) >= 1

    def test_sam2_box_request_has_box(self):
        req = load_request("sam2", "box")
        assert req.box is not None

    def test_clipseg_text_request_has_text_prompt(self):
        req = load_request("clipseg", "text")
        assert req.text_prompt


# ---------------------------------------------------------------------------
# SAM2 correctness probes (as used in evaluate_correctness.py)
# differ from the 1×1 stub — regression guard
# ---------------------------------------------------------------------------

class TestCorrectnessProbeSelection:
    """Ensure evaluate_correctness.py's sam2 probes use asset payloads."""

    def test_sam2_point_probe_is_not_1x1_stub(self):
        req = load_request("sam2", "point")
        assert req.image != _STUB_1x1

    def test_sam2_box_probe_is_not_1x1_stub(self):
        req = load_request("sam2", "box")
        assert req.image != _STUB_1x1

    def test_clipseg_text_probe_is_not_1x1_stub(self):
        req = load_request("clipseg", "text")
        assert req.image != _STUB_1x1


# ---------------------------------------------------------------------------
# BACKEND_PROBE_TYPES and DEFAULT_PROMPT_TYPE constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_mock_probe_types(self):
        assert set(BACKEND_PROBE_TYPES["mock"]) == {"point", "box", "text"}

    def test_sam2_probe_types(self):
        assert set(BACKEND_PROBE_TYPES["sam2"]) == {"point", "box"}

    def test_clipseg_probe_types(self):
        assert set(BACKEND_PROBE_TYPES["clipseg"]) == {"text"}

    def test_default_prompt_types(self):
        assert DEFAULT_PROMPT_TYPE["mock"] == "point"
        assert DEFAULT_PROMPT_TYPE["sam2"] == "point"
        assert DEFAULT_PROMPT_TYPE["clipseg"] == "text"
