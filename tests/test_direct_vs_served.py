"""Regression tests for benchmark/direct_vs_served.py payload selection.

Verifies that:
- mock requests keep the tiny built-in stub image.
- sam2-point and sam2-box use the asset-file images (not the 1×1 stub).
- clipseg-text uses the clipseg asset-file image (not the 1×1 stub).
- _REQUESTS contains entries for all expected backend/prompt-type combos.
- --payload-file loading parses correctly into a SegmentRequest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# benchmark/ is not a package, so we add it to sys.path to import it.
# typer and rich are CLI/display dependencies not installed in the test env;
# stub them out before importing so only the payload-selection logic is exercised.
from unittest.mock import MagicMock

for _mod in ("typer", "rich", "rich.console", "rich.table", "httpx"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

import direct_vs_served as dvs  # type: ignore[import]

from segmentation_service.eval.probe_payloads import _ASSETS_DIR, _MOCK_IMAGE
from segmentation_service.schemas.segment import PromptType, SegmentRequest

# The 1×1 stub that was previously (incorrectly) used for all backends.
_STUB_1x1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


# ---------------------------------------------------------------------------
# _REQUESTS dict — expected keys exist
# ---------------------------------------------------------------------------

class TestRequestsKeys:
    def test_has_mock_point(self):
        assert "mock-point" in dvs._REQUESTS

    def test_has_mock_box(self):
        assert "mock-box" in dvs._REQUESTS

    def test_has_mock_text(self):
        assert "mock-text" in dvs._REQUESTS

    def test_has_sam2_point(self):
        assert "sam2-point" in dvs._REQUESTS

    def test_has_sam2_box(self):
        assert "sam2-box" in dvs._REQUESTS

    def test_has_clipseg_text(self):
        assert "clipseg-text" in dvs._REQUESTS

    def test_all_values_are_segment_requests(self):
        for key, req in dvs._REQUESTS.items():
            assert isinstance(req, SegmentRequest), f"{key} is not a SegmentRequest"


# ---------------------------------------------------------------------------
# Mock entries keep the built-in stub image
# ---------------------------------------------------------------------------

class TestMockRequestImages:
    def test_mock_point_uses_mock_image(self):
        assert dvs._REQUESTS["mock-point"].image == _MOCK_IMAGE

    def test_mock_box_uses_mock_image(self):
        assert dvs._REQUESTS["mock-box"].image == _MOCK_IMAGE

    def test_mock_text_uses_mock_image(self):
        assert dvs._REQUESTS["mock-text"].image == _MOCK_IMAGE


# ---------------------------------------------------------------------------
# SAM2 entries use asset files, NOT the 1×1 stub
# ---------------------------------------------------------------------------

class TestSam2RequestImages:
    def test_sam2_point_is_not_1x1_stub(self):
        req = dvs._REQUESTS["sam2-point"]
        assert req.image != _STUB_1x1, (
            "sam2-point in _REQUESTS is still using the 1×1 stub image. "
            "It should use eval_assets/requests/sam2_point.json."
        )

    def test_sam2_box_is_not_1x1_stub(self):
        req = dvs._REQUESTS["sam2-box"]
        assert req.image != _STUB_1x1, (
            "sam2-box in _REQUESTS is still using the 1×1 stub image. "
            "It should use eval_assets/requests/sam2_box.json."
        )

    def test_sam2_point_matches_asset_file(self):
        expected = json.loads((_ASSETS_DIR / "sam2_point.json").read_text())
        assert dvs._REQUESTS["sam2-point"].image == expected["image"]

    def test_sam2_box_matches_asset_file(self):
        expected = json.loads((_ASSETS_DIR / "sam2_box.json").read_text())
        assert dvs._REQUESTS["sam2-box"].image == expected["image"]

    def test_sam2_point_prompt_type(self):
        assert dvs._REQUESTS["sam2-point"].prompt_type == PromptType.point

    def test_sam2_box_prompt_type(self):
        assert dvs._REQUESTS["sam2-box"].prompt_type == PromptType.box


# ---------------------------------------------------------------------------
# CLIPSeg entry uses the asset file, NOT the 1×1 stub
# ---------------------------------------------------------------------------

class TestClipsegRequestImage:
    def test_clipseg_text_is_not_1x1_stub(self):
        req = dvs._REQUESTS["clipseg-text"]
        assert req.image != _STUB_1x1, (
            "clipseg-text in _REQUESTS is still using the 1×1 stub image. "
            "It should use eval_assets/requests/clipseg_text.json."
        )

    def test_clipseg_text_matches_asset_file(self):
        expected = json.loads((_ASSETS_DIR / "clipseg_text.json").read_text())
        assert dvs._REQUESTS["clipseg-text"].image == expected["image"]

    def test_clipseg_text_prompt_type(self):
        assert dvs._REQUESTS["clipseg-text"].prompt_type == PromptType.text

    def test_clipseg_text_has_text_prompt(self):
        assert dvs._REQUESTS["clipseg-text"].text_prompt


# ---------------------------------------------------------------------------
# --payload-file loading
# ---------------------------------------------------------------------------

class TestPayloadFileLoading:
    def test_sam2_point_asset_parses_to_segment_request(self):
        payload = json.loads((_ASSETS_DIR / "sam2_point.json").read_text())
        req = SegmentRequest.model_validate(payload)
        assert req.prompt_type == PromptType.point
        assert req.points

    def test_sam2_box_asset_parses_to_segment_request(self):
        payload = json.loads((_ASSETS_DIR / "sam2_box.json").read_text())
        req = SegmentRequest.model_validate(payload)
        assert req.prompt_type == PromptType.box
        assert req.box is not None

    def test_clipseg_text_asset_parses_to_segment_request(self):
        payload = json.loads((_ASSETS_DIR / "clipseg_text.json").read_text())
        req = SegmentRequest.model_validate(payload)
        assert req.prompt_type == PromptType.text
        assert req.text_prompt


# ---------------------------------------------------------------------------
# _BACKEND_DEFAULT_KEY still maps correctly
# ---------------------------------------------------------------------------

class TestBackendDefaultKey:
    def test_mock_default_key(self):
        assert dvs._BACKEND_DEFAULT_KEY["mock"] == "mock-point"

    def test_sam2_default_key(self):
        assert dvs._BACKEND_DEFAULT_KEY["sam2"] == "sam2-point"

    def test_clipseg_default_key(self):
        assert dvs._BACKEND_DEFAULT_KEY["clipseg"] == "clipseg-text"
