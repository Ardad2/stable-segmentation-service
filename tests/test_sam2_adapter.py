"""Unit tests for SAM2SegmentationAdapter.

The SAM2 library and model weights are NOT expected to be present in CI.
All tests inject a mock predictor directly into the adapter instance so
that inference never actually runs.  Only the adapter's own translation
logic (prompt building, mask encoding, schema mapping) is exercised.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from segmentation_service.adapters.sam2_adapter import (
    SAM2SegmentationAdapter,
    _b64_to_numpy,
    _logits_to_b64,
    _mask_to_b64,
)
from segmentation_service.schemas.segment import BoxPrompt, PointPrompt, PromptType, SegmentRequest

# Minimal valid 1×1 PNG (base64) — used as a stand-in image in all tests.
_STUB_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# Pre-built numpy arrays that the mock predictor returns.
_MASK_1x1 = np.array([[[True]]], dtype=bool)       # shape (1, 1, 1)
_SCORES_1 = np.array([0.9], dtype=np.float32)
_LOGITS_1 = np.zeros((1, 256, 256), dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_predictor() -> MagicMock:
    """Mock SAM2ImagePredictor that returns deterministic output."""
    p = MagicMock()
    p.predict.return_value = (_MASK_1x1, _SCORES_1, _LOGITS_1)
    return p


@pytest.fixture
def adapter(mock_predictor: MagicMock) -> SAM2SegmentationAdapter:
    """SAM2 adapter with the predictor already injected (skips model loading)."""
    a = SAM2SegmentationAdapter()
    a._predictor = mock_predictor
    return a


# ---------------------------------------------------------------------------
# capabilities()
# ---------------------------------------------------------------------------

def test_capabilities_backend_name(adapter: SAM2SegmentationAdapter) -> None:
    assert adapter.capabilities().backend == "sam2"


def test_capabilities_supports_point_and_box(adapter: SAM2SegmentationAdapter) -> None:
    caps = adapter.capabilities()
    assert "point" in caps.supported_prompt_types
    assert "box" in caps.supported_prompt_types


def test_capabilities_excludes_text(adapter: SAM2SegmentationAdapter) -> None:
    assert "text" not in adapter.capabilities().supported_prompt_types


def test_capabilities_max_dimensions(adapter: SAM2SegmentationAdapter) -> None:
    caps = adapter.capabilities()
    assert caps.max_image_width == 4096
    assert caps.max_image_height == 4096


# ---------------------------------------------------------------------------
# text prompt rejection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_prompt_raises_value_error(adapter: SAM2SegmentationAdapter) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.text,
        text_prompt="a cat",
    )
    with pytest.raises(ValueError, match="text prompts"):
        await adapter.segment(req)


# ---------------------------------------------------------------------------
# point prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_point_prompt_returns_response(adapter: SAM2SegmentationAdapter) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.point,
        points=[PointPrompt(x=10, y=20, label=1)],
    )
    resp = await adapter.segment(req)
    assert resp.backend == "sam2"
    assert len(resp.masks) == 1


@pytest.mark.asyncio
async def test_point_prompt_mask_fields(adapter: SAM2SegmentationAdapter) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.point,
        points=[PointPrompt(x=0, y=0, label=1)],
    )
    resp = await adapter.segment(req)
    mask = resp.masks[0]
    assert mask.mask_b64  # non-empty base64 string
    assert 0.0 <= mask.score <= 1.0
    assert mask.area == 1  # _MASK_1x1 has exactly 1 True pixel


@pytest.mark.asyncio
async def test_point_prompt_passes_coords_to_predictor(
    adapter: SAM2SegmentationAdapter, mock_predictor: MagicMock
) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.point,
        points=[PointPrompt(x=10, y=20, label=1), PointPrompt(x=30, y=40, label=0)],
    )
    await adapter.segment(req)
    kwargs = mock_predictor.predict.call_args.kwargs
    np.testing.assert_array_almost_equal(kwargs["point_coords"], [[10, 20], [30, 40]])
    np.testing.assert_array_equal(kwargs["point_labels"], [1, 0])


@pytest.mark.asyncio
async def test_multimask_output_flag_forwarded(
    adapter: SAM2SegmentationAdapter, mock_predictor: MagicMock
) -> None:
    mock_predictor.predict.return_value = (
        np.ones((3, 1, 1), dtype=bool),
        np.array([0.9, 0.8, 0.7], dtype=np.float32),
        np.zeros((3, 256, 256), dtype=np.float32),
    )
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.point,
        points=[PointPrompt(x=0, y=0)],
        multimask_output=True,
    )
    resp = await adapter.segment(req)
    assert len(resp.masks) == 3
    assert mock_predictor.predict.call_args.kwargs["multimask_output"] is True


@pytest.mark.asyncio
async def test_single_mask_by_default(
    adapter: SAM2SegmentationAdapter, mock_predictor: MagicMock
) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.point,
        points=[PointPrompt(x=0, y=0)],
        multimask_output=False,
    )
    await adapter.segment(req)
    assert mock_predictor.predict.call_args.kwargs["multimask_output"] is False


# ---------------------------------------------------------------------------
# box prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_box_prompt_returns_response(adapter: SAM2SegmentationAdapter) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.box,
        box=BoxPrompt(x_min=0, y_min=0, x_max=100, y_max=100),
    )
    resp = await adapter.segment(req)
    assert resp.backend == "sam2"
    assert len(resp.masks) == 1


@pytest.mark.asyncio
async def test_box_prompt_passes_coords_to_predictor(
    adapter: SAM2SegmentationAdapter, mock_predictor: MagicMock
) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.box,
        box=BoxPrompt(x_min=10, y_min=20, x_max=110, y_max=120),
    )
    await adapter.segment(req)
    kwargs = mock_predictor.predict.call_args.kwargs
    np.testing.assert_array_almost_equal(kwargs["box"], [10, 20, 110, 120])


@pytest.mark.asyncio
async def test_box_prompt_forces_single_mask(
    adapter: SAM2SegmentationAdapter, mock_predictor: MagicMock
) -> None:
    """Box prompts must always use multimask_output=False (SAM2 limitation)."""
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.box,
        box=BoxPrompt(x_min=0, y_min=0, x_max=50, y_max=50),
        multimask_output=True,  # user requests multi — adapter must override
    )
    await adapter.segment(req)
    assert mock_predictor.predict.call_args.kwargs["multimask_output"] is False


# ---------------------------------------------------------------------------
# logits
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_return_logits_true(adapter: SAM2SegmentationAdapter) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.point,
        points=[PointPrompt(x=0, y=0)],
        return_logits=True,
    )
    resp = await adapter.segment(req)
    assert resp.masks[0].logits_b64 is not None


@pytest.mark.asyncio
async def test_return_logits_false(adapter: SAM2SegmentationAdapter) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.point,
        points=[PointPrompt(x=0, y=0)],
        return_logits=False,
    )
    resp = await adapter.segment(req)
    assert resp.masks[0].logits_b64 is None


# ---------------------------------------------------------------------------
# mask / logit encoding helpers
# ---------------------------------------------------------------------------

def test_mask_to_b64_area_matches_true_count() -> None:
    """_mask_to_b64 must produce a valid PNG; area must equal True pixel count."""
    mask = np.array([[True, False], [True, True]], dtype=bool)
    b64 = _mask_to_b64(mask)
    # Decode and verify
    import base64, io
    from PIL import Image
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    arr = np.array(img)
    assert arr.sum() // 255 == 3  # 3 white pixels


@pytest.mark.asyncio
async def test_mask_area_equals_true_pixels(
    adapter: SAM2SegmentationAdapter, mock_predictor: MagicMock
) -> None:
    mask = np.array([[[True, False], [True, True]]], dtype=bool)  # area = 3
    mock_predictor.predict.return_value = (
        mask,
        np.array([0.9], dtype=np.float32),
        np.zeros((1, 256, 256), dtype=np.float32),
    )
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.point,
        points=[PointPrompt(x=0, y=0)],
    )
    resp = await adapter.segment(req)
    assert resp.masks[0].area == 3


def test_logits_to_b64_handles_uniform_logits() -> None:
    """Uniform logits (hi == lo) should not cause division-by-zero."""
    uniform = np.zeros((256, 256), dtype=np.float32)
    b64 = _logits_to_b64(uniform)
    assert b64  # non-empty


def test_b64_to_numpy_returns_rgb_array() -> None:
    arr = _b64_to_numpy(_STUB_IMAGE)
    assert arr.ndim == 3
    assert arr.shape[2] == 3  # RGB channels


# ---------------------------------------------------------------------------
# response metadata
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_response_has_unique_request_ids(adapter: SAM2SegmentationAdapter) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.point,
        points=[PointPrompt(x=0, y=0)],
    )
    resp1 = await adapter.segment(req)
    resp2 = await adapter.segment(req)
    assert resp1.request_id != resp2.request_id


@pytest.mark.asyncio
async def test_response_latency_is_positive(adapter: SAM2SegmentationAdapter) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.point,
        points=[PointPrompt(x=0, y=0)],
    )
    resp = await adapter.segment(req)
    assert resp.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_response_score_clamped(
    adapter: SAM2SegmentationAdapter, mock_predictor: MagicMock
) -> None:
    """Scores outside [0, 1] returned by the model must be clamped."""
    mock_predictor.predict.return_value = (
        _MASK_1x1,
        np.array([1.5], dtype=np.float32),  # out-of-range score
        _LOGITS_1,
    )
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.point,
        points=[PointPrompt(x=0, y=0)],
    )
    resp = await adapter.segment(req)
    assert resp.masks[0].score <= 1.0


# ---------------------------------------------------------------------------
# model loading errors (no mock predictor injected)
# ---------------------------------------------------------------------------

def test_load_predictor_raises_without_sam2_installed() -> None:
    """When sam2 is not importable, _load_predictor raises RuntimeError."""
    import sys
    import importlib

    # Make 'sam2' unimportable for this test
    original = sys.modules.get("sam2")
    sys.modules["sam2"] = None  # type: ignore[assignment]
    # Also remove sub-modules if cached
    for key in list(sys.modules):
        if key.startswith("sam2."):
            sys.modules[key] = None  # type: ignore[assignment]

    try:
        fresh = SAM2SegmentationAdapter()
        # _predictor is None, so _load_predictor will attempt the import
        with pytest.raises(RuntimeError, match="not installed"):
            fresh._load_predictor()
    finally:
        if original is None:
            sys.modules.pop("sam2", None)
        else:
            sys.modules["sam2"] = original
        for key in list(sys.modules):
            if key.startswith("sam2.") and sys.modules[key] is None:
                sys.modules.pop(key)
