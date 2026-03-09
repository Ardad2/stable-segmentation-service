"""Unit tests for CLIPSegSegmentationAdapter.

The CLIPSeg library (transformers + torch) and model weights are NOT expected
to be present in CI.  All tests either:

- inject mock _processor / _model attributes to skip model loading, then
  patch _infer to return predetermined numpy arrays, OR
- test pure-Python helper functions directly.

No GPU, no HuggingFace download, no torch required.
"""

from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from segmentation_service.adapters.clipseg_adapter import (
    CLIPSegSegmentationAdapter,
    _b64_to_pil,
    _logits_to_b64,
    _mask_to_b64,
)
from segmentation_service.schemas.segment import (
    BoxPrompt,
    PointPrompt,
    PromptType,
    SegmentRequest,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# Minimal valid 1×1 PNG (base64) reused as a stand-in image.
_STUB_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# Predetermined outputs that the mock _infer returns.
_STUB_MASK = np.ones((64, 64), dtype=bool)        # all foreground
_STUB_LOGITS = np.ones((64, 64), dtype=np.float32) * 3.0  # logit=3 → sigmoid≈0.95
_STUB_SCORE = 0.95


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter() -> CLIPSegSegmentationAdapter:
    """Adapter with processor/model set so _load_model() short-circuits."""
    a = CLIPSegSegmentationAdapter()
    a._processor = MagicMock()
    a._model = MagicMock()
    return a


@pytest.fixture
def infer_patch(adapter: CLIPSegSegmentationAdapter):
    """Context manager that patches _infer on the adapter instance."""
    with patch.object(
        adapter,
        "_infer",
        return_value=(_STUB_MASK, _STUB_LOGITS, _STUB_SCORE),
    ) as mock_infer:
        yield mock_infer


# ---------------------------------------------------------------------------
# capabilities()
# ---------------------------------------------------------------------------

def test_capabilities_backend_name(adapter: CLIPSegSegmentationAdapter) -> None:
    assert adapter.capabilities().backend == "clipseg"


def test_capabilities_supports_text(adapter: CLIPSegSegmentationAdapter) -> None:
    assert "text" in adapter.capabilities().supported_prompt_types


def test_capabilities_excludes_point(adapter: CLIPSegSegmentationAdapter) -> None:
    assert "point" not in adapter.capabilities().supported_prompt_types


def test_capabilities_excludes_box(adapter: CLIPSegSegmentationAdapter) -> None:
    assert "box" not in adapter.capabilities().supported_prompt_types


def test_capabilities_max_dimensions(adapter: CLIPSegSegmentationAdapter) -> None:
    caps = adapter.capabilities()
    assert caps.max_image_width == 4096
    assert caps.max_image_height == 4096


def test_capabilities_notes_mentions_text(adapter: CLIPSegSegmentationAdapter) -> None:
    assert "text" in adapter.capabilities().notes.lower()


# ---------------------------------------------------------------------------
# Prompt-type validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_point_prompt_raises_value_error(
    adapter: CLIPSegSegmentationAdapter,
) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.point,
        points=[PointPrompt(x=10, y=20, label=1)],
    )
    with pytest.raises(ValueError, match="text prompts"):
        await adapter.segment(req)


@pytest.mark.asyncio
async def test_box_prompt_raises_value_error(
    adapter: CLIPSegSegmentationAdapter,
) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.box,
        box=BoxPrompt(x_min=0, y_min=0, x_max=100, y_max=100),
    )
    with pytest.raises(ValueError, match="text prompts"):
        await adapter.segment(req)


@pytest.mark.asyncio
async def test_empty_text_prompt_raises(
    adapter: CLIPSegSegmentationAdapter,
) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.text,
        text_prompt="",          # explicitly empty
    )
    with pytest.raises(ValueError, match="non-empty"):
        await adapter.segment(req)


# ---------------------------------------------------------------------------
# Text prompt — segment() orchestration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_prompt_returns_response(
    adapter: CLIPSegSegmentationAdapter, infer_patch: MagicMock
) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.text,
        text_prompt="a cat",
    )
    resp = await adapter.segment(req)
    assert resp.backend == "clipseg"
    assert len(resp.masks) == 1


@pytest.mark.asyncio
async def test_text_prompt_mask_fields(
    adapter: CLIPSegSegmentationAdapter, infer_patch: MagicMock
) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.text,
        text_prompt="a dog",
    )
    resp = await adapter.segment(req)
    mask = resp.masks[0]
    assert mask.mask_b64                    # non-empty base64 string
    assert 0.0 <= mask.score <= 1.0
    assert mask.area == 64 * 64             # _STUB_MASK is all-True


@pytest.mark.asyncio
async def test_text_prompt_passes_to_infer(
    adapter: CLIPSegSegmentationAdapter, infer_patch: MagicMock
) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.text,
        text_prompt="wooden table",
    )
    await adapter.segment(req)
    infer_patch.assert_called_once()
    call_args = infer_patch.call_args
    assert call_args.args[0] == "wooden table"


@pytest.mark.asyncio
async def test_response_metadata_contains_text_prompt(
    adapter: CLIPSegSegmentationAdapter, infer_patch: MagicMock
) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.text,
        text_prompt="red bicycle",
    )
    resp = await adapter.segment(req)
    assert resp.metadata.get("text_prompt") == "red bicycle"


@pytest.mark.asyncio
async def test_response_has_unique_request_ids(
    adapter: CLIPSegSegmentationAdapter, infer_patch: MagicMock
) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.text,
        text_prompt="sky",
    )
    r1 = await adapter.segment(req)
    r2 = await adapter.segment(req)
    assert r1.request_id != r2.request_id


@pytest.mark.asyncio
async def test_response_latency_is_non_negative(
    adapter: CLIPSegSegmentationAdapter, infer_patch: MagicMock
) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.text,
        text_prompt="sky",
    )
    resp = await adapter.segment(req)
    assert resp.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# Logits passthrough
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_return_logits_true(
    adapter: CLIPSegSegmentationAdapter, infer_patch: MagicMock
) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.text,
        text_prompt="water",
        return_logits=True,
    )
    resp = await adapter.segment(req)
    assert resp.masks[0].logits_b64 is not None


@pytest.mark.asyncio
async def test_return_logits_false(
    adapter: CLIPSegSegmentationAdapter, infer_patch: MagicMock
) -> None:
    req = SegmentRequest(
        image=_STUB_IMAGE,
        prompt_type=PromptType.text,
        text_prompt="water",
        return_logits=False,
    )
    resp = await adapter.segment(req)
    assert resp.masks[0].logits_b64 is None


# ---------------------------------------------------------------------------
# Score clamping
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_score_clamped_above_one(
    adapter: CLIPSegSegmentationAdapter,
) -> None:
    """Scores > 1 from the model must be clamped to 1.0."""
    with patch.object(
        adapter, "_infer", return_value=(_STUB_MASK, _STUB_LOGITS, 1.5)
    ):
        req = SegmentRequest(
            image=_STUB_IMAGE,
            prompt_type=PromptType.text,
            text_prompt="sky",
        )
        resp = await adapter.segment(req)
        assert resp.masks[0].score <= 1.0


@pytest.mark.asyncio
async def test_score_clamped_below_zero(
    adapter: CLIPSegSegmentationAdapter,
) -> None:
    """Scores < 0 from the model must be clamped to 0.0."""
    with patch.object(
        adapter, "_infer", return_value=(_STUB_MASK, _STUB_LOGITS, -0.1)
    ):
        req = SegmentRequest(
            image=_STUB_IMAGE,
            prompt_type=PromptType.text,
            text_prompt="sky",
        )
        resp = await adapter.segment(req)
        assert resp.masks[0].score >= 0.0


# ---------------------------------------------------------------------------
# Area calculation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_area_counts_true_pixels(
    adapter: CLIPSegSegmentationAdapter,
) -> None:
    partial_mask = np.zeros((4, 4), dtype=bool)
    partial_mask[0, 0] = True
    partial_mask[1, 1] = True  # area = 2
    with patch.object(
        adapter, "_infer", return_value=(partial_mask, _STUB_LOGITS, 0.9)
    ):
        req = SegmentRequest(
            image=_STUB_IMAGE,
            prompt_type=PromptType.text,
            text_prompt="object",
        )
        resp = await adapter.segment(req)
        assert resp.masks[0].area == 2


# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------

def test_mask_to_b64_roundtrip() -> None:
    """Boolean mask → base64 PNG → decoded array must preserve foreground."""
    mask = np.array([[True, False], [False, True]], dtype=bool)
    b64 = _mask_to_b64(mask)
    raw = base64.b64decode(b64)
    arr = np.array(Image.open(io.BytesIO(raw)))
    # White pixels (255) correspond to True, black (0) to False
    assert arr[0, 0] == 255
    assert arr[0, 1] == 0
    assert arr[1, 0] == 0
    assert arr[1, 1] == 255


def test_logits_to_b64_uniform_input() -> None:
    """Uniform logit map must not raise division-by-zero."""
    uniform = np.zeros((16, 16), dtype=np.float32)
    b64 = _logits_to_b64(uniform)
    assert b64  # non-empty


def test_logits_to_b64_preserves_relative_magnitude() -> None:
    """The brighter pixels in the encoded PNG correspond to higher logits."""
    logits = np.zeros((4, 4), dtype=np.float32)
    logits[0, 0] = 10.0   # high
    logits[3, 3] = -10.0  # low
    b64 = _logits_to_b64(logits)
    raw = base64.b64decode(b64)
    arr = np.array(Image.open(io.BytesIO(raw)))
    assert arr[0, 0] > arr[3, 3]


def test_b64_to_pil_returns_rgb_image() -> None:
    img = _b64_to_pil(_STUB_IMAGE)
    assert img.mode == "RGB"


# ---------------------------------------------------------------------------
# Model loading error paths (no mock injected)
# ---------------------------------------------------------------------------

def test_load_model_raises_when_transformers_missing() -> None:
    """_load_model raises RuntimeError when 'transformers' is not installed."""
    import sys

    # Shadow the transformers module to force ImportError.
    original = sys.modules.get("transformers")
    sys.modules["transformers"] = None  # type: ignore[assignment]

    try:
        fresh = CLIPSegSegmentationAdapter()
        with pytest.raises(RuntimeError, match="transformers"):
            fresh._load_model()
    finally:
        if original is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original


def test_load_model_raises_when_clipseg_model_empty() -> None:
    """_load_model raises RuntimeError when CLIPSEG_MODEL is empty."""
    import sys
    from unittest.mock import MagicMock

    # Provide a fake transformers module so the import succeeds.
    fake_transformers = MagicMock()
    original = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers

    try:
        fresh = CLIPSegSegmentationAdapter()
        # Override the setting to simulate an empty model name.
        from segmentation_service.config import Settings
        from unittest.mock import PropertyMock

        mock_settings = MagicMock(spec=Settings)
        mock_settings.clipseg_model = ""   # empty → should raise
        fresh._settings = mock_settings

        with pytest.raises(RuntimeError, match="CLIPSEG_MODEL"):
            fresh._load_model()
    finally:
        if original is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original
