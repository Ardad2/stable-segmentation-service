"""Direct (non-HTTP) adapter invocation helpers for evaluation.

These functions bypass the FastAPI/HTTP layer and call adapter.segment()
directly in a fresh asyncio event loop.  They are evaluation-oriented
and NOT part of the public service API.

They are used by:
- benchmark/direct_vs_served.py    — latency-overhead comparison
- scripts/evaluate_correctness.py  — output-correctness comparison

Design
------
- Reuse the existing adapter classes (Mock, SAM2, CLIPSeg) for model
  loading.  This shares the same lazy-loading logic as the production service
  so results are directly comparable.
- asyncio.run() creates a fresh event loop per call, which is correct for
  synchronous benchmark scripts.  Do not use from inside an already-running
  event loop (Jupyter, pytest with asyncio mode) — in those contexts,
  call adapter.segment() via await directly.
- Each runner can be asked to warm up by calling warm_up() once before
  timing starts.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from segmentation_service.adapters.base import BaseSegmentationAdapter
from segmentation_service.schemas.segment import SegmentRequest, SegmentResponse


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class DirectResult:
    """Result of a single direct (non-HTTP) adapter invocation."""

    response: SegmentResponse
    latency_ms: float
    backend: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_direct(
    adapter: BaseSegmentationAdapter,
    request: SegmentRequest,
) -> DirectResult:
    """Call adapter.segment(request) synchronously via a fresh event loop.

    Wall-clock time around the call is recorded as ``latency_ms``.

    Returns
    -------
    DirectResult
        On adapter error, ``error`` is set to the exception message and
        ``response`` is a zero-mask placeholder.
    """
    t0 = time.perf_counter()
    try:
        response = asyncio.run(adapter.segment(request))
        latency_ms = (time.perf_counter() - t0) * 1000
        return DirectResult(
            response=response,
            latency_ms=round(latency_ms, 3),
            backend=adapter.name,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        # Return a placeholder so callers can still inspect the error without
        # branching on None.
        placeholder = SegmentResponse(
            request_id="error",
            backend=getattr(adapter, "name", "unknown"),
            masks=[],
            latency_ms=round(latency_ms, 3),
        )
        return DirectResult(
            response=placeholder,
            latency_ms=round(latency_ms, 3),
            backend=getattr(adapter, "name", "unknown"),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# DirectRunner class
# ---------------------------------------------------------------------------

class DirectRunner:
    """Wrapper around a BaseSegmentationAdapter for benchmark / eval use.

    Example
    -------
    ::

        runner = DirectRunner.for_mock()
        result = runner.run(request)
        print(result.latency_ms, result.response.masks[0].score)
    """

    def __init__(self, adapter: BaseSegmentationAdapter) -> None:
        self._adapter = adapter

    @property
    def backend(self) -> str:
        return self._adapter.name

    def run(self, request: SegmentRequest) -> DirectResult:
        """Run one inference call directly, returning timing and result."""
        return run_direct(self._adapter, request)

    def warm_up(self, request: SegmentRequest) -> None:
        """Run one warm-up inference call (result discarded).

        Call this once before timing measurements to ensure model weights
        are loaded and caches are warm.
        """
        run_direct(self._adapter, request)

    # ------------------------------------------------------------------
    # Named constructors
    # ------------------------------------------------------------------

    @classmethod
    def for_mock(cls) -> "DirectRunner":
        """Return a runner backed by MockSegmentationAdapter.

        Never fails — no ML dependencies required.
        """
        from segmentation_service.adapters.mock_adapter import MockSegmentationAdapter

        return cls(MockSegmentationAdapter())

    @classmethod
    def for_sam2(
        cls,
        *,
        checkpoint: str,
        config: str,
        device: str = "cpu",
    ) -> "DirectRunner":
        """Return a runner backed by SAM2SegmentationAdapter.

        Parameters
        ----------
        checkpoint:
            Path to the SAM2 .pt weights file (SAM2_CHECKPOINT).
        config:
            SAM2 YAML config name (SAM2_CONFIG).
        device:
            ``"cpu"``, ``"cuda"``, or ``"mps"``.

        Raises
        ------
        RuntimeError
            If the ``sam2`` package is not installed or the checkpoint is missing.
        """
        import os

        from segmentation_service.adapters.sam2_adapter import SAM2SegmentationAdapter

        # Temporarily set env vars so the adapter's Settings reads them.
        # This is evaluation-only code — not part of the production path.
        os.environ["SAM2_CHECKPOINT"] = checkpoint
        os.environ["SAM2_CONFIG"] = config
        os.environ["MODEL_DEVICE"] = device
        os.environ["SEGMENTATION_BACKEND"] = "sam2"

        from segmentation_service.config import get_settings

        get_settings.cache_clear()  # type: ignore[attr-defined]

        adapter = SAM2SegmentationAdapter()
        adapter._load_predictor()  # eager-load so warm_up() skip is optional
        return cls(adapter)

    @classmethod
    def for_clipseg(
        cls,
        *,
        model: str = "CIDAS/clipseg-rd64-refined",
        device: str = "cpu",
    ) -> "DirectRunner":
        """Return a runner backed by CLIPSegSegmentationAdapter.

        Parameters
        ----------
        model:
            HuggingFace model ID or local path (CLIPSEG_MODEL).
        device:
            ``"cpu"``, ``"cuda"``, or ``"mps"``.

        Raises
        ------
        RuntimeError
            If ``transformers`` / ``torch`` are not installed.
        """
        import os

        from segmentation_service.adapters.clipseg_adapter import CLIPSegSegmentationAdapter

        os.environ["CLIPSEG_MODEL"] = model
        os.environ["MODEL_DEVICE"] = device
        os.environ["SEGMENTATION_BACKEND"] = "clipseg"

        from segmentation_service.config import get_settings

        get_settings.cache_clear()  # type: ignore[attr-defined]

        adapter = CLIPSegSegmentationAdapter()
        adapter._load_model()  # eager-load
        return cls(adapter)
