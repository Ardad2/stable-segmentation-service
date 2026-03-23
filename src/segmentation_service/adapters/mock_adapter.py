"""Mock adapter — returns synthetic masks without loading any real model.

Useful for:
- Local development without a GPU or model weights.
- Unit / integration tests that need deterministic output.
- Smoke-testing the API layer end-to-end.
"""

from __future__ import annotations

import time
import uuid

from segmentation_service.adapters.base import BaseSegmentationAdapter
from segmentation_service.logging_config import LogContext, get_logger
from segmentation_service.schemas.capabilities import CapabilitiesResponse
from segmentation_service.schemas.segment import MaskResult, SegmentRequest, SegmentResponse

log = LogContext(get_logger(__name__))

# 4×4 all-white PNG (Pillow-decodable, base64-encoded)
_STUB_MASK_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAAAAACMmsGiAAAAEUlEQVR4nGP8"
    "z8DAwMSAQgAAE1EBB4BOjR4AAAAASUVORK5CYII="
)

class MockSegmentationAdapter(BaseSegmentationAdapter):
    """Deterministic stub — no GPU, no weights required."""

    name = "mock"

    def capabilities(self) -> CapabilitiesResponse:
        return CapabilitiesResponse(
            backend=self.name,
            supported_input_types=["base64", "url"],
            supported_prompt_types=["point", "box", "text"],
            max_image_width=4096,
            max_image_height=4096,
            notes="Mock adapter — returns synthetic masks for development/testing.",
        )

    async def segment(self, request: SegmentRequest) -> SegmentResponse:
        t0 = time.perf_counter()
        log.debug("MockAdapter.segment called", prompt_type=request.prompt_type)

        # Simulate a small fixed latency so benchmarks have something realistic.
        # In a real adapter you would await your model inference here.
        mask = MaskResult(
            mask_b64=_STUB_MASK_B64,
            score=0.99,
            area=16,
            logits_b64=_STUB_MASK_B64 if request.return_logits else None,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        return SegmentResponse(
            request_id=str(uuid.uuid4()),
            backend=self.name,
            masks=[mask] * (3 if request.multimask_output else 1),
            latency_ms=round(latency_ms, 3),
            metadata={"stub": True},
        )
