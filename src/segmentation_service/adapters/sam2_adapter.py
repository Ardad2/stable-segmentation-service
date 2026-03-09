"""SAM2 segmentation backend adapter.

This adapter wraps the Meta SAM2 (Segment Anything Model 2) library and
translates its output into the service's common SegmentResponse schema.

Requirements
------------
Install the SAM2 library and its dependencies before enabling this backend::

    pip install 'git+https://github.com/facebookresearch/sam2.git'

Download model weights from the SAM2 release page, then set:

    SAM2_CHECKPOINT=weights/sam2_hiera_large.pt   # path to .pt file
    SAM2_CONFIG=sam2_hiera_l.yaml                 # config name (no path)
    SEGMENTATION_BACKEND=sam2
    MODEL_DEVICE=cuda                              # cpu | cuda | mps

Supported prompt types
----------------------
* point — one or more (x, y, label) coordinates
* box   — a single axis-aligned bounding box

Text prompts are NOT supported by SAM2 and will raise ValueError, which the
endpoint layer converts to an HTTP 500. Clients should check /capabilities
before sending a request to know which prompt types are accepted.

Model loading
-------------
The SAM2 predictor is loaded lazily on the first call to segment().
This lets the service start up and pass health checks before weights
are fully loaded, and avoids penalising the mock backend with an import
that would fail if sam2 is not installed.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from segmentation_service.adapters.base import BaseSegmentationAdapter
from segmentation_service.config import get_settings
from segmentation_service.logging_config import LogContext, get_logger
from segmentation_service.schemas.capabilities import CapabilitiesResponse
from segmentation_service.schemas.segment import (
    MaskResult,
    PromptType,
    SegmentRequest,
    SegmentResponse,
)

if TYPE_CHECKING:
    pass

log = LogContext(get_logger(__name__))


# ---------------------------------------------------------------------------
# Private image helpers
# ---------------------------------------------------------------------------

def _b64_to_numpy(b64_str: str) -> np.ndarray:
    """Decode a base64-encoded image string to an RGB uint8 numpy array."""
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)


async def _url_to_numpy(url: str) -> np.ndarray:
    """Fetch an image from a URL and return an RGB uint8 numpy array."""
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return np.array(img)


def _mask_to_b64(mask: np.ndarray) -> str:
    """Encode a boolean (H, W) mask as a base64 PNG (L-mode, 0/255)."""
    arr = (mask.astype(np.uint8)) * 255
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _logits_to_b64(logits: np.ndarray) -> str:
    """Normalize float32 logits (H, W) to [0, 255] uint8 and return base64 PNG.

    SAM2 returns low-resolution logits shaped (256, 256).  We store them as
    a single-channel PNG so downstream consumers can apply their own threshold.
    """
    lo, hi = float(logits.min()), float(logits.max())
    if hi > lo:
        norm = ((logits - lo) / (hi - lo) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(logits, dtype=np.uint8)
    img = Image.fromarray(norm, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class SAM2SegmentationAdapter(BaseSegmentationAdapter):
    """Adapter for the SAM2 image segmentation model.

    The adapter is intentionally thin: it only handles the translation
    between the service schema and SAM2's numpy-based API.  All
    backend-specific logic lives here; the API layer is unaware of SAM2.
    """

    name = "sam2"

    def __init__(self) -> None:
        self._settings = get_settings()
        # Predictor is None until the first segment() call (lazy loading).
        self._predictor: Any | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_predictor(self) -> Any:
        """Instantiate and cache the SAM2ImagePredictor (thread-safe via GIL).

        Raises RuntimeError if:
        - The ``sam2`` package is not installed.
        - SAM2_CHECKPOINT or SAM2_CONFIG env vars are not set.
        """
        if self._predictor is not None:
            return self._predictor

        try:
            from sam2.build_sam import build_sam2  # type: ignore[import]
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "The 'sam2' package is not installed. "
                "Install it with:\n"
                "  pip install 'git+https://github.com/facebookresearch/sam2.git'\n"
                "then download the model weights before switching to this backend."
            ) from exc

        checkpoint = self._settings.sam2_checkpoint
        config = self._settings.sam2_config

        if not checkpoint:
            raise RuntimeError(
                "SAM2_CHECKPOINT is not set. "
                "Point it to a SAM2 .pt weights file "
                "(e.g. SAM2_CHECKPOINT=weights/sam2_hiera_large.pt)."
            )
        if not config:
            raise RuntimeError(
                "SAM2_CONFIG is not set. "
                "Set it to the SAM2 YAML config name "
                "(e.g. SAM2_CONFIG=sam2_hiera_l.yaml)."
            )

        device = self._settings.model_device
        log.info(
            "Loading SAM2 model",
            checkpoint=checkpoint,
            config=config,
            device=device,
        )
        model = build_sam2(config, checkpoint, device=device)
        self._predictor = SAM2ImagePredictor(model)
        log.info("SAM2 model ready", device=device)
        return self._predictor

    # ------------------------------------------------------------------
    # BaseSegmentationAdapter interface
    # ------------------------------------------------------------------

    def capabilities(self) -> CapabilitiesResponse:
        return CapabilitiesResponse(
            backend=self.name,
            supported_input_types=["base64", "url"],
            supported_prompt_types=["point", "box"],
            max_image_width=4096,
            max_image_height=4096,
            notes=(
                "Text prompts are not supported by SAM2. "
                "Multi-mask output is available for point prompts only. "
                "Pass return_logits=true to receive low-resolution (256×256) "
                "logit maps alongside each mask."
            ),
        )

    async def segment(self, request: SegmentRequest) -> SegmentResponse:
        if request.prompt_type == PromptType.text:
            raise ValueError(
                "SAM2 does not support text prompts. "
                "Use prompt_type='point' or prompt_type='box'."
            )

        # Decode image to numpy outside the thread pool (httpx is async-native).
        image_np = await self._decode_image(request)

        # SAM2 inference is synchronous and CPU/GPU-bound; run in a thread
        # so the event loop is not blocked.
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        masks_np, scores_np, logits_np = await loop.run_in_executor(
            None, self._infer, request, image_np
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        mask_results = [
            MaskResult(
                mask_b64=_mask_to_b64(masks_np[i]),
                score=float(np.clip(scores_np[i], 0.0, 1.0)),
                area=int(masks_np[i].sum()),
                logits_b64=_logits_to_b64(logits_np[i]) if request.return_logits else None,
            )
            for i in range(len(masks_np))
        ]

        return SegmentResponse(
            request_id=str(uuid.uuid4()),
            backend=self.name,
            masks=mask_results,
            latency_ms=round(latency_ms, 3),
            metadata={
                "device": self._settings.model_device,
                "checkpoint": self._settings.sam2_checkpoint,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _decode_image(self, request: SegmentRequest) -> np.ndarray:
        """Decode the image field to an RGB uint8 numpy array.

        Detects whether the field contains an HTTP(S) URL or a raw base64
        string, and handles each case appropriately.
        """
        img = request.image
        if img.startswith("http://") or img.startswith("https://"):
            return await _url_to_numpy(img)
        return _b64_to_numpy(img)

    def _infer(
        self,
        request: SegmentRequest,
        image_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run SAM2 inference synchronously.

        Intended to be called via ``loop.run_in_executor`` so it doesn't
        block the event loop.

        Returns
        -------
        masks   : bool ndarray of shape (N, H, W)
        scores  : float ndarray of shape (N,)
        logits  : float32 ndarray of shape (N, 256, 256)  — low-res SAM2 logits
        """
        predictor = self._load_predictor()
        predictor.set_image(image_np)

        if request.prompt_type == PromptType.point:
            point_coords = np.array(
                [[p.x, p.y] for p in request.points],  # type: ignore[union-attr]
                dtype=np.float32,
            )
            point_labels = np.array(
                [p.label for p in request.points],  # type: ignore[union-attr]
                dtype=np.int32,
            )
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=request.multimask_output,
            )
        else:  # box
            box = request.box
            box_arr = np.array(
                [box.x_min, box.y_min, box.x_max, box.y_max],  # type: ignore[union-attr]
                dtype=np.float32,
            )
            # SAM2 does not support multi-mask output for box prompts.
            masks, scores, logits = predictor.predict(
                box=box_arr,
                multimask_output=False,
            )

        return masks, scores, logits
