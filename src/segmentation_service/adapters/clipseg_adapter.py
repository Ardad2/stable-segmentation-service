"""CLIPSeg segmentation backend adapter.

CLIPSeg (CLIP-guided Segmentation) is a text-driven segmentation model from
CIDAS/HuggingFace that produces a dense probability map for any natural-language
description of an image region.

This adapter is the complement to SAM2: where SAM2 requires geometric prompts
(points / boxes), CLIPSeg requires a text prompt — which is the prompt modality
the API already models but SAM2 rejects.

Requirements
------------
Install the CLIPSeg extras before enabling this backend::

    pip install -e ".[clipseg]"

Then set the following environment variables:

    SEGMENTATION_BACKEND=clipseg
    CLIPSEG_MODEL=CIDAS/clipseg-rd64-refined   # HuggingFace model ID or local path
    MODEL_DEVICE=cpu                            # cpu | cuda | mps

Supported prompt types
----------------------
* text — a natural-language description of the region to segment
         (e.g. "the cat", "a wooden chair", "water in the background")

Point and box prompts are NOT supported by CLIPSeg and will raise ValueError,
which the endpoint converts to HTTP 500. Clients should check /capabilities
before sending requests to know which prompt types are accepted.

Output format
-------------
CLIPSeg produces a single (H, W) logit map per text prompt. This adapter:
- applies sigmoid to obtain a probability map
- thresholds at 0.5 to produce a binary mask
- reports score = max sigmoid activation across the image
- optionally returns the raw logit map via return_logits=true

Model loading
-------------
The CLIPSeg processor and model are loaded lazily on the first segment() call,
consistent with the SAM2 adapter pattern.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
import uuid
from typing import Any

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

log = LogContext(get_logger(__name__))

# Sigmoid threshold: pixels with sigmoid(logit) > this value belong to the mask.
_MASK_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Private image helpers  (mirrors sam2_adapter conventions)
# ---------------------------------------------------------------------------

def _b64_to_pil(b64_str: str) -> Image.Image:
    """Decode a base64-encoded image string to an RGB PIL Image."""
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


async def _url_to_pil(url: str) -> Image.Image:
    """Fetch an image from a URL and return an RGB PIL Image."""
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _mask_to_b64(mask: np.ndarray) -> str:
    """Encode a boolean (H, W) mask as a base64 PNG (L-mode, 0/255)."""
    arr = mask.astype(np.uint8) * 255
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _logits_to_b64(logits: np.ndarray) -> str:
    """Normalize a float32 (H, W) logit map to [0, 255] uint8 and return base64 PNG.

    The normalised image allows downstream consumers to apply any threshold
    without needing to know the original logit scale.
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

class CLIPSegSegmentationAdapter(BaseSegmentationAdapter):
    """Adapter for CLIPSeg text-guided image segmentation.

    The adapter is intentionally thin: it only handles the translation
    between the service schema and CLIPSeg's HuggingFace API.  All
    backend-specific logic lives here; the API layer is unaware of CLIPSeg.
    """

    name = "clipseg"

    def __init__(self) -> None:
        self._settings = get_settings()
        # Processor and model are None until the first segment() call.
        self._processor: Any | None = None
        self._model: Any | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> tuple[Any, Any]:
        """Load and cache the CLIPSeg processor and model.

        Returns (processor, model). Subsequent calls return the cached pair.

        Raises RuntimeError if:
        - The ``transformers`` package is not installed.
        - CLIPSEG_MODEL env var is empty.
        """
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        try:
            from transformers import (  # type: ignore[import]
                CLIPSegForImageSegmentation,
                CLIPSegProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' package is not installed. "
                "Install the CLIPSeg extras with:\n"
                "  pip install -e '.[clipseg]'\n"
                "  # or: pip install transformers torch"
            ) from exc

        model_name = self._settings.clipseg_model
        if not model_name:
            raise RuntimeError(
                "CLIPSEG_MODEL is not set. "
                "Set it to a HuggingFace model ID or a local path "
                "(e.g. CLIPSEG_MODEL=CIDAS/clipseg-rd64-refined)."
            )

        device = self._settings.model_device
        log.info("Loading CLIPSeg model", model=model_name, device=device)

        processor = CLIPSegProcessor.from_pretrained(model_name)
        model = CLIPSegForImageSegmentation.from_pretrained(model_name)
        model = model.to(device)
        model.eval()

        self._processor = processor
        self._model = model
        log.info("CLIPSeg model ready", model=model_name, device=device)
        return processor, model

    # ------------------------------------------------------------------
    # BaseSegmentationAdapter interface
    # ------------------------------------------------------------------

    def capabilities(self) -> CapabilitiesResponse:
        return CapabilitiesResponse(
            backend=self.name,
            supported_input_types=["base64", "url"],
            supported_prompt_types=["text"],
            max_image_width=4096,
            max_image_height=4096,
            notes=(
                "CLIPSeg supports text-guided segmentation only. "
                "Point and box prompts are not supported. "
                "One mask is returned per request, corresponding to the text prompt. "
                "Pass return_logits=true to receive the raw (H×W) logit map "
                "alongside the thresholded binary mask."
            ),
        )

    async def segment(self, request: SegmentRequest) -> SegmentResponse:
        if request.prompt_type != PromptType.text:
            raise ValueError(
                f"CLIPSeg only supports text prompts. "
                f"Received prompt_type='{request.prompt_type.value}'. "
                "Use prompt_type='text' with a non-empty 'text_prompt'."
            )

        if not request.text_prompt:
            raise ValueError(
                "CLIPSeg requires a non-empty 'text_prompt' "
                "when prompt_type='text'."
            )

        pil_image = await self._decode_image(request)

        # CLIPSeg inference is synchronous and CPU/GPU-bound; run in a thread
        # to avoid blocking the event loop.
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        mask_np, logits_np, score = await loop.run_in_executor(
            None, self._infer, request.text_prompt, pil_image
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        mask_result = MaskResult(
            mask_b64=_mask_to_b64(mask_np),
            score=float(np.clip(score, 0.0, 1.0)),
            area=int(mask_np.sum()),
            logits_b64=_logits_to_b64(logits_np) if request.return_logits else None,
        )

        return SegmentResponse(
            request_id=str(uuid.uuid4()),
            backend=self.name,
            masks=[mask_result],
            latency_ms=round(latency_ms, 3),
            metadata={
                "device": self._settings.model_device,
                "model": self._settings.clipseg_model,
                "text_prompt": request.text_prompt,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _decode_image(self, request: SegmentRequest) -> Image.Image:
        """Decode the image field to an RGB PIL Image.

        Detects whether the field contains an HTTP(S) URL or a raw base64
        string, and handles each case appropriately.
        """
        img = request.image
        if img.startswith("http://") or img.startswith("https://"):
            return await _url_to_pil(img)
        return _b64_to_pil(img)

    def _infer(
        self,
        text_prompt: str,
        pil_image: Image.Image,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Run CLIPSeg inference synchronously.

        Intended to be called via ``loop.run_in_executor`` so it does not
        block the event loop.

        Parameters
        ----------
        text_prompt : str
            Natural-language description of the region to segment.
        pil_image : PIL.Image.Image
            RGB image to segment.

        Returns
        -------
        mask_np   : bool ndarray of shape (H, W) — thresholded binary mask
        logits_np : float32 ndarray of shape (H, W) — raw model logits
        score     : float in [0, 1] — peak sigmoid activation (confidence)
        """
        import torch  # type: ignore[import]

        processor, model = self._load_model()
        device = self._settings.model_device

        inputs = processor(
            text=[text_prompt],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )
        # Move all tensor inputs to the target device.
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # outputs.logits shape: (batch, H, W) — squeeze the batch dimension.
        logits = outputs.logits
        if logits.dim() == 3:
            logits = logits.squeeze(0)  # → (H, W)

        logits_np = logits.float().cpu().numpy()

        # Compute sigmoid probabilities in numpy to avoid importing torch outside _infer.
        probs = 1.0 / (1.0 + np.exp(-logits_np))
        mask_np = probs > _MASK_THRESHOLD
        score = float(probs.max())

        return mask_np, logits_np, score
