"""v2 segmentation request / response schemas.

Breaking change vs v1
---------------------
v1 uses a *flat* request layout::

    {
      "image": "...",
      "prompt_type": "point",
      "points": [{"x": 10, "y": 20, "label": 1}]
    }

v2 requires a *nested prompt envelope*::

    {
      "image": "...",
      "prompt": {
        "type": "point",
        "points": [{"x": 10, "y": 20, "label": 1}]
      }
    }

The ``prompt`` field is **required** and has no default.  A v1-style request
that sends ``prompt_type`` + ``points`` at the top level will receive a 422
response from the v2 endpoint because ``prompt`` is missing.

Sending a v2-style request to the v1 endpoint also fails: the top-level
``prompt`` field is an unknown extra field (ignored by Pydantic), but
``prompt_type`` defaults to ``"point"`` and ``points`` is absent, so the v1
route guard returns 422 for a missing point list.

All other fields (``image_format``, ``multimask_output``, ``return_logits``)
carry over unchanged from v1.

The response renames the inner mask field ``mask_b64`` → ``mask_data`` so that
any v1 client code that hard-codes ``response["masks"][0]["mask_b64"]`` will
fail with a KeyError rather than silently receiving wrong data.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from segmentation_service.schemas.segment import BoxPrompt, MaskResult, PointPrompt, PromptType


# ---------------------------------------------------------------------------
# v2 request
# ---------------------------------------------------------------------------

class V2PromptEnvelope(BaseModel):
    """Unified prompt object — the single breaking change vs v1.

    v1 scattered prompt data across four flat fields (``prompt_type``,
    ``points``, ``box``, ``text_prompt``).  v2 groups them into one object.
    The inner field ``text`` replaces v1's ``text_prompt``.
    """

    type: PromptType
    points: list[PointPrompt] | None = None
    box: BoxPrompt | None = None
    text: str | None = None  # renamed from v1's text_prompt


class V2SegmentRequest(BaseModel):
    """Payload sent to POST /api/v2/segment.

    The only structural difference from v1 is that the five flat prompt fields
    (``prompt_type``, ``points``, ``box``, ``text_prompt``) are replaced by
    a single required ``prompt`` envelope.  Wire-incompatible with v1.
    """

    image: str = Field(..., description="Base64-encoded image or HTTP(S) URL")
    image_format: str = Field(default="png", description="png | jpeg | webp")

    # BREAKING CHANGE: required prompt envelope (was flat fields in v1)
    prompt: V2PromptEnvelope

    multimask_output: bool = False
    return_logits: bool = False

    @field_validator("image_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        allowed = {"png", "jpeg", "jpg", "webp"}
        if v.lower() not in allowed:
            raise ValueError(f"image_format must be one of {allowed}")
        return v.lower()


# ---------------------------------------------------------------------------
# v2 response
# ---------------------------------------------------------------------------

class V2MaskResult(BaseModel):
    """A single predicted mask — v2 wire format.

    The mask payload field is renamed from v1's ``mask_b64`` to ``mask_data``.
    This is a deliberate breaking change: any v1 client code that accesses
    ``mask["mask_b64"]`` will receive a ``KeyError`` rather than silently
    continuing with wrong data.
    """

    mask_data: str = Field(..., description="Base64-encoded binary mask (PNG)")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    area: int = Field(..., description="Number of foreground pixels")
    logits_b64: str | None = Field(
        default=None,
        description="Base64-encoded raw logits (float32 PNG), if requested",
    )

    @classmethod
    def from_v1(cls, mask: MaskResult) -> "V2MaskResult":
        """Convert a v1 MaskResult to v2 wire format."""
        return cls(
            mask_data=mask.mask_b64,
            score=mask.score,
            area=mask.area,
            logits_b64=mask.logits_b64,
        )


class V2SegmentResponse(BaseModel):
    """Response from POST /api/v2/segment."""

    request_id: str
    api_version: str = "2.0"
    backend: str
    masks: list[V2MaskResult]
    latency_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)
