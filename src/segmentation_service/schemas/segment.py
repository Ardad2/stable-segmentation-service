"""Segmentation request / response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PromptType(str, Enum):
    point = "point"
    box = "box"
    text = "text"


class PointPrompt(BaseModel):
    """A single (x, y) coordinate with an optional foreground/background label."""

    x: float
    y: float
    label: int = Field(default=1, description="1=foreground, 0=background")


class BoxPrompt(BaseModel):
    """Axis-aligned bounding box [x_min, y_min, x_max, y_max]."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


class SegmentRequest(BaseModel):
    """Payload sent to POST /segment."""

    # Image encoded as base64 string or a URL
    image: str = Field(..., description="Base64-encoded image or HTTP(S) URL")
    image_format: str = Field(default="png", description="png | jpeg | webp")

    prompt_type: PromptType = PromptType.point
    points: list[PointPrompt] | None = None
    box: BoxPrompt | None = None
    text_prompt: str | None = None

    # Optional per-request overrides
    multimask_output: bool = False
    return_logits: bool = False

    @field_validator("image_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        allowed = {"png", "jpeg", "jpg", "webp"}
        if v.lower() not in allowed:
            raise ValueError(f"image_format must be one of {allowed}")
        return v.lower()


class MaskResult(BaseModel):
    """A single predicted mask."""

    mask_b64: str = Field(..., description="Base64-encoded binary mask (PNG)")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    area: int = Field(..., description="Number of foreground pixels")
    logits_b64: str | None = Field(
        default=None, description="Base64-encoded raw logits (float32 PNG), if requested"
    )


class SegmentResponse(BaseModel):
    """Response from POST /segment."""

    request_id: str
    backend: str
    masks: list[MaskResult]
    latency_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Additive v1 evolution: lets clients verify the API contract version.
    # Old clients that ignore unknown fields are unaffected.
    api_version: str = "1.0"
