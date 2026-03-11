"""Capabilities schemas."""

from __future__ import annotations

from pydantic import BaseModel


class CapabilitiesResponse(BaseModel):
    backend: str
    supported_input_types: list[str]
    supported_prompt_types: list[str]
    max_image_width: int
    max_image_height: int
    notes: str = ""
    # Additive v1 evolution: lets clients verify the API contract version.
    # Old clients that ignore unknown fields are unaffected.
    api_version: str = "1.0"
