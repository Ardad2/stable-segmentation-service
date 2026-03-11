"""Health-check schemas."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    backend: str
    # Additive v1 evolution: lets clients verify the API contract version.
    # Old clients that ignore unknown fields are unaffected.
    api_version: str = "1.0"
