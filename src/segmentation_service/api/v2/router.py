"""Aggregates all v2 endpoint routers into a single v2 router."""

from __future__ import annotations

from fastapi import APIRouter

from segmentation_service.api.v2 import capabilities, health, segment

v2_router = APIRouter(prefix="/v2")

v2_router.include_router(health.router)
v2_router.include_router(capabilities.router)
v2_router.include_router(segment.router)
