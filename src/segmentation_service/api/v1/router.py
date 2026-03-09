"""Aggregates all v1 endpoint routers into a single v1 router."""

from __future__ import annotations

from fastapi import APIRouter

from segmentation_service.api.v1 import capabilities, health, segment

v1_router = APIRouter(prefix="/v1")

v1_router.include_router(health.router)
v1_router.include_router(capabilities.router)
v1_router.include_router(segment.router)
