"""Top-level API router that mounts all versioned sub-routers."""

from __future__ import annotations

from fastapi import APIRouter

from segmentation_service.api.v1.router import v1_router
from segmentation_service.api.v2.router import v2_router

root_router = APIRouter()
root_router.include_router(v1_router)
root_router.include_router(v2_router)
