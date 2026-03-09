"""GET /capabilities endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from segmentation_service.adapters import get_adapter
from segmentation_service.schemas.capabilities import CapabilitiesResponse

router = APIRouter()


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    summary="Describe what the active backend supports",
    tags=["ops"],
)
async def capabilities() -> CapabilitiesResponse:
    """Returns the active backend's supported prompt types, image limits, etc."""
    return get_adapter().capabilities()
