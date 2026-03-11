"""GET /api/v2/capabilities endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from segmentation_service.adapters import get_adapter
from segmentation_service.schemas.capabilities import CapabilitiesResponse

router = APIRouter()


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    summary="Describe what the active backend supports (v2)",
    tags=["ops"],
)
async def capabilities() -> CapabilitiesResponse:
    """Returns the active backend's supported prompt types, image limits, etc."""
    caps = get_adapter().capabilities()
    # Override api_version to "2.0" so clients can distinguish the version.
    return CapabilitiesResponse(
        backend=caps.backend,
        supported_input_types=caps.supported_input_types,
        supported_prompt_types=caps.supported_prompt_types,
        max_image_width=caps.max_image_width,
        max_image_height=caps.max_image_height,
        notes=caps.notes,
        api_version="2.0",
    )
