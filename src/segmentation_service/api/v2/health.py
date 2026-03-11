"""GET /api/v2/health endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from segmentation_service import __version__
from segmentation_service.adapters import get_adapter
from segmentation_service.schemas.health import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check (v2)",
    tags=["ops"],
)
async def health() -> HealthResponse:
    """Returns service status, version, and active backend name."""
    adapter = get_adapter()
    return HealthResponse(
        status="ok",
        version=__version__,
        backend=adapter.name,
        api_version="2.0",
    )
