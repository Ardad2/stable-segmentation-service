"""POST /segment endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from segmentation_service.adapters import get_adapter
from segmentation_service.logging_config import LogContext, get_logger
from segmentation_service.schemas.segment import SegmentRequest, SegmentResponse

router = APIRouter()
log = LogContext(get_logger(__name__))


@router.post(
    "/segment",
    response_model=SegmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Run segmentation inference",
    tags=["inference"],
)
async def segment(request: SegmentRequest) -> SegmentResponse:
    """
    Accepts an image (base64 or URL) plus prompts (point / box / text) and
    returns one or more predicted masks with confidence scores.
    """
    # Basic prompt validation — the adapter should not need to worry about this.
    if request.prompt_type == "point" and not request.points:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="prompt_type='point' requires at least one entry in 'points'.",
        )
    if request.prompt_type == "box" and request.box is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="prompt_type='box' requires a 'box' object.",
        )
    if request.prompt_type == "text" and not request.text_prompt:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="prompt_type='text' requires a non-empty 'text_prompt'.",
        )

    try:
        adapter = get_adapter()
        response = await adapter.segment(request)
    except Exception as exc:
        log.error("Segmentation failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Segmentation inference failed. Check server logs.",
        ) from exc

    log.info(
        "Segment request completed",
        backend=response.backend,
        num_masks=len(response.masks),
        latency_ms=response.latency_ms,
    )
    return response
