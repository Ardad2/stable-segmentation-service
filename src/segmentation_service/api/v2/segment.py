"""POST /api/v2/segment endpoint.

Breaking change vs v1
---------------------
This endpoint accepts ``V2SegmentRequest``, which requires a nested ``prompt``
envelope object instead of the flat ``prompt_type`` + ``points``/``box``/
``text_prompt`` fields used in v1.

A v1-style request body sent to this endpoint will receive HTTP 422 because
the required ``prompt`` field is missing.

Internally the route converts the v2 request to a v1 ``SegmentRequest`` before
calling the adapter — the adapter layer is unchanged and backend-agnostic.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from segmentation_service.adapters import get_adapter
from segmentation_service.logging_config import LogContext, get_logger
from segmentation_service.schemas.segment import SegmentRequest
from segmentation_service.schemas.v2segment import (
    V2MaskResult,
    V2SegmentRequest,
    V2SegmentResponse,
)

router = APIRouter()
log = LogContext(get_logger(__name__))


def _to_v1_request(req: V2SegmentRequest) -> SegmentRequest:
    """Convert a v2 request envelope into the v1 SegmentRequest the adapters expect."""
    return SegmentRequest(
        image=req.image,
        image_format=req.image_format,
        prompt_type=req.prompt.type,
        points=req.prompt.points,
        box=req.prompt.box,
        text_prompt=req.prompt.text,
        multimask_output=req.multimask_output,
        return_logits=req.return_logits,
    )


@router.post(
    "/segment",
    response_model=V2SegmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Run segmentation inference (v2)",
    tags=["inference"],
)
async def segment(request: V2SegmentRequest) -> V2SegmentResponse:
    """
    v2 endpoint — accepts a unified ``prompt`` envelope object.

    Wire-incompatible with v1: a v1 client sending flat ``prompt_type`` +
    ``points``/``box``/``text_prompt`` fields will receive HTTP 422 because
    the required ``prompt`` field is absent.

    Internally converts to the v1 adapter interface so all backends work
    without modification.
    """
    v1_req = _to_v1_request(request)

    # Validate prompt contents (mirrors the v1 route guard).
    if v1_req.prompt_type == "point" and not v1_req.points:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="prompt.type='point' requires at least one entry in 'prompt.points'.",
        )
    if v1_req.prompt_type == "box" and v1_req.box is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="prompt.type='box' requires a 'prompt.box' object.",
        )
    if v1_req.prompt_type == "text" and not v1_req.text_prompt:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="prompt.type='text' requires a non-empty 'prompt.text'.",
        )

    try:
        adapter = get_adapter()
        v1_resp = await adapter.segment(v1_req)
    except Exception as exc:
        log.error("Segmentation failed (v2)", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Segmentation inference failed. Check server logs.",
        ) from exc

    log.info(
        "Segment request completed (v2)",
        backend=v1_resp.backend,
        num_masks=len(v1_resp.masks),
        latency_ms=v1_resp.latency_ms,
    )

    return V2SegmentResponse(
        request_id=v1_resp.request_id,
        api_version="2.0",
        backend=v1_resp.backend,
        masks=[V2MaskResult.from_v1(m) for m in v1_resp.masks],
        latency_ms=v1_resp.latency_ms,
        metadata=v1_resp.metadata,
    )
