"""Output comparison utilities for direct-vs-served evaluation.

These are pure functions that work on numpy boolean masks and on
base64-encoded mask strings (as returned by SegmentResponse).  All
functions are backend-agnostic.

Usage
-----
Typical call chain::

    from segmentation_service.eval.correctness import (
        decode_mask_b64,
        compare_responses,
        CorrectnessReport,
    )

    report = compare_responses(
        direct_masks_b64=[...],   # from adapter.segment() call
        served_masks_b64=[...],   # from /api/v1/segment HTTP response
        backend="mock",
        prompt_type="point",
    )
    print(report.all_passed)
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Mask decoding
# ---------------------------------------------------------------------------

def decode_mask_b64(b64_str: str) -> np.ndarray:
    """Decode a base64 PNG mask string to a boolean (H, W) numpy array.

    The PNG is expected to be an L-mode (greyscale) image where 255 = foreground
    and 0 = background, consistent with how both the SAM2 and CLIPSeg adapters
    encode masks.
    """
    raw = base64.b64decode(b64_str)
    arr = np.array(Image.open(io.BytesIO(raw)).convert("L"))
    return arr > 0


# ---------------------------------------------------------------------------
# Per-mask metrics
# ---------------------------------------------------------------------------

def coverage_ratio(mask: np.ndarray) -> float:
    """Fraction of pixels that are True (foreground) in the mask.

    Returns 0.0 for an empty mask.
    """
    if mask.size == 0:
        return 0.0
    return float(mask.sum()) / float(mask.size)


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection over Union for two boolean masks of the same shape.

    Returns 1.0 if both masks are all-zero (trivially identical).

    Raises
    ------
    ValueError
        If the two masks have different shapes.
    """
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"Mask shapes do not match for IoU: {mask_a.shape} vs {mask_b.shape}"
        )
    intersection = int((mask_a & mask_b).sum())
    union = int((mask_a | mask_b).sum())
    if union == 0:
        return 1.0  # both all-zero — perfectly identical
    return float(intersection) / float(union)


def pixel_agreement(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Fraction of pixels where mask_a and mask_b agree.

    A pixel "agrees" when both are True (both foreground) or both are False
    (both background).

    Raises
    ------
    ValueError
        If the two masks have different shapes.
    """
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"Mask shapes do not match for pixel agreement: "
            f"{mask_a.shape} vs {mask_b.shape}"
        )
    if mask_a.size == 0:
        return 1.0
    return float((mask_a == mask_b).sum()) / float(mask_a.size)


def is_all_zero(mask: np.ndarray) -> bool:
    """Return True if the mask contains no foreground pixels."""
    return bool(mask.sum() == 0)


def masks_have_same_dimensions(
    masks_a: list[np.ndarray],
    masks_b: list[np.ndarray],
) -> bool:
    """Return True if both lists have the same count and matching shapes."""
    if len(masks_a) != len(masks_b):
        return False
    return all(a.shape == b.shape for a, b in zip(masks_a, masks_b))


# ---------------------------------------------------------------------------
# Comparison report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MaskComparison:
    """Comparison result for a single mask (one index in the masks list)."""

    mask_index: int
    direct_shape: tuple[int, ...]
    served_shape: tuple[int, ...]
    dimensions_match: bool
    direct_all_zero: bool
    served_all_zero: bool
    direct_coverage: float
    served_coverage: float
    # None when shapes mismatch (comparison not possible)
    iou_score: float | None
    pixel_agreement_score: float | None

    @property
    def passed(self) -> bool:
        """True when dimensions match and neither mask is all-zero."""
        return self.dimensions_match and not self.served_all_zero


@dataclass
class CorrectnessReport:
    """Full comparison between one direct adapter call and one served response."""

    backend: str
    prompt_type: str
    num_masks_direct: int
    num_masks_served: int
    mask_count_match: bool
    mask_comparisons: list[MaskComparison] = field(default_factory=list)
    metadata_notes: str = ""
    error: str = ""

    @property
    def all_passed(self) -> bool:
        """True if every mask comparison passed and no top-level error occurred."""
        if self.error:
            return False
        if not self.mask_count_match:
            return False
        return all(mc.passed for mc in self.mask_comparisons)

    @property
    def mean_iou(self) -> float | None:
        """Average IoU across all valid mask comparisons, or None if none."""
        scores = [mc.iou_score for mc in self.mask_comparisons if mc.iou_score is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def mean_pixel_agreement(self) -> float | None:
        """Average pixel agreement across all valid comparisons, or None."""
        scores = [
            mc.pixel_agreement_score
            for mc in self.mask_comparisons
            if mc.pixel_agreement_score is not None
        ]
        return sum(scores) / len(scores) if scores else None


# ---------------------------------------------------------------------------
# High-level comparison function
# ---------------------------------------------------------------------------

def compare_responses(
    direct_masks_b64: list[str],
    served_masks_b64: list[str],
    backend: str,
    prompt_type: str,
    metadata_notes: str = "",
) -> CorrectnessReport:
    """Compare masks from a direct adapter call vs a served HTTP response.

    Both inputs are lists of base64-encoded PNG mask strings as they appear
    in ``SegmentResponse.masks[*].mask_b64``.

    Parameters
    ----------
    direct_masks_b64:
        Mask strings from direct ``adapter.segment()`` invocation.
    served_masks_b64:
        Mask strings from the ``/api/v1/segment`` HTTP response.
    backend, prompt_type:
        Metadata stored in the returned report.
    metadata_notes:
        Optional freetext context (e.g. checkpoint name, device).

    Returns
    -------
    CorrectnessReport
        Contains per-mask IoU, pixel agreement, and a top-level ``all_passed``
        flag.
    """
    num_direct = len(direct_masks_b64)
    num_served = len(served_masks_b64)

    report = CorrectnessReport(
        backend=backend,
        prompt_type=prompt_type,
        num_masks_direct=num_direct,
        num_masks_served=num_served,
        mask_count_match=(num_direct == num_served),
        metadata_notes=metadata_notes,
    )

    n = min(num_direct, num_served)
    for i in range(n):
        try:
            dm = decode_mask_b64(direct_masks_b64[i])
            sm = decode_mask_b64(served_masks_b64[i])
        except Exception as exc:
            report.error = f"Mask decode error at index {i}: {exc}"
            report.mask_comparisons.append(
                MaskComparison(
                    mask_index=i,
                    direct_shape=(),
                    served_shape=(),
                    dimensions_match=False,
                    direct_all_zero=True,
                    served_all_zero=True,
                    direct_coverage=0.0,
                    served_coverage=0.0,
                    iou_score=None,
                    pixel_agreement_score=None,
                )
            )
            continue

        dim_match = dm.shape == sm.shape
        report.mask_comparisons.append(
            MaskComparison(
                mask_index=i,
                direct_shape=tuple(dm.shape),
                served_shape=tuple(sm.shape),
                dimensions_match=dim_match,
                direct_all_zero=is_all_zero(dm),
                served_all_zero=is_all_zero(sm),
                direct_coverage=coverage_ratio(dm),
                served_coverage=coverage_ratio(sm),
                iou_score=iou(dm, sm) if dim_match else None,
                pixel_agreement_score=pixel_agreement(dm, sm) if dim_match else None,
            )
        )

    return report


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------

def validate_response_metadata(response_dict: dict[str, Any]) -> dict[str, bool]:
    """Check that a SegmentResponse JSON dict has all expected top-level fields.

    Returns a dict mapping ``"has_<field>"`` → bool.

    Example
    -------
    ::

        checks = validate_response_metadata(response.json())
        assert all(checks.values())
    """
    required_top = {"request_id", "backend", "masks", "latency_ms"}
    required_mask = {"mask_b64", "score", "area"}

    results: dict[str, bool] = {}

    for f in required_top:
        results[f"has_{f}"] = f in response_dict

    masks = response_dict.get("masks", [])
    results["has_at_least_one_mask"] = len(masks) > 0

    if masks:
        first = masks[0]
        for f in required_mask:
            results[f"mask_has_{f}"] = f in first
        results["mask_score_in_range"] = 0.0 <= first.get("score", -1) <= 1.0
        results["mask_area_non_negative"] = first.get("area", -1) >= 0

    return results
