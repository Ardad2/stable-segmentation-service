"""Unit tests for segmentation_service.eval.correctness utilities.

All tests use small synthetic numpy masks — no GPU, no model weights, no HTTP.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from segmentation_service.eval.correctness import (
    CorrectnessReport,
    MaskComparison,
    compare_responses,
    coverage_ratio,
    decode_mask_b64,
    iou,
    is_all_zero,
    masks_have_same_dimensions,
    pixel_agreement,
    validate_response_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_mask(arr: np.ndarray) -> str:
    """Encode a boolean (H, W) numpy array to a base64 PNG string (L-mode)."""
    img = Image.fromarray((arr.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_mask(h: int, w: int, fill: bool = True) -> np.ndarray:
    return np.full((h, w), fill, dtype=bool)


def _make_checkerboard(h: int, w: int) -> np.ndarray:
    """Alternating True/False checkerboard."""
    idx = np.indices((h, w)).sum(axis=0)
    return (idx % 2 == 0)


# ---------------------------------------------------------------------------
# decode_mask_b64
# ---------------------------------------------------------------------------

class TestDecodeMaskB64:
    def test_all_foreground_roundtrip(self):
        arr = _make_mask(4, 4, fill=True)
        decoded = decode_mask_b64(_encode_mask(arr))
        assert decoded.shape == (4, 4)
        assert decoded.all()

    def test_all_background_roundtrip(self):
        arr = _make_mask(4, 4, fill=False)
        decoded = decode_mask_b64(_encode_mask(arr))
        assert decoded.shape == (4, 4)
        assert not decoded.any()

    def test_partial_mask_roundtrip(self):
        arr = np.zeros((4, 4), dtype=bool)
        arr[1:3, 1:3] = True
        decoded = decode_mask_b64(_encode_mask(arr))
        np.testing.assert_array_equal(decoded, arr)

    def test_returns_bool_dtype(self):
        arr = _make_mask(2, 2, fill=True)
        decoded = decode_mask_b64(_encode_mask(arr))
        assert decoded.dtype == bool

    def test_single_pixel(self):
        arr = np.array([[True]], dtype=bool)
        decoded = decode_mask_b64(_encode_mask(arr))
        assert decoded.shape == (1, 1)
        assert decoded[0, 0]

    def test_invalid_b64_raises(self):
        with pytest.raises(Exception):
            decode_mask_b64("not-valid-base64!!!")


# ---------------------------------------------------------------------------
# coverage_ratio
# ---------------------------------------------------------------------------

class TestCoverageRatio:
    def test_all_foreground(self):
        mask = _make_mask(4, 4, fill=True)
        assert coverage_ratio(mask) == pytest.approx(1.0)

    def test_all_background(self):
        mask = _make_mask(4, 4, fill=False)
        assert coverage_ratio(mask) == pytest.approx(0.0)

    def test_half_foreground(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :] = True  # top half
        assert coverage_ratio(mask) == pytest.approx(0.5)

    def test_empty_mask_returns_zero(self):
        mask = np.array([], dtype=bool).reshape(0, 0)
        assert coverage_ratio(mask) == pytest.approx(0.0)

    def test_single_true_pixel(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True
        assert coverage_ratio(mask) == pytest.approx(1 / 100)


# ---------------------------------------------------------------------------
# is_all_zero
# ---------------------------------------------------------------------------

class TestIsAllZero:
    def test_all_false_is_zero(self):
        assert is_all_zero(_make_mask(3, 3, fill=False))

    def test_all_true_is_not_zero(self):
        assert not is_all_zero(_make_mask(3, 3, fill=True))

    def test_one_true_pixel_is_not_zero(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[2, 2] = True
        assert not is_all_zero(mask)


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------

class TestIoU:
    def test_identical_all_foreground(self):
        mask = _make_mask(4, 4, fill=True)
        assert iou(mask, mask) == pytest.approx(1.0)

    def test_identical_all_background(self):
        mask = _make_mask(4, 4, fill=False)
        assert iou(mask, mask) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.zeros((4, 4), dtype=bool)
        b = np.zeros((4, 4), dtype=bool)
        a[:2, :] = True
        b[2:, :] = True
        assert iou(a, b) == pytest.approx(0.0)

    def test_half_overlap(self):
        a = np.zeros((4, 4), dtype=bool)
        b = np.zeros((4, 4), dtype=bool)
        # a: columns 0-1, b: columns 1-2 → intersection=col1, union=col0+1+2
        a[:, :2] = True   # 8 pixels
        b[:, 1:3] = True  # 8 pixels, overlap col 1 (4 pixels)
        expected = 4 / 12
        assert iou(a, b) == pytest.approx(expected)

    def test_shape_mismatch_raises(self):
        a = _make_mask(2, 2)
        b = _make_mask(3, 3)
        with pytest.raises(ValueError, match="shapes do not match"):
            iou(a, b)

    def test_both_empty_returns_one(self):
        a = _make_mask(4, 4, fill=False)
        b = _make_mask(4, 4, fill=False)
        assert iou(a, b) == pytest.approx(1.0)

    def test_one_subset_of_other(self):
        a = _make_mask(4, 4, fill=True)
        b = np.zeros((4, 4), dtype=bool)
        b[1:3, 1:3] = True  # 4 pixels, subset of a (16 pixels)
        assert iou(a, b) == pytest.approx(4 / 16)


# ---------------------------------------------------------------------------
# pixel_agreement
# ---------------------------------------------------------------------------

class TestPixelAgreement:
    def test_identical_masks(self):
        mask = _make_mask(4, 4, fill=True)
        assert pixel_agreement(mask, mask) == pytest.approx(1.0)

    def test_opposite_masks(self):
        a = _make_mask(4, 4, fill=True)
        b = _make_mask(4, 4, fill=False)
        assert pixel_agreement(a, b) == pytest.approx(0.0)

    def test_half_agreement(self):
        a = np.zeros((4, 4), dtype=bool)
        b = np.zeros((4, 4), dtype=bool)
        a[:, :2] = True   # left half
        b[:, 2:] = True   # right half — agree on 0 foreground, 0 background?
        # a=T,b=F: disagree (left half, 8 pixels)
        # a=F,b=T: disagree (right half, 8 pixels)
        # → 0/16 = 0.0
        assert pixel_agreement(a, b) == pytest.approx(0.0)

    def test_checkerboard_vs_inverse(self):
        cb = _make_checkerboard(4, 4)
        inv = ~cb
        # Every pixel disagrees → agreement = 0.0
        assert pixel_agreement(cb, inv) == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        a = _make_mask(2, 2)
        b = _make_mask(3, 3)
        with pytest.raises(ValueError, match="shapes do not match"):
            pixel_agreement(a, b)

    def test_empty_mask_returns_one(self):
        a = np.array([], dtype=bool).reshape(0, 0)
        b = np.array([], dtype=bool).reshape(0, 0)
        assert pixel_agreement(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# masks_have_same_dimensions
# ---------------------------------------------------------------------------

class TestMasksHaveSameDimensions:
    def test_matching_single(self):
        a = [_make_mask(4, 4)]
        b = [_make_mask(4, 4)]
        assert masks_have_same_dimensions(a, b)

    def test_matching_multiple(self):
        a = [_make_mask(4, 4), _make_mask(8, 8)]
        b = [_make_mask(4, 4), _make_mask(8, 8)]
        assert masks_have_same_dimensions(a, b)

    def test_different_count(self):
        a = [_make_mask(4, 4), _make_mask(4, 4)]
        b = [_make_mask(4, 4)]
        assert not masks_have_same_dimensions(a, b)

    def test_same_count_different_shapes(self):
        a = [_make_mask(4, 4)]
        b = [_make_mask(8, 8)]
        assert not masks_have_same_dimensions(a, b)

    def test_empty_lists(self):
        assert masks_have_same_dimensions([], [])

    def test_one_empty_one_not(self):
        assert not masks_have_same_dimensions([], [_make_mask(4, 4)])


# ---------------------------------------------------------------------------
# MaskComparison.passed property
# ---------------------------------------------------------------------------

class TestMaskComparisonPassed:
    def _mc(self, *, dim_match=True, served_zero=False) -> MaskComparison:
        return MaskComparison(
            mask_index=0,
            direct_shape=(4, 4),
            served_shape=(4, 4),
            dimensions_match=dim_match,
            direct_all_zero=False,
            served_all_zero=served_zero,
            direct_coverage=0.5,
            served_coverage=0.5,
            iou_score=1.0,
            pixel_agreement_score=1.0,
        )

    def test_passes_when_dim_match_and_served_nonzero(self):
        assert self._mc(dim_match=True, served_zero=False).passed

    def test_fails_when_dim_mismatch(self):
        assert not self._mc(dim_match=False, served_zero=False).passed

    def test_fails_when_served_all_zero(self):
        assert not self._mc(dim_match=True, served_zero=True).passed


# ---------------------------------------------------------------------------
# CorrectnessReport
# ---------------------------------------------------------------------------

class TestCorrectnessReport:
    def _passing_report(self) -> CorrectnessReport:
        mc = MaskComparison(
            mask_index=0,
            direct_shape=(4, 4), served_shape=(4, 4),
            dimensions_match=True,
            direct_all_zero=False, served_all_zero=False,
            direct_coverage=0.5, served_coverage=0.5,
            iou_score=1.0, pixel_agreement_score=1.0,
        )
        return CorrectnessReport(
            backend="mock", prompt_type="point",
            num_masks_direct=1, num_masks_served=1,
            mask_count_match=True,
            mask_comparisons=[mc],
        )

    def test_all_passed_when_ok(self):
        assert self._passing_report().all_passed

    def test_all_passed_false_when_error(self):
        r = self._passing_report()
        r.error = "something went wrong"
        assert not r.all_passed

    def test_all_passed_false_when_count_mismatch(self):
        r = self._passing_report()
        r.mask_count_match = False
        assert not r.all_passed

    def test_all_passed_false_when_mask_fails(self):
        r = self._passing_report()
        r.mask_comparisons[0].served_all_zero = True
        assert not r.all_passed

    def test_mean_iou_single(self):
        r = self._passing_report()
        assert r.mean_iou == pytest.approx(1.0)

    def test_mean_iou_multiple(self):
        mc1 = MaskComparison(
            mask_index=0, direct_shape=(4,4), served_shape=(4,4),
            dimensions_match=True, direct_all_zero=False, served_all_zero=False,
            direct_coverage=0.5, served_coverage=0.5,
            iou_score=0.8, pixel_agreement_score=0.9,
        )
        mc2 = MaskComparison(
            mask_index=1, direct_shape=(4,4), served_shape=(4,4),
            dimensions_match=True, direct_all_zero=False, served_all_zero=False,
            direct_coverage=0.5, served_coverage=0.5,
            iou_score=0.4, pixel_agreement_score=0.6,
        )
        r = CorrectnessReport(
            backend="mock", prompt_type="point",
            num_masks_direct=2, num_masks_served=2,
            mask_count_match=True,
            mask_comparisons=[mc1, mc2],
        )
        assert r.mean_iou == pytest.approx(0.6)
        assert r.mean_pixel_agreement == pytest.approx(0.75)

    def test_mean_iou_none_when_no_comparisons(self):
        r = CorrectnessReport(
            backend="mock", prompt_type="point",
            num_masks_direct=0, num_masks_served=0,
            mask_count_match=True,
        )
        assert r.mean_iou is None
        assert r.mean_pixel_agreement is None

    def test_mean_iou_skips_none_iou_scores(self):
        mc = MaskComparison(
            mask_index=0, direct_shape=(4,4), served_shape=(8,8),
            dimensions_match=False, direct_all_zero=False, served_all_zero=False,
            direct_coverage=0.5, served_coverage=0.5,
            iou_score=None, pixel_agreement_score=None,
        )
        r = CorrectnessReport(
            backend="mock", prompt_type="point",
            num_masks_direct=1, num_masks_served=1,
            mask_count_match=True,
            mask_comparisons=[mc],
        )
        assert r.mean_iou is None


# ---------------------------------------------------------------------------
# compare_responses
# ---------------------------------------------------------------------------

class TestCompareResponses:
    def _mask_b64(self, h: int = 4, w: int = 4, fill: bool = True) -> str:
        return _encode_mask(_make_mask(h, w, fill=fill))

    def test_identical_masks_pass(self):
        b64 = self._mask_b64(4, 4, fill=True)
        report = compare_responses([b64], [b64], backend="mock", prompt_type="point")
        assert report.all_passed
        assert report.mask_comparisons[0].iou_score == pytest.approx(1.0)
        assert report.mask_comparisons[0].pixel_agreement_score == pytest.approx(1.0)

    def test_count_mismatch_recorded(self):
        b64 = self._mask_b64()
        report = compare_responses([b64, b64], [b64], backend="mock", prompt_type="box")
        assert not report.mask_count_match
        assert not report.all_passed

    def test_served_all_zero_causes_failure(self):
        direct_b64 = self._mask_b64(4, 4, fill=True)
        served_b64 = self._mask_b64(4, 4, fill=False)
        report = compare_responses([direct_b64], [served_b64], backend="mock", prompt_type="point")
        assert not report.all_passed
        assert report.mask_comparisons[0].served_all_zero

    def test_metadata_notes_stored(self):
        b64 = self._mask_b64()
        report = compare_responses([b64], [b64], backend="x", prompt_type="y", metadata_notes="notes here")
        assert report.metadata_notes == "notes here"

    def test_backend_and_prompt_type_stored(self):
        b64 = self._mask_b64()
        report = compare_responses([b64], [b64], backend="sam2", prompt_type="box")
        assert report.backend == "sam2"
        assert report.prompt_type == "box"

    def test_shape_mismatch_sets_dim_mismatch(self):
        d_b64 = self._mask_b64(4, 4)
        s_b64 = self._mask_b64(8, 8)
        report = compare_responses([d_b64], [s_b64], backend="mock", prompt_type="point")
        mc = report.mask_comparisons[0]
        assert not mc.dimensions_match
        assert mc.iou_score is None
        assert mc.pixel_agreement_score is None

    def test_empty_lists_pass(self):
        report = compare_responses([], [], backend="mock", prompt_type="point")
        assert report.mask_count_match
        assert report.all_passed
        assert len(report.mask_comparisons) == 0

    def test_coverage_values_correct(self):
        # Top half = True (8/16 pixels)
        arr = np.zeros((4, 4), dtype=bool)
        arr[:2, :] = True
        b64 = _encode_mask(arr)
        report = compare_responses([b64], [b64], backend="mock", prompt_type="text")
        mc = report.mask_comparisons[0]
        assert mc.direct_coverage == pytest.approx(0.5)
        assert mc.served_coverage == pytest.approx(0.5)

    def test_multiple_masks_all_identical(self):
        b1 = self._mask_b64(4, 4, fill=True)
        b2 = self._mask_b64(4, 4, fill=True)
        report = compare_responses([b1, b2], [b1, b2], backend="mock", prompt_type="point")
        assert report.all_passed
        assert len(report.mask_comparisons) == 2


# ---------------------------------------------------------------------------
# validate_response_metadata
# ---------------------------------------------------------------------------

class TestValidateResponseMetadata:
    def _valid_response(self) -> dict:
        b64 = _encode_mask(_make_mask(4, 4, fill=True))
        return {
            "request_id": "abc123",
            "backend": "mock",
            "masks": [{"mask_b64": b64, "score": 0.9, "area": 16}],
            "latency_ms": 5.0,
        }

    def test_fully_valid_response_all_true(self):
        checks = validate_response_metadata(self._valid_response())
        assert all(checks.values()), f"Some checks failed: {checks}"

    def test_missing_request_id(self):
        resp = self._valid_response()
        del resp["request_id"]
        checks = validate_response_metadata(resp)
        assert not checks["has_request_id"]

    def test_missing_backend(self):
        resp = self._valid_response()
        del resp["backend"]
        checks = validate_response_metadata(resp)
        assert not checks["has_backend"]

    def test_missing_masks(self):
        resp = self._valid_response()
        del resp["masks"]
        checks = validate_response_metadata(resp)
        assert not checks["has_masks"]

    def test_missing_latency(self):
        resp = self._valid_response()
        del resp["latency_ms"]
        checks = validate_response_metadata(resp)
        assert not checks["has_latency_ms"]

    def test_empty_masks_list(self):
        resp = self._valid_response()
        resp["masks"] = []
        checks = validate_response_metadata(resp)
        assert not checks["has_at_least_one_mask"]

    def test_mask_missing_score(self):
        resp = self._valid_response()
        del resp["masks"][0]["score"]
        checks = validate_response_metadata(resp)
        assert not checks["mask_has_score"]

    def test_mask_missing_area(self):
        resp = self._valid_response()
        del resp["masks"][0]["area"]
        checks = validate_response_metadata(resp)
        assert not checks["mask_has_area"]

    def test_mask_missing_mask_b64(self):
        resp = self._valid_response()
        del resp["masks"][0]["mask_b64"]
        checks = validate_response_metadata(resp)
        assert not checks["mask_has_mask_b64"]

    def test_score_out_of_range_high(self):
        resp = self._valid_response()
        resp["masks"][0]["score"] = 1.5
        checks = validate_response_metadata(resp)
        assert not checks["mask_score_in_range"]

    def test_score_out_of_range_low(self):
        resp = self._valid_response()
        resp["masks"][0]["score"] = -0.1
        checks = validate_response_metadata(resp)
        assert not checks["mask_score_in_range"]

    def test_score_boundary_values(self):
        resp = self._valid_response()
        resp["masks"][0]["score"] = 0.0
        checks = validate_response_metadata(resp)
        assert checks["mask_score_in_range"]

        resp["masks"][0]["score"] = 1.0
        checks = validate_response_metadata(resp)
        assert checks["mask_score_in_range"]

    def test_negative_area(self):
        resp = self._valid_response()
        resp["masks"][0]["area"] = -1
        checks = validate_response_metadata(resp)
        assert not checks["mask_area_non_negative"]

    def test_empty_response_dict(self):
        checks = validate_response_metadata({})
        assert not checks["has_request_id"]
        assert not checks["has_backend"]
        assert not checks["has_masks"]
        assert not checks["has_latency_ms"]
        assert not checks["has_at_least_one_mask"]
        # No mask-level checks when masks absent
        assert "mask_has_score" not in checks
