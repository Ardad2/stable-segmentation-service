# API Versioning Compatibility Matrix

Generated: 2026-03-26 00:48 UTC  |  URL: `http://127.0.0.1:8003`  |  Backend: `sam2`

**8/8 rows passed**

| Client | Server | Scenario | Expected | Observed | Result | Notes |
|--------|--------|----------|----------|----------|--------|-------|
| v1 | v1 | v1 payload → /api/v1/segment (exact match) | PASS | PASS | ✅ PASS | api_version=1.0 masks=1 |
| v1 | v1 | v1 response includes api_version='1.0' (additive) | PASS | PASS | ✅ PASS | api_version='1.0' |
| v1 | v1 | v1 mask field is 'mask_b64' (v1 contract preserved) | PASS | PASS | ✅ PASS | mask keys: ['mask_b64', 'score', 'area', 'logits_b64'] |
| v2 | v2 | v2 payload → /api/v2/segment (exact match) | PASS | PASS | ✅ PASS | api_version=2.0 masks=1 |
| v2 | v2 | v2 response includes api_version='2.0' | PASS | PASS | ✅ PASS | api_version='2.0' |
| v2 | v2 | v2 mask field is 'mask_data' not 'mask_b64' | PASS | PASS | ✅ PASS | mask keys: ['mask_data', 'score', 'area', 'logits_b64'] |
| v1 | v2 | v1 payload → /api/v2/segment (BREAKING — missing 'prompt') | FAIL | FAIL | ✅ PASS | status=422 (expected 422) |
| v2 | v1 | v2 payload → /api/v1/segment (BREAKING — no prompt_type field) | FAIL | FAIL | ✅ PASS | status=422 (expected 422) |

**Overall: ✅ ALL PASSED**

