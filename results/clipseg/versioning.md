# API Versioning Compatibility Matrix

Generated: 2026-03-27 03:32 UTC  |  URL: `http://127.0.0.1:8004`  |  Backend: `clipseg`

**2/8 rows passed**

| Client | Server | Scenario | Expected | Observed | Result | Notes |
|--------|--------|----------|----------|----------|--------|-------|
| v1 | v1 | v1 payload → /api/v1/segment (exact match) | PASS | FAIL (500) | ❌ FAIL | status=500 |
| v1 | v1 | v1 response includes api_version='1.0' (additive) | PASS | FAIL (500) | ❌ FAIL | status=500 |
| v1 | v1 | v1 mask field is 'mask_b64' (v1 contract preserved) | PASS | FAIL (500) | ❌ FAIL | status=500 |
| v2 | v2 | v2 payload → /api/v2/segment (exact match) | PASS | FAIL (500) | ❌ FAIL | status=500 |
| v2 | v2 | v2 response includes api_version='2.0' | PASS | FAIL (500) | ❌ FAIL | status=500 |
| v2 | v2 | v2 mask field is 'mask_data' not 'mask_b64' | PASS | FAIL (500) | ❌ FAIL | status=500 |
| v1 | v2 | v1 payload → /api/v2/segment (BREAKING — missing 'prompt') | FAIL | FAIL | ✅ PASS | status=422 (expected 422) |
| v2 | v1 | v2 payload → /api/v1/segment (BREAKING — no prompt_type field) | FAIL | FAIL | ✅ PASS | status=422 (expected 422) |

**Overall: ❌ 6 FAILED**

