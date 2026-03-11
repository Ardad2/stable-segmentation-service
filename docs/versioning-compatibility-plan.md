# API Versioning Compatibility Plan

## Overview

This document describes the API versioning experiment for the Stable
Segmentation Service.  It covers:

1. The exact backward-compatible change made to v1.
2. The exact breaking change introduced in v2.
3. Why each change is classified the way it is.
4. The compatibility matrix and pass/fail criteria.
5. How to run the tests and the evaluation script.

---

## Background

The service already has a stable `/api/v1` surface.  To demonstrate rigorous
versioning practice, the project introduces:

- A **backward-compatible additive change** within v1 (no new major version
  required).
- A **new major version `/api/v2`** containing one deliberate breaking change,
  co-existing with v1.

Both versions share the same adapter layer and the same set of backends
(mock, SAM2, CLIPSeg).  There is no backend-specific branching in the route
handlers.

---

## 1. Backward-Compatible v1 Change

### What changed

An `api_version` field (type `str`, default `"1.0"`) was added to all three
v1 response schemas:

| Schema | File |
|--------|------|
| `HealthResponse` | `src/segmentation_service/schemas/health.py` |
| `CapabilitiesResponse` | `src/segmentation_service/schemas/capabilities.py` |
| `SegmentResponse` | `src/segmentation_service/schemas/segment.py` |

### Why it is backward-compatible

1. **Additive only** — no existing field was renamed, removed, or had its type
   changed.
2. **Has a default** — the field is always present in responses with value
   `"1.0"`.  Clients that do not look for this field are unaffected.
3. **No change to request schemas** — senders do not need to supply
   `api_version`.
4. **JSON tolerant** — clients that deserialise the response into a fixed
   struct (e.g. a TypeScript or Python model) will ignore the new field if
   their schema does not mention it; they will not receive an error.

### Which existing clients pass unchanged

- All existing tests in `tests/test_health.py`, `tests/test_capabilities.py`,
  and `tests/test_segment.py` — they use `"field" in data` assertions, not
  exact schema equality.
- The CLI client (`seg-client`) — it reads `capabilities` and `masks`, not
  `api_version`.
- The benchmark and correctness scripts — they check mask content, not
  response envelope shape.

---

## 2. Breaking Change in v2

### What changed

The `POST /api/v2/segment` endpoint accepts **`V2SegmentRequest`** instead of
v1's `SegmentRequest`.

**v1 request (flat layout):**
```json
{
  "image": "...",
  "image_format": "png",
  "prompt_type": "point",
  "points": [{"x": 10, "y": 20, "label": 1}]
}
```

**v2 request (nested prompt envelope):**
```json
{
  "image": "...",
  "image_format": "png",
  "prompt": {
    "type": "point",
    "points": [{"x": 10, "y": 20, "label": 1}]
  }
}
```

Additionally, the per-mask response field `mask_b64` is renamed to
`mask_data` in `V2MaskResult`.

### Why each sub-change is breaking

#### Prompt envelope (`prompt` replaces five flat fields)

| | v1 | v2 |
|-|----|----|
| Field | `prompt_type`, `points`, `box`, `text_prompt` | `prompt.type`, `prompt.points`, `prompt.box`, `prompt.text` |
| Required | No — `prompt_type` has a default | Yes — `prompt` has no default |

A v1 client sending `{"prompt_type": "point", "points": [...]}` to `/api/v2/segment`:
- Pydantic sees no `prompt` field → **HTTP 422**, `{"detail": [{"loc": ["body", "prompt"], "msg": "Field required"}]}`.
- The error is explicit and refers to the missing field by name.

A v2 client sending `{"prompt": {...}}` to `/api/v1/segment`:
- Pydantic ignores the unknown `prompt` field (not in `SegmentRequest`).
- `prompt_type` defaults to `"point"`, `points` is `None`.
- The v1 route guard raises **HTTP 422**: `"prompt_type='point' requires at least one entry in 'points'."`.

#### `mask_b64` → `mask_data`

A v1 client code path that does `response["masks"][0]["mask_b64"]` will raise
`KeyError` when receiving a v2 response (the key is `mask_data`).  The failure
is immediate and clearly attributable to the version mismatch.

### Why this design is defensible

The `prompt_type` + four optional flat fields pattern is a common anti-pattern
that grows poorly as new prompt types are added.  The nested envelope makes the
prompt a first-class object, easier to validate, extend, and document.

Renaming `mask_b64` → `mask_data` removes the implementation hint (`b64`) from
the public API; the encoding is an implementation detail.

---

## 3. API Version vs Model/Backend Version

These are deliberately kept separate:

| Concept | Field | Example |
|---------|-------|---------|
| API contract version | `api_version` in every response | `"1.0"`, `"2.0"` |
| Service code version | `version` in `/health` response | `"0.1.0"` |
| Backend / model name | `backend` in every response | `"mock"`, `"sam2"`, `"clipseg"` |

Upgrading the backend (e.g. from SAM2 `hiera_small` to `hiera_large`) does
**not** change `api_version` — that is a model version decision, not an API
contract decision.

---

## 4. Compatibility Matrix

| Client version | Server endpoint | Backend | Scenario | Expected |
|---------------|-----------------|---------|----------|----------|
| v1 | `/api/v1/segment` | any | Exact version match | **PASS** |
| v1 | `/api/v1/segment` | any | Additive: `api_version` field present | **PASS** |
| v1 | `/api/v1/segment` | any | `mask_b64` field in mask | **PASS** |
| v2 | `/api/v2/segment` | any | Exact version match | **PASS** |
| v2 | `/api/v2/segment` | any | `api_version: "2.0"` in response | **PASS** |
| v2 | `/api/v2/segment` | any | `mask_data` field in mask | **PASS** |
| v1 | `/api/v2/segment` | any | Breaking: `prompt` field missing | **FAIL 422** |
| v2 | `/api/v1/segment` | any | Breaking: `points` not found | **FAIL 422** |

"FAIL" rows are **expected** failures — the service must reject incompatible
requests clearly rather than silently accepting them.

---

## 5. Pass/Fail Criteria

A row **PASSES** when:
- Compatible pairs (`v1→v1`, `v2→v2`): HTTP 200 and the expected response
  schema is present.
- Incompatible pairs (`v1→v2`, `v2→v1`): HTTP 422 with a body that references
  the mismatched field.

A row **FAILS** when:
- A compatible pair returns anything other than 200.
- An incompatible pair returns 200 (silently accepting a wrong version).
- An incompatible pair returns 500 (crashing instead of validating).
- A network/connection error occurs.

---

## 6. Running the Tests

### Unit / integration tests (no server required)

```bash
pytest tests/test_versioning.py -v
```

Key test classes:

| Class | Scenario |
|-------|----------|
| `TestV1ClientV1Server` | v1 → v1 (PASS) |
| `TestV1BackwardCompatibleAddition` | `api_version` field present in v1 (PASS) |
| `TestV2ClientV2Server` | v2 → v2 (PASS) |
| `TestV1ClientV2Server` | v1 → v2 (FAIL 422) |
| `TestV2ClientV1Server` | v2 → v1 (FAIL 422) |
| `TestAPIVersionBackendIndependence` | Same schema, different backends (PASS) |
| `TestV2PromptValidation` | Envelope content guards (PASS) |

### Live evaluation script

Requires a running service:

```bash
# Start the service (mock backend — no GPU required)
SEGMENTATION_BACKEND=mock uvicorn segmentation_service.main:app --port 8000

# In another terminal:
python scripts/evaluate_versioning.py --url http://localhost:8000

# CSV output
python scripts/evaluate_versioning.py --url http://localhost:8000 --format csv

# Save Markdown report
python scripts/evaluate_versioning.py --url http://localhost:8000 \
    --output docs/versioning-matrix-observed.md
```

Exit code `0` = all rows matched their expected result.
Exit code `1` = one or more rows diverged from expectations.

---

## 7. Files Changed / Created

### Modified
| File | Change |
|------|--------|
| `src/segmentation_service/schemas/health.py` | Added `api_version: str = "1.0"` |
| `src/segmentation_service/schemas/capabilities.py` | Added `api_version: str = "1.0"` |
| `src/segmentation_service/schemas/segment.py` | Added `api_version: str = "1.0"` |
| `src/segmentation_service/api/router.py` | Mounted `v2_router` |

### Created
| File | Purpose |
|------|---------|
| `src/segmentation_service/schemas/v2segment.py` | `V2PromptEnvelope`, `V2SegmentRequest`, `V2MaskResult`, `V2SegmentResponse` |
| `src/segmentation_service/api/v2/__init__.py` | Package marker |
| `src/segmentation_service/api/v2/health.py` | `GET /api/v2/health` |
| `src/segmentation_service/api/v2/capabilities.py` | `GET /api/v2/capabilities` |
| `src/segmentation_service/api/v2/segment.py` | `POST /api/v2/segment` |
| `src/segmentation_service/api/v2/router.py` | v2 router with `/v2` prefix |
| `tests/test_versioning.py` | 44 compatibility matrix tests |
| `scripts/evaluate_versioning.py` | Live-server evaluation script |
| `docs/versioning-compatibility-plan.md` | This document |
