# Backend Compatibility Matrix

This document records the expected and observed segmentation behaviour for every
combination of backend × prompt type.  The table is seeded from the known adapter
implementations; for live observed results, run the evaluation script (see below).

---

## Static expected behaviour

| Backend | Prompt type | Expected support | Notes |
|---------|-------------|:----------------:|-------|
| `mock`    | `point` | ✅ supported   | Returns a synthetic mask; all three types are accepted |
| `mock`    | `box`   | ✅ supported   | Returns a synthetic mask |
| `mock`    | `text`  | ✅ supported   | Returns a synthetic mask |
| `sam2`    | `point` | ✅ supported   | Single or multi-mask; uses (x, y, label) coordinates |
| `sam2`    | `box`   | ✅ supported   | Single mask; multi-mask output is ignored |
| `sam2`    | `text`  | ❌ unsupported | SAM2 is geometric-only; adapter raises `ValueError` → HTTP 500 |
| `clipseg` | `point` | ❌ unsupported | CLIPSeg is text-driven; adapter raises `ValueError` → HTTP 500 |
| `clipseg` | `box`   | ❌ unsupported | CLIPSeg is text-driven; adapter raises `ValueError` → HTTP 500 |
| `clipseg` | `text`  | ✅ supported   | Returns one mask per text description |

---

## Live evaluation

Run the evaluation script against a live server to populate observed results:

```bash
# Start the server first (pick any backend via SEGMENTATION_BACKEND env var)
uvicorn segmentation_service.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal:
python scripts/evaluate_compatibility.py --url http://localhost:8000
```

Sample output for the **mock** backend:

```
# Compatibility Report — mock

Generated: 2026-03-10 12:00 UTC  |  URL: `http://localhost:8000`

| Prompt type | Advertised | Observed | HTTP | Latency (ms) | Masks | Aligned | Notes |
|-------------|:----------:|:--------:|:----:|:------------:|:-----:|:-------:|-------|
| `point`     | ✅ yes     | ✅       | 200  | 2.3          | 1     | ✅       |       |
| `box`       | ✅ yes     | ✅       | 200  | 1.8          | 1     | ✅       |       |
| `text`      | ✅ yes     | ✅       | 200  | 2.1          | 1     | ✅       |       |

**All behaviours aligned with advertised capabilities: Yes ✅**
```

Sample output for the **sam2** backend:

```
| Prompt type | Advertised | Observed | HTTP | Latency (ms) | Masks | Aligned |
|-------------|:----------:|:--------:|:----:|:------------:|:-----:|:-------:|
| `point`     | ✅ yes     | ✅       | 200  | …            | 1–3   | ✅       |
| `box`       | ✅ yes     | ✅       | 200  | …            | 1     | ✅       |
| `text`      | ❌ no      | ❌       | 500  | —            | —     | ✅       |
```

Sample output for the **clipseg** backend:

```
| Prompt type | Advertised | Observed | HTTP | Latency (ms) | Masks | Aligned |
|-------------|:----------:|:--------:|:----:|:------------:|:-----:|:-------:|
| `point`     | ❌ no      | ❌       | 500  | —            | —     | ✅       |
| `box`       | ❌ no      | ❌       | 500  | —            | —     | ✅       |
| `text`      | ✅ yes     | ✅       | 200  | …            | 1     | ✅       |
```

---

## Automated test coverage

`tests/test_compatibility.py` enforces the "capabilities ↔ segment alignment"
invariant automatically in CI (no running server, no GPU — all adapters are
mocked):

```
tests/test_compatibility.py::test_mock_capabilities_matches_segment
tests/test_compatibility.py::test_sam2_capabilities_matches_segment
tests/test_compatibility.py::test_clipseg_capabilities_matches_segment
tests/test_compatibility.py::test_sam2_known_support_matrix[point-True]
tests/test_compatibility.py::test_sam2_known_support_matrix[box-True]
tests/test_compatibility.py::test_sam2_known_support_matrix[text-False]
tests/test_compatibility.py::test_clipseg_known_support_matrix[text-True]
tests/test_compatibility.py::test_clipseg_known_support_matrix[point-False]
tests/test_compatibility.py::test_clipseg_known_support_matrix[box-False]
```

---

## How to update this document

1. Run the evaluation script against each backend.
2. Paste the Markdown tables into the "Live observed results" section above.
3. If a new backend is added, append rows to the static table and run the
   script against the new backend to capture observed results.
