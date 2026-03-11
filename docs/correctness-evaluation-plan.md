# Correctness Evaluation Plan — Direct-vs-Served Output Comparison

## Purpose

Verify that the service layer is **output-transparent**: for the same model and
the same input, the masks returned via `POST /api/v1/segment` must be
byte-for-byte identical to the masks produced by calling the adapter directly.

Any divergence indicates a bug in the service layer (e.g. incorrect
serialisation, response builder truncating a mask, or a different model being
loaded between the direct and served paths).

---

## Core claim

> For any backend B and any valid request R:
> `adapter.segment(R).masks == serve(R).masks`  (IoU = 1.0, pixel agreement = 1.0)

---

## Metrics

For each (backend, prompt_type) pair the script computes:

| Metric | Symbol | Perfect value | Fail threshold |
|--------|--------|--------------|----------------|
| IoU (Intersection over Union) | IoU | 1.0 | < 1.0 |
| Pixel agreement | PA | 1.0 | < 1.0 |
| Coverage ratio (direct) | cov_d | — | report only |
| Coverage ratio (served) | cov_s | — | report only |
| Mask dimensions match | dim | True | False |
| Served mask all-zero | zero_s | False | True |
| Mask count match | count | True | False |

A probe **passes** if and only if:
- Mask counts are equal.
- For every mask pair: dimensions match AND served mask is not all-zero.

IoU and pixel agreement are computed only when shapes match (logged as `—` otherwise).

---

## Probe inputs

| Backend | Prompt type | Image | Prompt |
|---------|------------|-------|--------|
| `mock` | `point` | 1×1 stub PNG | `(0,0,1)` |
| `mock` | `box` | 1×1 stub PNG | `(0,0)→(1,1)` |
| `mock` | `text` | 1×1 stub PNG | `"object"` |
| `sam2` | `point` | 1×1 stub PNG | `(0,0,1)` |
| `sam2` | `box` | 1×1 stub PNG | `(0,0)→(1,1)` |
| `clipseg` | `text` | 1×1 stub PNG | `"object"` |

Richer 16×16 pixel inputs for SAM2 and CLIPSeg are available in
`eval_assets/requests/` for manual or extended evaluation.

---

## Running the evaluation

### Prerequisites

```bash
pip install -e ".[dev]"
# The service must already be running with the same backend:
SEGMENTATION_BACKEND=mock uvicorn segmentation_service.main:app --port 8000
```

### Mock backend (no GPU required)

```bash
python scripts/evaluate_correctness.py --backend mock \
    --url http://localhost:8000
```

### SAM2 backend

```bash
python scripts/evaluate_correctness.py --backend sam2 \
    --url http://localhost:8000 \
    --sam2-checkpoint weights/sam2_hiera_large.pt \
    --sam2-config sam2_hiera_l.yaml
```

### CLIPSeg backend

```bash
python scripts/evaluate_correctness.py --backend clipseg \
    --url http://localhost:8000
```

### Save a Markdown report

```bash
python scripts/evaluate_correctness.py --backend mock \
    --url http://localhost:8000 --output report.md
```

### CSV output

```bash
python scripts/evaluate_correctness.py --backend mock \
    --url http://localhost:8000 --format csv
```

---

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | `mock` | `mock` \| `sam2` \| `clipseg` |
| `--url` | `http://localhost:8000` | Service base URL |
| `--sam2-checkpoint PATH` | _(required for sam2)_ | SAM2 `.pt` weights file |
| `--sam2-config NAME` | _(required for sam2)_ | SAM2 YAML config name |
| `--clipseg-model ID` | `CIDAS/clipseg-rd64-refined` | HuggingFace model ID or path |
| `--device` | `cpu` | `cpu` \| `cuda` \| `mps` |
| `--format` | `markdown` | `markdown` \| `csv` |
| `--output FILE` | _(stdout)_ | Write report to FILE |
| `--timeout` | `60.0` | Per-request timeout (seconds) |

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | All probes passed |
| `1` | One or more probes failed, or a runtime error occurred |

---

## Pass / fail criteria

**PASS**: Every probe in `--backend` returns IoU = 1.0 and pixel agreement = 1.0.

**FAIL (any of)**:
- Direct adapter returns an error.
- HTTP response returns a non-2xx status code.
- Mask counts differ between direct and served responses.
- At least one served mask has different dimensions from the direct mask.
- At least one served mask is all-zero when the direct mask is not.

---

## Report format (Markdown example)

```markdown
# Correctness Report — mock

Generated: 2024-01-15 10:30 UTC  |  URL: `http://localhost:8000`

## Mask comparison

| Prompt | Masks (D/S) | Dims match | Non-zero | IoU   | Pixel agree | Coverage (D/S) | Result |
|--------|:-----------:|:----------:|:--------:|:-----:|:-----------:|:--------------:|:------:|
| `point` | 1 / 1 | ✅ | ✅ | 1.000 | 1.000 | 1.00 / 1.00 | ✅ PASS |
| `box`   | 1 / 1 | ✅ | ✅ | 1.000 | 1.000 | 1.00 / 1.00 | ✅ PASS |
| `text`  | 1 / 1 | ✅ | ✅ | 1.000 | 1.000 | 1.00 / 1.00 | ✅ PASS |

**Overall result: PASS ✅**
```

---

## Implementation notes

- `evaluate_correctness.py` uses `DirectRunner` from
  `segmentation_service.eval.direct_runners` to call the adapter synchronously
  and `httpx.Client` to call the served endpoint.
- Mask comparison logic lives in `segmentation_service.eval.correctness`:
  `compare_responses()` decodes both sets of base64 PNG masks and computes
  per-pair IoU and pixel agreement.
- `validate_response_metadata()` checks that the served response contains all
  required top-level fields (`request_id`, `backend`, `masks`, `latency_ms`)
  and that each mask has `mask_b64`, `score`, and `area`.
- Unit tests for all comparison utilities are in `tests/test_eval_utils.py`.
