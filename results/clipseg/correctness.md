# Correctness Report — clipseg

Generated: 2026-03-27 03:39 UTC  |  URL: `http://127.0.0.1:8004`

## Mask comparison

| Prompt | Masks (D/S) | Dims match | Non-zero | IoU | Pixel agree | Coverage (D/S) | Result |
|--------|:-----------:|:----------:|:--------:|:---:|:-----------:|:--------------:|:------:|
| `text` | 1 / 1 | ✅ | ✅ | 1.000 | 1.000 | 0.24 / 0.24 | ✅ PASS |

**Overall result: PASS ✅**

## Served response metadata

| Prompt | has_at_least_one_mask | has_backend | has_latency_ms | has_masks | has_request_id | mask_area_non_negative | mask_has_area | mask_has_mask_b64 | mask_has_score | mask_score_in_range |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `text` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
