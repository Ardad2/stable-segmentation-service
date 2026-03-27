# Compatibility Report — clipseg

Generated: 2026-03-27 03:32 UTC  |  URL: `http://127.0.0.1:8004`

| Prompt type | Advertised | Observed | HTTP | Latency (ms) | Masks | Aligned | Notes |
|-------------|:----------:|:--------:|:----:|:-----------:|:-----:|:-------:|-------|
| `point` | ❌ no | ❌ | 500 | 1.2 | — | ✅ | Segmentation inference failed. Check server logs. |
| `box` | ❌ no | ❌ | 500 | 0.8 | — | ✅ | Segmentation inference failed. Check server logs. |
| `text` | ✅ yes | ❌ | 500 | 475.1 | — | ⚠️ | Segmentation inference failed. Check server logs. |

**All behaviours aligned with advertised capabilities: No ⚠️**

