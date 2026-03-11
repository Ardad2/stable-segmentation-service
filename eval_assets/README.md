# eval_assets

Lightweight, reproducible inputs for direct-vs-served evaluation.

## images/

| File | Size | Description |
|------|------|-------------|
| `sample_rgb.png` | 16×16 px | Natural RGB gradient with noise — suitable for SAM2 point/box prompts |
| `sample_text.png` | 16×16 px | Four bold colour quadrants (red, green, blue, yellow) — suitable for CLIPSeg text prompts (e.g. "red square") |

## requests/

Pre-built JSON request payloads. The `image` field contains the base64-encoded
PNG already embedded, so no separate image file lookup is needed at runtime.

| File | Backend | Prompt type | Prompt |
|------|---------|-------------|--------|
| `sam2_point.json` | SAM2 | point | `(4, 4, label=1)` |
| `sam2_box.json` | SAM2 | box | `(2,2) → (14,14)` |
| `clipseg_text.json` | CLIPSeg | text | `"red square"` |

## Regenerating

```bash
python3 - <<'EOF'
# ... (see scripts/evaluate_correctness.py for the generation code)
EOF
```

All assets are deterministically generated from a fixed random seed (42) so
they are stable across platforms.
