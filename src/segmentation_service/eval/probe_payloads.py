"""Backend-specific probe payloads for evaluation and benchmark scripts.

Provides canonical per-backend probe requests used by:
  - scripts/evaluate_correctness.py
  - scripts/evaluate_compatibility.py
  - benchmark/latency.py
  - benchmark/throughput.py

Design
------
- ``mock`` uses small built-in synthetic payloads (no asset files needed).
- ``sam2`` and ``clipseg`` load from ``eval_assets/requests/*.json``, which
  contain backend-appropriate images and prompts (real-sized images, valid
  coordinate ranges).
- For (backend, prompt_type) combinations without a dedicated asset file the
  module falls back to the mock payload so that compatibility probes can still
  send a structurally-valid request and observe the expected rejection.
"""

from __future__ import annotations

import json
from pathlib import Path

from segmentation_service.schemas.segment import SegmentRequest

# eval_assets/requests/ lives three packages above this file:
#   src/segmentation_service/eval/probe_payloads.py
#   parents[0] = eval/
#   parents[1] = segmentation_service/
#   parents[2] = src/
#   parents[3] = repo root
_ASSETS_DIR = Path(__file__).resolve().parents[3] / "eval_assets" / "requests"

# ---------------------------------------------------------------------------
# Built-in mock payloads (tiny 1×1 PNG, no file I/O required)
# ---------------------------------------------------------------------------

_MOCK_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

_MOCK_PAYLOADS: dict[str, dict] = {
    "point": {
        "image": _MOCK_IMAGE,
        "image_format": "png",
        "prompt_type": "point",
        "points": [{"x": 0, "y": 0, "label": 1}],
    },
    "box": {
        "image": _MOCK_IMAGE,
        "image_format": "png",
        "prompt_type": "box",
        "box": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
    },
    "text": {
        "image": _MOCK_IMAGE,
        "image_format": "png",
        "prompt_type": "text",
        "text_prompt": "object",
    },
}

# ---------------------------------------------------------------------------
# Asset-file mapping: backend → prompt_type → filename under _ASSETS_DIR
# ---------------------------------------------------------------------------

_ASSET_FILES: dict[str, dict[str, str]] = {
    "sam2": {
        "point": "sam2_point.json",
        "box": "sam2_box.json",
    },
    "clipseg": {
        "text": "clipseg_text.json",
    },
}

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Canonical probe types per backend (in preferred probe order).
BACKEND_PROBE_TYPES: dict[str, list[str]] = {
    "mock": ["point", "box", "text"],
    "sam2": ["point", "box"],
    "clipseg": ["text"],
}

# Default (first) prompt type used when a single probe is needed per backend.
DEFAULT_PROMPT_TYPE: dict[str, str] = {
    "mock": "point",
    "sam2": "point",
    "clipseg": "text",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_payload(backend: str, prompt_type: str) -> dict:
    """Return a probe payload dict for ``(backend, prompt_type)``.

    Resolution order:
    1. ``mock`` → built-in tiny synthetic payload.
    2. Asset file exists in ``eval_assets/requests/`` → load from JSON.
    3. Fallback → built-in mock payload for the given prompt_type (or point).

    The returned dict is a fresh copy safe to mutate.
    """
    if backend == "mock":
        return dict(_MOCK_PAYLOADS[prompt_type])

    asset_map = _ASSET_FILES.get(backend, {})
    filename = asset_map.get(prompt_type)
    if filename:
        asset_path = _ASSETS_DIR / filename
        return json.loads(asset_path.read_text(encoding="utf-8"))

    # Fallback: use mock payload so the probe is at least structurally valid.
    fallback = _MOCK_PAYLOADS.get(prompt_type, _MOCK_PAYLOADS["point"])
    return dict(fallback)


def load_request(backend: str, prompt_type: str) -> SegmentRequest:
    """Return a ``SegmentRequest`` for ``(backend, prompt_type)``."""
    return SegmentRequest.model_validate(load_payload(backend, prompt_type))
