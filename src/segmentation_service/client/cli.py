"""Minimal CLI client for the stable segmentation API.

The client is intentionally backend-agnostic.  It calls /api/v1/capabilities
first, then selects an appropriate prompt type at runtime based on what the
active backend advertises.  The exact same invocation works unchanged with
the mock, SAM2, and CLIPSeg backends.

Prompt selection
----------------
The client picks the *first* compatible prompt from the user-supplied options,
evaluated in this order: text → point → box.

- If the user supplied ``--text-prompt`` AND the backend supports ``text``,
  a text-prompt request is sent.
- Else if ``--point`` was given AND the backend supports ``point``, a
  point-prompt request is sent.
- Else if ``--box`` was given AND the backend supports ``box``, a box-prompt
  request is sent.
- If the user supplied prompt data but none of it is supported, the client
  exits with a clear error message.
- If no prompt data was provided at all, a sensible synthetic placeholder is
  chosen from the supported types (useful for smoke-testing the server).

Usage
-----
::

    seg-client --base-url http://localhost:8000 \\
               --image path/to/image.png \\
               --text-prompt "the cat"

    seg-client --base-url http://localhost:8000 \\
               --image path/to/image.png \\
               --point "320,240,1" \\
               --box "10,20,200,300" \\
               --output-dir ./masks \\
               --return-logits
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Synthetic smoke-test payloads — used when the user provides no prompt data
# ---------------------------------------------------------------------------

# Minimal valid 1×1 transparent PNG.
_SMOKE_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


# ---------------------------------------------------------------------------
# Pure helpers (testable without HTTP)
# ---------------------------------------------------------------------------

def load_image_b64(path: str) -> str:
    """Read an image file and return it as a base64-encoded string."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def select_prompt(
    supported_prompt_types: list[str],
    *,
    text_prompt: str | None = None,
    point: str | None = None,
    box: str | None = None,
) -> dict[str, Any]:
    """Build the prompt-related fields for a /segment request payload.

    Parameters
    ----------
    supported_prompt_types:
        The list from ``GET /api/v1/capabilities``.
    text_prompt:
        Natural-language description (e.g. ``"the cat"``).
    point:
        Comma-separated ``"x,y,label"`` string (e.g. ``"320,240,1"``).
    box:
        Comma-separated ``"xmin,ymin,xmax,ymax"`` string (e.g. ``"0,0,100,100"``).

    Returns
    -------
    dict
        Prompt-related keys ready to merge into a full segment request payload.
        Keys vary by prompt_type:
        - text:  ``{prompt_type, text_prompt}``
        - point: ``{prompt_type, points}``
        - box:   ``{prompt_type, box}``

    Raises
    ------
    ValueError
        If the user supplied prompt data but none of it matches what the backend
        supports.
    ValueError
        If a supplied prompt string cannot be parsed.
    """
    user_supplied = {
        "text": text_prompt,
        "point": point,
        "box": box,
    }
    user_gave_any = any(user_supplied.values())

    # Evaluate candidates in priority order: text > point > box.
    for prompt_type in ("text", "point", "box"):
        value = user_supplied[prompt_type]
        if value and prompt_type in supported_prompt_types:
            return _build_prompt_payload(prompt_type, value)

    if user_gave_any:
        provided = [t for t, v in user_supplied.items() if v]
        raise ValueError(
            f"Prompt types {provided} are not supported by the active backend.\n"
            f"Supported types: {supported_prompt_types}\n"
            "Tip: check GET /api/v1/capabilities before sending a request."
        )

    # No user prompts given — pick a smoke-test placeholder from supported types.
    for prompt_type in ("text", "point", "box"):
        if prompt_type in supported_prompt_types:
            return _build_synthetic_prompt(prompt_type)

    raise ValueError(
        f"No recognised prompt type in supported list: {supported_prompt_types}"
    )


def _build_prompt_payload(prompt_type: str, raw_value: str) -> dict[str, Any]:
    """Parse a raw CLI value and return prompt fields for the given type."""
    if prompt_type == "text":
        return {"prompt_type": "text", "text_prompt": raw_value}

    if prompt_type == "point":
        parts = raw_value.split(",")
        if len(parts) != 3:
            raise ValueError(
                f"--point must be 'x,y,label' (got {raw_value!r})."
            )
        x, y, label = float(parts[0]), float(parts[1]), int(parts[2])
        return {
            "prompt_type": "point",
            "points": [{"x": x, "y": y, "label": label}],
        }

    if prompt_type == "box":
        parts = raw_value.split(",")
        if len(parts) != 4:
            raise ValueError(
                f"--box must be 'xmin,ymin,xmax,ymax' (got {raw_value!r})."
            )
        xmin, ymin, xmax, ymax = (float(p) for p in parts)
        return {
            "prompt_type": "box",
            "box": {"x_min": xmin, "y_min": ymin, "x_max": xmax, "y_max": ymax},
        }

    raise ValueError(f"Unknown prompt type: {prompt_type!r}")


def _build_synthetic_prompt(prompt_type: str) -> dict[str, Any]:
    """Return a harmless placeholder prompt for smoke-testing."""
    if prompt_type == "text":
        return {"prompt_type": "text", "text_prompt": "object"}
    if prompt_type == "point":
        return {
            "prompt_type": "point",
            "points": [{"x": 0, "y": 0, "label": 1}],
        }
    if prompt_type == "box":
        return {
            "prompt_type": "box",
            "box": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
        }
    raise ValueError(f"Unknown prompt type: {prompt_type!r}")


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class SegmentationClient:
    """Thin synchronous HTTP client for the segmentation service stable API.

    Uses only the three stable endpoints:
    - ``GET  /api/v1/health``
    - ``GET  /api/v1/capabilities``
    - ``POST /api/v1/segment``

    No backend-specific knowledge lives here.
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    def get_health(self) -> dict[str, Any]:
        """Call GET /api/v1/health."""
        return self._get("/api/v1/health")

    def get_capabilities(self) -> dict[str, Any]:
        """Call GET /api/v1/capabilities."""
        return self._get("/api/v1/capabilities")

    def segment(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call POST /api/v1/segment with *payload*."""
        return self._post("/api/v1/segment", payload)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as http:
            resp = http.get(self._base_url + path)
            resp.raise_for_status()
            return resp.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as http:
            resp = http.post(self._base_url + path, json=payload)
            resp.raise_for_status()
            return resp.json()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="seg-client",
        description=(
            "Stable-API client for the segmentation service. "
            "Calls /capabilities first, then sends a /segment request "
            "using the most appropriate prompt type the backend supports."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        metavar="URL",
        help="Service base URL (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--image",
        metavar="PATH",
        help=(
            "Path to the image file to segment. "
            "If omitted, a tiny synthetic 1×1 PNG is used (smoke-test mode)."
        ),
    )
    parser.add_argument(
        "--text-prompt",
        metavar="TEXT",
        help='Natural-language prompt, e.g. "the cat". Used when backend supports text.',
    )
    parser.add_argument(
        "--point",
        metavar="X,Y,LABEL",
        help='Point prompt as "x,y,label", e.g. "320,240,1". Used when backend supports point.',
    )
    parser.add_argument(
        "--box",
        metavar="XMIN,YMIN,XMAX,YMAX",
        help='Box prompt as "xmin,ymin,xmax,ymax". Used when backend supports box.',
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Directory in which to save returned masks as PNG files.",
    )
    parser.add_argument(
        "--return-logits",
        action="store_true",
        help="Request raw logit maps alongside binary masks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Print the full JSON response instead of the human-readable summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    """Entry point for the ``seg-client`` console script.

    Returns an exit code: 0 for success, 1 for any error.
    """
    args = _parse_args(argv)
    client = SegmentationClient(args.base_url)

    # ── 1. Capabilities ──────────────────────────────────────────────────
    try:
        caps = client.get_capabilities()
    except httpx.HTTPError as exc:
        print(f"ERROR: Could not reach {args.base_url}: {exc}", file=sys.stderr)
        return 1

    backend = caps.get("backend", "unknown")
    supported = caps.get("supported_prompt_types", [])

    # Informational output always goes to stderr so that --json produces
    # clean, machine-readable JSON on stdout.
    _info = sys.stderr if args.output_json else sys.stdout
    print(f"Backend:   {backend}", file=_info)
    print(f"Supported: {', '.join(supported) or '(none)'}", file=_info)

    # ── 2. Select prompt ─────────────────────────────────────────────────
    try:
        prompt_fields = select_prompt(
            supported,
            text_prompt=args.text_prompt,
            point=args.point,
            box=args.box,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Prompt:    {prompt_fields['prompt_type']}", end="", file=_info)
    if prompt_fields["prompt_type"] == "text":
        print(f' — "{prompt_fields["text_prompt"]}"', file=_info)
    elif prompt_fields["prompt_type"] == "point":
        p = prompt_fields["points"][0]
        print(f" — ({p['x']}, {p['y']}, label={p['label']})", file=_info)
    elif prompt_fields["prompt_type"] == "box":
        b = prompt_fields["box"]
        print(f" — ({b['x_min']}, {b['y_min']}, {b['x_max']}, {b['y_max']})", file=_info)
    else:
        print(file=_info)

    # ── 3. Build payload ─────────────────────────────────────────────────
    image_b64 = (
        load_image_b64(args.image) if args.image else _SMOKE_IMAGE_B64
    )
    payload: dict[str, Any] = {
        "image": image_b64,
        "image_format": "png",
        "return_logits": args.return_logits,
        **prompt_fields,
    }

    # ── 4. Send request ───────────────────────────────────────────────────
    print("Sending request…", file=_info)
    try:
        result = client.segment(payload)
    except httpx.HTTPStatusError as exc:
        print(
            f"ERROR: /segment returned {exc.response.status_code}: "
            f"{exc.response.text[:200]}",
            file=sys.stderr,
        )
        return 1
    except httpx.HTTPError as exc:
        print(f"ERROR: Request failed: {exc}", file=sys.stderr)
        return 1

    # ── 5. Display ────────────────────────────────────────────────────────
    if args.output_json:
        print(json.dumps(result, indent=2))
        return 0

    masks = result.get("masks", [])
    print(f"Masks:     {len(masks)}")
    for i, mask in enumerate(masks):
        score = mask.get("score", 0.0)
        area = mask.get("area", 0)
        has_logits = mask.get("logits_b64") is not None
        logit_tag = "  [has logits]" if has_logits else ""
        print(f"  [{i}] score={score:.4f}  area={area} px{logit_tag}")
    print(f"Latency:   {result.get('latency_ms', 0):.1f} ms")
    print(f"RequestID: {result.get('request_id', '?')}")

    # ── 6. Save masks ─────────────────────────────────────────────────────
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, mask in enumerate(masks):
            mask_b64 = mask.get("mask_b64", "")
            if mask_b64:
                mask_path = out / f"mask_{i:02d}.png"
                mask_path.write_bytes(base64.b64decode(mask_b64))
                print(f"  Saved: {mask_path}")
            logits_b64 = mask.get("logits_b64", "")
            if logits_b64:
                logit_path = out / f"logits_{i:02d}.png"
                logit_path.write_bytes(base64.b64decode(logits_b64))
                print(f"  Saved: {logit_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
