#!/usr/bin/env python
"""evaluate_compatibility.py — query a live segmentation service and emit a
compatibility report.

The script:
1. Calls GET /api/v1/capabilities to discover the active backend.
2. Probes POST /api/v1/segment with every known prompt type (point, box, text).
3. Records: backend, prompt type, supported (per capabilities), observed HTTP
   status, latency (ms), and any error details.
4. Prints a Markdown or CSV report to stdout (or a file).

The report can be pasted directly into docs/compatibility-matrix.md or
included in a CI artefact.

Usage
-----
::

    python scripts/evaluate_compatibility.py --url http://localhost:8000

    # CSV output
    python scripts/evaluate_compatibility.py --url http://localhost:8000 --format csv

    # Save to file
    python scripts/evaluate_compatibility.py --url http://localhost:8000 \\
        --output docs/compatibility-matrix.md

No ML dependencies or model weights are required — the script works against
any running instance of the service, including one using the mock backend.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Tiny synthetic image (1×1 PNG) used in all probe requests.
# ---------------------------------------------------------------------------
_PROBE_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# Probe payloads for each prompt type.
_PROBE_PAYLOADS: dict[str, dict[str, Any]] = {
    "point": {
        "image": _PROBE_IMAGE,
        "image_format": "png",
        "prompt_type": "point",
        "points": [{"x": 0, "y": 0, "label": 1}],
    },
    "box": {
        "image": _PROBE_IMAGE,
        "image_format": "png",
        "prompt_type": "box",
        "box": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
    },
    "text": {
        "image": _PROBE_IMAGE,
        "image_format": "png",
        "prompt_type": "text",
        "text_prompt": "object",
    },
}

# Probe order (text first: most distinguishing across backends).
_PROBE_ORDER = ("point", "box", "text")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    backend: str
    prompt_type: str
    claimed_supported: bool
    http_status: int | None
    latency_ms: float | None
    masks_returned: int | None
    error: str

    @property
    def observed_ok(self) -> bool:
        return self.http_status == 200

    @property
    def aligned(self) -> bool:
        """True when observed behaviour matches what capabilities advertised."""
        return self.claimed_supported == self.observed_ok

    @property
    def status_emoji(self) -> str:
        if self.observed_ok:
            return "✅"
        return "❌"

    @property
    def aligned_emoji(self) -> str:
        return "✅" if self.aligned else "⚠️"


# ---------------------------------------------------------------------------
# Probe logic
# ---------------------------------------------------------------------------

def _fetch_capabilities(base_url: str, timeout: float) -> dict[str, Any]:
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(f"{base_url}/api/v1/capabilities")
        resp.raise_for_status()
        return resp.json()


def _probe(
    base_url: str,
    prompt_type: str,
    timeout: float,
) -> tuple[int | None, float | None, int | None, str]:
    """Send a single /segment probe request.

    Returns (http_status, latency_ms, masks_count, error_detail).
    """
    payload = _PROBE_PAYLOADS[prompt_type]
    try:
        with httpx.Client(timeout=timeout) as client:
            t0 = time.perf_counter()
            resp = client.post(f"{base_url}/api/v1/segment", json=payload)
            latency_ms = (time.perf_counter() - t0) * 1000

        status = resp.status_code
        error = ""
        masks_count = None

        if status == 200:
            data = resp.json()
            masks_count = len(data.get("masks", []))
            latency_ms = data.get("latency_ms", latency_ms)
        else:
            detail = ""
            try:
                detail = resp.json().get("detail", resp.text[:120])
            except Exception:
                detail = resp.text[:120]
            error = detail

        return status, round(latency_ms, 1), masks_count, error

    except httpx.TimeoutException:
        return None, None, None, "Request timed out"
    except httpx.HTTPError as exc:
        return None, None, None, str(exc)


def run_evaluation(base_url: str, timeout: float = 30.0) -> list[ProbeResult]:
    """Query capabilities and probe all prompt types. Return a list of results."""
    caps = _fetch_capabilities(base_url, timeout)
    backend = caps.get("backend", "unknown")
    supported_types: list[str] = caps.get("supported_prompt_types", [])

    results: list[ProbeResult] = []
    for pt in _PROBE_ORDER:
        claimed = pt in supported_types
        status, latency, masks, error = _probe(base_url, pt, timeout)
        results.append(
            ProbeResult(
                backend=backend,
                prompt_type=pt,
                claimed_supported=claimed,
                http_status=status,
                latency_ms=latency,
                masks_returned=masks,
                error=error,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_markdown(results: list[ProbeResult], base_url: str) -> str:
    lines: list[str] = []
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    backend = results[0].backend if results else "unknown"

    lines.append(f"# Compatibility Report — {backend}")
    lines.append(f"\nGenerated: {ts}  |  URL: `{base_url}`\n")
    lines.append(
        "| Prompt type | Advertised | Observed | HTTP | "
        "Latency (ms) | Masks | Aligned | Notes |"
    )
    lines.append("|-------------|:----------:|:--------:|:----:|:-----------:|:-----:|:-------:|-------|")

    for r in results:
        adv = "✅ yes" if r.claimed_supported else "❌ no"
        obs = r.status_emoji
        http = str(r.http_status) if r.http_status else "—"
        lat = f"{r.latency_ms:.1f}" if r.latency_ms is not None else "—"
        masks = str(r.masks_returned) if r.masks_returned is not None else "—"
        aligned = r.aligned_emoji
        notes = r.error[:60] if r.error else ""
        lines.append(
            f"| `{r.prompt_type}` | {adv} | {obs} | {http} | {lat} | {masks} | {aligned} | {notes} |"
        )

    # Summary line
    all_aligned = all(r.aligned for r in results)
    lines.append(
        f"\n**All behaviours aligned with advertised capabilities: "
        f"{'Yes ✅' if all_aligned else 'No ⚠️'}**"
    )
    return "\n".join(lines) + "\n"


def _fmt_csv(results: list[ProbeResult]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "backend", "prompt_type", "claimed_supported", "observed_ok",
        "http_status", "latency_ms", "masks_returned", "aligned", "error",
    ])
    for r in results:
        writer.writerow([
            r.backend, r.prompt_type, r.claimed_supported, r.observed_ok,
            r.http_status, r.latency_ms, r.masks_returned, r.aligned, r.error,
        ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate_compatibility",
        description="Probe a running segmentation service and emit a compatibility report.",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Service base URL (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "csv"],
        default="markdown",
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write output to FILE instead of stdout.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Per-request timeout in seconds (default: 30).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    base_url = args.url.rstrip("/")

    print(f"Probing: {base_url}", file=sys.stderr)

    try:
        results = run_evaluation(base_url, timeout=args.timeout)
    except httpx.HTTPError as exc:
        print(f"ERROR: Could not reach {base_url}: {exc}", file=sys.stderr)
        return 1

    if args.format == "markdown":
        report = _fmt_markdown(results, base_url)
    else:
        report = _fmt_csv(results)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)

    # Print a quick summary to stderr.
    backend = results[0].backend if results else "unknown"
    ok = sum(1 for r in results if r.observed_ok)
    total = len(results)
    aligned = all(r.aligned for r in results)
    print(
        f"Backend: {backend}  |  Successful probes: {ok}/{total}  |  "
        f"All aligned: {'yes' if aligned else 'NO — check report'}",
        file=sys.stderr,
    )

    return 0 if aligned else 1


if __name__ == "__main__":
    sys.exit(main())
