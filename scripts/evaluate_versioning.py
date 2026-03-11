#!/usr/bin/env python
"""evaluate_versioning.py — exercise client/server version-pair combinations
against a live segmentation service and emit a compatibility matrix.

The script:
1. Discovers the active backend via GET /api/v1/capabilities.
2. Probes every row of the compatibility matrix:
   - v1 payload → /api/v1/segment          (expect PASS)
   - v1 payload + api_version field check   (expect PASS — additive)
   - v2 payload → /api/v2/segment          (expect PASS)
   - v1 payload → /api/v2/segment          (expect FAIL 422 — breaking)
   - v2 payload → /api/v1/segment          (expect FAIL 422 — breaking)
3. Prints a Markdown or CSV compatibility matrix to stdout (or a file).

Output columns
--------------
client_version | server_api_version | backend | scenario | expected_result
| observed_result | pass_fail | notes

Usage
-----
::

    # Start the service first (any backend):
    SEGMENTATION_BACKEND=mock uvicorn segmentation_service.main:app --port 8000

    python scripts/evaluate_versioning.py --url http://localhost:8000

    # CSV
    python scripts/evaluate_versioning.py --url http://localhost:8000 --format csv

    # Save to file
    python scripts/evaluate_versioning.py --url http://localhost:8000 \\
        --output docs/versioning-matrix-observed.md

Exit codes
----------
0 — all rows match their expected result
1 — one or more rows did not match, or a runtime error occurred
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Tiny 1×1 PNG used in all probe requests.
# ---------------------------------------------------------------------------
_PROBE_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# ---------------------------------------------------------------------------
# Probe payloads
# ---------------------------------------------------------------------------

_V1_POINT_PAYLOAD: dict[str, Any] = {
    "image": _PROBE_IMAGE,
    "image_format": "png",
    "prompt_type": "point",
    "points": [{"x": 0, "y": 0, "label": 1}],
}

_V2_POINT_PAYLOAD: dict[str, Any] = {
    "image": _PROBE_IMAGE,
    "image_format": "png",
    "prompt": {"type": "point", "points": [{"x": 0, "y": 0, "label": 1}]},
}


# ---------------------------------------------------------------------------
# Row dataclass
# ---------------------------------------------------------------------------

@dataclass
class VersioningProbeResult:
    client_version: str
    server_api_version: str
    backend: str
    scenario: str
    expected_result: str      # "PASS" or "FAIL"
    observed_result: str      # "PASS" or "FAIL <status_code>"
    pass_fail: str            # "PASS" or "FAIL"
    notes: str = ""

    @property
    def ok(self) -> bool:
        return self.pass_fail == "PASS"


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def _probe(
    client: httpx.Client,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[int, dict[str, Any], float]:
    """Issue one HTTP request; return (status_code, response_body, latency_ms)."""
    t0 = time.perf_counter()
    try:
        if method == "GET":
            resp = client.get(path, timeout=timeout)
        else:
            resp = client.post(path, json=payload, timeout=timeout)
    except httpx.RequestError as exc:
        return -1, {"error": str(exc)}, (time.perf_counter() - t0) * 1000
    latency_ms = (time.perf_counter() - t0) * 1000
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    return resp.status_code, body, latency_ms


def _discover_backend(client: httpx.Client, base_url: str, timeout: float) -> str:
    status, body, _ = _probe(client, "GET", f"{base_url}/api/v1/capabilities", None, timeout)
    if status == 200:
        return body.get("backend", "unknown")
    return "unknown"


# ---------------------------------------------------------------------------
# Matrix definition
# ---------------------------------------------------------------------------

def _build_matrix(base_url: str, backend: str) -> list[dict]:
    """Return a list of probe specs — each describes one row of the matrix."""
    seg_v1 = f"{base_url}/api/v1/segment"
    seg_v2 = f"{base_url}/api/v2/segment"

    return [
        {
            "client_version": "v1",
            "server_api_version": "v1",
            "backend": backend,
            "scenario": "v1 payload → /api/v1/segment (exact match)",
            "expected_result": "PASS",
            "method": "POST",
            "path": seg_v1,
            "payload": _V1_POINT_PAYLOAD,
            "check": lambda status, body: status == 200,
            "notes_fn": lambda status, body: (
                f"api_version={body.get('api_version')} masks={len(body.get('masks', []))}"
                if status == 200 else f"status={status}"
            ),
        },
        {
            "client_version": "v1",
            "server_api_version": "v1",
            "backend": backend,
            "scenario": "v1 response includes api_version='1.0' (additive)",
            "expected_result": "PASS",
            "method": "POST",
            "path": seg_v1,
            "payload": _V1_POINT_PAYLOAD,
            "check": lambda status, body: (
                status == 200 and body.get("api_version") == "1.0"
            ),
            "notes_fn": lambda status, body: (
                f"api_version={body.get('api_version')!r}"
                if status == 200 else f"status={status}"
            ),
        },
        {
            "client_version": "v1",
            "server_api_version": "v1",
            "backend": backend,
            "scenario": "v1 mask field is 'mask_b64' (v1 contract preserved)",
            "expected_result": "PASS",
            "method": "POST",
            "path": seg_v1,
            "payload": _V1_POINT_PAYLOAD,
            "check": lambda status, body: (
                status == 200
                and body.get("masks")
                and "mask_b64" in body["masks"][0]
            ),
            "notes_fn": lambda status, body: (
                f"mask keys: {list(body['masks'][0].keys())}"
                if status == 200 and body.get("masks") else f"status={status}"
            ),
        },
        {
            "client_version": "v2",
            "server_api_version": "v2",
            "backend": backend,
            "scenario": "v2 payload → /api/v2/segment (exact match)",
            "expected_result": "PASS",
            "method": "POST",
            "path": seg_v2,
            "payload": _V2_POINT_PAYLOAD,
            "check": lambda status, body: status == 200,
            "notes_fn": lambda status, body: (
                f"api_version={body.get('api_version')} masks={len(body.get('masks', []))}"
                if status == 200 else f"status={status}"
            ),
        },
        {
            "client_version": "v2",
            "server_api_version": "v2",
            "backend": backend,
            "scenario": "v2 response includes api_version='2.0'",
            "expected_result": "PASS",
            "method": "POST",
            "path": seg_v2,
            "payload": _V2_POINT_PAYLOAD,
            "check": lambda status, body: (
                status == 200 and body.get("api_version") == "2.0"
            ),
            "notes_fn": lambda status, body: (
                f"api_version={body.get('api_version')!r}"
                if status == 200 else f"status={status}"
            ),
        },
        {
            "client_version": "v2",
            "server_api_version": "v2",
            "backend": backend,
            "scenario": "v2 mask field is 'mask_data' not 'mask_b64'",
            "expected_result": "PASS",
            "method": "POST",
            "path": seg_v2,
            "payload": _V2_POINT_PAYLOAD,
            "check": lambda status, body: (
                status == 200
                and body.get("masks")
                and "mask_data" in body["masks"][0]
                and "mask_b64" not in body["masks"][0]
            ),
            "notes_fn": lambda status, body: (
                f"mask keys: {list(body['masks'][0].keys())}"
                if status == 200 and body.get("masks") else f"status={status}"
            ),
        },
        {
            "client_version": "v1",
            "server_api_version": "v2",
            "backend": backend,
            "scenario": "v1 payload → /api/v2/segment (BREAKING — missing 'prompt')",
            "expected_result": "FAIL",
            "method": "POST",
            "path": seg_v2,
            "payload": _V1_POINT_PAYLOAD,
            "check": lambda status, body: status == 422,
            "notes_fn": lambda status, body: f"status={status} (expected 422)",
        },
        {
            "client_version": "v2",
            "server_api_version": "v1",
            "backend": backend,
            "scenario": "v2 payload → /api/v1/segment (BREAKING — no prompt_type field)",
            "expected_result": "FAIL",
            "method": "POST",
            "path": seg_v1,
            "payload": _V2_POINT_PAYLOAD,
            "check": lambda status, body: status == 422,
            "notes_fn": lambda status, body: f"status={status} (expected 422)",
        },
    ]


# ---------------------------------------------------------------------------
# Run all probes
# ---------------------------------------------------------------------------

def run_matrix(
    base_url: str,
    timeout: float = 30.0,
) -> list[VersioningProbeResult]:
    results: list[VersioningProbeResult] = []

    with httpx.Client() as client:
        backend = _discover_backend(client, base_url, timeout)

        for spec in _build_matrix(base_url, backend):
            status, body, _lat = _probe(
                client, spec["method"], spec["path"], spec["payload"], timeout
            )
            passed = spec["check"](status, body)
            notes = spec["notes_fn"](status, body)

            if status == -1:
                observed = "ERROR"
                pf = "FAIL"
            elif spec["expected_result"] == "PASS":
                observed = "PASS" if passed else f"FAIL ({status})"
                pf = "PASS" if passed else "FAIL"
            else:  # expected FAIL
                observed = "FAIL" if passed else f"PASS (unexpected {status})"
                pf = "PASS" if passed else "FAIL"

            results.append(
                VersioningProbeResult(
                    client_version=spec["client_version"],
                    server_api_version=spec["server_api_version"],
                    backend=backend,
                    scenario=spec["scenario"],
                    expected_result=spec["expected_result"],
                    observed_result=observed,
                    pass_fail=pf,
                    notes=notes,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_markdown(results: list[VersioningProbeResult], base_url: str) -> str:
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = len(results)
    passed = sum(1 for r in results if r.ok)
    backend = results[0].backend if results else "unknown"

    lines = [
        "# API Versioning Compatibility Matrix",
        "",
        f"Generated: {now}  |  URL: `{base_url}`  |  Backend: `{backend}`",
        "",
        f"**{passed}/{total} rows passed**",
        "",
        "| Client | Server | Scenario | Expected | Observed | Result | Notes |",
        "|--------|--------|----------|----------|----------|--------|-------|",
    ]
    for r in results:
        icon = "✅" if r.ok else "❌"
        scenario = r.scenario.replace("|", "\\|")
        notes = r.notes.replace("|", "\\|")
        lines.append(
            f"| {r.client_version} | {r.server_api_version} | {scenario} "
            f"| {r.expected_result} | {r.observed_result} | {icon} {r.pass_fail} | {notes} |"
        )

    overall = "✅ ALL PASSED" if passed == total else f"❌ {total - passed} FAILED"
    lines += ["", f"**Overall: {overall}**", ""]
    return "\n".join(lines)


def _fmt_csv(results: list[VersioningProbeResult]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "client_version", "server_api_version", "backend", "scenario",
        "expected_result", "observed_result", "pass_fail", "notes",
    ])
    for r in results:
        w.writerow([
            r.client_version, r.server_api_version, r.backend, r.scenario,
            r.expected_result, r.observed_result, r.pass_fail, r.notes,
        ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Probe a live service and emit a versioning compatibility matrix."
    )
    p.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the running service (default: http://localhost:8000)",
    )
    p.add_argument(
        "--format",
        choices=["markdown", "csv"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write output to FILE instead of stdout",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds (default: 30)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        results = run_matrix(base_url=args.url, timeout=args.timeout)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.format == "csv":
        output = _fmt_csv(results)
    else:
        output = _fmt_markdown(results, args.url)

    if args.output:
        import pathlib
        pathlib.Path(args.output).write_text(output, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)

    all_ok = all(r.ok for r in results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
