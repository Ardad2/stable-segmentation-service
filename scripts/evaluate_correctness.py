#!/usr/bin/env python
"""evaluate_correctness.py — compare direct vs served segmentation outputs.

For each supported prompt type the script:
1. Runs inference directly via the adapter (no HTTP).
2. Calls the live service endpoint ``POST /api/v1/segment``.
3. Compares the resulting masks using IoU, pixel agreement, coverage, and
   dimension checks.
4. Emits a Markdown or CSV report.

The key expectation is that, for the same model and the same input, direct
and served outputs must be **identical** (IoU = 1.0, pixel agreement = 1.0).
Any divergence indicates a bug in the service layer.

Usage
-----
::

    # Mock backend (always available — no GPU needed)
    python scripts/evaluate_correctness.py \\
        --backend mock --url http://localhost:8000

    # SAM2 (service must be running with SEGMENTATION_BACKEND=sam2)
    python scripts/evaluate_correctness.py --backend sam2 \\
        --url http://localhost:8000 \\
        --sam2-checkpoint weights/sam2_hiera_large.pt \\
        --sam2-config sam2_hiera_l.yaml

    # CLIPSeg (service must be running with SEGMENTATION_BACKEND=clipseg)
    python scripts/evaluate_correctness.py --backend clipseg \\
        --url http://localhost:8000

    # Save Markdown report
    python scripts/evaluate_correctness.py --backend mock \\
        --url http://localhost:8000 --output report.md

    # CSV output
    python scripts/evaluate_correctness.py --backend mock \\
        --url http://localhost:8000 --format csv

Requirements
------------
- The service must be running at ``--url``.
- For SAM2/CLIPSeg the same model must be loaded both locally (direct) and
  in the running service.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Allow running from any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from segmentation_service.eval.correctness import (
    CorrectnessReport,
    MaskComparison,
    compare_responses,
    validate_response_metadata,
)
from segmentation_service.eval.direct_runners import DirectRunner
from segmentation_service.eval.probe_payloads import (
    BACKEND_PROBE_TYPES,
    load_request,
)
from segmentation_service.schemas.segment import (
    BoxPrompt,
    PointPrompt,
    PromptType,
    SegmentRequest,
)

# ---------------------------------------------------------------------------
# Probe inputs per backend × prompt type
# ---------------------------------------------------------------------------
# mock uses tiny built-in synthetic payloads (no asset files needed).
# sam2/clipseg load from eval_assets/requests/*.json so that probes use
# backend-appropriate images and coordinate ranges rather than a 1×1 stub
# that can produce unstable or degenerate model outputs.

_STUB_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

_PROBE_REQUESTS: dict[str, dict[str, SegmentRequest]] = {
    "mock": {
        "point": SegmentRequest(
            image=_STUB_IMAGE, image_format="png",
            prompt_type=PromptType.point,
            points=[PointPrompt(x=0, y=0, label=1)],
        ),
        "box": SegmentRequest(
            image=_STUB_IMAGE, image_format="png",
            prompt_type=PromptType.box,
            box=BoxPrompt(x_min=0, y_min=0, x_max=1, y_max=1),
        ),
        "text": SegmentRequest(
            image=_STUB_IMAGE, image_format="png",
            prompt_type=PromptType.text,
            text_prompt="object",
        ),
    },
    "sam2": {
        pt: load_request("sam2", pt)
        for pt in BACKEND_PROBE_TYPES["sam2"]
    },
    "clipseg": {
        pt: load_request("clipseg", pt)
        for pt in BACKEND_PROBE_TYPES["clipseg"]
    },
}


def _request_to_payload(req: SegmentRequest) -> dict:
    payload = req.model_dump(exclude_none=True)
    payload["prompt_type"] = req.prompt_type.value
    if req.points:
        payload["points"] = [p.model_dump() for p in req.points]
    if req.box:
        payload["box"] = req.box.model_dump()
    return payload


# ---------------------------------------------------------------------------
# Per-probe evaluation
# ---------------------------------------------------------------------------

def _run_probe(
    runner: DirectRunner,
    request: SegmentRequest,
    endpoint: str,
    timeout: float,
) -> tuple[CorrectnessReport, dict]:
    """Run one probe: direct + served, then compare.

    Returns (CorrectnessReport, metadata_checks_dict).
    """
    # Direct invocation.
    direct_result = runner.run(request)
    if direct_result.error:
        report = CorrectnessReport(
            backend=runner.backend,
            prompt_type=request.prompt_type.value,
            num_masks_direct=0,
            num_masks_served=0,
            mask_count_match=False,
            error=f"Direct error: {direct_result.error}",
        )
        return report, {}

    direct_masks_b64 = [m.mask_b64 for m in direct_result.response.masks]

    # Served invocation.
    payload = _request_to_payload(request)
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(endpoint, json=payload)
            resp.raise_for_status()
            served_data = resp.json()
    except httpx.HTTPError as exc:
        report = CorrectnessReport(
            backend=runner.backend,
            prompt_type=request.prompt_type.value,
            num_masks_direct=len(direct_masks_b64),
            num_masks_served=0,
            mask_count_match=False,
            error=f"Served error: {exc}",
        )
        return report, {}

    served_masks_b64 = [m["mask_b64"] for m in served_data.get("masks", [])]
    meta_notes = (
        f"direct_latency={direct_result.latency_ms:.1f}ms  "
        f"served_latency={served_data.get('latency_ms', '?')}ms"
    )

    report = compare_responses(
        direct_masks_b64=direct_masks_b64,
        served_masks_b64=served_masks_b64,
        backend=runner.backend,
        prompt_type=request.prompt_type.value,
        metadata_notes=meta_notes,
    )

    meta_checks = validate_response_metadata(served_data)
    return report, meta_checks


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_markdown(
    reports: list[CorrectnessReport],
    meta_checks_list: list[dict],
    base_url: str,
) -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    backend = reports[0].backend if reports else "unknown"

    lines = [
        f"# Correctness Report — {backend}",
        f"\nGenerated: {ts}  |  URL: `{base_url}`\n",
        "## Mask comparison\n",
        "| Prompt | Masks (D/S) | Dims match | Non-zero | IoU | Pixel agree | Coverage (D/S) | Result |",
        "|--------|:-----------:|:----------:|:--------:|:---:|:-----------:|:--------------:|:------:|",
    ]

    for report in reports:
        mc_list = report.mask_comparisons
        if mc_list:
            mc = mc_list[0]
            dims = "✅" if mc.dimensions_match else "❌"
            non_zero = "✅" if not mc.served_all_zero else "❌"
            iou_s = f"{mc.iou_score:.3f}" if mc.iou_score is not None else "—"
            pa_s = f"{mc.pixel_agreement_score:.3f}" if mc.pixel_agreement_score is not None else "—"
            cov_s = f"{mc.direct_coverage:.2f} / {mc.served_coverage:.2f}"
        else:
            dims = non_zero = "—"
            iou_s = pa_s = cov_s = "—"

        count = f"{report.num_masks_direct} / {report.num_masks_served}"
        result = "✅ PASS" if report.all_passed else "❌ FAIL"
        err = f" `{report.error[:40]}`" if report.error else ""
        lines.append(
            f"| `{report.prompt_type}` | {count} | {dims} | {non_zero} | "
            f"{iou_s} | {pa_s} | {cov_s} | {result}{err} |"
        )

    all_ok = all(r.all_passed for r in reports)
    lines.append(
        f"\n**Overall result: {'PASS ✅' if all_ok else 'FAIL ❌'}**\n"
    )

    if meta_checks_list:
        lines.append("## Served response metadata\n")
        all_keys: list[str] = sorted({k for d in meta_checks_list for k in d})
        lines.append("| Prompt | " + " | ".join(all_keys) + " |")
        lines.append("|--------|" + "|".join(":---:" for _ in all_keys) + "|")
        for report, meta in zip(reports, meta_checks_list):
            vals = " | ".join("✅" if meta.get(k) else "❌" for k in all_keys)
            lines.append(f"| `{report.prompt_type}` | {vals} |")

    return "\n".join(lines) + "\n"


def _fmt_csv(reports: list[CorrectnessReport]) -> str:
    buf = io.StringIO()
    fields = [
        "backend", "prompt_type",
        "num_masks_direct", "num_masks_served", "mask_count_match",
        "mask_index", "dimensions_match", "direct_all_zero", "served_all_zero",
        "direct_coverage", "served_coverage", "iou_score", "pixel_agreement_score",
        "all_passed", "error",
    ]
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()

    for r in reports:
        if not r.mask_comparisons:
            writer.writerow({
                "backend": r.backend, "prompt_type": r.prompt_type,
                "num_masks_direct": r.num_masks_direct,
                "num_masks_served": r.num_masks_served,
                "mask_count_match": r.mask_count_match,
                "mask_index": "", "dimensions_match": "",
                "direct_all_zero": "", "served_all_zero": "",
                "direct_coverage": "", "served_coverage": "",
                "iou_score": "", "pixel_agreement_score": "",
                "all_passed": r.all_passed, "error": r.error,
            })
        else:
            for mc in r.mask_comparisons:
                writer.writerow({
                    "backend": r.backend, "prompt_type": r.prompt_type,
                    "num_masks_direct": r.num_masks_direct,
                    "num_masks_served": r.num_masks_served,
                    "mask_count_match": r.mask_count_match,
                    "mask_index": mc.mask_index,
                    "dimensions_match": mc.dimensions_match,
                    "direct_all_zero": mc.direct_all_zero,
                    "served_all_zero": mc.served_all_zero,
                    "direct_coverage": mc.direct_coverage,
                    "served_coverage": mc.served_coverage,
                    "iou_score": mc.iou_score,
                    "pixel_agreement_score": mc.pixel_agreement_score,
                    "all_passed": r.all_passed, "error": r.error,
                })

    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="evaluate_correctness",
        description="Compare direct adapter output vs served /segment response.",
    )
    p.add_argument("--backend", default="mock",
                   help="Backend to test: mock | sam2 | clipseg  (default: mock)")
    p.add_argument("--url", default="http://localhost:8000",
                   help="Service base URL (default: http://localhost:8000)")
    p.add_argument("--sam2-checkpoint", default="",
                   help="Path to SAM2 .pt weights (required when --backend=sam2)")
    p.add_argument("--sam2-config", default="",
                   help="SAM2 YAML config name (required when --backend=sam2)")
    p.add_argument("--clipseg-model", default="CIDAS/clipseg-rd64-refined",
                   help="CLIPSeg model ID or path (default: CIDAS/clipseg-rd64-refined)")
    p.add_argument("--device", default="cpu",
                   help="Model device: cpu | cuda | mps  (default: cpu)")
    p.add_argument("--format", choices=["markdown", "csv"], default="markdown",
                   help="Output format (default: markdown)")
    p.add_argument("--output", metavar="FILE",
                   help="Write report to FILE instead of stdout")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="Per-request timeout in seconds (default: 60)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    base_url = args.url.rstrip("/")
    endpoint = f"{base_url}/api/v1/segment"

    print(f"Backend: {args.backend}  |  URL: {base_url}", file=sys.stderr)

    # Build runner.
    try:
        if args.backend == "mock":
            runner = DirectRunner.for_mock()
        elif args.backend == "sam2":
            if not args.sam2_checkpoint or not args.sam2_config:
                print("ERROR: --sam2-checkpoint and --sam2-config are required for --backend=sam2",
                      file=sys.stderr)
                return 1
            runner = DirectRunner.for_sam2(
                checkpoint=args.sam2_checkpoint,
                config=args.sam2_config,
                device=args.device,
            )
        elif args.backend == "clipseg":
            runner = DirectRunner.for_clipseg(
                model=args.clipseg_model, device=args.device
            )
        else:
            print(f"ERROR: Unknown backend: {args.backend!r}", file=sys.stderr)
            return 1
    except RuntimeError as exc:
        print(f"ERROR: Could not initialise runner: {exc}", file=sys.stderr)
        return 1

    probes = _PROBE_REQUESTS.get(args.backend, {})
    if not probes:
        print(f"ERROR: No probe requests defined for backend: {args.backend!r}", file=sys.stderr)
        return 1

    reports: list[CorrectnessReport] = []
    meta_checks_list: list[dict] = []

    for prompt_type, request in probes.items():
        print(f"  Probing {args.backend} / {prompt_type}…", file=sys.stderr)
        report, meta_checks = _run_probe(runner, request, endpoint, args.timeout)
        reports.append(report)
        meta_checks_list.append(meta_checks)
        status = "PASS" if report.all_passed else f"FAIL ({report.error or 'see report'})"
        print(f"    → {status}", file=sys.stderr)

    # Format report.
    if args.format == "markdown":
        report_text = _fmt_markdown(reports, meta_checks_list, base_url)
    else:
        report_text = _fmt_csv(reports)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(report_text)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report_text)

    all_passed = all(r.all_passed for r in reports)
    summary = "PASS" if all_passed else "FAIL"
    print(f"Overall: {summary}", file=sys.stderr)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
