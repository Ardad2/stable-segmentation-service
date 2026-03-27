"""Direct-vs-served latency comparison benchmark.

Measures the overhead introduced by the HTTP / FastAPI service layer by running
the same input through:

    A. Direct local model invocation   (bypasses all HTTP, ASGI, routing)
    B. Served endpoint invocation      (/api/v1/segment over HTTP)

Usage
-----
::

    # Mock backend — no GPU or model weights required (start the service first)
    python benchmark/direct_vs_served.py --backend mock \\
        --url http://localhost:8000 --n 20

    # SAM2 backend (service must be running with SEGMENTATION_BACKEND=sam2)
    python benchmark/direct_vs_served.py --backend sam2 \\
        --url http://localhost:8000 \\
        --sam2-checkpoint weights/sam2_hiera_large.pt \\
        --sam2-config sam2_hiera_l.yaml \\
        --n 10 --warmup 3

    # CLIPSeg backend
    python benchmark/direct_vs_served.py --backend clipseg \\
        --url http://localhost:8000 \\
        --clipseg-model CIDAS/clipseg-rd64-refined \\
        --n 10 --warmup 3

    # Supply an arbitrary payload from a JSON file
    python benchmark/direct_vs_served.py --backend clipseg \\
        --url http://localhost:8000 \\
        --payload-file eval_assets/requests/clipseg_text.json \\
        --n 10 --warmup 3

    # Export CSV for the final report
    python benchmark/direct_vs_served.py --backend mock \\
        --url http://localhost:8000 --n 20 --csv results.csv

Notes
-----
- The service must be running at --url before this script starts.
- Direct latency includes the asyncio event loop creation overhead for each
  call (``asyncio.run()``) but not network I/O.
- Reported "overhead" = median(served) - median(direct).  It includes ASGI
  routing, request serialisation/deserialisation, and TCP stack overhead.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

# Locate eval assets relative to this script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from segmentation_service.eval.direct_runners import DirectRunner
from segmentation_service.eval.probe_payloads import load_request
from segmentation_service.schemas.segment import SegmentRequest

app = typer.Typer(add_completion=False)
console = Console()

# ---------------------------------------------------------------------------
# Shared test inputs — loaded via probe_payloads so that sam2/clipseg use
# backend-appropriate asset images rather than a 1×1 synthetic stub.
# ---------------------------------------------------------------------------

_REQUESTS: dict[str, SegmentRequest] = {
    "mock-point":   load_request("mock",    "point"),
    "mock-box":     load_request("mock",    "box"),
    "mock-text":    load_request("mock",    "text"),
    "sam2-point":   load_request("sam2",    "point"),
    "sam2-box":     load_request("sam2",    "box"),
    "clipseg-text": load_request("clipseg", "text"),
}

_BACKEND_DEFAULT_KEY = {
    "mock": "mock-point",
    "sam2": "sam2-point",
    "clipseg": "clipseg-text",
}

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_served(endpoint: str, payload: dict, n: int) -> list[float]:
    """Run n HTTP requests and return latencies in ms."""
    latencies = []
    with httpx.Client(timeout=60) as client:
        for _ in range(n):
            t0 = time.perf_counter()
            resp = client.post(endpoint, json=payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            latencies.append(round(elapsed_ms, 3))
    return latencies


def _time_direct(runner: DirectRunner, request: SegmentRequest, n: int) -> list[float]:
    """Run n direct invocations and return latencies in ms."""
    latencies = []
    for _ in range(n):
        result = runner.run(request)
        if result.error:
            console.print(f"[red]Direct error:[/red] {result.error}")
        latencies.append(result.latency_ms)
    return latencies


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    return {
        "n": len(s),
        "min": min(s),
        "p50": statistics.median(s),
        "p95": s[max(0, int(len(s) * 0.95) - 1)],
        "p99": s[max(0, int(len(s) * 0.99) - 1)],
        "max": max(s),
        "mean": statistics.mean(s),
        "stdev": statistics.stdev(s) if len(s) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


@app.command()
def main(  # noqa: C901
    backend: Annotated[str, typer.Option(help="Backend: mock | sam2 | clipseg")] = "mock",
    url: Annotated[str, typer.Option(help="Service base URL")] = "http://localhost:8000",
    n: Annotated[int, typer.Option(help="Measurement iterations (after warm-up)")] = 20,
    warmup: Annotated[int, typer.Option(help="Warm-up iterations (discarded)")] = 3,
    prompt_type: Annotated[str, typer.Option(help="Override prompt type: point | box | text")] = "",
    payload_file: Annotated[str, typer.Option(help="Path to a JSON payload file (overrides built-in probe)")] = "",
    sam2_checkpoint: Annotated[str, typer.Option(help="SAM2 checkpoint path")] = "",
    sam2_config: Annotated[str, typer.Option(help="SAM2 config name")] = "",
    clipseg_model: Annotated[str, typer.Option(help="CLIPSeg model ID or path")] = "CIDAS/clipseg-rd64-refined",
    device: Annotated[str, typer.Option(help="cpu | cuda | mps")] = "cpu",
    csv_out: Annotated[str, typer.Option("--csv", help="Path for CSV output")] = "",
) -> None:
    endpoint = f"{url.rstrip('/')}/api/v1/segment"
    console.print(f"[bold]Direct-vs-Served Benchmark[/bold]  backend={backend}  n={n}  warmup={warmup}")
    console.print(f"  served endpoint: {endpoint}")

    # ── 1. Build request ──────────────────────────────────────────────────
    if payload_file:
        # Explicit file takes precedence over everything else.
        payload = json.loads(Path(payload_file).read_text(encoding="utf-8"))
        request = SegmentRequest.model_validate(payload)
        key = f"{backend}-{request.prompt_type.value}"
    else:
        key = _BACKEND_DEFAULT_KEY.get(backend, "mock-point")
        if prompt_type:
            # Allow overriding the prompt type for the given backend.
            overrides = {"mock": "mock", "sam2": "sam2", "clipseg": "clipseg"}
            key = f"{overrides.get(backend, 'mock')}-{prompt_type}"
        request = _REQUESTS.get(key)
        if request is None:
            console.print(f"[red]Unknown request key:[/red] {key!r}. Available: {list(_REQUESTS)}")
            raise typer.Exit(1)
        payload = request.model_dump(exclude_none=True)
        # JSON-serialise enum values.
        payload["prompt_type"] = request.prompt_type.value
        if request.points:
            payload["points"] = [p.model_dump() for p in request.points]
        if request.box:
            payload["box"] = request.box.model_dump()

    # ── 2. Build direct runner ────────────────────────────────────────────
    console.print(f"  Building direct runner for '{backend}'…")
    try:
        if backend == "mock":
            runner = DirectRunner.for_mock()
        elif backend == "sam2":
            if not sam2_checkpoint or not sam2_config:
                console.print("[red]--sam2-checkpoint and --sam2-config are required for backend=sam2[/red]")
                raise typer.Exit(1)
            runner = DirectRunner.for_sam2(checkpoint=sam2_checkpoint, config=sam2_config, device=device)
        elif backend == "clipseg":
            runner = DirectRunner.for_clipseg(model=clipseg_model, device=device)
        else:
            console.print(f"[red]Unknown backend:[/red] {backend!r}.  Use: mock | sam2 | clipseg")
            raise typer.Exit(1)
    except RuntimeError as exc:
        console.print(f"[red]Runner init failed:[/red] {exc}")
        raise typer.Exit(1)

    # ── 3. Warm-up ────────────────────────────────────────────────────────
    if warmup > 0:
        console.print(f"  Warming up ({warmup} iterations)…")
        # Warm-up direct.
        for _ in range(warmup):
            runner.run(request)
        # Warm-up served.
        with httpx.Client(timeout=60) as client:
            for _ in range(warmup):
                try:
                    client.post(endpoint, json=payload).raise_for_status()
                except Exception:
                    pass

    # ── 4. Measure ────────────────────────────────────────────────────────
    console.print(f"  Measuring direct   ({n} iterations)…")
    direct_ms = _time_direct(runner, request, n)

    console.print(f"  Measuring served   ({n} iterations)…")
    try:
        served_ms = _time_served(endpoint, payload, n)
    except httpx.HTTPError as exc:
        console.print(f"[red]Served request failed:[/red] {exc}")
        raise typer.Exit(1)

    # ── 5. Compute stats ──────────────────────────────────────────────────
    d = _stats(direct_ms)
    s = _stats(served_ms)

    overhead_ms = s["p50"] - d["p50"]
    overhead_pct = (overhead_ms / d["p50"] * 100) if d["p50"] > 0 else float("inf")

    # ── 6. Display ────────────────────────────────────────────────────────
    table = Table(title=f"Direct vs Served — {backend} / {key} (ms)")
    table.add_column("Metric", style="cyan")
    table.add_column("Direct", justify="right")
    table.add_column("Served", justify="right")
    table.add_column("Overhead", justify="right", style="yellow")

    def _row(label: str, dk: str) -> None:
        ov = s[dk] - d[dk]
        table.add_row(label, f"{d[dk]:.2f}", f"{s[dk]:.2f}", f"{ov:+.2f}")

    _row("min (ms)", "min")
    _row("p50 (ms)", "p50")
    _row("p95 (ms)", "p95")
    _row("p99 (ms)", "p99")
    _row("max (ms)", "max")
    _row("mean (ms)", "mean")
    _row("stdev (ms)", "stdev")

    console.print(table)
    console.print(
        f"[bold]Service overhead (p50):[/bold]  "
        f"{overhead_ms:+.2f} ms  ({overhead_pct:+.1f}%)"
    )

    # ── 7. CSV export ─────────────────────────────────────────────────────
    if csv_out:
        rows = [
            {
                "backend": backend,
                "request_key": key,
                "side": "direct",
                "iteration": i + 1,
                "latency_ms": v,
            }
            for i, v in enumerate(direct_ms)
        ] + [
            {
                "backend": backend,
                "request_key": key,
                "side": "served",
                "iteration": i + 1,
                "latency_ms": v,
            }
            for i, v in enumerate(served_ms)
        ]
        with open(csv_out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["backend", "request_key", "side", "iteration", "latency_ms"])
            writer.writeheader()
            writer.writerows(rows)

        # Also append a summary row.
        summary_path = csv_out.replace(".csv", "_summary.csv")
        summary_rows = [
            {
                "backend": backend, "request_key": key,
                "side": side, "metric": metric, "value_ms": val,
            }
            for side, stats in (("direct", d), ("served", s))
            for metric, val in stats.items()
            if metric != "n"
        ] + [
            {"backend": backend, "request_key": key, "side": "overhead", "metric": "p50", "value_ms": overhead_ms},
            {"backend": backend, "request_key": key, "side": "overhead", "metric": "p50_pct", "value_ms": overhead_pct},
        ]
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["backend", "request_key", "side", "metric", "value_ms"])
            writer.writeheader()
            writer.writerows(summary_rows)

        console.print(f"  Raw data  → {csv_out}")
        console.print(f"  Summary   → {summary_path}")


if __name__ == "__main__":
    app()
