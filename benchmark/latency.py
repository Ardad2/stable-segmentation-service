"""Single-request latency benchmark.

Usage:
    # Default mock probe
    python benchmark/latency.py --url http://localhost:8000 --n 50

    # Use a built-in backend probe payload
    python benchmark/latency.py --url http://localhost:8000 --backend sam2 --n 20

    # Supply an arbitrary payload from a JSON file
    python benchmark/latency.py --url http://localhost:8000 \\
        --payload-file eval_assets/requests/sam2_point.json --n 20
"""

from __future__ import annotations

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

# Allow running from any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from segmentation_service.eval.probe_payloads import (  # noqa: E402
    DEFAULT_PROMPT_TYPE,
    load_payload,
)

app = typer.Typer(add_completion=False)
console = Console()


def _resolve_payload(
    backend: str | None,
    payload_file: str | None,
) -> dict:
    """Return the request payload dict to use for the benchmark.

    Priority: --payload-file > --backend > built-in mock/point default.
    """
    if payload_file:
        return json.loads(Path(payload_file).read_text(encoding="utf-8"))
    if backend:
        prompt_type = DEFAULT_PROMPT_TYPE.get(backend, "point")
        return load_payload(backend, prompt_type)
    # Default: mock point probe
    return load_payload("mock", "point")


@app.command()
def main(
    url: Annotated[str, typer.Option(help="Base URL of the service")] = "http://localhost:8000",
    n: Annotated[int, typer.Option(help="Number of requests")] = 50,
    concurrency: Annotated[int, typer.Option(help="(unused in latency mode — serial only)")] = 1,
    backend: Annotated[str | None, typer.Option(help="Built-in probe backend: mock | sam2 | clipseg")] = None,
    payload_file: Annotated[str | None, typer.Option(help="Path to a JSON payload file")] = None,
) -> None:
    endpoint = f"{url.rstrip('/')}/api/v1/segment"
    payload = _resolve_payload(backend, payload_file)
    source = payload_file or (f"backend={backend}" if backend else "backend=mock (default)")
    console.print(f"[bold]Latency benchmark[/bold]  endpoint={endpoint}  n={n}  payload={source}")

    latencies: list[float] = []

    with httpx.Client(timeout=30) as client:
        for i in range(n):
            t0 = time.perf_counter()
            resp = client.post(endpoint, json=payload)
            elapsed = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            latencies.append(elapsed)
            if (i + 1) % 10 == 0:
                console.print(f"  {i+1}/{n} done, last={elapsed:.1f} ms")

    table = Table(title="Latency Results (ms)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    p = sorted(latencies)
    table.add_row("n", str(n))
    table.add_row("min", f"{min(p):.2f}")
    table.add_row("p50", f"{statistics.median(p):.2f}")
    table.add_row("p95", f"{p[int(len(p) * 0.95)]:.2f}")
    table.add_row("p99", f"{p[int(len(p) * 0.99)]:.2f}")
    table.add_row("max", f"{max(p):.2f}")
    table.add_row("mean", f"{statistics.mean(p):.2f}")
    table.add_row("stdev", f"{statistics.stdev(p):.2f}")

    console.print(table)


if __name__ == "__main__":
    app()
