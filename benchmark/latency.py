"""Single-request latency benchmark.

Usage:
    python benchmark/latency.py --url http://localhost:8000 --n 50
"""

from __future__ import annotations

import statistics
import time
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()

_STUB_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

_PAYLOAD = {
    "image": _STUB_IMAGE,
    "image_format": "png",
    "prompt_type": "point",
    "points": [{"x": 50, "y": 50, "label": 1}],
}


@app.command()
def main(
    url: Annotated[str, typer.Option(help="Base URL of the service")] = "http://localhost:8000",
    n: Annotated[int, typer.Option(help="Number of requests")] = 50,
    concurrency: Annotated[int, typer.Option(help="(unused in latency mode — serial only)")] = 1,
) -> None:
    endpoint = f"{url.rstrip('/')}/api/v1/segment"
    console.print(f"[bold]Latency benchmark[/bold]  endpoint={endpoint}  n={n}")

    latencies: list[float] = []

    with httpx.Client(timeout=30) as client:
        for i in range(n):
            t0 = time.perf_counter()
            resp = client.post(endpoint, json=_PAYLOAD)
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
