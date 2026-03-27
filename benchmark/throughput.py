"""Concurrent throughput benchmark.

Usage:
    # Default mock probe
    python benchmark/throughput.py --url http://localhost:8000 --duration 10 --concurrency 8

    # Use a built-in backend probe payload
    python benchmark/throughput.py --url http://localhost:8000 --backend sam2 --duration 10

    # Supply an arbitrary payload from a JSON file
    python benchmark/throughput.py --url http://localhost:8000 \\
        --payload-file eval_assets/requests/sam2_point.json --duration 10
"""

from __future__ import annotations

import asyncio
import json
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


async def _worker(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict,
    stop_event: asyncio.Event,
    results: list[float],
    errors: list[str],
) -> None:
    while not stop_event.is_set():
        t0 = time.perf_counter()
        try:
            resp = await client.post(endpoint, json=payload)
            resp.raise_for_status()
            results.append((time.perf_counter() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))


async def _run(url: str, duration: int, concurrency: int, payload: dict) -> None:
    endpoint = f"{url.rstrip('/')}/api/v1/segment"
    console.print(
        f"[bold]Throughput benchmark[/bold]  endpoint={endpoint}  "
        f"duration={duration}s  concurrency={concurrency}"
    )

    results: list[float] = []
    errors: list[str] = []
    stop_event = asyncio.Event()

    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [
            asyncio.create_task(
                _worker(client, endpoint, payload, stop_event, results, errors)
            )
            for _ in range(concurrency)
        ]
        t_start = time.perf_counter()
        await asyncio.sleep(duration)
        stop_event.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.perf_counter() - t_start

    total = len(results)
    rps = total / elapsed if elapsed > 0 else 0.0

    table = Table(title="Throughput Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("duration (s)", f"{elapsed:.2f}")
    table.add_row("concurrency", str(concurrency))
    table.add_row("total requests", str(total))
    table.add_row("errors", str(len(errors)))
    table.add_row("req/s (RPS)", f"{rps:.2f}")
    if results:
        import statistics

        table.add_row("mean latency (ms)", f"{statistics.mean(results):.2f}")
        table.add_row("p95 latency (ms)", f"{sorted(results)[int(len(results)*0.95)]:.2f}")
    console.print(table)


@app.command()
def main(
    url: Annotated[str, typer.Option()] = "http://localhost:8000",
    duration: Annotated[int, typer.Option(help="Test duration in seconds")] = 10,
    concurrency: Annotated[int, typer.Option(help="Number of parallel workers")] = 4,
    backend: Annotated[str | None, typer.Option(help="Built-in probe backend: mock | sam2 | clipseg")] = None,
    payload_file: Annotated[str | None, typer.Option(help="Path to a JSON payload file")] = None,
) -> None:
    payload = _resolve_payload(backend, payload_file)
    asyncio.run(_run(url, duration, concurrency, payload))


if __name__ == "__main__":
    app()
