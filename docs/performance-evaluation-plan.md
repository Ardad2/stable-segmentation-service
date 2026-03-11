# Performance Evaluation Plan — Direct-vs-Served Latency

## Purpose

Measure the overhead introduced by the HTTP / FastAPI / ASGI service layer relative
to calling the segmentation adapter directly (no network, no serialisation).

The benchmark isolates the _service layer cost_ — not the model inference cost.

---

## What is being measured

| Path | Label | Description |
|------|-------|-------------|
| `adapter.segment(request)` via `asyncio.run()` | **direct** | Model inference + sync event-loop overhead; no HTTP |
| `POST /api/v1/segment` via `httpx.Client` | **served** | Full round-trip: TCP, ASGI, routing, JSON ser/deser, model inference, response |

**Service overhead** = median(served) − median(direct)

Because both paths run the same model, the overhead attributable to the service
layer alone can be read directly from this difference.

---

## Metrics collected

| Metric | Description |
|--------|-------------|
| `min` | Fastest observed response time (ms) |
| `p50` | Median latency (primary comparison metric) |
| `p95` | 95th-percentile — tail latency |
| `p99` | 99th-percentile — worst realistic case |
| `max` | Slowest observed response (ms) |
| `mean` | Arithmetic mean |
| `stdev` | Standard deviation (stability indicator) |
| overhead p50 | `p50(served) − p50(direct)` in ms and % |

---

## Running the benchmark

### Prerequisites

```bash
pip install -e ".[benchmark]"
# The service must already be running:
SEGMENTATION_BACKEND=mock uvicorn segmentation_service.main:app --port 8000
```

### Mock backend (no GPU required)

```bash
python benchmark/direct_vs_served.py --backend mock \
    --url http://localhost:8000 --n 20
```

### SAM2 backend

```bash
python benchmark/direct_vs_served.py --backend sam2 \
    --url http://localhost:8000 \
    --sam2-checkpoint weights/sam2_hiera_large.pt \
    --sam2-config sam2_hiera_l.yaml \
    --n 10 --warmup 3
```

### CLIPSeg backend

```bash
python benchmark/direct_vs_served.py --backend clipseg \
    --url http://localhost:8000 \
    --clipseg-model CIDAS/clipseg-rd64-refined \
    --n 10 --warmup 3
```

### Export results to CSV

```bash
python benchmark/direct_vs_served.py --backend mock \
    --url http://localhost:8000 --n 20 --csv results.csv
# Produces: results.csv and results_summary.csv
```

---

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | `mock` | `mock` \| `sam2` \| `clipseg` |
| `--url` | `http://localhost:8000` | Service base URL |
| `--n` | `20` | Number of measurement iterations (after warm-up) |
| `--warmup` | `3` | Warm-up iterations (results discarded) |
| `--prompt-type` | _(backend default)_ | Override: `point` \| `box` \| `text` |
| `--device` | `cpu` | `cpu` \| `cuda` \| `mps` |
| `--csv PATH` | _(none)_ | Write raw latency rows and summary to CSV |

---

## Success criteria

| Condition | Acceptable | Action if exceeded |
|-----------|-----------|-------------------|
| p50 overhead (mock) | ≤ 5 ms | Profile ASGI / routing layer |
| p50 overhead (SAM2/CLIPSeg) | ≤ 10 ms | Profile JSON ser/deser or response builder |
| p99 overhead | ≤ 3× p50 overhead | Investigate GC or event-loop contention |

_These thresholds are guidelines for the mock backend running on local hardware.
Real-model benchmarks dominate with model inference time; overhead fractions
will naturally be much lower._

---

## Interpreting results

**Overhead < 5 ms**: The service layer is negligible — model inference dominates.

**Overhead 5–20 ms**: Acceptable for most production workloads. Review if
latency is a hard SLA requirement.

**Overhead > 20 ms**: Investigate JSON serialisation (`model_dump()` on large
mask arrays), Pydantic validation, or ASGI middleware.

**High stdev**: Indicates GC pressure, OS scheduling jitter, or shared
resources. Increase `--n`, isolate the machine, or run in a container.

---

## Implementation notes

- `DirectRunner.run()` wraps `asyncio.run(adapter.segment(request))` — this
  includes event-loop creation overhead (~0.1 ms on CPython 3.11+) but no
  network I/O.
- Warm-up iterations prime the model (lazy loading) and OS-level TCP state.
  Always use `--warmup ≥ 3` for real models.
- `DirectRunner.for_sam2()` / `for_clipseg()` set env vars and call
  `get_settings.cache_clear()` so the adapter uses the correct configuration.
- CSV output (`--csv`) writes one row per iteration for statistical analysis
  and a companion `*_summary.csv` with aggregated metrics.
