# stable-segmentation-service

A modular, production-ready FastAPI service for image segmentation inference.
Designed for easy backend swapping (mock → SAM-2 → custom model) without touching the API layer.

---

## Directory structure

```
stable-segmentation-service/
├── src/
│   └── segmentation_service/
│       ├── __init__.py
│       ├── main.py                  # FastAPI app factory + uvicorn entrypoint
│       ├── config.py                # Pydantic-Settings config (env / .env)
│       ├── logging_config.py        # Structured logging helpers
│       ├── api/
│       │   ├── router.py            # Mounts all versioned routers
│       │   ├── v1/
│       │   │   ├── router.py        # Aggregates v1 endpoints
│       │   │   ├── health.py        # GET  /api/v1/health
│       │   │   ├── capabilities.py  # GET  /api/v1/capabilities
│       │   │   └── segment.py       # POST /api/v1/segment
│       │   └── v2/
│       │       ├── router.py        # Aggregates v2 endpoints
│       │       ├── health.py        # GET  /api/v2/health
│       │       ├── capabilities.py  # GET  /api/v2/capabilities
│       │       └── segment.py       # POST /api/v2/segment (breaking: prompt envelope)
│       ├── schemas/
│       │   ├── health.py            # HealthResponse
│       │   ├── capabilities.py      # CapabilitiesResponse
│       │   ├── segment.py           # SegmentRequest / SegmentResponse / MaskResult (v1)
│       │   └── v2segment.py         # V2SegmentRequest / V2SegmentResponse (v2)
│       └── adapters/
│           ├── base.py              # BaseSegmentationAdapter (ABC)
│           ├── mock_adapter.py      # Stub adapter — no GPU required
│           ├── sam2_adapter.py      # SAM2 (Meta) — point/box prompts
│           ├── clipseg_adapter.py   # CLIPSeg (HuggingFace) — text prompts
│           └── registry.py          # Maps Backend enum → adapter class
│       ├── client/
│       │   └── cli.py               # seg-client: backend-agnostic CLI client
│       └── eval/
│           ├── correctness.py       # IoU, pixel agreement, coverage utilities
│           └── direct_runners.py    # Direct (non-HTTP) adapter invocation helpers
├── tests/
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_capabilities.py
│   ├── test_segment.py
│   ├── test_sam2_adapter.py         # SAM2 adapter unit tests (mocked predictor)
│   ├── test_sam2_endpoint.py        # SAM2 HTTP-level tests (mocked predictor)
│   ├── test_clipseg_adapter.py      # CLIPSeg adapter unit tests (mocked model)
│   ├── test_clipseg_endpoint.py     # CLIPSeg HTTP-level tests (mocked model)
│   ├── test_client.py               # CLI client unit tests (mocked HTTP)
│   ├── test_compatibility.py        # Cross-backend capability/behavior alignment
│   ├── test_eval_utils.py           # Unit tests for correctness comparison utilities
│   └── test_versioning.py           # API version × client/server compatibility matrix
├── scripts/
│   ├── evaluate_compatibility.py    # Probe a live server; emit compatibility report
│   ├── evaluate_correctness.py      # Compare direct vs served outputs; emit report
│   └── evaluate_versioning.py       # Client/server version-pair compatibility matrix
├── eval_assets/
│   ├── images/                      # Reproducible 16×16 PNG test images (seed=42)
│   └── requests/                    # Pre-built JSON request payloads
├── docs/
│   ├── adapter-integration-notes.md      # Per-adapter file change log
│   ├── compatibility-matrix.md           # Expected + observed backend×prompt matrix
│   ├── client-stability-notes.md         # Which files must stay stable across swaps
│   ├── performance-evaluation-plan.md    # Direct-vs-served latency benchmark guide
│   ├── correctness-evaluation-plan.md    # Output correctness evaluation guide
│   └── versioning-compatibility-plan.md  # API versioning experiment design + matrix
├── benchmark/
│   ├── latency.py                   # Serial latency measurements
│   ├── throughput.py                # Concurrent RPS measurement
│   └── direct_vs_served.py          # Direct vs HTTP service overhead comparison
├── .env.example
├── .gitignore
└── pyproject.toml
```

---

## Requirements

- Python 3.10+
- [hatch](https://hatch.pypa.io/) (recommended) **or** pip

---

## Setup

### 1 — Clone and create environment

```bash
git clone <repo-url>
cd stable-segmentation-service

# with hatch (manages its own venv automatically)
pip install hatch
hatch env create

# — OR — plain pip / venv
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2 — Configure environment

```bash
cp .env.example .env
# Edit .env as needed (defaults work out-of-the-box with the mock backend)
```

### 3 — Run the server

```bash
# hatch
hatch run serve

# plain uvicorn
uvicorn segmentation_service.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs are available at http://localhost:8000/docs (development mode only).

---

## API endpoints

Two API versions co-exist under the same `/api` prefix.

**v1 routes**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/health` | Service liveness check |
| `GET`  | `/api/v1/capabilities` | Active backend's supported features |
| `POST` | `/api/v1/segment` | Run segmentation inference (flat prompt fields) |

**v2 routes**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v2/health` | Service liveness check |
| `GET`  | `/api/v2/capabilities` | Active backend's supported features |
| `POST` | `/api/v2/segment` | Run segmentation inference (nested prompt envelope) |

All responses carry an `api_version` field (`"1.0"` or `"2.0"`).

See the [API versioning section](#api-versioning) for the full compatibility
matrix and an explanation of the breaking change.

### POST /api/v1/segment — example

```bash
curl -s -X POST http://localhost:8000/api/v1/segment \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64-encoded-png>",
    "image_format": "png",
    "prompt_type": "point",
    "points": [{"x": 320, "y": 240, "label": 1}]
  }' | python -m json.tool
```

**Prompt types**

| `prompt_type` | Required field |
|---------------|---------------|
| `point` | `points` list (x, y, label) |
| `box`   | `box` object (x_min, y_min, x_max, y_max) |
| `text`  | `text_prompt` string |

---

## SAM2 backend

### Prerequisites

1. **Install the SAM2 library** (not on PyPI — install from source):

```bash
pip install 'git+https://github.com/facebookresearch/sam2.git'
```

2. **Install the service with SAM2 extras** (numpy, Pillow, httpx):

```bash
pip install -e ".[sam2]"
```

3. **Download model weights** from the
   [SAM2 releases page](https://github.com/facebookresearch/sam2/releases) and
   place them somewhere accessible (e.g. `weights/`).

### Configuration

Set the following environment variables (or add them to `.env`):

```bash
SEGMENTATION_BACKEND=sam2
SAM2_CHECKPOINT=weights/sam2_hiera_large.pt   # path to the downloaded .pt file
SAM2_CONFIG=sam2_hiera_l.yaml                 # SAM2 YAML config name (no path prefix)
MODEL_DEVICE=cuda                             # cpu | cuda | mps
```

Available config names and their weight files:

| Config | Weights file |
|--------|-------------|
| `sam2_hiera_t.yaml` | `sam2_hiera_tiny.pt` |
| `sam2_hiera_s.yaml` | `sam2_hiera_small.pt` |
| `sam2_hiera_b+.yaml` | `sam2_hiera_base_plus.pt` |
| `sam2_hiera_l.yaml` | `sam2_hiera_large.pt` |

### Supported prompt types

| `prompt_type` | Supported | Notes |
|---------------|-----------|-------|
| `point` | ✅ | Multiple (x, y, label) coordinates; supports `multimask_output=true` |
| `box` | ✅ | Single axis-aligned bounding box; `multimask_output` is ignored |
| `text` | ❌ | Not supported by SAM2 — check `/api/v1/capabilities` before sending |

### Running with SAM2

```bash
# Copy and edit .env
cp .env.example .env
# Set SEGMENTATION_BACKEND=sam2 and the SAM2_* vars in .env

uvicorn segmentation_service.main:app --reload --host 0.0.0.0 --port 8000
```

Verify the active backend:

```bash
curl http://localhost:8000/api/v1/capabilities | python -m json.tool
```

---

## CLIPSeg backend

CLIPSeg is a text-guided segmentation model that complements SAM2: where SAM2
requires geometric prompts (points / boxes), CLIPSeg accepts a free-text
description of the region to segment.

### Prerequisites

1. **Install PyTorch** (version matching your CUDA toolkit if using GPU):
   See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for
   the correct install command for your platform.

2. **Install the CLIPSeg extras** (transformers, numpy, Pillow, httpx):

```bash
pip install -e ".[clipseg]"
```

### Configuration

Set the following environment variables (or add them to `.env`):

```bash
SEGMENTATION_BACKEND=clipseg
CLIPSEG_MODEL=CIDAS/clipseg-rd64-refined   # HuggingFace model ID or local path
MODEL_DEVICE=cpu                           # cpu | cuda | mps
```

The default model (`CIDAS/clipseg-rd64-refined`) is downloaded automatically
from HuggingFace on the first request. To use a locally cached copy, set
`CLIPSEG_MODEL` to the directory path where the model was saved.

### Supported prompt types

| `prompt_type` | Supported | Notes |
|---------------|-----------|-------|
| `text` | ✅ | Natural-language description of the region (e.g. `"the cat"`) |
| `point` | ❌ | Not supported by CLIPSeg |
| `box` | ❌ | Not supported by CLIPSeg |

### Example request

```bash
curl -s -X POST http://localhost:8000/api/v1/segment \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64-encoded-png>",
    "image_format": "png",
    "prompt_type": "text",
    "text_prompt": "the wooden chair"
  }' | python -m json.tool
```

The response includes a single mask covering the region that best matches the
text description, together with a confidence score (peak sigmoid activation).

### Running with CLIPSeg

```bash
cp .env.example .env
# Set SEGMENTATION_BACKEND=clipseg (and optionally CLIPSEG_MODEL) in .env

uvicorn segmentation_service.main:app --reload --host 0.0.0.0 --port 8000
```

Verify the active backend:

```bash
curl http://localhost:8000/api/v1/capabilities | python -m json.tool
```

---

## CLI client

A small command-line client is included.  It calls `/api/v1/capabilities` first
and selects the appropriate prompt type at runtime — the exact same invocation
works unchanged with mock, SAM2, and CLIPSeg backends.

### Installation

```bash
pip install -e ".[dev]"   # httpx is included in dev extras
```

`seg-client` is registered as a console entry point and is available immediately
after installation.

### Prompt selection

The client picks the first compatible prompt in priority order: **text → point → box**.

- If you supply `--text-prompt` and the backend supports `text`, a text request is sent.
- If you supply `--point` and the backend supports `point`, a point request is sent.
- If a supplied prompt is not supported, the client exits with a clear error message.
- If no prompt is supplied, a harmless synthetic placeholder is chosen automatically (smoke-test mode).

### Examples

```bash
# CLIPSeg backend — text prompt selected automatically
seg-client --base-url http://localhost:8000 --image img.png --text-prompt "the cat"

# SAM2 or mock backend — point prompt
seg-client --base-url http://localhost:8000 --image img.png --point "320,240,1"

# Supply both; client picks whichever the backend supports
seg-client --base-url http://localhost:8000 --image img.png \
    --text-prompt "the cat" --point "320,240,1"

# Smoke-test mode (no image or prompt — uses synthetic defaults)
seg-client --base-url http://localhost:8000

# Save masks to disk
seg-client --base-url http://localhost:8000 --image img.png \
    --text-prompt "wooden chair" --output-dir ./out --return-logits

# Full JSON response
seg-client --base-url http://localhost:8000 --json
```

Sample output:

```
Backend:   clipseg
Supported: text
Prompt:    text — "the cat"
Sending request…
Masks:     1
  [0] score=0.9300  area=18432 px
Latency:   41.2 ms
RequestID: a1b2c3d4-…
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--base-url URL` | `http://localhost:8000` | Service base URL |
| `--image PATH` | _(1×1 smoke PNG)_ | Image file to segment |
| `--text-prompt TEXT` | _(none)_ | Natural-language prompt |
| `--point X,Y,LABEL` | _(none)_ | Point prompt e.g. `"320,240,1"` |
| `--box XMIN,YMIN,XMAX,YMAX` | _(none)_ | Box prompt |
| `--output-dir DIR` | _(none)_ | Save masks as PNGs here |
| `--return-logits` | false | Include raw logit maps |
| `--json` | false | Print full JSON response |

---

## Compatibility evaluation

### Automated tests (no server required)

`tests/test_compatibility.py` proves pluggability in CI without a running server:

```bash
pytest tests/test_compatibility.py -v
```

Key invariants enforced for mock, SAM2, and CLIPSeg:
- `/capabilities` accurately reflects what `/segment` accepts.
- `/segment` accepts every prompt type listed in capabilities.
- `/segment` returns 4xx/5xx for every prompt type NOT in capabilities.
- `select_prompt()` adapts correctly to each backend's capabilities.
- `SegmentResponse` schema is identical across all backends.

### Live evaluation script

`scripts/evaluate_compatibility.py` probes a **running** service and emits a
Markdown or CSV report:

```bash
# Start the server (any backend)
SEGMENTATION_BACKEND=mock uvicorn segmentation_service.main:app --port 8000

# In another terminal:
python scripts/evaluate_compatibility.py --url http://localhost:8000

# CSV
python scripts/evaluate_compatibility.py --url http://localhost:8000 --format csv

# Save to file
python scripts/evaluate_compatibility.py --url http://localhost:8000 \
    --output docs/compatibility-matrix.md
```

The script probes all three prompt types and reports whether each observed
result aligns with the backend's advertised capabilities.

See `docs/compatibility-matrix.md` for the pre-seeded expected matrix and
sample output for each backend.

---

## API versioning

Two API major versions coexist.  The versioning experiment demonstrates:

1. **Backward-compatible v1 evolution** — an additive `api_version` field was
   added to all v1 response schemas.  Old clients that ignore unknown JSON keys
   are unaffected.

2. **Breaking v2 change** — `/api/v2/segment` replaces the five flat prompt
   fields (`prompt_type`, `points`, `box`, `text_prompt`) with a single
   required nested `prompt` envelope object.  Additionally, the per-mask field
   `mask_b64` is renamed to `mask_data`.

### API version vs model/backend version

| Concept | Response field | Example |
|---------|---------------|---------|
| API contract | `api_version` | `"1.0"` / `"2.0"` |
| Service code | `version` (health only) | `"0.1.0"` |
| Active backend | `backend` | `"mock"` / `"sam2"` / `"clipseg"` |

Swapping the backend does **not** change `api_version`.

### v2 request format

```bash
# v1 — flat fields
curl -s -X POST http://localhost:8000/api/v1/segment \
  -H "Content-Type: application/json" \
  -d '{"image":"...","prompt_type":"point","points":[{"x":10,"y":20,"label":1}]}'

# v2 — nested prompt envelope (breaking change)
curl -s -X POST http://localhost:8000/api/v2/segment \
  -H "Content-Type: application/json" \
  -d '{"image":"...","prompt":{"type":"point","points":[{"x":10,"y":20,"label":1}]}}'
```

A v1 payload sent to `/api/v2/segment` returns **HTTP 422** (missing `prompt`
field).  A v2 payload sent to `/api/v1/segment` also returns **HTTP 422**
(prompt_type='point' with no points — the `prompt` field is an unknown extra
that Pydantic ignores).

### Compatibility matrix (automated)

```bash
pytest tests/test_versioning.py -v
```

### Live versioning evaluation

```bash
# Start the service
SEGMENTATION_BACKEND=mock uvicorn segmentation_service.main:app --port 8000

python scripts/evaluate_versioning.py --url http://localhost:8000

# CSV output
python scripts/evaluate_versioning.py --url http://localhost:8000 --format csv
```

See `docs/versioning-compatibility-plan.md` for the full compatibility matrix,
pass/fail criteria, and design rationale.

---

## Running tests

```bash
# hatch
hatch run test

# plain pytest
pytest
```

---

## Benchmarks

The service must be running before executing benchmark scripts.

```bash
# Serial latency (50 requests)
python benchmark/latency.py --url http://localhost:8000 --n 50

# Concurrent throughput (8 workers, 10 seconds)
python benchmark/throughput.py --url http://localhost:8000 --concurrency 8 --duration 10
```

Install benchmark extras if needed:

```bash
pip install -e ".[benchmark]"
```

### Direct-vs-served overhead

`benchmark/direct_vs_served.py` compares the latency of calling the adapter
directly (no HTTP) against calling the served endpoint. This isolates the cost
of the service layer (ASGI routing, JSON serialisation, TCP stack).

```bash
# Mock backend — no GPU required (start the service first)
python benchmark/direct_vs_served.py --backend mock \
    --url http://localhost:8000 --n 20

# SAM2 backend
python benchmark/direct_vs_served.py --backend sam2 \
    --url http://localhost:8000 \
    --sam2-checkpoint weights/sam2_hiera_large.pt \
    --sam2-config sam2_hiera_l.yaml \
    --n 10 --warmup 3

# Export CSV
python benchmark/direct_vs_served.py --backend mock \
    --url http://localhost:8000 --n 20 --csv results.csv
```

See `docs/performance-evaluation-plan.md` for metrics, success criteria, and
how to interpret results.

---

## Correctness evaluation

`scripts/evaluate_correctness.py` checks that the service layer preserves model
outputs exactly: the masks from a direct adapter call must be identical to those
returned by the served endpoint (IoU = 1.0, pixel agreement = 1.0).

```bash
# Mock backend (no GPU required)
python scripts/evaluate_correctness.py --backend mock \
    --url http://localhost:8000

# SAM2 backend
python scripts/evaluate_correctness.py --backend sam2 \
    --url http://localhost:8000 \
    --sam2-checkpoint weights/sam2_hiera_large.pt \
    --sam2-config sam2_hiera_l.yaml

# CLIPSeg backend
python scripts/evaluate_correctness.py --backend clipseg \
    --url http://localhost:8000

# Save Markdown report
python scripts/evaluate_correctness.py --backend mock \
    --url http://localhost:8000 --output report.md

# CSV output
python scripts/evaluate_correctness.py --backend mock \
    --url http://localhost:8000 --format csv
```

The script exits `0` if all probes pass, `1` if any fail.

See `docs/correctness-evaluation-plan.md` for metrics, pass/fail criteria, and
report format details.

Unit tests for the comparison utilities are in `tests/test_eval_utils.py` and
run without a running server or GPU:

```bash
pytest tests/test_eval_utils.py -v
```

---

## Adding a new backend

1. Create `src/segmentation_service/adapters/my_backend.py` and subclass `BaseSegmentationAdapter`:

```python
from segmentation_service.adapters.base import BaseSegmentationAdapter
from segmentation_service.schemas.capabilities import CapabilitiesResponse
from segmentation_service.schemas.segment import SegmentRequest, SegmentResponse

class MyBackendAdapter(BaseSegmentationAdapter):
    name = "my_backend"

    def capabilities(self) -> CapabilitiesResponse:
        ...

    async def segment(self, request: SegmentRequest) -> SegmentResponse:
        ...
```

2. Register it in `src/segmentation_service/adapters/registry.py`:

```python
from segmentation_service.adapters.my_backend import MyBackendAdapter

_REGISTRY = {
    Backend.mock: MockSegmentationAdapter,
    Backend.my_backend: MyBackendAdapter,   # add this line
}
```

3. Add `my_backend` to the `Backend` enum in `config.py`.

4. Set `SEGMENTATION_BACKEND=my_backend` in your `.env`.

---

## Adding an API v2

1. Create `src/segmentation_service/api/v2/` mirroring the `v1/` layout.
2. Add a `v2_router` in `v2/router.py`.
3. Mount it in `src/segmentation_service/api/router.py`:

```python
from segmentation_service.api.v2.router import v2_router
root_router.include_router(v2_router)
```

---

## Configuration reference

All settings can be set via environment variables or a `.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | `development` / `staging` / `production` |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `SEGMENTATION_BACKEND` | `mock` | `mock` / `sam2` / `clipseg` / `custom` |
| `MODEL_DEVICE` | `cpu` | `cpu` / `cuda` / `mps` |
| `SAM2_CHECKPOINT` | _(empty)_ | **Required for sam2.** Filesystem path to a SAM2 `.pt` weights file (e.g. `weights/sam2_hiera_large.pt`) |
| `SAM2_CONFIG` | _(empty)_ | **Required for sam2.** SAM2 YAML config name without path prefix (e.g. `sam2_hiera_l.yaml`) |
| `CLIPSEG_MODEL` | `CIDAS/clipseg-rd64-refined` | **Used by clipseg.** HuggingFace model ID or local path |
