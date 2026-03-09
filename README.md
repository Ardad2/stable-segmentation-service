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
│       │   └── v1/
│       │       ├── router.py        # Aggregates v1 endpoints
│       │       ├── health.py        # GET  /api/v1/health
│       │       ├── capabilities.py  # GET  /api/v1/capabilities
│       │       └── segment.py       # POST /api/v1/segment
│       ├── schemas/
│       │   ├── health.py            # HealthResponse
│       │   ├── capabilities.py      # CapabilitiesResponse
│       │   └── segment.py           # SegmentRequest / SegmentResponse / MaskResult
│       └── adapters/
│           ├── base.py              # BaseSegmentationAdapter (ABC)
│           ├── mock_adapter.py      # Stub adapter — no GPU required
│           ├── sam2_adapter.py      # SAM2 (Meta) backend
│           └── registry.py          # Maps Backend enum → adapter class
├── tests/
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_capabilities.py
│   ├── test_segment.py
│   ├── test_sam2_adapter.py         # SAM2 adapter unit tests (mocked predictor)
│   └── test_sam2_endpoint.py        # SAM2 HTTP-level tests (mocked predictor)
├── benchmark/
│   ├── latency.py                   # Serial latency measurements
│   └── throughput.py                # Concurrent RPS measurement
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

All routes are prefixed with `/api/v1`.

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/health` | Service liveness check |
| `GET`  | `/api/v1/capabilities` | Active backend's supported features |
| `POST` | `/api/v1/segment` | Run segmentation inference |

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
| `SEGMENTATION_BACKEND` | `mock` | `mock` / `sam2` / `custom` |
| `MODEL_DEVICE` | `cpu` | `cpu` / `cuda` / `mps` |
| `SAM2_CHECKPOINT` | _(empty)_ | **Required for sam2.** Filesystem path to a SAM2 `.pt` weights file (e.g. `weights/sam2_hiera_large.pt`) |
| `SAM2_CONFIG` | _(empty)_ | **Required for sam2.** SAM2 YAML config name without path prefix (e.g. `sam2_hiera_l.yaml`) |
