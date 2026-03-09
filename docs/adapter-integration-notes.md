# Adapter Integration Notes

This document records exactly which files were touched to integrate each
backend adapter, and separates adapter-specific changes from framework/core
changes.  It is intended to help maintainers evaluate the pluggability of the
architecture and to serve as a guide for adding future adapters.

---

## SAM2 adapter (commit: `feat(sam2): integrate SAM2 as a selectable backend`)

### Adapter-specific files (new)

| File | Purpose |
|------|---------|
| `src/segmentation_service/adapters/sam2_adapter.py` | SAM2SegmentationAdapter — lazy model loading, point/box prompt handling, mask + logit encoding |
| `tests/test_sam2_adapter.py` | 29 unit tests for the adapter in isolation (mocked predictor) |
| `tests/test_sam2_endpoint.py` | 16 HTTP-level integration tests via patched `get_adapter` |

### Framework / core files changed

| File | Change |
|------|--------|
| `src/segmentation_service/adapters/registry.py` | Added `SAM2SegmentationAdapter` import and `Backend.sam2 → SAM2SegmentationAdapter` registry entry |
| `src/segmentation_service/adapters/__init__.py` | Exported `SAM2SegmentationAdapter` in `__all__` |
| `pyproject.toml` | Added `[sam2]` optional-dependency group (numpy, Pillow, httpx); added numpy + Pillow to `[dev]` for test imports |
| `README.md` | Added SAM2 setup section, env-var table, and supported prompt-type table |

### Files NOT touched

- `src/segmentation_service/config.py` — `Backend.sam2` enum value and `sam2_checkpoint` / `sam2_config` settings were pre-existing scaffolding
- All API route modules (`api/v1/health.py`, `api/v1/capabilities.py`, `api/v1/segment.py`)
- All existing schema modules
- Existing mock adapter and its tests

---

## CLIPSeg adapter (commit: `feat(clipseg): integrate CLIPSeg text-guided segmentation backend`)

### Adapter-specific files (new)

| File | Purpose |
|------|---------|
| `src/segmentation_service/adapters/clipseg_adapter.py` | CLIPSegSegmentationAdapter — lazy HuggingFace model loading, text prompt validation, logit→mask conversion, logit encoding |
| `tests/test_clipseg_adapter.py` | 27 unit tests for the adapter in isolation (mocked _infer) |
| `tests/test_clipseg_endpoint.py` | 17 HTTP-level integration tests via patched `get_adapter` |

### Framework / core files changed

| File | Change |
|------|--------|
| `src/segmentation_service/config.py` | Added `Backend.clipseg` enum value; added `clipseg_model` setting (default `CIDAS/clipseg-rd64-refined`) |
| `src/segmentation_service/adapters/registry.py` | Added `CLIPSegSegmentationAdapter` import and `Backend.clipseg → CLIPSegSegmentationAdapter` entry |
| `src/segmentation_service/adapters/__init__.py` | Exported `CLIPSegSegmentationAdapter` in `__all__` |
| `pyproject.toml` | Added `[clipseg]` optional-dependency group (transformers, numpy, Pillow, httpx) |
| `README.md` | Added CLIPSeg setup section, env-var entry, prompt-type table, and example curl command |
| `docs/adapter-integration-notes.md` | This file (new) |

### Files NOT touched

- All API route modules (`api/v1/health.py`, `api/v1/capabilities.py`, `api/v1/segment.py`)
- All existing schema modules (`schemas/segment.py`, `schemas/capabilities.py`, `schemas/health.py`)
- SAM2 adapter and its tests
- Mock adapter and its tests
- `src/segmentation_service/main.py`
- `src/segmentation_service/logging_config.py`

---

## Architecture observations

### What worked well

- **Zero API-layer changes across both integrations.** The three route handlers
  (`health`, `capabilities`, `segment`) are genuinely backend-agnostic.
  Adding a new backend touches no file under `api/`.

- **Config is the single source of truth.** Backend selection, model paths,
  and device are all driven by `Settings` in `config.py`. No adapter reads
  environment variables directly.

- **Lazy loading is the correct default.** Both SAM2 and CLIPSeg load their
  models on the first `segment()` call, allowing the service to start and pass
  `/health` without any weights on disk.

- **Tests require no real ML libraries.** Both adapters are fully testable by
  injecting mocks at the `_predictor` / `_model` / `_infer` level. Neither
  `sam2` nor `transformers` need to be installed to run the test suite.

### What required touching core files

- `config.py`: One enum value and one settings field per new backend.
- `registry.py`: One import and one dict entry per new backend.
- `adapters/__init__.py`: One import and one `__all__` entry per new backend.
- `pyproject.toml`: One optional-dependency group per new backend.
- `README.md`: Documentation.

### Checklist for the next backend

1. Create `src/segmentation_service/adapters/<name>_adapter.py` and subclass
   `BaseSegmentationAdapter` (implement `capabilities()` and `segment()`).
2. Add `<name>` to the `Backend` enum in `config.py`.
3. Add any backend-specific settings to the `Settings` class in `config.py`.
4. Import the new adapter in `registry.py` and add it to `_REGISTRY`.
5. Export it from `adapters/__init__.py`.
6. Add an optional-dependency group in `pyproject.toml`.
7. Write unit tests (`tests/test_<name>_adapter.py`) and HTTP tests
   (`tests/test_<name>_endpoint.py`), following the SAM2 / CLIPSeg patterns.
8. Update `README.md` and this file.
