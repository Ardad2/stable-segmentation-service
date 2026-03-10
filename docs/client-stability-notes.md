# Client Stability Notes

This document describes which files make up the CLI client, which files must
remain stable across backend swaps, and how to measure client code changes
when evaluating pluggability.

---

## Client files created

| File | Purpose |
|------|---------|
| `src/segmentation_service/client/__init__.py` | Package marker (empty) |
| `src/segmentation_service/client/cli.py` | All client logic: `SegmentationClient`, `select_prompt`, `main()` |

---

## Files that must NOT change when swapping backends

These files contain zero backend-specific knowledge.  A maintainer reviewing a
backend swap should be able to verify this by diffing the branch against `main`:

| File | Why it must stay unchanged |
|------|---------------------------|
| `src/segmentation_service/client/cli.py` | Client depends only on `/capabilities` and `/segment` contracts |
| `src/segmentation_service/api/v1/health.py` | Route is backend-agnostic |
| `src/segmentation_service/api/v1/capabilities.py` | Route is backend-agnostic |
| `src/segmentation_service/api/v1/segment.py` | Route is backend-agnostic |
| `src/segmentation_service/schemas/segment.py` | Schema is backend-agnostic |
| `src/segmentation_service/schemas/capabilities.py` | Schema is backend-agnostic |
| `src/segmentation_service/schemas/health.py` | Schema is backend-agnostic |
| `tests/test_compatibility.py` | Tests the contract, not an implementation |
| `tests/test_client.py` | Tests client logic in isolation; no adapter imports |

---

## How to measure "client LOC changed"

When adding a new backend, run:

```bash
git diff main -- src/segmentation_service/client/
```

For a correctly implemented adapter, this diff should be **empty**.  The client
is only allowed to change if the *stable API contract itself* changes (e.g. a
new field is added to `CapabilitiesResponse`).

To count changed lines automatically:

```bash
git diff --stat main -- src/segmentation_service/client/
```

A clean integration will show `0 insertions(+), 0 deletions(-)` for all client
files.

---

## How to measure "API surface changed"

```bash
git diff main -- \
  src/segmentation_service/api/ \
  src/segmentation_service/schemas/
```

Again, this diff should be empty for a backend-only change.

---

## Procedure for comparing backend integration diffs

Use this procedure to evaluate the effort and scope of adding the next backend:

1. **Identify the adapter branch** (e.g. `feature/my-new-backend`).

2. **Compute the diff against the integration point**:
   ```bash
   git diff main..feature/my-new-backend --stat
   ```

3. **Separate adapter-specific from framework files**:
   ```bash
   # Framework files (should be minimal):
   git diff main..feature/my-new-backend -- \
       src/segmentation_service/config.py \
       src/segmentation_service/adapters/registry.py \
       src/segmentation_service/adapters/__init__.py \
       pyproject.toml README.md docs/

   # Adapter-specific files (all new, not modifications):
   git diff main..feature/my-new-backend -- \
       src/segmentation_service/adapters/<name>_adapter.py \
       tests/test_<name>_adapter.py \
       tests/test_<name>_endpoint.py
   ```

4. **Check client is unmodified**:
   ```bash
   git diff main..feature/my-new-backend -- src/segmentation_service/client/
   # Expected: (empty output)
   ```

5. **Run the compatibility tests** to confirm behavioural alignment:
   ```bash
   pytest tests/test_compatibility.py -v
   ```

6. **Run the evaluation script** against the new backend to capture live
   observed results:
   ```bash
   SEGMENTATION_BACKEND=<name> uvicorn segmentation_service.main:app &
   python scripts/evaluate_compatibility.py --url http://localhost:8000
   ```

---

## Design principles preserved by the client

| Principle | Evidence |
|-----------|---------|
| Zero backend-specific imports | `cli.py` imports only `httpx` and stdlib |
| Capability-driven prompt selection | `select_prompt()` reads `/capabilities` at runtime |
| Graceful unsupported-prompt handling | `select_prompt()` raises `ValueError` with a clear message |
| Uniform response handling | `SegmentationClient.segment()` parses the same `SegmentResponse` schema regardless of backend |
| No hardcoded backend names | The client never branches on `backend == "sam2"` etc. |

---

## Test coverage for client stability

`tests/test_client.py` (27 tests) exercises:
- `select_prompt` — 14 tests covering all priority rules, error paths, and
  synthetic fallback modes.
- `SegmentationClient` — tests that HTTP calls target the correct paths and
  that responses are parsed without backend knowledge.
- `main()` — end-to-end CLI tests with fully mocked HTTP, covering success
  paths for all three backends, unsupported prompt errors, `--json` output,
  and `--output-dir` mask saving.

`tests/test_compatibility.py` (23 tests) exercises:
- Capabilities ↔ segment alignment for mock, SAM2, and CLIPSeg.
- Known support matrix (point/box/text) for each backend.
- `select_prompt()` against real capabilities responses.
- Response schema is backend-agnostic.
- All API paths are stable.
