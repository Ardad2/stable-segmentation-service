"""Adapter registry — maps Backend enum values to concrete adapter classes.

To add a new backend:
1. Create your adapter module (e.g. my_backend_adapter.py).
2. Import it here and add an entry to _REGISTRY.
"""

from __future__ import annotations

from functools import lru_cache

from segmentation_service.adapters.base import BaseSegmentationAdapter
from segmentation_service.adapters.clipseg_adapter import CLIPSegSegmentationAdapter
from segmentation_service.adapters.mock_adapter import MockSegmentationAdapter
from segmentation_service.adapters.sam2_adapter import SAM2SegmentationAdapter
from segmentation_service.config import Backend, get_settings
from segmentation_service.logging_config import LogContext, get_logger

log = LogContext(get_logger(__name__))

# Registry: Backend enum value -> adapter class.
# To add a new backend, create its module and add an entry here.
_REGISTRY: dict[Backend, type[BaseSegmentationAdapter]] = {
    Backend.mock: MockSegmentationAdapter,
    Backend.sam2: SAM2SegmentationAdapter,
    Backend.clipseg: CLIPSegSegmentationAdapter,
    # Backend.custom: CustomSegmentationAdapter,
}


@lru_cache(maxsize=1)
def get_adapter() -> BaseSegmentationAdapter:
    """Instantiate and cache the adapter selected by config."""
    settings = get_settings()
    backend = settings.segmentation_backend

    adapter_cls = _REGISTRY.get(backend)
    if adapter_cls is None:
        raise RuntimeError(
            f"No adapter registered for backend '{backend}'. "
            f"Available: {list(_REGISTRY)}"
        )

    log.info("Loading segmentation adapter", backend=backend.value)
    return adapter_cls()
