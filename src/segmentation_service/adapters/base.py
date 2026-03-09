"""Abstract base class that every segmentation backend must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

from segmentation_service.schemas.capabilities import CapabilitiesResponse
from segmentation_service.schemas.segment import SegmentRequest, SegmentResponse


class BaseSegmentationAdapter(ABC):
    """Contract that all segmentation backends must fulfil.

    A concrete adapter is responsible for:
    - Loading / initialising the model (in __init__ or a lazy property).
    - Running inference given a SegmentRequest.
    - Reporting its own capabilities via capabilities().

    Adding a new backend is as simple as:
    1. Create  src/segmentation_service/adapters/my_backend.py
    2. Subclass BaseSegmentationAdapter and implement the two abstract methods.
    3. Register the class in registry.py.
    """

    # Subclasses should override this with a human-readable identifier.
    name: str = "base"

    @abstractmethod
    async def segment(self, request: SegmentRequest) -> SegmentResponse:
        """Run segmentation inference and return a SegmentResponse."""
        ...

    @abstractmethod
    def capabilities(self) -> CapabilitiesResponse:
        """Return a description of what this backend supports."""
        ...
