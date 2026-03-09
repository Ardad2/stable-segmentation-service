"""Segmentation backend adapters."""

from segmentation_service.adapters.base import BaseSegmentationAdapter
from segmentation_service.adapters.mock_adapter import MockSegmentationAdapter
from segmentation_service.adapters.registry import get_adapter

__all__ = [
    "BaseSegmentationAdapter",
    "MockSegmentationAdapter",
    "get_adapter",
]
