"""Segmentation backend adapters."""

from segmentation_service.adapters.base import BaseSegmentationAdapter
from segmentation_service.adapters.clipseg_adapter import CLIPSegSegmentationAdapter
from segmentation_service.adapters.mock_adapter import MockSegmentationAdapter
from segmentation_service.adapters.registry import get_adapter
from segmentation_service.adapters.sam2_adapter import SAM2SegmentationAdapter

__all__ = [
    "BaseSegmentationAdapter",
    "CLIPSegSegmentationAdapter",
    "MockSegmentationAdapter",
    "SAM2SegmentationAdapter",
    "get_adapter",
]
