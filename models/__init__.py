"""Model entry points."""

from .carl_d_detector import (
    RESNET50_ARCHITECTURE,
    SUPPORTED_ARCHITECTURES,
    CARLDDetector,
    create_carl_d_detector,
)

__all__ = [
    "CARLDDetector",
    "RESNET50_ARCHITECTURE",
    "SUPPORTED_ARCHITECTURES",
    "create_carl_d_detector",
]
