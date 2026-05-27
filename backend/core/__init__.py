"""Core detector and classifier contracts."""

from .base_classifier import BaseClassifier
from .base_detector import BaseDetector, Detection

__all__ = ["BaseClassifier", "BaseDetector", "Detection"]

