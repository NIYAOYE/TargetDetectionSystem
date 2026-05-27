"""Algorithm layer used by the browser/server application."""

from .feature_extractor import (
    extract_sar_features,
    extract_scatter_features_from_bbox,
    extract_scatter_features_from_patch,
)
from .image_processing import CoordinateMapper, ImageSlicer

__all__ = [
    "CoordinateMapper",
    "ImageSlicer",
    "RFClassifier",
    "YOLODetector",
    "extract_sar_features",
    "extract_scatter_features_from_bbox",
    "extract_scatter_features_from_patch",
]


def __getattr__(name):
    if name == "YOLODetector":
        from .yolo_detector import YOLODetector

        return YOLODetector
    if name == "RFClassifier":
        from .rf_classifier import RFClassifier

        return RFClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

