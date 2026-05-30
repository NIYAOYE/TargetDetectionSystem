"""Tests for YOLO detector coordinate normalization."""

import numpy as np

from backend.algorithms.yolo_detector import YOLODetector


class TestYOLODetectorBoxClipping:
    def test_clips_box_to_image_bounds(self):
        xyxy = np.array([-5.0, 10.0, 120.0, 90.0])
        assert YOLODetector._clip_xyxy(xyxy, image_width=100, image_height=80) == (0.0, 10.0, 100.0, 80.0)

    def test_collapses_inverted_box_at_clipped_origin(self):
        xyxy = np.array([30.0, 40.0, 20.0, 10.0])
        assert YOLODetector._clip_xyxy(xyxy, image_width=100, image_height=80) == (30.0, 40.0, 30.0, 40.0)
