"""Tests for backend/pipeline.py — detectors, NMS, coordinate clipping, device resolution."""

import pytest
import numpy as np
from backend.pipeline import (
    SyntheticDetector,
    clip_bbox,
    global_nms,
    resolve_device,
)
from backend.core.base_detector import Detection


def _make_detection(x, y, w, h, confidence, class_id=0, class_name="target"):
    return Detection(
        bbox=(float(x), float(y), float(w), float(h)),
        confidence=float(confidence),
        class_id=int(class_id),
        class_name=class_name,
    )


class TestSyntheticDetector:
    def test_returns_two_detections(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = SyntheticDetector().detect(img)
        assert len(detections) == 2

    def test_class_names(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = SyntheticDetector().detect(img)
        names = {d.class_name for d in detections}
        assert names == {"demo_target", "demo_ship"}

    def test_confidences_valid(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = SyntheticDetector().detect(img)
        for d in detections:
            assert 0.0 <= d.confidence <= 1.0

    def test_boxes_within_image(self):
        img = np.zeros((800, 600, 3), dtype=np.uint8)
        detections = SyntheticDetector().detect(img)
        for d in detections:
            x, y, w, h = d.bbox
            assert x >= 0 and y >= 0
            assert x + w <= 600  # width
            assert y + h <= 800  # height


class TestGlobalNMS:
    def test_empty_input(self):
        assert global_nms([], 0.5) == []

    def test_single_detection_survives(self):
        d = [_make_detection(0, 0, 100, 100, 0.9)]
        result = global_nms(d, 0.5)
        assert len(result) == 1

    def test_identical_boxes_merged(self):
        detections = [
            _make_detection(0, 0, 100, 100, 0.9),
            _make_detection(0, 0, 100, 100, 0.8),
        ]
        result = global_nms(detections, 0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.9  # Higher confidence survives

    def test_non_overlapping_both_keep(self):
        detections = [
            _make_detection(0, 0, 50, 50, 0.9),
            _make_detection(200, 200, 50, 50, 0.8),
        ]
        result = global_nms(detections, 0.5)
        assert len(result) == 2

    def test_partially_overlapping(self):
        detections = [
            _make_detection(0, 0, 100, 100, 0.9),
            _make_detection(30, 30, 100, 100, 0.7),
        ]
        # IOU = intersection/(union) ≈ (70*70)/(10000+10000-4900) ≈ 4900/15100 ≈ 0.32
        # With iou_threshold=0.5, 0.32 < 0.5 so both survive
        result = global_nms(detections, 0.5)
        assert len(result) == 2


class TestClipBBox:
    def test_fully_inside(self):
        result = clip_bbox((50, 50, 100, 100), 300, 300)
        assert result == (50, 50, 100, 100)

    def test_exceeds_right_edge(self):
        result = clip_bbox((250, 50, 100, 100), 300, 300)
        assert result == (250, 50, 50, 100)  # clipped w to 300-250=50

    def test_exceeds_bottom_edge(self):
        result = clip_bbox((50, 250, 100, 100), 300, 300)
        assert result == (50, 250, 100, 50)  # clipped h to 300-250=50

    def test_negative_coords(self):
        result = clip_bbox((-10, -10, 50, 50), 300, 300)
        assert result == (0, 0, 40, 40)  # x1 clamped to 0

    def test_zero_size_bbox(self):
        result = clip_bbox((0, 0, -1, -1), 300, 300)
        assert result[2] >= 0
        assert result[3] >= 0


class TestResolveDevice:
    def test_auto_resolves(self):
        device = resolve_device("auto")
        assert device in ("cpu", "cuda")

    def test_empty_string_resolves(self):
        device = resolve_device("")
        assert device in ("cpu", "cuda")

    def test_cpu_passthrough(self):
        assert resolve_device("cpu") == "cpu"

    def test_unknown_device_passthrough(self):
        assert resolve_device("mps") == "mps"
