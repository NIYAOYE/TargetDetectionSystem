"""Tests for target segmentation and clean-background cutout."""

import cv2
import numpy as np

from backend.algorithms.feature_extractor import extract_sar_features, segment_target_mask
from backend.image_io import segment_target_on_background


def _blob_crop(size=64, dtype=np.uint8):
    crop = np.zeros((size, size), dtype=dtype)
    high = 255 if dtype == np.uint8 else 4000
    cv2.rectangle(crop, (size // 3, size // 3), (2 * size // 3, 2 * size // 3), high, -1)
    return crop


class TestSegmentTargetMask:
    def test_bright_blob_is_segmented(self):
        mask = segment_target_mask(_blob_crop())
        assert mask is not None
        assert mask.dtype == np.uint8
        assert mask[32, 32] == 255  # blob centre is foreground
        assert mask[1, 1] == 0  # corner is background

    def test_empty_crop_returns_none(self):
        assert segment_target_mask(np.zeros((64, 64), dtype=np.uint8)) is None

    def test_none_input_returns_none(self):
        assert segment_target_mask(None) is None

    def test_16bit_blob_is_segmented(self):
        mask = segment_target_mask(_blob_crop(dtype=np.uint16))
        assert mask is not None
        assert mask[32, 32] == 255

    def test_features_consistent_with_mask(self):
        # The same blob must yield non-zero RF features (shared segmentation core).
        features = extract_sar_features(_blob_crop(), bbox=(0, 0, 64, 64))
        assert np.any(features != 0)


class TestSegmentOnBackground:
    def test_returns_decodable_png_of_crop_size(self, tmp_path):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (70, 70), 255, -1)
        path = tmp_path / "t.png"
        cv2.imwrite(str(path), img)

        png = segment_target_on_background(path, (25, 25, 50, 50))
        decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == (50, 50, 3)

    def test_background_pixels_match_clean_color(self, tmp_path):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(img, (40, 40), (60, 60), 255, -1)
        path = tmp_path / "t.png"
        cv2.imwrite(str(path), img)

        png = segment_target_on_background(path, (30, 30, 40, 40), feather=0.0)
        decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        # Top-left corner of the crop is outside the blob -> clean background.
        assert tuple(int(c) for c in decoded[0, 0]) == (22, 15, 10)
