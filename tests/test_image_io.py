"""Tests for backend/image_io.py — file listing, path safety, detection serialisation."""

import pytest
from pathlib import Path
from backend.image_io import (
    detection_to_dict,
    detections_to_dicts,
    list_files,
    resolve_storage_file,
)
from backend.core.base_detector import Detection


class TestListFiles:
    def test_empty_directory(self, tmp_path):
        files = list_files(tmp_path, {".png"})
        assert files == []

    def test_filters_by_extension(self, tmp_path):
        (tmp_path / "img1.png").write_text("")
        (tmp_path / "img2.jpg").write_text("")
        (tmp_path / "doc.txt").write_text("")
        files = list_files(tmp_path, {".png", ".jpg"})
        names = {f["name"] for f in files}
        assert names == {"img1.png", "img2.jpg"}

    def test_file_metadata(self, tmp_path):
        (tmp_path / "test.png").write_text("abc")
        files = list_files(tmp_path, {".png"})
        assert len(files) == 1
        f = files[0]
        assert f["name"] == "test.png"
        assert f["size"] == 3
        assert f["kind"] == "png"
        assert f["modified_at"] is not None

    def test_sorted_by_name(self, tmp_path):
        (tmp_path / "c.png").write_text("")
        (tmp_path / "a.png").write_text("")
        (tmp_path / "B.png").write_text("")
        files = list_files(tmp_path, {".png"})
        names = [f["name"] for f in files]
        # Case-insensitive sort
        assert names[0].lower() == "a.png"
        assert names[1].lower() == "b.png"
        assert names[2].lower() == "c.png"


class TestResolveStorageFile:
    def test_finds_existing_file(self, tmp_path):
        (tmp_path / "image.png").write_text("")
        resolved = resolve_storage_file(tmp_path, "image.png", {".png"})
        assert resolved.name == "image.png"

    def test_raises_if_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_storage_file(tmp_path, "missing.png", {".png"})

    def test_raises_on_wrong_extension(self, tmp_path):
        (tmp_path / "bad.txt").write_text("")
        with pytest.raises(FileNotFoundError):
            resolve_storage_file(tmp_path, "bad.txt", {".png"})

    def test_blocks_path_traversal(self, tmp_path):
        # resolve_storage_file rejects paths with "/" or ".."
        with pytest.raises(FileNotFoundError):
            resolve_storage_file(tmp_path, "../etc/passwd", {".png"})

    def test_blocks_absolute_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_storage_file(tmp_path, "/etc/passwd", {".png"})


class TestDetectionSerialisation:
    def test_detection_to_dict(self):
        d = Detection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.95,
            class_id=0,
            class_name="target",
        )
        result = detection_to_dict(5, d)
        assert result["id"] == 5
        assert result["bbox"] == [10.0, 20.0, 30.0, 40.0]
        assert result["confidence"] == 0.95
        assert result["class_name"] == "target"
        assert result["source"] == "auto"

    def test_detection_with_rf_result(self):
        d = Detection(
            bbox=(10, 20, 30, 40),
            confidence=0.8,
            class_id=1,
            class_name="ship",
        )
        d.rf_result = 1
        result = detection_to_dict(0, d)
        assert result["rf_result"] == 1

    def test_detections_to_dicts(self):
        d1 = Detection(bbox=(0, 0, 10, 10), confidence=0.9, class_id=0, class_name="a")
        d2 = Detection(bbox=(0, 0, 10, 10), confidence=0.8, class_id=1, class_name="b", source="manual")
        results = detections_to_dicts([d1, d2])
        assert len(results) == 2
        assert results[0]["id"] == 0
        assert results[1]["id"] == 1
        assert results[1]["source"] == "manual"
