"""Tests for backend/config.py — configuration loading, merging, defaults."""

import pytest
from backend.config import DEFAULT_CONFIG, ensure_storage_dirs, merge_dict, load_config


class TestMergeDict:
    def test_shallow_override(self):
        result = merge_dict({"a": 1, "b": 2}, {"b": 99})
        assert result == {"a": 1, "b": 99}

    def test_deep_merge(self):
        base = {"x": {"y": 1, "z": 2}}
        override = {"x": {"y": 10}}
        result = merge_dict(base, override)
        assert result == {"x": {"y": 10, "z": 2}}

    def test_new_key_added(self):
        result = merge_dict({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_empty_override(self):
        result = merge_dict({"a": 1}, {})
        assert result == {"a": 1}


class TestDefaultConfig:
    def test_has_required_sections(self):
        for section in ["yolo", "random_forest", "image_processing", "ui"]:
            assert section in DEFAULT_CONFIG, f"Missing section: {section}"

    def test_yolo_defaults(self):
        yolo = DEFAULT_CONFIG["yolo"]
        assert isinstance(yolo["conf_threshold"], float)
        assert isinstance(yolo["iou_threshold"], float)
        assert 0 <= yolo["conf_threshold"] <= 1
        assert 0 <= yolo["iou_threshold"] <= 1

    def test_image_processing_defaults(self):
        ip = DEFAULT_CONFIG["image_processing"]
        assert ip["patch_size"] > ip["overlap"]
        assert 0 <= ip["nms_threshold"] <= 1


class TestEnsureStorageDirs:
    def test_dirs_created(self, tmp_path):
        # ensure_storage_dirs uses BASE_DIR (project root) — test that it doesn't crash
        ensure_storage_dirs()


class TestLoadConfig:
    def test_load_defaults_when_no_file(self):
        # Without a config file, load_config returns defaults.
        # Not testing with tmp_path because CONFIG_PATH is module-level.
        config = load_config()
        assert isinstance(config, dict)
        assert "yolo" in config
