from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"
STORAGE_DIR = BASE_DIR / "storage"
IMAGE_DIR = STORAGE_DIR / "images"
DETECTOR_MODEL_DIR = STORAGE_DIR / "models" / "detectors"
CLASSIFIER_MODEL_DIR = STORAGE_DIR / "models" / "classifiers"
OUTPUTS_DIR = STORAGE_DIR / "outputs"
CONFIG_PATH = BASE_DIR / "config" / "default_config.yaml"

DEMO_DETECTOR_NAME = "__demo__"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DETECTOR_EXTENSIONS = {".pt", ".pth"}
CLASSIFIER_EXTENSIONS = {".pkl", ".joblib"}

DEFAULT_CONFIG = {
    "yolo": {
        "device": "auto",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
    },
    "random_forest": {
        "enabled": True,
    },
    "image_processing": {
        "patch_size": 640,
        "overlap": 64,
        "nms_threshold": 0.45,
    },
    "segmentation": {
        "background": [10, 15, 22],  # clean cutout backdrop, RGB
        "feather": 1.5,  # gaussian sigma for soft mask edges; 0 = hard edge
    },
    "ui": {
        "default_rf_enabled": True,
    },
}


def ensure_storage_dirs() -> None:
    for directory in [
        STATIC_DIR,
        IMAGE_DIR,
        DETECTOR_MODEL_DIR,
        CLASSIFIER_MODEL_DIR,
        OUTPUTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG
    try:
        import logging

        import yaml

        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        return merge_dict(DEFAULT_CONFIG, loaded)
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to load config %s: %s, using defaults", CONFIG_PATH, exc)
        return DEFAULT_CONFIG

