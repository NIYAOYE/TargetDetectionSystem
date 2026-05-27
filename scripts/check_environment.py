#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import platform
import sys


REQUIRED_MODULES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("numpy", "numpy"),
    ("yaml", "pyyaml"),
    ("cv2", "opencv-python-headless"),
    ("rasterio", "rasterio"),
    ("ultralytics", "ultralytics"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("pandas", "pandas"),
]


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> int:
    print("SAR BS environment check")
    print(f"Python: {sys.version.split()[0]} ({platform.system()} {platform.release()})")

    if sys.version_info[:2] != (3, 11):
        print("[WARN] Python 3.11 is recommended for the SAR model stack.")

    missing = []
    for module_name, package_name in REQUIRED_MODULES:
        if module_available(module_name):
            print(f"[OK] {package_name}")
        else:
            print(f"[MISS] {package_name}")
            missing.append(package_name)

    if missing:
        print("")
        print("Install missing Python packages inside the virtual environment:")
        print("  pip install -r requirements.txt")
        print("")
        print("On Ubuntu, OpenCV may also need system libraries:")
        print("  sudo apt install -y libgl1 libglib2.0-0")
        return 1

    try:
        import torch

        print(f"[INFO] Torch CUDA available: {torch.cuda.is_available()}")
    except Exception as exc:
        print(f"[WARN] Could not check Torch CUDA: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
