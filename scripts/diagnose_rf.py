"""Diagnose why RF classification yields all 未分类 (rf_result == -1).

Usage (from project root, in the SarProject env):
    python scripts/diagnose_rf.py <image> <detector.pt> <classifier.joblib>

The detector may be "__demo__" to use the synthetic detector.
"""
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

from backend.algorithms.rf_classifier import RFClassifier
from backend.config import DEMO_DETECTOR_NAME
from backend.pipeline import DetectionPipeline, ModelRegistry


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    image_path = Path(sys.argv[1])
    detector_arg = sys.argv[2]
    classifier_path = Path(sys.argv[3])

    print("=" * 70)
    print("IMAGE")
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  cv2 could not read {image_path}")
    else:
        print(f"  shape={img.shape} dtype={img.dtype} channels={1 if img.ndim == 2 else img.shape[2]}")

    print("=" * 70)
    print("CLASSIFIER MODEL")
    clf = RFClassifier(model_path=str(classifier_path))
    model = clf.model
    print(f"  type={type(model)}")
    print(f"  n_features_in_={getattr(model, 'n_features_in_', 'MISSING')}")
    names = getattr(model, "feature_names_in_", None)
    print(f"  feature_names_in_={list(names) if names is not None else 'MISSING (no name alignment!)'}")
    classes = getattr(model, "classes_", None)
    print(f"  classes_={classes} (dtype={getattr(classes, 'dtype', '?')})")
    if classes is not None:
        try:
            [int(c) for c in classes]
            print("  -> labels are int-castable: OK")
        except (TypeError, ValueError):
            print("  -> labels are NOT int-castable: int(prediction) will THROW -> all -1")

    print("=" * 70)
    print("PIPELINE RUN")
    detector_path = None if detector_arg == DEMO_DETECTOR_NAME else Path(detector_arg)
    pipeline = DetectionPipeline(ModelRegistry())
    detections = pipeline.run(
        image_path=image_path,
        detector_path=detector_path,
        classifier_path=classifier_path,
        patch_size=640,
        overlap=64,
        conf_threshold=0.25,
        iou_threshold=0.45,
        nms_threshold=0.45,
        use_classifier=True,
        device="auto",
        progress=lambda *a: None,
    )
    print("-" * 70)
    print(f"  total detections: {len(detections)}")
    print(f"  rf_result distribution: {Counter(d.rf_result for d in detections)}")
    print("    (1 = 真目标/green, 0 = 虚警/red, -1 = 未分类/yellow)")


if __name__ == "__main__":
    main()
