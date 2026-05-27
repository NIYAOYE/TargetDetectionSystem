from pathlib import Path
from threading import RLock
from typing import Callable, Optional
import logging

import cv2
import numpy as np

from backend.algorithms.feature_extractor import (
    crop_detection_patch,
    extract_sar_features,
    extract_scatter_features_from_bbox,
)
from backend.algorithms.image_processing import ImageSlicer
from backend.config import DEMO_DETECTOR_NAME
from backend.core.base_classifier import BaseClassifier
from backend.core.base_detector import BaseDetector, Detection

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class SyntheticDetector(BaseDetector):
    model_path = DEMO_DETECTOR_NAME

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        boxes = [
            (0.18 * w, 0.20 * h, 0.18 * w, 0.12 * h, 0.91, 0, "demo_target"),
            (0.56 * w, 0.46 * h, 0.16 * w, 0.14 * h, 0.84, 1, "demo_ship"),
        ]
        detections = []
        for x, y, bw, bh, confidence, class_id, class_name in boxes:
            detections.append(
                Detection(
                    bbox=(float(x), float(y), float(bw), float(bh)),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=class_name,
                )
            )
        return detections


class ModelRegistry:
    def __init__(self):
        self._detectors: dict[tuple, BaseDetector] = {}
        self._classifiers: dict[Path, BaseClassifier] = {}
        self._lock = RLock()

    def get_detector(
        self,
        detector_path: Optional[Path],
        conf_threshold: float,
        iou_threshold: float,
        device: str,
    ) -> BaseDetector:
        if detector_path is None:
            return SyntheticDetector()

        resolved_device = resolve_device(device)
        key = (detector_path.resolve(), conf_threshold, iou_threshold, resolved_device)
        with self._lock:
            if key not in self._detectors:
                from backend.algorithms.yolo_detector import YOLODetector

                self._detectors[key] = YOLODetector(
                    model_path=str(detector_path),
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    device=resolved_device,
                )
            return self._detectors[key]

    def get_classifier(self, classifier_path: Optional[Path]) -> Optional[BaseClassifier]:
        if classifier_path is None:
            return None

        key = classifier_path.resolve()
        with self._lock:
            if key not in self._classifiers:
                from backend.algorithms.rf_classifier import RFClassifier

                self._classifiers[key] = RFClassifier(model_path=str(classifier_path))
            return self._classifiers[key]


def resolve_device(device: str) -> str:
    if device in {"", "auto"}:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as exc:
            logger.debug("torch import/check failed, falling back to CPU: %s", exc)
            return "cpu"
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except Exception as exc:
            logger.debug("CUDA check failed, falling back to CPU: %s", exc)
            return "cpu"
    return device


def global_nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    if not detections:
        return []

    boxes = []
    scores = []
    for detection in detections:
        x, y, w, h = detection.bbox
        boxes.append([float(x), float(y), float(w), float(h)])
        scores.append(float(detection.confidence))

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=0.0,
        nms_threshold=float(iou_threshold),
    )
    if len(indices) > 0:
        return [detections[int(index)] for index in np.array(indices).flatten()]
    return []


def clip_bbox(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    x1 = max(0.0, min(float(x), float(image_width)))
    y1 = max(0.0, min(float(y), float(image_height)))
    x2 = max(x1, min(float(x + w), float(image_width)))
    y2 = max(y1, min(float(y + h), float(image_height)))
    return (x1, y1, x2 - x1, y2 - y1)


class DetectionPipeline:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def run(
        self,
        image_path: Path,
        detector_path: Optional[Path],
        classifier_path: Optional[Path],
        patch_size: int,
        overlap: int,
        conf_threshold: float,
        iou_threshold: float,
        nms_threshold: float,
        use_classifier: bool,
        device: str,
        progress: ProgressCallback,
    ) -> list[Detection]:
        progress(0, 100, "Initializing image slicer...")
        detector = self.registry.get_detector(detector_path, conf_threshold, iou_threshold, device)
        classifier = self.registry.get_classifier(classifier_path) if use_classifier else None

        slicer = ImageSlicer(str(image_path), patch_size=patch_size, overlap=overlap)
        total_patches = len(slicer)
        progress(10, 100, f"Processing {total_patches} patches...")

        all_detections: list[Detection] = []
        patches = []
        metadata = []
        batch_size = 16

        for index, (row, col, patch, bbox) in enumerate(slicer):
            del row, col
            patches.append(patch)
            metadata.append(bbox)

            if len(patches) >= batch_size or index == total_patches - 1:
                batch_results = detector.detect_batch(patches)
                for detections, patch_bbox in zip(batch_results, metadata):
                    y_start, _, x_start, _ = patch_bbox
                    for detection in detections:
                        x, y, w, h = detection.bbox
                        global_bbox = (
                            float(x + x_start),
                            float(y + y_start),
                            float(w),
                            float(h),
                        )
                        clipped_bbox = clip_bbox(global_bbox, slicer.width, slicer.height)
                        if clipped_bbox[2] > 0 and clipped_bbox[3] > 0:
                            all_detections.append(
                                Detection(
                                    bbox=clipped_bbox,
                                    confidence=detection.confidence,
                                    class_id=detection.class_id,
                                    class_name=detection.class_name,
                                )
                            )

                patches = []
                metadata = []
                current = 10 + int(70 * (index + 1) / total_patches)
                progress(current, 100, f"Detected on {index + 1}/{total_patches} patches")

        progress(85, 100, "Applying global NMS...")
        final_detections = global_nms(all_detections, nms_threshold)

        full_image = None
        if final_detections:
            progress(88, 100, "Extracting scatter features...")
            full_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if full_image is not None:
                self._attach_scatter_features(final_detections, full_image)

        if use_classifier and classifier is not None:
            progress(92, 100, "Extracting features and classifying...")
            final_detections = self._classify_detections(
                final_detections,
                classifier=classifier,
                full_image=full_image,
                image_path=image_path,
            )

        progress(100, 100, f"Complete! Found {len(final_detections)} targets")
        return final_detections

    @staticmethod
    def _attach_scatter_features(detections: list[Detection], full_image: np.ndarray) -> None:
        for detection in detections:
            try:
                scatter_features = extract_scatter_features_from_bbox(full_image, detection.bbox)
                detection.scatter_point_count = int(scatter_features["scatter_point_count"])
                detection.scatter_mean_amplitude = float(scatter_features["scatter_mean_amplitude"])
                detection.hu_moment_1 = float(scatter_features["hu_moment_1"])
                detection.hu_moment_2 = float(scatter_features["hu_moment_2"])
            except Exception as exc:
                detection.scatter_point_count = 0
                detection.scatter_mean_amplitude = 0.0
                detection.hu_moment_1 = 0.0
                detection.hu_moment_2 = 0.0
                logger.debug("scatter feature extraction failed for bbox %s: %s", detection.bbox, exc)

    @staticmethod
    def _classify_detections(
        detections: list[Detection],
        classifier: BaseClassifier,
        full_image: Optional[np.ndarray],
        image_path: Path,
    ) -> list[Detection]:
        if not detections:
            return []

        if full_image is None:
            full_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if full_image is None:
            for detection in detections:
                detection.rf_result = -1
            return detections

        try:
            import pandas as pd
        except ImportError:
            for detection in detections:
                detection.rf_result = -1
            return detections

        feature_rows = []
        valid_detections = []

        for detection in detections:
            try:
                crop = crop_detection_patch(full_image, detection.bbox)
                sar_features = extract_sar_features(crop, detection.bbox)
                if np.any(sar_features != 0):
                    feature_rows.append(
                        {
                            "area": float(sar_features[0]),
                            "perimeter": float(sar_features[1]),
                            "length": float(sar_features[2]),
                            "width": float(sar_features[3]),
                            "aspect_ratio": float(sar_features[4]),
                            "mean_intensity": float(sar_features[5]),
                            "max_intensity": float(sar_features[6]),
                            "std_intensity": float(sar_features[7]),
                            "confidence": float(detection.confidence),
                            "class_index": int(detection.class_id),
                        }
                    )
                    valid_detections.append(detection)
                else:
                    detection.rf_result = -1
            except Exception as exc:
                detection.rf_result = -1
                logger.debug("SAR feature extraction failed for detection: %s", exc)

        if not feature_rows:
            return detections

        try:
            df = pd.DataFrame(feature_rows)
            base_features = [
                "confidence",
                "area",
                "perimeter",
                "length",
                "width",
                "aspect_ratio",
                "mean_intensity",
                "max_intensity",
                "std_intensity",
            ]
            x_base = df[base_features].copy()
            class_dummies = pd.get_dummies(df["class_index"], prefix="yolo_cls")
            x_temp = pd.concat([x_base, class_dummies], axis=1)
            if hasattr(classifier.model, "feature_names_in_"):
                trained_features = classifier.model.feature_names_in_
                x_final = x_temp.reindex(columns=trained_features, fill_value=0)
            else:
                x_final = x_temp
            predictions = classifier.predict(x_final)
            for detection, prediction in zip(valid_detections, predictions):
                detection.rf_result = int(prediction)
        except Exception as exc:
            for detection in valid_detections:
                detection.rf_result = -1
            logger.warning("RF classification batch failed: %s", exc)

        return detections

