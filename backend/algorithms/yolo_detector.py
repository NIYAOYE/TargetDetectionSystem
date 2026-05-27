from typing import List

import numpy as np

from backend.core.base_detector import BaseDetector, Detection


class YOLODetector(BaseDetector):
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)
        self.class_names = self.model.names

    def _class_name(self, class_id: int) -> str:
        if isinstance(self.class_names, dict):
            return self.class_names.get(class_id, f"class_{class_id}")
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def detect(self, image: np.ndarray) -> List[Detection]:
        if image is None or image.size == 0:
            raise ValueError("Invalid image: empty input")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel BGR image, got {image.shape}")

        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        if not results:
            return detections

        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            x1, y1, x2, y2 = xyxy
            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                    confidence=conf,
                    class_id=class_id,
                    class_name=self._class_name(class_id),
                )
            )

        return detections

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        if not images:
            return []

        for index, image in enumerate(images):
            if image is None or image.size == 0:
                raise ValueError(f"Invalid image at index {index}: empty input")
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected 3-channel BGR image at index {index}, got {image.shape}")

        results = self.model.predict(
            images,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        all_detections = []
        for result in results:
            detections = []
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                x1, y1, x2, y2 = xyxy
                detections.append(
                    Detection(
                        bbox=(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                        confidence=conf,
                        class_id=class_id,
                        class_name=self._class_name(class_id),
                    )
                )
            all_detections.append(detections)

        return all_detections

