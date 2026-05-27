from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    source: str = "auto"
    rf_result: int = -1
    scatter_point_count: int = 0
    scatter_mean_amplitude: float = 0.0
    hu_moment_1: float = 0.0
    hu_moment_2: float = 0.0


class BaseDetector:
    def detect(self, image: np.ndarray) -> List[Detection]:
        raise NotImplementedError

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        return [self.detect(image) for image in images]

