from typing import Optional

import numpy as np

from backend.core.base_classifier import BaseClassifier


class RFClassifier(BaseClassifier):
    def __init__(self, model_path: Optional[str] = None, model=None):
        if model is not None:
            self.model = model
        elif model_path is not None:
            import joblib

            self.model = joblib.load(model_path)
        else:
            raise ValueError("Must provide either model_path or model instance")

        self.n_features = self.model.n_features_in_

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != self.n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features}, got {features.shape[1]}"
            )
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != self.n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features}, got {features.shape[1]}"
            )
        return self.model.predict_proba(features)

