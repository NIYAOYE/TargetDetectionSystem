import numpy as np


class BaseClassifier:
    def predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

