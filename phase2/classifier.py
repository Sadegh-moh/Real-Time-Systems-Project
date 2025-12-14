# phase2/classifier.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class DistanceToMeanClassifier:
    """
    Proposed fast online method:
      - compute mean feature vector for each class (0=big, 1=little)
      - assign by smallest distance to mean

    Optional standardization improves stability.
    """
    standardize: bool = True
    eps: float = 1e-8

    mean0: Optional[np.ndarray] = None
    mean1: Optional[np.ndarray] = None
    mu: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DistanceToMeanClassifier":
        if X.ndim != 2:
            raise ValueError("X must be 2D (N, D).")
        if y.ndim != 1:
            raise ValueError("y must be 1D (N,).")

        if self.standardize:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0) + self.eps
            Xn = (X - self.mu) / self.sigma
        else:
            Xn = X

        X0 = Xn[y == 0]
        X1 = Xn[y == 1]
        if len(X0) == 0 or len(X1) == 0:
            raise ValueError("Need at least one sample for both classes 0 and 1.")

        self.mean0 = X0.mean(axis=0)
        self.mean1 = X1.mean(axis=0)
        return self

    def predict_one(self, x: np.ndarray) -> int:
        if self.mean0 is None or self.mean1 is None:
            raise RuntimeError("Classifier not fitted yet.")

        if self.standardize:
            x = (x - self.mu) / self.sigma

        d0 = float(np.linalg.norm(x - self.mean0))
        d1 = float(np.linalg.norm(x - self.mean1))
        return 0 if d0 <= d1 else 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(x) for x in X], dtype=np.int64)

    def means(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.mean0 is None or self.mean1 is None:
            raise RuntimeError("Classifier not fitted yet.")
        return self.mean0, self.mean1
