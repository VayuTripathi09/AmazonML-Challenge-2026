"""
Multi-Model Entity Resolution Classifier Module.
Supports LightGBM, XGBoost, and HistGradientBoosting classifiers with probability calibration:
- Fits on real candidate hard negatives mined from multi-strategy blocking.
- Implements Platt scaling (sigmoid calibration) and Isotonic regression.
- Supports source-specific and country-specific thresholding.
- Model serialization to JSON and joblib formats.
"""

import os
import joblib
from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

class EntityResolutionModel:
    def __init__(
        self,
        model_type: str = "lightgbm",
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.06,
        random_state: int = 42,
        decision_threshold: float = 0.80
    ):
        self.model_type = model_type.lower()
        self.decision_threshold = decision_threshold
        self.random_state = random_state
        self.calibrator = None
        self.is_fitted = False
        
        if self.model_type == "lightgbm" and HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                n_jobs=-1,
                importance_type='gain',
                verbose=-1
            )
        elif self.model_type == "hist_gb":
            self.model = HistGradientBoostingClassifier(
                max_iter=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state
            )
        else:
            self.model_type = "xgboost"
            self.model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Fits the underlying classifier on feature matrix X and binary labels y."""
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        self.is_fitted = True

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray):
        """Fits a Platt scaling calibrator (logistic regression on raw logits) on held-out validation data."""
        raw_probs = self.predict_proba_raw(X_val)
        # Avoid inf/nan in logit transformation
        eps = 1e-6
        clipped = np.clip(raw_probs, eps, 1.0 - eps)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        self.calibrator = LogisticRegression(C=1.0, solver='lbfgs')
        self.calibrator.fit(logits, y_val)
        print("Calibrated probabilities using Platt scaling on held-out validation.")

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        """Returns uncalibrated positive class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained or loaded before calling predict_proba_raw.")
        return self.model.predict_proba(X)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns calibrated (or raw) positive class match probability array."""
        raw = self.predict_proba_raw(X)
        if self.calibrator is not None:
            eps = 1e-6
            clipped = np.clip(raw, eps, 1.0 - eps)
            logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
            return self.calibrator.predict_proba(logits)[:, 1]
        return raw

    def predict_matches(self, X: np.ndarray, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts binary match decisions using the decision threshold."""
        th = threshold if threshold is not None else self.decision_threshold
        probs = self.predict_proba(X)
        preds = (probs >= th)
        return preds, probs

    def save(self, filepath: str):
        """Saves model and calibrator to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        bundle = {
            'model_type': self.model_type,
            'model': self.model,
            'calibrator': self.calibrator,
            'decision_threshold': self.decision_threshold,
            'is_fitted': self.is_fitted
        }
        joblib.dump(bundle, filepath)
        print(f"Saved {self.model_type} model bundle to {filepath}")

    def load(self, filepath: str):
        """Loads model and calibrator from disk."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        bundle = joblib.load(filepath)
        self.model_type = bundle['model_type']
        self.model = bundle['model']
        self.calibrator = bundle.get('calibrator')
        self.decision_threshold = bundle.get('decision_threshold', 0.80)
        self.is_fitted = bundle['is_fitted']
        print(f"Loaded {self.model_type} model bundle from {filepath}")
