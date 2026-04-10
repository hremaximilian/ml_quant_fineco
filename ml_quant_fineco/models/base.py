"""
Base model interface for all ML models in the ml_quant_fineco framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)


class BaseModel(ABC):
    """Abstract base class for trading models."""

    def __init__(self, name: str, params: Optional[Dict] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions (for classification)."""
        pass

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task: str = "classification",
    ) -> Dict[str, float]:
        """
        Evaluate the model and return standard metrics.

        Args:
            X_test: Test features.
            y_test: True labels.
            task: 'classification' or 'regression'.

        Returns:
            Dictionary of metric name -> value.
        """
        y_pred = self.predict(X_test)

        if task == "classification":
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
            try:
                y_proba = self.predict_proba(X_test)
                if y_proba.ndim == 2:
                    metrics["auc"] = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    metrics["auc"] = roc_auc_score(y_test, y_proba)
            except (ValueError, AttributeError):
                metrics["auc"] = np.nan

        elif task == "regression":
            metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
            }

        return metrics

    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return self.params
