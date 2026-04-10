"""
Random Forest and SVM model wrappers.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler

from .base import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest classifier or regressor."""

    def __init__(
        self,
        task: str = "classification",
        params: Optional[Dict] = None,
    ):
        """
        Args:
            task: 'classification' or 'regression'.
            params: Hyperparameters for the sklearn model.
        """
        default_params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)

        super().__init__("RandomForest", default_params)
        self.task = task

        if task == "classification":
            self.model = RandomForestClassifier(**default_params)
        else:
            self.model = RandomForestRegressor(**default_params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        logger.info(f"Fitting RandomForest ({self.task}) on {X_train.shape[0]} samples, {X_train.shape[1]} features")
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)

    def feature_importance(self) -> pd.Series:
        """Return feature importances."""
        return pd.Series(self.model.feature_importances_)


class SVMModel(BaseModel):
    """Support Vector Machine classifier or regressor."""

    def __init__(
        self,
        task: str = "classification",
        params: Optional[Dict] = None,
    ):
        default_params = {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "random_state": 42,
        }
        if params:
            default_params.update(params)

        super().__init__("SVM", default_params)
        self.task = task
        self.scaler = StandardScaler()

        if task == "classification":
            default_params["probability"] = True
            self.model = SVC(**default_params)
        else:
            self.model = SVR(C=default_params["C"], kernel=default_params["kernel"])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        logger.info(f"Fitting SVM ({self.task}) on {X_train.shape[0]} samples")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
