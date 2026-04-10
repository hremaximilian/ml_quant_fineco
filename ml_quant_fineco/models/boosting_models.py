"""
XGBoost and LightGBM model wrappers.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost classifier or regressor."""

    def __init__(
        self,
        task: str = "classification",
        params: Optional[Dict] = None,
    ):
        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)

        super().__init__("XGBoost", default_params)
        self.task = task

        if task == "classification":
            from xgboost import XGBClassifier
            self.model = XGBClassifier(**default_params)
        else:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(**default_params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        logger.info(f"Fitting XGBoost ({self.task}) on {X_train.shape[0]} samples, {X_train.shape[1]} features")

        eval_set = kwargs.get("eval_set")
        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False

        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_)


class LightGBMModel(BaseModel):
    """LightGBM classifier or regressor."""

    def __init__(
        self,
        task: str = "classification",
        params: Optional[Dict] = None,
    ):
        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)

        super().__init__("LightGBM", default_params)
        self.task = task

        if task == "classification":
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(**default_params)
        else:
            from lightgbm import LGBMRegressor
            self.model = LGBMRegressor(**default_params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        logger.info(f"Fitting LightGBM ({self.task}) on {X_train.shape[0]} samples, {X_train.shape[1]} features")

        eval_set = kwargs.get("eval_set")
        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set

        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_)
