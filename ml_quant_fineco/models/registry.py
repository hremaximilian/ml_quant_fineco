"""
Model registry: factory for creating models by name.
"""

from typing import Dict, Optional

from .base import BaseModel
from .sklearn_models import RandomForestModel, SVMModel
from .boosting_models import XGBoostModel, LightGBMModel
from .dl_models import LSTMModel


MODEL_REGISTRY: Dict[str, type] = {
    "random_forest": RandomForestModel,
    "rf": RandomForestModel,
    "xgboost": XGBoostModel,
    "xgb": XGBoostModel,
    "lightgbm": LightGBMModel,
    "lgbm": LightGBMModel,
    "svm": SVMModel,
    "lstm": LSTMModel,
}


def create_model(
    model_name: str,
    task: str = "classification",
    params: Optional[Dict] = None,
    **kwargs,
) -> BaseModel:
    """
    Create a model instance by name.

    Args:
        model_name: Name of the model (e.g. 'xgboost', 'lstm', 'random_forest').
        task: 'classification' or 'regression'.
        params: Optional hyperparameter overrides.
        **kwargs: Additional keyword arguments passed to the model constructor.

    Returns:
        An instantiated BaseModel subclass.

    Raises:
        ValueError: If model_name is not recognized.
    """
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(set(MODEL_REGISTRY.values()), key=lambda c: c.__name__))
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available}"
        )

    cls = MODEL_REGISTRY[key]
    return cls(task=task, params=params, **kwargs)


def list_models() -> Dict[str, str]:
    """Return a mapping of alias -> full class name."""
    return {alias: cls.__name__ for alias, cls in MODEL_REGISTRY.items()}
