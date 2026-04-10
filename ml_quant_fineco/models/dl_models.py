"""
LSTM deep learning model wrapper for time-series prediction.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseModel

logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):
    """LSTM neural network for time-series classification or regression."""

    def __init__(
        self,
        task: str = "classification",
        params: Optional[Dict] = None,
        sequence_length: int = 20,
    ):
        default_params = {
            "lstm_units_1": 64,
            "lstm_units_2": 32,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "patience": 10,
        }
        if params:
            default_params.update(params)

        super().__init__("LSTM", default_params)
        self.task = task
        self.sequence_length = sequence_length
        self.model = None

    def _build_model(self, input_shape: tuple) -> None:
        """Build the LSTM architecture."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping

        p = self.params
        model = Sequential([
            Input(shape=input_shape),
            KerasLSTM(p["lstm_units_1"], return_sequences=True),
            Dropout(p["dropout_rate"]),
            KerasLSTM(p["lstm_units_2"]),
            Dropout(p["dropout_rate"]),
        ])

        if self.task == "classification":
            model.add(Dense(1, activation="sigmoid"))
            model.compile(
                optimizer=Adam(learning_rate=p["learning_rate"]),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
        else:
            model.add(Dense(1, activation="linear"))
            model.compile(
                optimizer=Adam(learning_rate=p["learning_rate"]),
                loss="mse",
                metrics=["mae"],
            )

        self.model = model

    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """Transform 2D data into 3D sequences for LSTM input."""
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length : i])
            if y is not None:
                y_seq.append(y[i])
        X_seq = np.array(X_seq)
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        return X_seq, None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Train the LSTM model."""
        from tensorflow.keras.callbacks import EarlyStopping

        X_arr = X_train.values.astype(np.float32)
        y_arr = y_train.values.astype(np.float32)

        X_seq, y_seq = self._create_sequences(X_arr, y_arr)
        logger.info(
            f"Fitting LSTM ({self.task}) on {X_seq.shape[0]} sequences "
            f"of length {self.sequence_length}, {X_seq.shape[2]} features"
        )

        self._build_model((X_seq.shape[1], X_seq.shape[2]))

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.params["patience"],
            restore_best_weights=True,
        )

        self.model.fit(
            X_seq,
            y_seq,
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0,
        )
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.values.astype(np.float32)
        X_seq, _ = self._create_sequences(X_arr)

        raw = self.model.predict(X_seq, verbose=0)
        if self.task == "classification":
            return (raw.flatten() > 0.5).astype(int)
        return raw.flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")

        X_arr = X.values.astype(np.float32)
        X_seq, _ = self._create_sequences(X_arr)

        proba_1 = self.model.predict(X_seq, verbose=0).flatten()
        proba_0 = 1 - proba_1
        return np.column_stack([proba_0, proba_1])
