"""
Feature engineering module for generating technical indicators and ML-ready features.

Produces labeled datasets suitable for classification (up/down) or regression
(next-day return) models.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generate technical features and ML labels from OHLCV data."""

    def __init__(self, label_method: str = "return_sign", forward_period: int = 1):
        """
        Args:
            label_method: How to construct the target label.
                - 'return_sign': 1 if next-period return > 0, else 0 (classification).
                - 'return': raw next-period percentage return (regression).
                - 'excess_return': return minus a baseline (classification vs baseline).
            forward_period: Number of periods ahead for the target.
        """
        self.label_method = label_method
        self.forward_period = forward_period

    def build(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Build all features and labels from OHLCV data.

        Args:
            df: DataFrame with Open, High, Low, Close, Volume columns.
            drop_na: Whether to drop rows with NaN after feature construction.

        Returns:
            DataFrame with all features and a 'target' column.
        """
        result = df.copy()

        # --- Price-based features ---
        result = self._add_price_features(result)

        # --- Volume-based features ---
        result = self._add_volume_features(result)

        # --- Technical indicators ---
        result = self._add_technical_indicators(result)

        # --- Lag features ---
        result = self._add_lag_features(result)

        # --- Rolling statistics ---
        result = self._add_rolling_features(result)

        # --- Target label ---
        result = self._add_target(result)

        if drop_na:
            result = result.dropna()

        logger.info(f"Feature matrix shape: {result.shape}")
        return result

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return the list of feature column names (excluding OHLCV and target)."""
        exclude = {"Open", "High", "Low", "Close", "Volume", "target"}
        return [c for c in df.columns if c not in exclude]

    # ------------------------------------------------------------------
    # Internal feature builders
    # ------------------------------------------------------------------

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic return and price-ratio features."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        opn = df["Open"]

        # Simple returns at various lags
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f"ret_{lag}d"] = close.pct_change(lag)

        # Intraday features
        df["intraday_range"] = (high - low) / close
        df["gap"] = (opn - close.shift(1)) / close.shift(1)
        df["oc_ratio"] = (close - opn) / opn  # close vs open
        df["hl_ratio"] = (high - low) / (high + low + 1e-10)

        # Log returns
        df["log_ret_1d"] = np.log(close / close.shift(1))

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features."""
        vol = df["Volume"].astype(float)
        close = df["Close"]

        df["vol_change_1d"] = vol.pct_change(1)
        df["vol_ma_5"] = vol.rolling(5).mean()
        df["vol_ma_20"] = vol.rolling(20).mean()
        df["vol_ratio"] = vol / (df["vol_ma_20"] + 1e-10)

        # Volume-Price Trend (VPT)
        df["vpt"] = ((close.pct_change(1)) * vol).cumsum()

        # On-Balance Volume (simplified)
        obv = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + vol.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - vol.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]
        df["obv"] = obv

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Common technical indicators computed with numpy/pandas only."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # --- Moving Averages ---
        for window in [5, 10, 20, 60]:
            df[f"sma_{window}"] = close.rolling(window).mean()
        df["ema_12"] = close.ewm(span=12, adjust=False).mean()
        df["ema_26"] = close.ewm(span=26, adjust=False).mean()

        # --- MACD ---
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # --- RSI ---
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # --- Bollinger Bands ---
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (sma20 + 1e-10)

        # --- Stochastic Oscillator ---
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        df["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # --- ATR (Average True Range) ---
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

        # --- ADX (Average Directional Index) ---
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        atr14 = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-10)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-10)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df["adx"] = dx.rolling(14).mean()

        # --- Williams %R ---
        df["willr"] = -100 * (high14 - close) / (high14 - low14 + 1e-10)

        # --- CCI (Commodity Channel Index) ---
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df["cci"] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lagged features for key indicators."""
        lag_cols = ["ret_1d", "rsi_14", "macd_hist", "bb_pct", "vol_ratio"]
        for col in lag_cols:
            if col in df.columns:
                for lag in [1, 2, 3]:
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling statistical features."""
        close = df["Close"]
        for window in [5, 10, 20]:
            df[f"rolling_skew_{window}"] = close.pct_change().rolling(window).skew()
            df[f"rolling_kurt_{window}"] = close.pct_change().rolling(window).kurt()
            df[f"rolling_vol_{window}"] = close.pct_change().rolling(window).std()

        # Momentum
        for period in [5, 10, 20]:
            df[f"momentum_{period}"] = close - close.shift(period)

        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct the prediction target."""
        close = df["Close"]
        future_return = close.shift(-self.forward_period) / close - 1

        if self.label_method == "return_sign":
            df["target"] = (future_return > 0).astype(int)
        elif self.label_method == "return":
            df["target"] = future_return
        elif self.label_method == "excess_return":
            # Use a simple zero baseline; can be replaced with benchmark return
            df["target"] = (future_return > 0).astype(int)
        else:
            raise ValueError(f"Unknown label_method: {self.label_method}")

        return df


def train_test_split_time(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-aware train/test split.  Earlier data -> train, later data -> test.

    Args:
        df: DataFrame sorted by datetime index.
        test_ratio: Fraction of data reserved for testing.

    Returns:
        (train_df, test_df)
    """
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    return train_df, test_df
