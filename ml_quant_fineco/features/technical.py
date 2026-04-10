"""
Feature engineering module for generating technical indicators and ML-ready features.

Uses the ``ta`` library (https://github.com/bukosabino/ta) for standard-compliant
technical indicator computation.  All indicators follow widely-accepted formulas
(Wilder smoothing for RSI/ATR, standard MACD parameters, etc.).

Produces labeled datasets suitable for classification (up/down) or regression
(next-day return) models.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from ta.trend import (
    SMAIndicator,
    EMAIndicator,
    WMAIndicator,
    MACD,
    ADXIndicator,
    CCIIndicator,
    IchimokuIndicator,
    PSARIndicator,
    TRIXIndicator,
    VortexIndicator,
    AroonIndicator,
    KSTIndicator,
    stc,
)
from ta.momentum import (
    RSIIndicator,
    StochasticOscillator,
    WilliamsRIndicator,
    StochRSIIndicator,
    ROCIndicator,
    UltimateOscillator,
    TSIIndicator,
    KAMAIndicator,
    PercentagePriceOscillator,
    AwesomeOscillatorIndicator,
)
from ta.volatility import (
    BollingerBands,
    AverageTrueRange,
    KeltnerChannel,
    DonchianChannel,
    UlcerIndex,
)
from ta.volume import (
    OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator,
    ChaikinMoneyFlowIndicator,
    MFIIndicator,
    ForceIndexIndicator,
    EaseOfMovementIndicator,
    VolumeWeightedAveragePrice,
    NegativeVolumeIndexIndicator,
    AccDistIndexIndicator,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generate technical features and ML labels from OHLCV data.

    All indicators are computed via the ``ta`` library, ensuring correct
    mathematical definitions (e.g. Wilder-smoothed RSI, EMA-based MACD).
    """

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

        # --- Trend indicators (MA, MACD, ADX, Ichimoku, etc.) ---
        result = self._add_trend_indicators(result)

        # --- Momentum indicators (RSI, Stochastic, Williams %R, etc.) ---
        result = self._add_momentum_indicators(result)

        # --- Volatility indicators (Bollinger, ATR, Keltner, etc.) ---
        result = self._add_volatility_indicators(result)

        # --- Volume indicators (OBV, VPT, CMF, MFI, etc.) ---
        result = self._add_volume_indicators(result)

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
        df["oc_ratio"] = (close - opn) / opn
        df["hl_ratio"] = (high - low) / (high + low + 1e-10)

        # Log returns
        df["log_ret_1d"] = np.log(close / close.shift(1))

        return df

    # ------------------------------------------------------------------
    # Trend indicators
    # ------------------------------------------------------------------

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend-based technical indicators via ``ta.trend``."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        opn = df["Open"]

        # --- Simple Moving Averages (SMA) ---
        for window in [5, 10, 20, 60]:
            df[f"sma_{window}"] = SMAIndicator(
                close=close, window=window
            ).sma_indicator()

        # --- Exponential Moving Averages (EMA) ---
        df["ema_12"] = EMAIndicator(close=close, window=12).ema_indicator()
        df["ema_26"] = EMAIndicator(close=close, window=26).ema_indicator()

        # --- Weighted Moving Average (WMA) ---
        df["wma_20"] = WMAIndicator(close=close, window=20).wma()

        # --- MACD (Moving Average Convergence Divergence) ---
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # --- SMA cross signals ---
        df["sma_5_20_cross"] = df["sma_5"] - df["sma_20"]
        df["sma_10_60_cross"] = df["sma_10"] - df["sma_60"]

        # --- ADX (Average Directional Index) with +DI / -DI ---
        adx = ADXIndicator(high=high, low=low, close=close, window=14)
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()

        # --- CCI (Commodity Channel Index) ---
        df["cci"] = CCIIndicator(
            high=high, low=low, close=close, window=20
        ).cci()

        # --- Ichimoku Cloud ---
        ichimoku = IchimokuIndicator(
            high=high, low=low, window1=9, window2=26, window3=52
        )
        df["ichimoku_a"] = ichimoku.ichimoku_a()
        df["ichimoku_b"] = ichimoku.ichimoku_b()
        df["ichimoku_base"] = ichimoku.ichimoku_base_line()
        df["ichimoku_conversion"] = ichimoku.ichimoku_conversion_line()

        # --- Parabolic SAR ---
        df["psar"] = PSARIndicator(
            high=high, low=low, close=close, step=0.02, max_step=0.2
        ).psar()
        # SAR-based direction signal: 1 if price above SAR (bullish)
        df["psar_signal"] = (close > df["psar"]).astype(int)

        # --- TRIX (Triple Exponential Moving Average) ---
        df["trix"] = TRIXIndicator(close=close, window=20).trix()

        # --- Vortex Indicator ---
        vortex = VortexIndicator(high=high, low=low, close=close, window=14)
        df["vortex_pos"] = vortex.vortex_indicator_pos()
        df["vortex_neg"] = vortex.vortex_indicator_neg()
        df["vortex_diff"] = df["vortex_pos"] - df["vortex_neg"]

        # --- Aroon Indicator ---
        aroon = AroonIndicator(high=high, low=low, window=25)
        df["aroon_up"] = aroon.aroon_up()
        df["aroon_down"] = aroon.aroon_down()
        df["aroon_indicator"] = aroon.aroon_indicator()

        # --- KST (Know Sure Thing) ---
        kst = KSTIndicator(close=close)
        df["kst"] = kst.kst()
        df["kst_signal"] = kst.kst_sig()
        df["kst_diff"] = kst.kst_diff()

        # --- Schaff Trend Cycle (STC) ---
        df["stc"] = stc(close, window_fast=23, window_slow=50, cycle=10)

        # --- Percentage Price Oscillator (PPO) ---
        ppo = PercentagePriceOscillator(close=close)
        df["ppo"] = ppo.ppo()
        df["ppo_signal"] = ppo.ppo_signal()
        df["ppo_hist"] = ppo.ppo_hist()

        return df

    # ------------------------------------------------------------------
    # Momentum indicators
    # ------------------------------------------------------------------

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum-based technical indicators via ``ta.momentum``."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # --- RSI (Wilder-smoothed Relative Strength Index) ---
        df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
        df["rsi_7"] = RSIIndicator(close=close, window=7).rsi()
        df["rsi_21"] = RSIIndicator(close=close, window=21).rsi()

        # --- Stochastic Oscillator (%K, %D) ---
        stoch = StochasticOscillator(
            high=high, low=low, close=close, window=14, smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # --- Stochastic RSI ---
        stochrsi = StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3)
        df["stochrsi_k"] = stochrsi.stochrsi_k()
        df["stochrsi_d"] = stochrsi.stochrsi_d()

        # --- Williams %R ---
        df["willr"] = WilliamsRIndicator(
            high=high, low=low, close=close, lbp=14
        ).williams_r()

        # --- ROC (Rate of Change) ---
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = ROCIndicator(close=close, window=period).roc()

        # --- Ultimate Oscillator ---
        df["ultimate_osc"] = UltimateOscillator(
            high=high, low=low, close=close
        ).ultimate_oscillator()

        # --- TSI (True Strength Index) ---
        df["tsi"] = TSIIndicator(close=close).tsi()

        # --- KAMA (Kaufman Adaptive Moving Average) ---
        df["kama_10"] = KAMAIndicator(close=close, window=10).kama()
        df["kama_20"] = KAMAIndicator(close=close, window=20).kama()

        # --- Awesome Oscillator ---
        df["awesome_osc"] = AwesomeOscillatorIndicator(
            high=high, low=low
        ).awesome_oscillator()

        return df

    # ------------------------------------------------------------------
    # Volatility indicators
    # ------------------------------------------------------------------

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility-based technical indicators via ``ta.volatility``."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # --- Bollinger Bands ---
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mavg"] = bb.bollinger_mavg()
        df["bb_pct"] = bb.bollinger_pband()
        df["bb_width"] = bb.bollinger_wband()

        # --- ATR (Average True Range, Wilder smoothed) ---
        atr_14 = AverageTrueRange(high=high, low=low, close=close, window=14)
        df["atr_14"] = atr_14.average_true_range()
        atr_7 = AverageTrueRange(high=high, low=low, close=close, window=7)
        df["atr_7"] = atr_7.average_true_range()
        # Normalized ATR (as percentage of closing price)
        df["atr_pct_14"] = df["atr_14"] / close

        # --- Keltner Channel ---
        keltner = KeltnerChannel(
            high=high, low=low, close=close, window=20, window_atr=10
        )
        df["keltner_upper"] = keltner.keltner_channel_hband()
        df["keltner_lower"] = keltner.keltner_channel_lband()
        df["keltner_pct"] = keltner.keltner_channel_pband()
        df["keltner_width"] = keltner.keltner_channel_wband()

        # --- Donchian Channel ---
        donchian = DonchianChannel(high=high, low=low, close=close, window=20)
        df["donchian_upper"] = donchian.donchian_channel_hband()
        df["donchian_lower"] = donchian.donchian_channel_lband()
        df["donchian_pct"] = donchian.donchian_channel_pband()
        df["donchian_width"] = donchian.donchian_channel_wband()

        # --- Ulcer Index ---
        df["ulcer_index"] = UlcerIndex(close=close, window=14).ulcer_index()

        return df

    # ------------------------------------------------------------------
    # Volume indicators
    # ------------------------------------------------------------------

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based technical indicators via ``ta.volume``."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        vol = df["Volume"].astype(float)

        # Basic volume stats (kept from original)
        df["vol_change_1d"] = vol.pct_change(1)
        df["vol_ma_5"] = vol.rolling(5).mean()
        df["vol_ma_20"] = vol.rolling(20).mean()
        df["vol_ratio"] = vol / (df["vol_ma_20"] + 1e-10)

        # --- On-Balance Volume (OBV) ---
        df["obv"] = OnBalanceVolumeIndicator(
            close=close, volume=vol
        ).on_balance_volume()

        # --- Volume Price Trend (VPT) ---
        df["vpt"] = VolumePriceTrendIndicator(
            close=close, volume=vol
        ).volume_price_trend()

        # --- Chaikin Money Flow (CMF) ---
        df["cmf"] = ChaikinMoneyFlowIndicator(
            high=high, low=low, close=close, volume=vol, window=20
        ).chaikin_money_flow()

        # --- Money Flow Index (MFI) ---
        df["mfi"] = MFIIndicator(
            high=high, low=low, close=close, volume=vol, window=14
        ).money_flow_index()

        # --- Force Index ---
        df["force_index"] = ForceIndexIndicator(
            close=close, volume=vol, window=13
        ).force_index()

        # --- Ease of Movement ---
        df["ease_of_movement"] = EaseOfMovementIndicator(
            high=high, low=low, volume=vol, window=14
        ).ease_of_movement()

        # --- Volume Weighted Average Price (VWAP) ---
        df["vwap"] = VolumeWeightedAveragePrice(
            high=high, low=low, close=close, volume=vol
        ).volume_weighted_average_price()

        # --- Accumulation/Distribution Index ---
        df["adi"] = AccDistIndexIndicator(
            high=high, low=low, close=close, volume=vol
        ).acc_dist_index()

        # --- Negative Volume Index ---
        df["nvi"] = NegativeVolumeIndexIndicator(
            close=close, volume=vol
        ).negative_volume_index()

        return df

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lagged features for key indicators."""
        lag_cols = [
            "ret_1d", "rsi_14", "macd_hist", "bb_pct", "vol_ratio",
            "atr_pct_14", "cci",
        ]
        for col in lag_cols:
            if col in df.columns:
                for lag in [1, 2, 3]:
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)
        return df

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Target label
    # ------------------------------------------------------------------

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct the prediction target."""
        close = df["Close"]
        future_return = close.shift(-self.forward_period) / close - 1

        if self.label_method == "return_sign":
            df["target"] = (future_return > 0).astype(int)
        elif self.label_method == "return":
            df["target"] = future_return
        elif self.label_method == "excess_return":
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
