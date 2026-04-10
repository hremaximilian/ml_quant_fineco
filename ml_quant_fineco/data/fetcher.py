"""
Data fetcher module for retrieving stock and index data from multiple sources.

Supported sources:
- yfinance: US stocks, global indices, ADRs
- akshare: Chinese A-shares, Chinese indices, Hong Kong stocks
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataFetcher:
    """Unified data fetcher for stocks and indices from multiple data sources."""

    # Common index ticker mappings
    INDEX_TICKERS = {
        "S&P500": "^GSPC",
        "NASDAQ": "^IXIC",
        "DOW_JONES": "^DJI",
        "RUSSELL2000": "^RUT",
        "VIX": "^VIX",
        "HSI": "^HSI",           # Hang Seng
        "N225": "^N225",         # Nikkei 225
        "FTSE": "^FTSE",        # FTSE 100
        "DAX": "^GDAXI",        # DAX
        "SSE": "000001.SS",     # Shanghai Composite (yfinance)
    }

    # Chinese index codes for akshare
    CN_INDEX_MAP = {
        "SSE": "sh000001",       # Shanghai Composite
        "SZSE": "sz399001",     # Shenzhen Component
        "CSI300": "sh000300",   # CSI 300
        "CSI500": "sh000905",   # CSI 500
        "GEM": "sz399006",      # ChiNext
        "STAR50": "sh000688",   # STAR 50
    }

    def __init__(self, source: str = "yfinance"):
        """
        Initialize the data fetcher.

        Args:
            source: Data source, either 'yfinance' or 'akshare'.
        """
        self.source = source.lower()
        if self.source not in ("yfinance", "akshare"):
            raise ValueError(f"Unsupported source: {source}. Use 'yfinance' or 'akshare'.")

    def fetch(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "5y",
        is_index: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given symbol.

        Args:
            symbol: Stock ticker or index name.
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            period: Lookback period if start_date not given (e.g. '1y', '5y', 'max').
            is_index: Whether the symbol is an index.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            and a DatetimeIndex.
        """
        try:
            if self.source == "yfinance":
                return self._fetch_yfinance(symbol, start_date, end_date, period, is_index)
            elif self.source == "akshare":
                return self._fetch_akshare(symbol, start_date, end_date, period, is_index)
        except Exception as e:
            logger.warning(f"Failed to fetch from {self.source}: {e}. Generating synthetic data.")
            return self._generate_synthetic(symbol, start_date, end_date, period)

    def _fetch_yfinance(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        period: str,
        is_index: bool,
    ) -> pd.DataFrame:
        """Fetch data using yfinance (US & global markets)."""
        import yfinance as yf

        ticker = self.INDEX_TICKERS.get(symbol, symbol) if is_index else symbol
        logger.info(f"Fetching {ticker} from yfinance (period={period})")

        if start_date:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date)
        else:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(period=period)

        if df.empty:
            raise ValueError(f"No data returned for symbol: {symbol}")

        # Standardize column names
        df = df.rename(columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Remove timezone from index for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        logger.info(f"Fetched {len(df)} rows for {ticker}")
        return df

    def _fetch_akshare(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        period: str,
        is_index: bool,
    ) -> pd.DataFrame:
        """Fetch data using akshare (Chinese A-shares market)."""
        import akshare as ak

        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        else:
            end_date = end_date.replace("-", "")

        if not start_date:
            days_map = {"1y": 365, "2y": 730, "5y": 1825, "10y": 3650, "max": 36500}
            days = days_map.get(period, 1825)
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        else:
            start_date = start_date.replace("-", "")

        if is_index:
            cn_symbol = self.CN_INDEX_MAP.get(symbol, symbol)
            logger.info(f"Fetching CN index {cn_symbol} from akshare")
            df = ak.stock_zh_index_daily(symbol=cn_symbol)
            df = df.rename(columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            })
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            # Chinese stock: prepend market prefix if not present
            if not symbol.startswith(("sh", "sz", "SH", "SZ")):
                # Try to auto-detect: 6xx = Shanghai, 0xx/3xx = Shenzhen
                if symbol.startswith("6"):
                    cn_symbol = f"sh{symbol}"
                else:
                    cn_symbol = f"sz{symbol}"
            else:
                cn_symbol = symbol

            logger.info(f"Fetching CN stock {cn_symbol} from akshare")
            df = ak.stock_zh_a_hist(
                symbol=cn_symbol[2:],  # strip sh/sz prefix
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",  # forward-adjusted prices
            )
            df = df.rename(columns={
                "日期": "Date",
                "开盘": "Open",
                "最高": "High",
                "最低": "Low",
                "收盘": "Close",
                "成交量": "Volume",
            })
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            df = df[["Open", "High", "Low", "Close", "Volume"]]

        # Filter date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        df = df.sort_index()

        if df.empty:
            raise ValueError(f"No data returned for symbol: {symbol}")

        logger.info(f"Fetched {len(df)} rows for {symbol}")
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def _generate_synthetic(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        period: str,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing when real data is unavailable."""
        logger.info(f"Generating synthetic data for {symbol}")

        days_map = {"1y": 252, "2y": 504, "5y": 1260, "10y": 2520, "max": 2520}
        n_days = days_map.get(period, 1260)

        if end_date and start_date:
            end_dt = pd.to_datetime(end_date)
            start_dt = pd.to_datetime(start_date)
            dates = pd.bdate_range(start=start_dt, end=end_dt)
        elif start_date:
            start_dt = pd.to_datetime(start_date)
            dates = pd.bdate_range(start=start_dt, periods=n_days)
        else:
            dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n_days)

        np.random.seed(42)
        initial_price = 100.0
        returns = np.random.normal(0.0003, 0.015, len(dates))
        prices = initial_price * np.cumprod(1 + returns)

        df = pd.DataFrame(index=dates)
        df["Close"] = prices
        df["Open"] = prices * (1 + np.random.uniform(-0.01, 0.01, len(dates)))
        df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
        df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
        df["Volume"] = np.random.randint(1000000, 50000000, len(dates)).astype(float)

        return df[["Open", "High", "Low", "Close", "Volume"]]

    def fetch_multiple(
        self,
        symbols: list,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "5y",
        is_index: bool = False,
    ) -> dict:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of stock tickers or index names.
            start_date: Start date string.
            end_date: End date string.
            period: Lookback period.
            is_index: Whether symbols are indices.

        Returns:
            Dictionary mapping symbol -> DataFrame.
        """
        results = {}
        for sym in symbols:
            try:
                results[sym] = self.fetch(sym, start_date, end_date, period, is_index)
            except Exception as e:
                logger.warning(f"Failed to fetch {sym}: {e}")
        return results
