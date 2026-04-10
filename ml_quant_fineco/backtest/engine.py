"""
Backtesting engine with benchmark comparison.

Simulates a trading strategy based on model signals and computes
comprehensive performance metrics against a buy-and-hold benchmark.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Event-driven backtester for ML-based trading strategies.

    Supports:
    - Long-only and long-short strategies
    - Configurable transaction costs
    - Buy-and-hold benchmark comparison
    - Multiple performance metrics (Sharpe, Sortino, max drawdown, etc.)
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        strategy: str = "long_only",
        position_size: float = 1.0,
        risk_free_rate: float = 0.03,
    ):
        """
        Args:
            initial_capital: Starting portfolio value.
            commission_rate: Commission per trade as fraction of trade value.
            slippage_rate: Slippage per trade as fraction of price.
            strategy: 'long_only' or 'long_short'.
            position_size: Fraction of capital to allocate per trade (0-1).
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.strategy = strategy
        self.position_size = position_size
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        prices: pd.Series,
        signals: pd.Series,
        probabilities: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Run the backtest.

        Args:
            prices: Series of closing prices, indexed by date.
            signals: Series of trading signals (1=buy, 0=sell/flat, -1=short).
                     Must align with prices index.
            probabilities: Optional series of predicted probabilities for
                           position sizing (confidence-weighted).

        Returns:
            DataFrame with columns:
                price, signal, position, cash, holdings, portfolio_value,
                daily_return, strategy_return, benchmark_return
        """
        # Align indices
        common_idx = prices.index.intersection(signals.index)
        prices = prices.loc[common_idx]
        signals = signals.loc[common_idx]

        if probabilities is not None:
            probabilities = probabilities.loc[
                probabilities.index.isin(common_idx)
            ]

        n = len(prices)
        records = []

        cash = self.initial_capital
        position = 0  # shares held
        portfolio_value = self.initial_capital

        for i in range(n):
            date = prices.index[i]
            price = prices.iloc[i]
            signal = int(signals.iloc[i])

            # Determine target position
            if self.strategy == "long_only":
                target_position = signal  # 1 or 0
            elif self.strategy == "long_short":
                target_position = signal  # 1, 0, or -1
            else:
                target_position = signal

            # Confidence-weighted sizing
            size_mult = 1.0
            if probabilities is not None and date in probabilities.index:
                prob = probabilities.loc[date]
                size_mult = min(abs(prob - 0.5) * 2, 1.0)  # scale by confidence
                size_mult = max(size_mult, 0.2)

            # Execute trades
            shares_to_hold = 0
            if target_position == 1:
                alloc = self.initial_capital * self.position_size * size_mult
                shares_to_hold = int(alloc / price)
            elif target_position == -1 and self.strategy == "long_short":
                shares_to_hold = -int(self.initial_capital * self.position_size * size_mult / price)

            trade_shares = shares_to_hold - position

            if trade_shares != 0:
                trade_value = abs(trade_shares) * price
                commission = trade_value * self.commission_rate
                slippage = trade_value * self.slippage_rate
                cost = commission + slippage

                if trade_shares > 0:  # buying
                    cash -= trade_shares * price + cost
                else:  # selling
                    cash -= trade_shares * price - cost

                position = shares_to_hold

            holdings = position * price
            portfolio_value = cash + holdings

            # Returns
            if i == 0:
                daily_ret = 0.0
                strat_ret = 0.0
                bench_ret = 0.0
            else:
                prev_pv = records[-1]["portfolio_value"]
                prev_price = records[-1]["price"]
                daily_ret = (portfolio_value - prev_pv) / prev_pv if prev_pv != 0 else 0
                bench_ret = (price - prev_price) / prev_price if prev_price != 0 else 0
                strat_ret = daily_ret

            records.append({
                "date": date,
                "price": price,
                "signal": signal,
                "position": position,
                "cash": cash,
                "holdings": holdings,
                "portfolio_value": portfolio_value,
                "daily_return": daily_ret,
                "strategy_return": strat_ret,
                "benchmark_return": bench_ret,
            })

        result = pd.DataFrame(records).set_index("date")
        self._result = result
        return result

    def compute_metrics(self, result: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Compute comprehensive performance metrics.

        Args:
            result: Backtest result DataFrame. If None, uses the last run result.

        Returns:
            Dictionary of metric name -> value.
        """
        if result is None:
            result = self._result

        strat_ret = result["strategy_return"]
        bench_ret = result["benchmark_return"]
        pv = result["portfolio_value"]

        metrics = {}

        # --- Basic ---
        metrics["total_return"] = (pv.iloc[-1] / self.initial_capital) - 1
        metrics["benchmark_return"] = (result["price"].iloc[-1] / result["price"].iloc[0]) - 1
        metrics["total_days"] = len(result)

        # Annualized return (assume 252 trading days)
        years = len(result) / 252
        if years > 0:
            metrics["annualized_return"] = (1 + metrics["total_return"]) ** (1 / years) - 1
            metrics["annualized_benchmark"] = (1 + metrics["benchmark_return"]) ** (1 / years) - 1

        # --- Risk ---
        metrics["annualized_volatility"] = strat_ret.std() * np.sqrt(252)
        metrics["benchmark_volatility"] = bench_ret.std() * np.sqrt(252)

        # Max Drawdown
        cummax = pv.cummax()
        drawdown = (pv - cummax) / cummax
        metrics["max_drawdown"] = drawdown.min()

        # Benchmark max drawdown
        bench_cummax = result["price"].cummax()
        bench_dd = (result["price"] - bench_cummax) / bench_cummax
        metrics["benchmark_max_drawdown"] = bench_dd.min()

        # --- Risk-Adjusted ---
        daily_rf = self.risk_free_rate / 252
        excess_ret = strat_ret - daily_rf
        if strat_ret.std() > 0:
            metrics["sharpe_ratio"] = (excess_ret.mean() / strat_ret.std()) * np.sqrt(252)
        else:
            metrics["sharpe_ratio"] = 0.0

        # Sortino ratio (downside deviation)
        downside = strat_ret[strat_ret < 0]
        downside_std = downside.std() if len(downside) > 0 else 1e-10
        metrics["sortino_ratio"] = (excess_ret.mean() / downside_std) * np.sqrt(252)

        # Calmar ratio
        if abs(metrics["max_drawdown"]) > 1e-10:
            metrics["calmar_ratio"] = metrics["annualized_return"] / abs(metrics["max_drawdown"])
        else:
            metrics["calmar_ratio"] = np.inf

        # --- Win/Loss ---
        winning_days = (strat_ret > 0).sum()
        total_trading_days = (strat_ret != 0).sum()
        metrics["win_rate"] = winning_days / max(total_trading_days, 1)

        # Alpha / Beta (vs benchmark)
        cov_matrix = np.cov(strat_ret, bench_ret)
        if cov_matrix[1, 1] > 0:
            metrics["beta"] = cov_matrix[0, 1] / cov_matrix[1, 1]
            metrics["alpha"] = metrics["annualized_return"] - (
                self.risk_free_rate + metrics["beta"] * (metrics["annualized_benchmark"] - self.risk_free_rate)
            )
        else:
            metrics["beta"] = 0.0
            metrics["alpha"] = 0.0

        # Information ratio
        active_ret = strat_ret - bench_ret
        tracking_error = active_ret.std()
        if tracking_error > 0:
            metrics["information_ratio"] = active_ret.mean() / tracking_error * np.sqrt(252)
        else:
            metrics["information_ratio"] = 0.0

        return metrics

    def compare_benchmark(
        self,
        result: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate a side-by-side comparison table of strategy vs benchmark.

        Returns:
            DataFrame with Metric, Strategy, Benchmark columns.
        """
        metrics = self.compute_metrics(result)

        comparison = pd.DataFrame({
            "Metric": [
                "Total Return",
                "Annualized Return",
                "Annualized Volatility",
                "Max Drawdown",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "Win Rate",
                "Beta",
                "Alpha",
                "Information Ratio",
            ],
            "Strategy": [
                f"{metrics['total_return']:.2%}",
                f"{metrics['annualized_return']:.2%}",
                f"{metrics['annualized_volatility']:.2%}",
                f"{metrics['max_drawdown']:.2%}",
                f"{metrics['sharpe_ratio']:.3f}",
                f"{metrics['sortino_ratio']:.3f}",
                f"{metrics['calmar_ratio']:.3f}",
                f"{metrics['win_rate']:.2%}",
                f"{metrics['beta']:.3f}",
                f"{metrics['alpha']:.2%}",
                f"{metrics['information_ratio']:.3f}",
            ],
            "Benchmark": [
                f"{metrics['benchmark_return']:.2%}",
                f"{metrics['annualized_benchmark']:.2%}",
                f"{metrics['benchmark_volatility']:.2%}",
                f"{metrics['benchmark_max_drawdown']:.2%}",
                "-",
                "-",
                "-",
                "-",
                "1.000",
                "0.00%",
                "-",
            ],
        })

        return comparison
